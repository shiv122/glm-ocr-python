"""
GLM OCR service — layout-aware OCR that returns text AND bounding boxes.

FastAPI in front of the **glmocr SDK** (PP-DocLayoutV3 layout detection + GLM-OCR
recognition). The OCR model is served by a local **vLLM** server (OpenAI-
compatible) on 127.0.0.1:VLLM_PORT; the layout model (PP-DocLayoutV3, torch +
transformers, NOT paddle) runs in-process on the GPU. Clients send an IMAGE URL
— not bytes; we fetch the bytes ourselves, run the pipeline, and return the
recognized regions with pixel bounding boxes.

    POST /parse  { image_url }
              -> { text, blocks: [ { index, label, content,
                                     bbox_2d: [x1, y1, x2, y2] } ],
                   model, timing_ms: { download_ms, inference_ms } }

    POST /ocr    { image_url, prompt?, ... }      # back-compat: text only.
              -> { text, model, timing_ms }       # prompt is IGNORED — the SDK
                                                  # drives its own per-region
                                                  # prompts.
    GET  /health

The boxes come from PP-DocLayoutV3 and are LAYOUT REGIONS (paragraph / table /
figure / formula …), not per-word boxes. `bbox_2d` is [x1, y1, x2, y2] in
absolute pixels of the original image; `index` is the reading order; `label` is
the region category; `content` is the recognized text (HTML for tables, LaTeX
for formulas).
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# vLLM serves the GLM-OCR model (OpenAI-compatible) on this host:port. The SDK
# talks to it as its "ocr_api" backend in selfhosted mode.
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
DEFAULT_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")  # must match --served-model-name
# Device the PP-DocLayoutV3 layout model runs on (shares the GPU with vLLM).
LAYOUT_DEVICE = os.getenv("GLMOCR_LAYOUT_DEVICE", "cuda:0")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "30"))

app = FastAPI(title="GLM OCR Service", version="2.0.0")

# The glmocr pipeline is expensive to construct (loads the layout model onto the
# GPU), so we build it once, lazily, and guard construction with a lock.
_PARSER = None
_PARSER_LOCK = threading.Lock()


def _get_parser():
    global _PARSER
    if _PARSER is None:
        with _PARSER_LOCK:
            if _PARSER is None:
                from glmocr import GlmOcr

                _PARSER = GlmOcr(
                    mode="selfhosted",      # use the local vLLM, not the cloud MaaS
                    ocr_api_host=VLLM_HOST,
                    ocr_api_port=VLLM_PORT,
                    model=DEFAULT_MODEL,
                    layout_device=LAYOUT_DEVICE,
                )
    return _PARSER


class OcrRequest(BaseModel):
    image_url: str
    # Accepted for backward compatibility with the old text-only API but
    # ignored: the glmocr pipeline uses its own per-region prompts.
    prompt: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@app.get("/health")
def health():
    try:
        r = httpx.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models", timeout=5)
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", [])]
        return {"status": "ok", "models": models}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "error": str(e)}
        )


def _fetch_image(url: str) -> tuple[bytes | None, JSONResponse | None]:
    try:
        with httpx.stream(
            "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
        ) as resp:
            if resp.status_code != 200:
                return None, JSONResponse(
                    status_code=400,
                    content={"error": f"image fetch returned HTTP {resp.status_code}"},
                )
            data = bytearray()
            for chunk in resp.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_IMAGE_BYTES:
                    return None, JSONResponse(
                        status_code=413,
                        content={"error": f"image exceeds {MAX_IMAGE_BYTES} bytes"},
                    )
            return bytes(data), None
    except Exception as e:  # noqa: BLE001
        return None, JSONResponse(
            status_code=400, content={"error": f"failed to fetch image: {e}"}
        )


def _run_pipeline(image_bytes: bytes) -> tuple[str, list, int]:
    """Run the glmocr pipeline on raw image bytes.

    Returns (markdown_text, blocks, inference_ms). `blocks` is the SDK's
    json_result: a list of { index, label, content, bbox_2d } regions.
    """
    parser = _get_parser()
    # The SDK loads pages from a path (PDF or image); write the fetched bytes to
    # a tempfile. Pillow sniffs the real format, so the suffix is cosmetic.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        t0 = time.monotonic()
        result = parser.parse(tmp.name, save_layout_visualization=False)
        inference_ms = int((time.monotonic() - t0) * 1000)

    blocks = list(result.json_result or [])
    text = result.markdown_result or "\n".join(
        str(b.get("content", "")) for b in blocks
    )
    return text, blocks, inference_ms


@app.post("/parse")
def parse(req: OcrRequest):
    """Layout-aware parse: returns recognized text AND per-region bounding boxes."""
    if not req.image_url:
        return JSONResponse(status_code=422, content={"error": "image_url is required"})

    t0 = time.monotonic()
    image_bytes, err = _fetch_image(req.image_url)
    if err is not None:
        return err
    download_ms = int((time.monotonic() - t0) * 1000)

    try:
        text, blocks, inference_ms = _run_pipeline(image_bytes)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=502, content={"error": f"GLM OCR pipeline failed: {e}"}
        )

    return {
        "text": text,
        "blocks": blocks,
        "model": req.model or DEFAULT_MODEL,
        "timing_ms": {"download_ms": download_ms, "inference_ms": inference_ms},
    }


@app.post("/ocr")
def ocr(req: OcrRequest):
    """Back-compat text-only endpoint. Runs the same pipeline, drops the boxes."""
    out = parse(req)
    if isinstance(out, JSONResponse):
        return out
    return {
        "text": out["text"],
        "model": out["model"],
        "timing_ms": out["timing_ms"],
    }
