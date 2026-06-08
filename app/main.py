"""
GLM OCR service.

A small FastAPI server in front of Ollama (which serves the `glm-ocr` vision
model, baked into the image). Clients send an IMAGE URL — not base64 — and this
service fetches the bytes itself, runs OCR, and returns the extracted text.

    POST /ocr   { image_url, prompt, model?, max_tokens?, temperature? }
             -> { text, model, timing_ms: { download_ms, inference_ms } }
    GET  /health
"""

from __future__ import annotations

import base64
import os
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "30"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

app = FastAPI(title="GLM OCR Service", version="1.0.0")


class OcrRequest(BaseModel):
    image_url: str
    prompt: str
    model: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0


@app.get("/health")
def health():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
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


@app.post("/ocr")
def ocr(req: OcrRequest):
    if not req.image_url:
        return JSONResponse(status_code=422, content={"error": "image_url is required"})

    t0 = time.monotonic()
    image_bytes, err = _fetch_image(req.image_url)
    if err is not None:
        return err
    download_ms = int((time.monotonic() - t0) * 1000)

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    body = {
        "model": req.model or DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": req.prompt, "images": [image_b64]}
        ],
        "stream": False,
        "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
    }

    t1 = time.monotonic()
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=502, content={"error": f"GLM inference failed: {e}"}
        )
    inference_ms = int((time.monotonic() - t1) * 1000)

    out = r.json()
    text = (out.get("message", {}).get("content") or "").strip()
    return {
        "text": text,
        "model": body["model"],
        "timing_ms": {"download_ms": download_ms, "inference_ms": inference_ms},
    }
