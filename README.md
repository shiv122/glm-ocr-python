# GLM OCR Service

A **FastAPI** service in front of the **glmocr SDK** — PP-DocLayoutV3 layout
detection + GLM-OCR recognition. The OCR model is served by a local **vLLM**
server; the layout model runs in the FastAPI process. Clients send an **image
URL** — not base64 — and the service fetches the bytes itself, runs the
pipeline, and returns recognized text **with per-region bounding boxes**.

Built to pair with the detector backend: detection uploads each frame to
DigitalOcean Spaces and sends that public URL here.

## API

### `POST /parse` — text + bounding boxes
```json
{ "image_url": "https://df-detection.blr1.digitaloceanspaces.com/frames/frame_x.jpg" }
```
Response:
```json
{
  "text": "<markdown of the whole page>",
  "blocks": [
    { "index": 0, "label": "table", "content": "<table>…</table>", "bbox_2d": [33, 317, 931, 887] },
    { "index": 1, "label": "text",  "content": "…",                 "bbox_2d": [68, 128, 445, 185] }
  ],
  "model": "glm-ocr",
  "timing_ms": { "download_ms": 120, "inference_ms": 4300 }
}
```
- `bbox_2d` is `[x1, y1, x2, y2]` in **absolute pixels** of the original image
  (top-left, bottom-right rectangle — not normalized, not a polygon).
- `index` is the **reading order**; `label` is the region category
  (`text` / `table` / `formula` / `figure` / …); `content` is the recognized
  text (HTML for tables, LaTeX for formulas).
- These are **layout regions**, not per-word boxes.

### `POST /ocr` — back-compat, text only
Same request body (a `prompt` field is accepted but **ignored** — the pipeline
uses its own per-region prompts). Returns `{ text, model, timing_ms }`.

### `GET /health`
```json
{ "status": "ok", "models": ["glm-ocr"] }   // queries the local vLLM /v1/models
```

Errors return `{ "error": "..." }` with a 4xx/5xx status.

## Run

Requires a GPU host (`nvidia-container-toolkit`). vLLM downloads the GLM-OCR
weights on first boot — mount a persistent HF cache to avoid re-downloading.

```bash
docker build -t glm-ocr-service .
docker run --gpus all -p 8080:8080 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  glm-ocr-service

# smoke test (cold start is slow — vLLM loads the model + layout weights first)
curl localhost:8080/health
curl -X POST localhost:8080/parse \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://.../frame.jpg"}'
```

> **Dependency note:** `glmocr[selfhosted]` wants `transformers>=5.3` and
> `torch>=2.10`. If pip reports a conflict with the torch baked into the vLLM
> base image, pin compatible versions in `requirements.txt` so glmocr doesn't
> upgrade torch out from under vLLM.

## Wiring into the detector

Point the detector backend at this service and turn on the boxes path:
```
GLM_OCR_HOST=http://<this-service-host>:8080
GLM_OCR_MODEL=glm-ocr
GLM_OCR_BOXES=1
```
With `GLM_OCR_BOXES=1` the backend POSTs `{image_url}` to `/parse` and threads
`blocks` (the boxes) into the OCR result (see `glm_ocr_client.py`). Left unset,
it uses the text-only `/ocr` path as before.

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `PORT` | `8080` | FastAPI listen port |
| `VLLM_HOST` / `VLLM_PORT` | `127.0.0.1` / `8000` | internal vLLM server |
| `GLM_OCR_MODEL` | `glm-ocr` | vLLM `--served-model-name` |
| `VLLM_MODEL_PATH` | `zai-org/GLM-OCR` | HF repo (or local path) vLLM loads |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.45` | GLM-OCR is ~0.9B; low value leaves room for the layout model |
| `GLMOCR_LAYOUT_DEVICE` | `cuda:0` | device for PP-DocLayoutV3 |
| `DOWNLOAD_TIMEOUT_SECONDS` | `30` | image fetch timeout |
| `MAX_IMAGE_BYTES` | `26214400` | 25 MB fetch cap |
