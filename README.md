# GLM OCR Service

A small **FastAPI** service in front of **Ollama** (serving the `glm-ocr` vision
model, baked into the image). Clients send an **image URL** — not base64 — and
the service fetches the bytes itself, runs OCR, and returns the extracted text.

Built to pair with the detector backend: detection uploads each frame to
DigitalOcean Spaces and sends that public URL here.

## API

### `POST /ocr`
```json
{
  "image_url": "https://df-detection.blr1.digitaloceanspaces.com/frames/frame_x.jpg",
  "prompt": "Extract all visible text from this image...",
  "model": "glm-ocr",          // optional, defaults to GLM_OCR_MODEL
  "max_tokens": 2048,           // optional
  "temperature": 0.0            // optional
}
```
Response:
```json
{
  "text": "<extracted text>",
  "model": "glm-ocr",
  "timing_ms": { "download_ms": 120, "inference_ms": 4300 }
}
```
Errors return `{ "error": "..." }` with a 4xx/5xx status.

### `GET /health`
```json
{ "status": "ok", "models": ["glm-ocr:latest"] }
```

## Run

Requires a GPU host (`nvidia-container-toolkit`). The image bakes in `glm-ocr`
(~several GB), so the first build is slow but cold starts are fast.

```bash
docker build -t glm-ocr-service .
docker run --gpus all -p 8080:8080 glm-ocr-service

# smoke test
curl localhost:8080/health
curl -X POST localhost:8080/ocr \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://.../frame.jpg","prompt":"Extract all text."}'
```

## Wiring into the detector

Point the detector backend at this service:
```
GLM_OCR_HOST=http://<this-service-host>:8080
GLM_OCR_MODEL=glm-ocr
```
The detector POSTs `{image_url, prompt}` to `/ocr` (see `glm_ocr_client.py`).

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `PORT` | `8080` | FastAPI listen port |
| `GLM_OCR_MODEL` | `glm-ocr` | Ollama model name |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | internal Ollama |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | per-inference ceiling |
| `DOWNLOAD_TIMEOUT_SECONDS` | `30` | image fetch timeout |
| `MAX_IMAGE_BYTES` | `26214400` | 25 MB fetch cap |
