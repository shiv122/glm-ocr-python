#!/usr/bin/env bash
# Start Ollama (loopback only) then the FastAPI server. Ollama holds the
# glm-ocr weights on the GPU; FastAPI is the public, URL-based API.
set -euo pipefail

echo "[entrypoint] starting ollama..."
OLLAMA_HOST=127.0.0.1:11434 ollama serve >/tmp/ollama.log 2>&1 &

echo "[entrypoint] waiting for ollama to accept connections..."
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "[entrypoint] ollama ready"
        break
    fi
    sleep 1
done

echo "[entrypoint] starting GLM OCR API on :${PORT:-8080}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
