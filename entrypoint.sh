#!/usr/bin/env bash
# Start vLLM (serves the GLM-OCR model, loopback only) then the FastAPI server.
# vLLM holds the GLM-OCR weights on the GPU; the FastAPI process also loads the
# PP-DocLayoutV3 layout model onto the same GPU and is the public, URL-based API.
set -euo pipefail

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLM_OCR_MODEL="${GLM_OCR_MODEL:-glm-ocr}"
# HF repo vLLM loads. Override with VLLM_MODEL_PATH for a local/cached path.
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-zai-org/GLM-OCR}"
# GLM-OCR is only ~0.9B params, so vLLM needs little — keep this low so the
# in-process PP-DocLayoutV3 layout model has room on the same GPU. ~0.45 of a
# 12GB card (~5.4GB) is plenty for weights + KV cache.
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"

echo "[entrypoint] starting vLLM (${VLLM_MODEL_PATH} as ${GLM_OCR_MODEL}) on ${VLLM_HOST}:${VLLM_PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_PATH}" \
    --served-model-name "${GLM_OCR_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    >/tmp/vllm.log 2>&1 &

echo "[entrypoint] waiting for vLLM to accept connections..."
for i in $(seq 1 600); do
    if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[entrypoint] vLLM ready"
        break
    fi
    if [ "$i" -eq 600 ]; then
        echo "[entrypoint] vLLM did not become ready in time; tail of /tmp/vllm.log:"
        tail -n 50 /tmp/vllm.log || true
        exit 1
    fi
    sleep 1
done

echo "[entrypoint] starting GLM OCR API on :${PORT:-8080}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
