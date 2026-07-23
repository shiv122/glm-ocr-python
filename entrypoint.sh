#!/usr/bin/env bash
# Start vLLM (serves the GLM-OCR model, loopback only) AND the FastAPI box API.
# vLLM holds the GLM-OCR weights on the GPU; the FastAPI process also loads the
# PP-DocLayoutV3 layout model onto the same GPU and is the public, URL-based API.
#
# Two deliberate operational choices:
#   * vLLM logs stream to the container's stdout (docker logs), NOT a file, so
#     download progress and crashes are visible.
#   * uvicorn starts IMMEDIATELY (not after vLLM), so :PORT/health is reachable
#     during model load and reports vLLM readiness (503 -> 200).
set -euo pipefail
export PYTHONUNBUFFERED=1

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLM_OCR_MODEL="${GLM_OCR_MODEL:-glm-ocr}"
# HF repo vLLM loads. Override with VLLM_MODEL_PATH for a local/cached path.
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-zai-org/GLM-OCR}"
# GLM-OCR is only ~0.9B params, so vLLM needs little — keep this low so the
# in-process PP-DocLayoutV3 layout model has room on the same GPU. ~0.45 of a
# 12GB card (~5.4GB) is plenty for weights + KV cache.
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
# GLM-OCR's native max context is 131072 tokens, whose KV cache (8 GiB) can't
# fit in 0.45 of a 12GB card — vLLM refuses to start. OCR requests are a few
# thousand tokens, so cap the context; 32768 fits the remaining ~2 GiB.
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# Speculative decoding (MTP) speeds GLM-OCR up but is an optimization and a
# common first-boot failure point — leave it OFF by default; set
# VLLM_SPEC_DECODING=1 once the plain path is confirmed working.
SPEC_ARGS=()
if [ -n "${VLLM_SPEC_DECODING:-}" ]; then
    SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
fi

echo "[entrypoint] starting vLLM (${VLLM_MODEL_PATH} as ${GLM_OCR_MODEL}) on ${VLLM_HOST}:${VLLM_PORT}..."
# Use the documented `vllm serve` CLI. Output inherits the container's stdout so
# it shows up in `docker logs`.
vllm serve "${VLLM_MODEL_PATH}" \
    --served-model-name "${GLM_OCR_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${SPEC_ARGS[@]}" &
VLLM_PID=$!

# Background readiness watcher — informational only, does NOT block the API.
(
    for i in $(seq 1 600); do
        if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "[entrypoint] vLLM ready on ${VLLM_HOST}:${VLLM_PORT}"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[entrypoint] ERROR: vLLM exited before becoming ready (see vLLM logs above)"
            break
        fi
        sleep 2
    done
) &

echo "[entrypoint] starting GLM OCR API on :${PORT:-8080} (/health reports vLLM readiness)..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
