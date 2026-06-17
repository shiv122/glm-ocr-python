# GLM OCR service: FastAPI (URL-based API) in front of the glmocr SDK
# (PP-DocLayoutV3 layout detection + GLM-OCR recognition). The OCR model is
# served by a local vLLM server; the layout model runs in the FastAPI process.
# Needs a GPU at runtime (nvidia-container-toolkit / a GPU host).
#
# The base image bundles vLLM + a CUDA torch. glmocr[selfhosted] then adds the
# layout stack (torch/transformers/opencv). If pip reports a torch/transformers
# conflict with the base image, pin compatible versions here rather than letting
# glmocr upgrade torch out from under vLLM.
FROM vllm/vllm-openai:v0.19.0-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8000 \
    GLM_OCR_MODEL=glm-ocr \
    GLMOCR_LAYOUT_DEVICE=cuda:0 \
    HF_HOME=/root/.cache/huggingface \
    PORT=8080

WORKDIR /app

# The vLLM base image ships an ENTRYPOINT that launches the API server; clear it
# so our entrypoint.sh can run both vLLM and FastAPI.
ENTRYPOINT []

# Service deps + the glmocr layout pipeline. The base image already has vLLM +
# torch; install on top.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the PP-DocLayoutV3 layout weights so the first request doesn't pay
# the download. The GLM-OCR model itself is pulled by vLLM at first boot — mount
# a persistent HF cache volume at $HF_HOME to skip that cold-start download.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('PaddlePaddle/PP-DocLayoutV3_safetensors')" \
    || echo "WARN: layout weight prefetch failed; will download at runtime"

COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
