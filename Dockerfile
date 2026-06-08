# GLM OCR service: FastAPI (URL-based API) in front of Ollama serving the
# glm-ocr vision model. The model is baked in at build time so cold starts
# don't pull from the registry. Needs a GPU at runtime (nvidia-container-toolkit
# / a GPU host on Coolify).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_URL=http://127.0.0.1:11434 \
    GLM_OCR_MODEL=glm-ocr \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl zstd tini python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ollama from the GitHub release tarball (avoids systemd).
ARG OLLAMA_VERSION=v0.24.0
RUN curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
        -o /tmp/ollama.tar.zst \
    && zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar \
    && tar -C /usr -xf /tmp/ollama.tar \
    && rm /tmp/ollama.tar /tmp/ollama.tar.zst \
    && ollama --version

WORKDIR /app

# Python deps (cached unless requirements change).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the glm-ocr model into the image (start ollama briefly, pull, stop).
RUN set -eux; \
    OLLAMA_HOST=127.0.0.1:11434 ollama serve > /tmp/ollama-build.log 2>&1 & \
    pid=$!; \
    for i in $(seq 1 30); do \
        if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi; \
        sleep 1; \
    done; \
    OLLAMA_HOST=127.0.0.1:11434 ollama pull "${GLM_OCR_MODEL}"; \
    OLLAMA_HOST=127.0.0.1:11434 ollama list; \
    kill $pid; wait $pid 2>/dev/null || true

COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
