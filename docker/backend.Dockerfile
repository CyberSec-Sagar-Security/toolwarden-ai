# Phase 12 backend: code + Python dependencies baked into the image,
# model weights NEVER baked in -- TOOLWARDEN_MODEL_DIR is bind-mounted at
# `docker run`/compose time from the host's real model directory (see
# docker-compose.yml). This is the storage-split rule (README.md /
# project memory) applied to the container: the image is reproducible and
# small without a multi-GB checkpoint baked in, and swapping model weights
# never requires a rebuild.
#
# CPU-first: base torch (the same one `pip install toolwarden-ai` resolves,
# see pyproject.toml's comment) has no CUDA build pinned. GPU passthrough
# is verified and documented as its own checkpoint on top of this image,
# not a blocker for the rest of the compose stack -- see
# docs/docker_walkthrough.md.

FROM python:3.13-slim

# LightGBM's compiled library needs libgomp (GNU OpenMP) -- not present in
# the slim base image. Caught via a real container run (OSError:
# libgomp.so.1: cannot open shared object file), not assumed.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[service]"

ENV TOOLWARDEN_MODEL_DIR=/models
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "toolwarden.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
