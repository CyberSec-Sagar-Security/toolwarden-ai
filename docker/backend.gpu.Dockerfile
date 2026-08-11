# GPU-enabled variant of backend.Dockerfile: installs a CUDA-matched torch
# build (matching this project's own training pin, requirements.txt's
# torch==2.13.0+cu132) instead of the CPU wheel the base `torch>=2.13`
# dependency resolves from plain PyPI. Everything else -- deps baked in,
# model weights volume-mounted, never baked in -- is identical.
#
# Kept as a separate file rather than a build ARG so the CPU path (what
# most consumers/CI would actually use) stays the simple, unconditional
# default in backend.Dockerfile. Used via docker-compose.gpu.yml, layered
# on top of docker-compose.yml -- see docs/docker_walkthrough.md for the
# verification actually run for this project (NVIDIA Container Toolkit /
# WSL2 GPU passthrough, confirmed working via `docker run --gpus all`
# before this image was built on top of it).

FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[service]" \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu132 \
       --force-reinstall "torch==2.13.0+cu132"

ENV TOOLWARDEN_MODEL_DIR=/models
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "toolwarden.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
