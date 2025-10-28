# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE} AS runtime

# Reproducibility and sane defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# System deps kept minimal for building scientific wheels where needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv
RUN python -m venv "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/pip" install --upgrade pip setuptools wheel

# Copy project (use .dockerignore to keep context lean)
COPY . .

# Install project with dev extras (tests/plots/tooling). For CPU-only PyTorch,
# the default indices typically suffice. If you need a CUDA wheel, consider
# preinstalling the appropriate torch/torchvision wheels in a derived image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .[dev]

# Default: print core versions to verify environment
CMD ["python", "-c", "import platform; print('Python', platform.python_version()); import torch, lightning; print('torch', torch.__version__, 'lightning', lightning.__version__)" ]
