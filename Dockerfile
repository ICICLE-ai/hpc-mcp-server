# syntax=docker/dockerfile:1

# hpc-mcp-server: execution-aware LLM MCP server for Tapis Pods.
# Multi-stage build using uv for reproducible, locked dependency installs.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.5.10

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies into a self-contained virtualenv.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# Pull the uv binary from the official image (pinned to match local tooling).
COPY --from=ghcr.io/astral-sh/uv:0.5.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# build-essential is required to compile deepspeed (and other) source packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only third-party dependencies first so this layer is cached across
# source-only changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the project (respecting .dockerignore) and install it into the venv.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# The vendored training estimator needs ijson, which is not a project dep.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install ijson

# ---------------------------------------------------------------------------
# Runtime: slim image carrying only the venv and application source.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/models/.cache/huggingface

USER app

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
