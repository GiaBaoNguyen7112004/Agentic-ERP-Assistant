# syntax=docker/dockerfile:1

# Stage 1: the React app. vite.config.ts writes to
# ../src/agentic_erp_assistant/web/static relative to ui/, so WORKDIR
# /build/ui puts the output at /build/src/agentic_erp_assistant/web/static
# -- the exact path web/app.py's STATIC_DIR expects once it lands at
# /app/src/....
FROM node:22-alpine AS ui
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# Stage 2: resolve the Python environment from uv.lock. Same base image as
# the runtime stage, so the venv's interpreter symlinks resolve identically.
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# Third-party deps first, so the layer is cached until uv.lock changes; the
# project itself second (uv installs it editable -- the lock pins it as
# source = { editable = "." }, which is what keeps __file__ pointing at
# /app/src/... and the parents[3]-relative data paths working).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Stage 3: runtime.
FROM python:3.14-slim AS runtime
# THE /app LAYOUT IS LOAD-BEARING: data paths (users.json, erp/project.json,
# documents/manifest.json) are computed as Path(__file__).resolve()
# .parents[3] / "data" from inside the package (composition/settings.py,
# erp/mock.py, rag/manifest.py), and the editable install puts __file__ at
# /app/src/agentic_erp_assistant/. Moving src/ or data/ breaks startup. See
# ADR 0029.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv ./.venv
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --from=ui --chown=appuser:appuser /build/src/agentic_erp_assistant/web/static src/agentic_erp_assistant/web/static
USER appuser
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# A 200 from /api/health proves the uvicorn lifespan finished -- the lexical
# index was built from Qdrant and Postgres connected -- so this is a
# readiness probe, not just a port check. python's urllib, not curl: the
# slim image has no curl, and the venv is already there.
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=90s \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
# --host 0.0.0.0 is required: the entry point defaults to 127.0.0.1, which is
# unreachable from outside the container. workers=1 stays (ADR 0004: the rate
# limiter, flaky-tool counter and ERP lock are in-process state).
CMD ["/app/.venv/bin/agentic-erp-assistant", "serve", "--host", "0.0.0.0", "--port", "8000"]