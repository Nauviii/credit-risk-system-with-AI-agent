# Multi-stage: the builder holds uv and the full toolchain, the runtime holds neither.
# Keeps the shipped image small and removes the build tooling from the attack surface.

# The uv in this image must be AT LEAST as new as the uv that wrote uv.lock. The lockfile
# carries a `revision` field, and an older uv cannot read a newer revision - `uv sync
# --frozen` then fails on a lockfile that is perfectly valid locally. Pin this to the
# output of `uv --version` on the machine that maintains the lock; `latest` is the safe
# default but gives up build reproducibility.
#
# The ARG sits before any FROM (global scope) and the image is pulled as its own stage,
# because BuildKit does not expand variables in `COPY --from=` - only in `FROM`.
ARG UV_VERSION=latest
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile BEFORE the source is copied, so a source
# change does not invalidate the dependency layer. --no-install-project skips the package
# itself at this stage for the same reason.
# The cache mount needs BuildKit; CI sets that up with docker/setup-buildx-action.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# libgomp1 is required by LightGBM at runtime and is not in the slim base image; without
# it, importing lightgbm fails with an OSError that is easy to misread as a build problem.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root by default. Nothing here needs to write outside /app.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/configs /app/configs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# No model artefacts are baked in - none are persisted yet (see PROJECT_HANDOFF section 8).
# Phase 8 replaces this CMD with `uvicorn credit_risk.serving.app:app --host 0.0.0.0 --port 8000`
# and mounts or copies the trained artefacts.
CMD ["python", "-c", "import credit_risk; print('credit_risk image ready')"]