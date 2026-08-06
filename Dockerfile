# Multi-stage: the builder holds uv and the full toolchain, the runtime holds neither.
# Keeps the shipped image small and removes the build tooling from the attack surface.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile BEFORE the source is copied, so a source
# change does not invalidate the dependency layer. --no-install-project skips the package
# itself at this stage for the same reason.
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
    
CMD ["python", "-c", "import credit_risk; print('credit_risk image ready')"]
