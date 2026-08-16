# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime

ARG BUILD_SOURCE=""
ARG BUILD_REVISION=""
ARG BUILD_CREATED=""

LABEL org.opencontainers.image.source="${BUILD_SOURCE}" \
    org.opencontainers.image.revision="${BUILD_REVISION}" \
    org.opencontainers.image.created="${BUILD_CREATED}" \
    org.opencontainers.image.title="ANewDawn"

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

USER app

ENTRYPOINT ["python"]
CMD ["main.py"]
