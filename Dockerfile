# syntax=docker/dockerfile:1
# check=error=true;experimental=all
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:02ab5961f98c58f6f604122755aa466f28687b656d6bd6ef0f6b8036dd6d34a2

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project

# Copy the application files
COPY main.py misc.py settings.py /app/

# Set the environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["uv", "run", "main.py"]
