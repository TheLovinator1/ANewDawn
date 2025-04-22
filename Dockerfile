# syntax=docker/dockerfile:1
# check=error=true;experimental=all
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:873073fa8e9a3a4dfa8af5e4c444640459c640afff11612bcb970e8f2fe5caa7

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
