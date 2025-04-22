# syntax=docker/dockerfile:1
# check=error=true;experimental=all
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:61f9b961f6fd782bdf4ce6e29a643aee2b7ef742149585903de371070c09f4b3

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
