# syntax=docker/dockerfile:1
# check=error=true;experimental=all
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:efe2350aa848bc60720f136d94f47480095d8231b17547e4369f6bc97df59701

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
