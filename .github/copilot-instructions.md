# Custom Instructions for GitHub Copilot

## Project Overview
This is a Python project named ANewDawn. It uses Docker for containerization (`Dockerfile`, `docker-compose.yml`). Key files include `main.py` and `settings.py`.

## Development Environment
- **Operating System:** Windows
- **Default Shell:** PowerShell (`pwsh.exe`). Please generate terminal commands compatible with PowerShell.

## Coding Standards
- **Linting & Formatting:** We use `ruff` for linting and formatting. Adhere to `ruff` standards. Configuration is in `.github/workflows/ruff.yml` and possibly `pyproject.toml` or `ruff.toml`.
- **Python Version:** 3.13
- **Dependencies:** Managed using `uv` and listed in `pyproject.toml`. Commands include:
  - `uv run pytest` for testing.
  - `uv add <package_name>` for package installation.
  - `uv sync --upgrade` for dependency updates.
  - `uv run python main.py` to run the project.

## General Guidelines
- Follow Python best practices.
- Write clear, concise code.
- Add comments only for complex logic.
- Ensure compatibility with the Docker environment.
- Use `uv` commands for package management and scripts.
- Use `docker` and `docker-compose` for container tasks:
  - Build: `docker build -t <image_name> .`
  - Run: `docker run <image_name>` or `docker-compose up`.
  - Stop/Remove: `docker stop <container_id>` and `docker rm <container_id>`.

## Discord Bot Functionality
- **Chat Interaction:** Responds to messages containing "lovibot" or its mention (`<@345000831499894795>`) using the OpenAI chat API (`gpt-4o-mini`). See `on_message` event handler and `misc.chat` function.
- **Slash Commands:**
  - `/ask <text>`: Directly ask the AI a question. Uses `misc.chat`.
- **Context Menu Commands:**
  - `Enhance Image`: Right-click on a message with an image to enhance it using OpenCV methods (`enhance_image1`, `enhance_image2`, `enhance_image3`).
- **User Restrictions:** Interaction is limited to users listed in `misc.get_allowed_users()`. Image creation has additional restrictions.
