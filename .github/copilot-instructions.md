# Custom Instructions for GitHub Copilot

## Project Overview
This is a Python project named ANewDawn. It utilizes Docker for containerization (`Dockerfile`, `docker-compose.yml`). Key files include `main.py` and `settings.py`.

## Development Environment
- **Operating System:** Windows
- **Default Shell:** PowerShell (`pwsh.exe`). Please generate terminal commands compatible with PowerShell.

## Coding Standards
- **Linting & Formatting:** We use `ruff` for linting and formatting. Please adhere to `ruff` standards. The configuration can be found in the `.github/workflows/ruff.yml` workflow and potentially a `pyproject.toml` or `ruff.toml` file.
- **Python Version:** 3.13
- **Dependencies:** Dependencies are managed using `uv` and listed in `pyproject.toml`. Some commands you can run are `uv run pytest`, `uv add <package_name>`, `uv sync --upgrade`, and `uv run python main.py`.

## General Guidelines
- Follow Python best practices.
- Write clear and concise code.
- Only add comments where necessary to explain complex logic.
- Ensure code is compatible with the Docker environment defined.
- Use `uv` commands for package management and running scripts.
    - For example, to run a script, use `uv run python main.py`.
    - For package installation, use `uv add <package_name>`.
    - For testing, use `uv run pytest`.
    - For linting and formatting, use `uv run ruff`.
- For Docker-related tasks, use the `docker` and `docker-compose` commands as needed.
- For example, to build the Docker image, use `docker build -t <image_name> .`.
- To run the Docker container, use `docker run <image_name>` or `docker-compose up` for multi-container setups.
- For stopping and removing containers, use `docker stop <container_id>` and `docker rm <container_id>`.

## Discord Bot Functionality
- **Chat Interaction:** The bot responds to messages containing "lovibot" or its mention (`<@345000831499894795>`) by using the OpenAI chat API (`gpt-4o-mini`). See the `on_message` event handler and `misc.chat` function.
- **Slash Commands:**
    - `/ask <text>`: Ask the AI a question directly. Uses `misc.chat`.
- **Context Menu Commands:**
    - `Enhance Image`: Right-click on a message containing an image to enhance it using three different OpenCV methods (`enhance_image1`, `enhance_image2`, `enhance_image3`).
- **User Restrictions:** Interaction is limited to users listed in `misc.get_allowed_users()`. Image creation has further restrictions.
