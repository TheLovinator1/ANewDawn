# Copilot Instructions for ANewDawn

## Project Overview

ANewDawn is a Discord bot written in Python 3.13+ using the discord.py library and Pydantic AI for AI-powered chat capabilities. The bot includes features such as:

- AI-powered chat responses using OpenAI and Grok models
- Conversation memory with reset/undo functionality
- Image enhancement using OpenCV
- Web search integration via Ollama
- Slash commands and context menus

## Development Environment

- **Python**: 3.13 or higher required
- **Package Manager**: Use `uv` for dependency management (see `pyproject.toml`)
- **Docker**: The project uses Docker for deployment (see `Dockerfile` and `docker-compose.yml`)
- **Environment Variables**: Copy `.env.example` to `.env` and fill in required tokens

## Code Style and Conventions

### Linting and Formatting

This project uses **Ruff** for linting and formatting with strict settings:

- All rules enabled (`lint.select = ["ALL"]`)
- Preview features enabled
- Auto-fix enabled
- Line length: 160 characters
- Google-style docstrings required

Run linting:
```bash
ruff check --exit-non-zero-on-fix --verbose
```

Run formatting check:
```bash
ruff format --check --verbose
```

### Python Conventions

- Use `from __future__ import annotations` at the top of all files (automatically added by Ruff)
- Use type hints for all function parameters and return types
- Follow Google docstring convention
- Use `logging` module for logging, not print statements
- Prefer explicit imports over wildcard imports

### Testing

- Tests use pytest
- Test files should be named `*_test.py` or `test_*.py`
- Run tests with: `pytest`

## Project Structure

- `main.py` - Main bot application with all commands and event handlers
- `pyproject.toml` - Project configuration and dependencies
- `Dockerfile` / `docker-compose.yml` - Container configuration
- `.github/workflows/` - CI/CD workflows

## Key Components

### Bot Client

The main bot client is `LoviBotClient` which extends `discord.Client`. It handles:
- Message events (`on_message`)
- Slash commands (`/ask`, `/grok`, `/reset`, `/undo`)
- Context menus (image enhancement)

### AI Integration

- `chatgpt_agent` - Pydantic AI agent using OpenAI
- `grok_it()` - Function for Grok model responses
- Message history is stored in `recent_messages` dict per channel

### Memory Management

- `add_message_to_memory()` - Store messages for context
- `reset_memory()` - Clear conversation history
- `undo_reset()` - Restore previous state

## CI/CD

The GitHub Actions workflow (`.github/workflows/docker-publish.yml`) runs:
1. Ruff linting and format check
2. Dockerfile validation
3. Docker image build and push to GitHub Container Registry

## Common Tasks

### Adding a New Slash Command

1. Add the command function with `@client.tree.command()` decorator
2. Include `@app_commands.allowed_installs()` and `@app_commands.allowed_contexts()` decorators
3. Use `await interaction.response.defer()` for long-running operations
4. Check user authorization with `get_allowed_users()`

### Adding a New AI Instruction

1. Create a function decorated with `@chatgpt_agent.instructions`
2. The function should return a string with the instruction content
3. Use `RunContext[BotDependencies]` parameter to access dependencies

### Modifying Image Enhancement

Image enhancement functions (`enhance_image1`, `enhance_image2`, `enhance_image3`) use OpenCV. Each returns WebP-encoded bytes.
