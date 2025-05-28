from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion


logger: logging.Logger = logging.getLogger(__name__)


def get_allowed_users() -> list[str]:
    """Get the list of allowed users to interact with the bot.

    Returns:
        The list of allowed users.
    """
    return [
        "thelovinator",
        "killyoy",
        "forgefilip",
        "plubplub",
        "nobot",
        "kao172",
    ]


def chat(user_message: str, openai_client: OpenAI) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        user_message: The message to send to OpenAI.
        openai_client: The OpenAI client to use.

    Returns:
        The response from the AI model.
    """
    completion: ChatCompletion = openai_client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {
                "role": "developer",
                "content": "You are in a Discord group chat. Use Discord Markdown to format messages if needed.",
            },
            {"role": "user", "content": user_message},
        ],
    )
    response: str | None = completion.choices[0].message.content
    logger.info("AI response: %s", response)

    return response
