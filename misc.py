from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import hikari
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


def get_trigger_keywords(bot: hikari.GatewayBotAware) -> list[str]:
    """Get the list of trigger keywords to respond to.

    Returns:
        The list of trigger keywords.
    """
    bot_user: hikari.OwnUser | None = bot.get_me()
    bot_mention_string: str = f"<@{bot_user.id}>" if bot_user else ""
    notification_keywords: list[str] = ["lovibot", bot_mention_string]
    return notification_keywords


def chat(user_message: str, openai_client: OpenAI) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        user_message: The message to send to OpenAI.
        openai_client: The OpenAI client to use.

    Returns:
        The response from the AI model.
    """
    completion: ChatCompletion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer",
                "content": "You are in a Discord group chat with people above the age of 30. Use Discord Markdown to format messages if needed.",  # noqa: E501
            },
            {"role": "user", "content": user_message},
        ],
    )
    response: str | None = completion.choices[0].message.content
    logger.info("AI response: %s", response)

    return response