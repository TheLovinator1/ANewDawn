from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import hikari
import lightbulb
import openai
from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv(verbose=True)

discord_token: str | None = os.getenv("DISCORD_TOKEN")
openai_api_key: str | None = os.getenv("OPENAI_TOKEN")
if not discord_token or not openai_api_key:
    logger.error("You haven't configured the bot correctly. Please set the environment variables.")
    sys.exit(1)


bot = lightbulb.BotApp(token=discord_token, intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.GUILD_MESSAGE_TYPING)
openai_client = OpenAI(api_key=openai_api_key)


def chat(user_message: str) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        user_message: The message to send to OpenAI.

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


@bot.listen(hikari.MessageCreateEvent)
async def on_message(event: hikari.MessageCreateEvent) -> None:
    """Respond to a message."""
    if not event.is_human:
        return

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if event.author.username not in allowed_users:
        logger.info("Ignoring message from: %s", event.author.username)
        return

    incoming_message: str | None = event.message.content
    if not incoming_message:
        logger.error("No message content found in the event: %s", event)
        return

    lowercase_message: str = incoming_message.lower() if incoming_message else ""
    trigger_keywords: list[str] = get_trigger_keywords()
    if any(trigger in lowercase_message for trigger in trigger_keywords):
        logger.info("Received message: %s from: %s", incoming_message, event.author.username)

        async with bot.rest.trigger_typing(event.channel_id):
            try:
                response: str | None = chat(incoming_message)
            except openai.OpenAIError as e:
                logger.exception("An error occurred while chatting with the AI model.")
                e.add_note(f"Message: {incoming_message}\nEvent: {event}\nWho: {event.author.username}")
                await bot.rest.create_message(
                    event.channel_id, f"An error occurred while chatting with the AI model. {e}"
                )
                return

            if response:
                logger.info("Responding to message: %s with: %s", incoming_message, response)
                await bot.rest.create_message(event.channel_id, response)
            else:
                logger.warning("No response from the AI model. Message: %s", incoming_message)
                await bot.rest.create_message(event.channel_id, "I forgor how to think 💀")


def get_trigger_keywords() -> list[str]:
    """Get the list of trigger keywords to respond to.

    Returns:
        The list of trigger keywords.
    """
    bot_user: hikari.OwnUser | None = bot.get_me()
    bot_mention_string: str = f"<@{bot_user.id}>" if bot_user else ""
    notification_keywords: list[str] = ["lovibot", bot_mention_string]
    return notification_keywords


if __name__ == "__main__":
    logger.info("Starting the bot.")
    bot.run(asyncio_debug=True, check_for_updates=True)
