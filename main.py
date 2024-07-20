from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import discord
from discord.ext import commands
from openai import OpenAI

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion

from dotenv import load_dotenv

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load the environment variables from the .env file
load_dotenv(verbose=True)

# Get the Discord token and OpenAI API key from the environment variables
discord_token: str | None = os.getenv("DISCORD_TOKEN")
openai_api_key: str | None = os.getenv("OPENAI_TOKEN")
if not discord_token or not openai_api_key:
    logger.error("You haven't configured the bot correctly. Please set the environment variables.")
    sys.exit(1)

# Use OpenAI for chatting with the bot
openai_client = OpenAI(api_key=openai_api_key)

# Create a bot with the necessary intents
# TODO(TheLovinator): We should only enable the intents we need  # noqa: TD003
intents: discord.Intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready() -> None:  # noqa: RUF029
    """Print a message when the bot is ready."""
    logger.info("Logged on as %s", bot.user)


def chat(msg: str) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        msg: The message to send to the bot.

    Returns:
        The response from the bot.
    """
    completion: ChatCompletion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a chatbot. Use Markdown to format your messages if you want.",
            },
            {"role": "user", "content": msg},
        ],
    )
    response: str | None = completion.choices[0].message.content
    logger.info("AI response: %s from message: %s", response, msg)

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


def remove_mentions(message_content: str) -> str:
    """Remove mentions of the bot from the message content.

    Args:
        message_content: The message content to process.

    Returns:
        The message content without the mentions of the bot.
    """
    message_content = message_content.removeprefix("lovibot").strip()
    message_content = message_content.removeprefix(",").strip()
    if bot.user:
        message_content = message_content.replace(f"<@!{bot.user.id}>", "").strip()
        message_content = message_content.replace(f"<@{bot.user.id}>", "").strip()

    return message_content


@bot.event
async def on_message(message: discord.Message) -> None:
    """Respond to a message."""
    logger.info("Message received: %s", message.content)

    message_content: str = message.content.lower()

    # Ignore messages from the bot itself to prevent an infinite loop
    if message.author == bot.user:
        return

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if message.author.name not in allowed_users:
        logger.info("Ignoring message from: %s", message.author.name)
        return

    # Check if the message mentions the bot or starts with the bot's name
    things_to_notify_on: list[str] = ["lovibot"]
    if bot.user:
        things_to_notify_on.extend((f"<@!{bot.user.id}>", f"<@{bot.user.id}>"))

    # Only respond to messages that mention the bot or are a reply to a bot message
    if any(thing.lower() in message_content for thing in things_to_notify_on) or message.reference:
        if message.reference:
            # Get the message that the current message is replying to
            message_id: int | None = message.reference.message_id
            if message_id is None:
                return

            try:
                reply_message: discord.Message | None = await message.channel.fetch_message(message_id)
            except discord.errors.NotFound:
                return

            # Get the message content and author
            reply_content: str = reply_message.content
            reply_author: str = reply_message.author.name

            # Add the reply message to the current message
            message.content = f"{reply_author}: {reply_content}\n{message.author.name}: {message.content}"

        # Remove the mention of the bot from the message
        message_content = remove_mentions(message_content)

        # Grab 10 messages before the current one to provide context
        old_messages: list[str] = [
            f"{old_message.author.name}: {old_message.content}"
            async for old_message in message.channel.history(limit=10)
        ]
        old_messages.reverse()

        # Get the response from OpenAI
        response: str | None = chat("\n".join(old_messages) + "\n" + f"{message.author.name}: {message.content}")

        # Remove LoviBot: from the response
        if response:
            response = response.removeprefix("LoviBot:").strip()

        if response:
            logger.info("Responding to message: %s with: %s", message.content, response)
            await message.channel.send(response)
        else:
            logger.warning("No response from the AI model. Message: %s", message.content)
            await message.channel.send("I forgor how to think 💀")


if __name__ == "__main__":
    logger.info("Starting the bot.")
    bot.run(token=discord_token, root_logger=True)
