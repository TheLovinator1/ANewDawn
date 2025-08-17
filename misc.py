from __future__ import annotations

import datetime
import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion


logger: logging.Logger = logging.getLogger(__name__)

# A dictionary to store recent messages per channel with a maximum length per channel
recent_messages: dict[str, deque[tuple[str, str, datetime.datetime]]] = {}


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


def add_message_to_memory(channel_id: str, user: str, message: str) -> None:
    """Add a message to the memory for a specific channel.

    Args:
        channel_id: The ID of the channel where the message was sent.
        user: The user who sent the message.
        message: The content of the message.
    """
    if channel_id not in recent_messages:
        recent_messages[channel_id] = deque(maxlen=50)

    timestamp: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
    recent_messages[channel_id].append((user, message, timestamp))

    logger.info("Added message to memory: %s from %s in channel %s", message, user, channel_id)


def get_recent_messages(channel_id: str, threshold_minutes: int = 10) -> list[tuple[str, str]]:
    """Retrieve messages from the last `threshold_minutes` minutes for a specific channel.

    Args:
        channel_id: The ID of the channel to retrieve messages for.
        threshold_minutes: The number of minutes to consider messages as recent.

    Returns:
        A list of tuples containing user and message content.
    """
    if channel_id not in recent_messages:
        return []

    threshold: datetime.datetime = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(
        minutes=threshold_minutes
    )
    return [(user, message) for user, message, timestamp in recent_messages[channel_id] if timestamp > threshold]


def chat(user_message: str, openai_client: OpenAI, channel_id: str) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        user_message: The message to send to OpenAI.
        openai_client: The OpenAI client to use.
        channel_id: The ID of the channel where the conversation is happening.

    Returns:
        The response from the AI model.
    """
    # Include recent messages in the prompt
    recent_context: str = "\n".join([f"{user}: {message}" for user, message in get_recent_messages(channel_id)])
    prompt: str = (
        "You are in a Discord group chat. People can ask you questions. "
        "Use Discord Markdown to format messages if needed.\n"
        f"Recent context:\n{recent_context}\n"
        f"User: {user_message}"
    )

    completion: ChatCompletion = openai_client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=[{"role": "system", "content": prompt}],
    )
    response: str | None = completion.choices[0].message.content
    logger.info("AI response: %s", response)

    return response
