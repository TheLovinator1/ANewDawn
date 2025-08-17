from __future__ import annotations

import datetime
import logging
from collections import deque
from typing import TYPE_CHECKING

import psutil
from discord import Emoji, Member, User, channel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from discord.abc import MessageableChannel
    from discord.guild import GuildChannel
    from discord.interactions import InteractionChannel
    from openai import OpenAI
    from openai.types.responses import Response


logger: logging.Logger = logging.getLogger(__name__)

# A dictionary to store recent messages per channel with a maximum length per channel
recent_messages: dict[str, deque[tuple[str, str, datetime.datetime]]] = {}

# A dictionary to track the last time each user triggered the bot in each channel
last_trigger_time: dict[str, dict[str, datetime.datetime]] = {}


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


def get_recent_messages(channel_id: int, threshold_minutes: int = 10) -> list[tuple[str, str]]:
    """Retrieve messages from the last `threshold_minutes` minutes for a specific channel.

    Args:
        channel_id: The ID of the channel to retrieve messages for.
        threshold_minutes: The number of minutes to consider messages as recent.

    Returns:
        A list of tuples containing user and message content.
    """
    if str(channel_id) not in recent_messages:
        return []

    threshold: datetime.datetime = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(minutes=threshold_minutes)
    return [(user, message) for user, message, timestamp in recent_messages[str(channel_id)] if timestamp > threshold]


def update_trigger_time(channel_id: str, user: str) -> None:
    """Update the last trigger time for a user in a specific channel.

    Args:
        channel_id: The ID of the channel.
        user: The user who triggered the bot.
    """
    if channel_id not in last_trigger_time:
        last_trigger_time[channel_id] = {}

    last_trigger_time[channel_id][user] = datetime.datetime.now(tz=datetime.UTC)
    logger.info("Updated trigger time for user %s in channel %s", user, channel_id)


def should_respond_without_trigger(channel_id: str, user: str, threshold_seconds: int = 40) -> bool:
    """Check if the bot should respond to a user without requiring trigger keywords.

    Args:
        channel_id: The ID of the channel.
        user: The user who sent the message.
        threshold_seconds: The number of seconds to consider as "recent trigger".

    Returns:
        True if the bot should respond without trigger keywords, False otherwise.
    """
    if channel_id not in last_trigger_time or user not in last_trigger_time[channel_id]:
        return False

    last_trigger: datetime.datetime = last_trigger_time[channel_id][user]
    threshold: datetime.datetime = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(seconds=threshold_seconds)

    should_respond: bool = last_trigger > threshold
    logger.info("User %s in channel %s last triggered at %s, should respond without trigger: %s", user, channel_id, last_trigger, should_respond)

    return should_respond


def extra_context(current_channel: MessageableChannel | InteractionChannel | None, user: User | Member) -> str:
    """Add extra context to the chat prompt.

    For example:
    - Current date and time
    - Channel name and server
    - User's current status (online/offline)
    - User's role in the server (e.g., admin, member)
    - CPU usage
    - Memory usage
    - Disk usage
    - How many messages saved in memory

    Args:
        current_channel: The channel where the conversation is happening.
        user: The user who is interacting with the bot.

    Returns:
        The extra context to include in the chat prompt.
    """
    context: str = ""

    # Information about the servers and channels:
    context += "KillYoy's Server Information:\n"
    context += "- Server is for friends to hang out and chat.\n"
    context += "- Server was created by KillYoy (<@98468214824001536>)\n"

    # Current date and time
    context += f"Current date and time: {datetime.datetime.now(tz=datetime.UTC)} UTC, but user is in CEST or CET\n"

    # Channel name and server
    if isinstance(current_channel, channel.TextChannel):
        context += f"Channel name: {current_channel.name}, channel ID: {current_channel.id}, Server: {current_channel.guild.name}\n"

    # User information
    context += f"User name: {user.name}, User ID: {user.id}\n"
    if isinstance(user, Member):
        context += f"User roles: {', '.join([role.name for role in user.roles])}\n"
        context += f"User status: {user.status}\n"
        context += f"User is currently {'on mobile' if user.is_on_mobile() else 'on desktop'}\n"
        context += f"User joined server at: {user.joined_at}\n"
        context += f"User's current activity: {user.activity}\n"
        context += f"User's username color: {user.color}\n"

    # System information
    context += f"CPU usage per core: {psutil.cpu_percent(percpu=True)}%\n"
    context += f"Memory usage: {psutil.virtual_memory().percent}%\n"
    context += f"Total memory: {psutil.virtual_memory().total / (1024 * 1024):.2f} MB\n"
    context += f"Swap memory usage: {psutil.swap_memory().percent}%\n"
    context += f"Swap memory total: {psutil.swap_memory().total / (1024 * 1024):.2f} MB\n"
    context += f"Bot memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB\n"
    uptime: datetime.timedelta = datetime.datetime.now(tz=datetime.UTC) - datetime.datetime.fromtimestamp(psutil.boot_time(), tz=datetime.UTC)
    context += f"System uptime: {uptime}\n"
    context += "Disk usage:\n"
    for partition in psutil.disk_partitions():
        try:
            context += f"  {partition.mountpoint}: {psutil.disk_usage(partition.mountpoint).percent}%\n"
        except PermissionError as e:
            context += f"  {partition.mountpoint} got PermissionError: {e}\n"

    if current_channel:
        context += f"Messages saved in memory: {len(get_recent_messages(channel_id=current_channel.id))}\n"

    return context


def chat(  # noqa: PLR0913, PLR0917
    user_message: str,
    openai_client: OpenAI,
    current_channel: MessageableChannel | InteractionChannel | None,
    user: User | Member,
    allowed_users: list[str],
    all_channels_in_guild: Sequence[GuildChannel] | None = None,
) -> str | None:
    """Chat with the bot using the OpenAI API.

    Args:
        user_message: The message to send to OpenAI.
        openai_client: The OpenAI client to use.
        current_channel: The channel where the conversation is happening.
        user: The user who is interacting with the bot.
        allowed_users: The list of allowed users to interact with the bot.
        all_channels_in_guild: The list of all channels in the guild.

    Returns:
        The response from the AI model.
    """
    recent_context: str = ""
    context: str = ""

    if current_channel:
        channel_id = int(current_channel.id)
        recent_context: str = "\n".join([f"{user}: {message}" for user, message in get_recent_messages(channel_id=channel_id)])

        context = extra_context(current_channel=current_channel, user=user)

        if current_channel.guild:
            server_emojis: list[Emoji] = list(current_channel.guild.emojis)
            if server_emojis:
                context += "\nEmojis with `kao` are pictures of kao172, he is our friend so you can use them to express yourself!\n"
                context += "\nYou can use the following server emojis:\n"
                for emoji in server_emojis:
                    context += f"  - {emoji!s}\n"

                context += "\n You can create bigger emojis by combining them:\n"
                context += "For example if you want to create a big rat emoji, you can combine the following emojis. The picture is three by three:\n"
                context += "  - <:rat1:1405292421742334116>: + <:rat2:1405292423373918258> + <:rat3:1405292425446031400>\n"
                context += "  - <:rat4:1405292427777933354>: + <:rat5:1405292430210891949>: + <:rat6:1405292433411145860>:\n"
                context += "  - <:rat7:1405292434883084409>: + <:rat8:1405292442181304320>: + <:rat9:1405292443619819631>:\n"
                context += "This will create a picture of Jane Does ass."
                context += " You can use it when we talk about coom, Zenless Zone Zero (ZZZ) or other related topics."
                context += "\n"

                context += "The following emojis needs to be on the same line to form a bigger emoji:\n"
                context += "<a:phibiscarf2:1050306159023759420><a:phibiscarf_mid:1050306153084637194><a:phibiscarf1:1050306156997918802>\n"

                context += "If you are using emoji combos, ONLY send the emoji itself and don't add unnecessary text.\n"
                context += "Remember that combo emojis need to be on a separate line to form a bigger emoji.\n"
                context += "But remember to not overuse them, remember that the user still can see the old message, so no need to write it again.\n"
                context += "Also remember that you cant put code blocks around emojis.\n"
                context += "Licka and Sniffa emojis are dogs that lick and sniff things. For example anime feet, butts and sweat.\n"
                context += "If you want to use them, just send the emoji itself without any extra text.\n"

    context += "The bot is in the following channels:\n"
    if all_channels_in_guild:
        for c in all_channels_in_guild:
            context += f"{c!r}\n"

    context += "\nThe bot responds to the following users:\n"
    for user_id in allowed_users:
        context += f"  - User ID: {user_id}\n"

    prompt: str = (
        "You are in a Discord group chat. People can ask you questions.\n"
        "Try to be brief, we don't want bloated messages. Be concise and to the point.\n"
        "Use Discord Markdown to format messages if needed.\n"
        "Don't use emojis.\n"
        "Extra context starts here:\n"
        f"{context}"
        "Extra context ends here.\n"
        "Recent context starts here:\n"
        f"{recent_context}\n"
        "Recent context ends here.\n"
    )

    logger.info("Sending request to OpenAI API with prompt: %s", prompt)

    resp: Response = openai_client.responses.create(
        model="gpt-5-chat-latest",
        instructions=prompt,
        input=user_message,
    )
    response: str | None = resp.output_text
    logger.info("AI response: %s", response)

    return response
