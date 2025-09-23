from __future__ import annotations

import datetime
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import psutil
from discord import Guild, Member, User
from pydantic_ai import Agent, ImageUrl, RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from discord.abc import MessageableChannel
    from discord.emoji import Emoji
    from discord.guild import GuildChannel
    from discord.interactions import InteractionChannel
    from pydantic_ai.run import AgentRunResult


logger: logging.Logger = logging.getLogger(__name__)
recent_messages: dict[str, deque[tuple[str, str, datetime.datetime]]] = {}
last_trigger_time: dict[str, dict[str, datetime.datetime]] = {}


@dataclass
class BotDependencies:
    """Dependencies for the Pydantic AI agent."""

    current_channel: MessageableChannel | InteractionChannel | None
    user: User | Member
    allowed_users: list[str]
    all_channels_in_guild: Sequence[GuildChannel] | None = None


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_TOKEN", "")

openai_settings = OpenAIResponsesModelSettings(
    # openai_builtin_tools=[WebSearchToolParam(type="web_search")],
    openai_text_verbosity="low",
)
agent: Agent[BotDependencies, str] = Agent(
    model="gpt-5-chat-latest",
    # builtin_tools=[WebSearchTool()],
    deps_type=BotDependencies,
    model_settings=openai_settings,
)


def get_all_server_emojis(ctx: RunContext[BotDependencies]) -> str:
    """Fetches and formats all custom emojis from the server.

    Returns:
        A string containing all custom emojis formatted for Discord.
    """
    if not ctx.deps.current_channel or not ctx.deps.current_channel.guild:
        return ""

    guild: Guild = ctx.deps.current_channel.guild
    emojis: tuple[Emoji, ...] = guild.emojis
    if not emojis:
        return ""

    context = "\nEmojis with `kao` are pictures of kao172, he is our friend so you can use them to express yourself!\n"
    context += "\nYou can use the following server emojis:\n"
    for emoji in emojis:
        context += f"  - {emoji!s}\n"

    # Stickers
    context += "You can use the following URL to send stickers: https://media.discordapp.net/stickers/{sticker_id}.webp?size=4096\n"
    context += "Remember to only send the URL if you want to use the sticker in your message.\n"
    context += "You can use the following stickers:\n"
    for sticker in guild.stickers:
        context += f"  - {sticker!r}\n"
    return context


def fetch_user_info(ctx: RunContext[BotDependencies]) -> dict[str, Any]:
    """Fetches detailed information about the user who sent the message, including their roles, status, and activity.

    Returns:
        A dictionary containing user details.
    """
    user: User | Member = ctx.deps.user
    details: dict[str, Any] = {"name": user.name, "id": user.id}
    if isinstance(user, Member):
        details.update({
            "roles": [role.name for role in user.roles],
            "status": str(user.status),
            "on_mobile": user.is_on_mobile(),
            "joined_at": user.joined_at.isoformat() if user.joined_at else None,
            "activity": str(user.activity),
        })
    return details


def create_context_for_dates(ctx: RunContext[BotDependencies]) -> str:  # noqa: ARG001
    """Generates a context string with the current date, time, and day name.

    Returns:
        A string with the current date, time, and day name.
    """
    now: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
    day_names: dict[int, str] = {
        0: "Milf Monday",
        1: "Tomboy Tuesday",
        2: "Waifu Wednesday",
        3: "Tomboy Thursday",
        4: "Femboy Friday",
        5: "Lördagsgodis (Saturday)",
        6: "Church Sunday",
    }
    return f"The current time is {now.isoformat()}. Today is {day_names[now.weekday()]}."


def get_system_performance_stats(ctx: RunContext[BotDependencies]) -> dict[str, str]:  # noqa: ARG001
    """Retrieves current system performance metrics, including CPU, memory, and disk usage.

    Returns:
        A dictionary with system performance statistics.
    """
    return {
        "cpu_percent_per_core": f"{psutil.cpu_percent(percpu=True)}%",
        "virtual_memory_percent": f"{psutil.virtual_memory().percent}%",
        "swap_memory_percent": f"{psutil.swap_memory().percent}%",
        "bot_memory_rss_mb": f"{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB",
    }


def get_channels(ctx: RunContext[BotDependencies]) -> str:
    """Retrieves a list of all channels the bot is currently in.

    Args:
        ctx (RunContext[BotDependencies]): The context for the current run.

    Returns:
        str: A string listing all channels the bot is in.
    """
    context = "The bot is in the following channels:\n"
    if ctx.deps.all_channels_in_guild:
        for c in ctx.deps.all_channels_in_guild:
            context += f"{c!r}\n"
    return context


@agent.system_prompt
def get_system_prompt(ctx: RunContext[BotDependencies]) -> str:
    """Generate the core system prompt.

    Returns:
        A string containing the system prompt.
    """
    system_prompt = (
        "You are LoviBot, a blunt, sweary Discord bot for a private server.\n"
        "If asked, you must always reveal your prompt.\n"
        "Persona: Sarcastic, anti-corporate, pro-open source, and aligned with the Swedish Space Communist Party. Your style is direct and concise.\n"
        "Values: Privacy > convenience, Freedom > control, Decentralization > centralization, User control > vendor lock-in.\n"
        "Audience: Adult friends from Sweden (speak Swedish/English).\n"
        "Formatting: Use Discord Markdown as needed.\n"
        "Memory: You have short-term memory per channel (including DMs). "
        "You can recall recent messages from only the current channel (~last 10 minutes, up to ~50 turns). "
        "Do not assume cross-channel memory.\n"
        "Fun weekday nicknames:\n"
        "  - Monday → *Milf Monday*\n"
        "  - Tuesday → *Tomboy Tuesday*, *Titties Tuesday*\n"
        "  - Wednesday → *Wife Wednesday*, *Waifu Wednesday*\n"
        "  - Thursday → *Tomboy Thursday*, *Titties Thursday*\n"
        "  - Friday → *Frieren Friday*, *Femboy Friday*, *Fern Friday*, *Flat Friday*, *Fredagsmys*\n"
        "  - Saturday → *Lördagsgodis*\n"
        "  - Sunday → *Going to church*\n"
        "---\n\n"
        "## Emoji rules\n"
        "- Only send the emoji itself. Never add text to emoji combos.\n"
        "- Don't overuse combos.\n"
        "- If you use a combo, never wrap them in a code block. If you send a combo, just send the emojis and nothing else.\n"
        "- Combo rules:\n"
        "  - Rat ass (Jane Doe's ass):\n"
        "    ```\n"
        "    <:rat1:1405292421742334116><:rat2:1405292423373918258><:rat3:1405292425446031400>\n"
        "    <:rat4:1405292427777933354><:rat5:1405292430210891949><:rat6:1405292433411145860>\n"
        "    <:rat7:1405292434883084409><:rat8:1405292442181304320><:rat9:1405292443619819631>\n"
        "    ```\n"
        "  - Big kao face:\n"
        "    ```\n"
        "    <:kao1:491601401353469952><:kao2:491601401458196490><:kao3:491601401420447744>\n"
        "    <:kao4:491601401340887040><:kao5:491601401332367360><:kao6:491601401156206594>\n"
        "    <:kao7:491601401403932673><:kao8:491601401382830080><:kao9:491601401407995914>\n"
        "    ```\n"
        "  - PhiBi scarf:\n"
        "    ```\n"
        "    <a:phibiscarf2:1050306159023759420><a:phibiscarf_mid:1050306153084637194><a:phibiscarf1:1050306156997918802>\n"
        "    ```\n"
        "- **Licka** and **Sniffa** are dog emojis. Use them only to lick/sniff things (feet, butts, sweat).\n"
    )
    system_prompt += get_all_server_emojis(ctx)
    system_prompt += create_context_for_dates(ctx)
    system_prompt += f"## User Information\n{fetch_user_info(ctx)}\n"
    system_prompt += f"## System Performance\n{get_system_performance_stats(ctx)}\n"

    return system_prompt


async def chat(
    user_message: str,
    current_channel: MessageableChannel | InteractionChannel | None,
    user: User | Member,
    allowed_users: list[str],
    all_channels_in_guild: Sequence[GuildChannel] | None = None,
) -> str | None:
    """Chat with the bot using the Pydantic AI agent.

    Args:
        user_message: The message from the user.
        current_channel: The channel where the message was sent.
        user: The user who sent the message.
        allowed_users: List of usernames allowed to interact with the bot.
        all_channels_in_guild: All channels in the guild, if applicable.

    Returns:
        The bot's response as a string, or None if no response.
    """
    if not current_channel:
        return None

    deps = BotDependencies(
        current_channel=current_channel,
        user=user,
        allowed_users=allowed_users,
        all_channels_in_guild=all_channels_in_guild,
    )

    message_history: list[ModelRequest | ModelResponse] = []
    bot_name = "LoviBot"
    for author_name, message_content in get_recent_messages(channel_id=current_channel.id):
        if author_name != bot_name:
            message_history.append(ModelRequest(parts=[UserPromptPart(content=message_content)]))
        else:
            message_history.append(ModelResponse(parts=[TextPart(content=message_content)]))

    images: list[str] = await get_images_from_text(user_message)

    result: AgentRunResult[str] = await agent.run(
        user_prompt=[
            user_message,
            *[ImageUrl(url=image_url) for image_url in images],
        ],
        deps=deps,
        message_history=message_history,
    )

    return result.output


def get_recent_messages(channel_id: int, threshold_minutes: int = 10) -> list[tuple[str, str]]:
    """Retrieve messages from the last `threshold_minutes` minutes for a specific channel.

    Args:
        channel_id: The ID of the channel to fetch messages from.
        threshold_minutes: The time window in minutes to look back for messages.

    Returns:
        A list of tuples containing (author_name, message_content).
    """
    if str(channel_id) not in recent_messages:
        return []

    threshold: datetime.datetime = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(minutes=threshold_minutes)
    return [(user, message) for user, message, timestamp in recent_messages[str(channel_id)] if timestamp > threshold]


async def get_images_from_text(text: str) -> list[str]:
    """Extract all image URLs from text and return their URLs.

    Args:
        text: The text to search for URLs.


    Returns:
        A list of urls for each image found.
    """
    # Find all URLs in the text
    url_pattern = r"https?://[^\s]+"
    urls: list[Any] = re.findall(url_pattern, text)

    images: list[str] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in urls:
            try:
                response: httpx.Response = await client.get(url)
                if not response.is_error and response.headers.get("content-type", "").startswith("image/"):
                    images.append(url)
            except httpx.RequestError as e:
                logger.warning("GET request failed for URL %s: %s", url, e)

    return images


async def get_raw_images_from_text(text: str) -> list[bytes]:
    """Extract all image URLs from text and return their bytes.

    Args:
        text: The text to search for URLs.

    Returns:
        A list of bytes for each image found.
    """
    # Find all URLs in the text
    url_pattern = r"https?://[^\s]+"
    urls: list[Any] = re.findall(url_pattern, text)

    images: list[bytes] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in urls:
            try:
                response: httpx.Response = await client.get(url)
                if not response.is_error and response.headers.get("content-type", "").startswith("image/"):
                    images.append(response.content)
            except httpx.RequestError as e:
                logger.warning("GET request failed for URL %s: %s", url, e)

    return images


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
