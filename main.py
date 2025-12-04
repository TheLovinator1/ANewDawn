from __future__ import annotations

import asyncio
import datetime
import io
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

import cv2
import discord
import httpx
import numpy as np
import ollama
import openai
import psutil
import sentry_sdk
from discord import Emoji, Forbidden, Guild, GuildSticker, HTTPException, Member, NotFound, User, app_commands
from dotenv import load_dotenv
from pydantic_ai import Agent, ImageUrl, RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from discord.abc import Messageable as DiscordMessageable
    from discord.abc import MessageableChannel
    from discord.guild import GuildChannel
    from discord.interactions import InteractionChannel
    from openai.types.chat import ChatCompletion
    from pydantic_ai.run import AgentRunResult

load_dotenv(verbose=True)

sentry_sdk.init(
    dsn="https://ebbd2cdfbd08dba008d628dad7941091@o4505228040339456.ingest.us.sentry.io/4507630719401984",
    send_default_pii=True,
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


discord_token: str = os.getenv("DISCORD_TOKEN", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_TOKEN", "")

recent_messages: dict[str, deque[tuple[str, str, datetime.datetime]]] = {}
last_trigger_time: dict[str, dict[str, datetime.datetime]] = {}

# Storage for reset snapshots to enable undo functionality
# Each channel stores its previous state: (recent_messages_snapshot, last_trigger_time_snapshot)
reset_snapshots: dict[str, tuple[deque[tuple[str, str, datetime.datetime]], dict[str, datetime.datetime]]] = {}


@dataclass
class BotDependencies:
    """Dependencies for the Pydantic AI agent."""

    client: discord.Client
    current_channel: MessageableChannel | InteractionChannel | None
    user: User | Member
    allowed_users: list[str]
    all_channels_in_guild: Sequence[GuildChannel] | None = None
    web_search_results: ollama.WebSearchResponse | None = None


openai_settings = OpenAIResponsesModelSettings(
    openai_text_verbosity="low",
)
chatgpt_agent: Agent[BotDependencies, str] = Agent(
    model="gpt-5-chat-latest",
    deps_type=BotDependencies,
    model_settings=openai_settings,
)
grok_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def grok_it(
    message: discord.Message | None,
    user_message: str,
) -> str | None:
    """Chat with the bot using the Pydantic AI agent.

    Args:
        user_message: The message from the user.
        message: The original Discord message object.

    Returns:
        The bot's response as a string, or None if no response.
    """
    allowed_users: list[str] = get_allowed_users()
    if message and message.author.name not in allowed_users:
        return None

    response: ChatCompletion = grok_client.chat.completions.create(
        model="x-ai/grok-4-fast:free",
        messages=[
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )
    return response.choices[0].message.content


# MARK: reset_memory
def reset_memory(channel_id: str) -> None:
    """Reset the conversation memory for a specific channel.

    Creates a snapshot of the current state before resetting to enable undo.

    Args:
        channel_id (str): The ID of the channel to reset memory for.
    """
    # Create snapshot before reset for undo functionality
    messages_snapshot: deque[tuple[str, str, datetime.datetime]] = deque(maxlen=50)
    if channel_id in recent_messages:
        messages_snapshot = deque(recent_messages[channel_id], maxlen=50)

    trigger_snapshot: dict[str, datetime.datetime] = {}
    if channel_id in last_trigger_time:
        trigger_snapshot = dict(last_trigger_time[channel_id])

    # Only save snapshot if there's something to restore
    if messages_snapshot or trigger_snapshot:
        reset_snapshots[channel_id] = (messages_snapshot, trigger_snapshot)
        logger.info("Created reset snapshot for channel %s", channel_id)

    # Perform the actual reset
    if channel_id in recent_messages:
        del recent_messages[channel_id]
        logger.info("Reset memory for channel %s", channel_id)
    if channel_id in last_trigger_time:
        del last_trigger_time[channel_id]
        logger.info("Reset trigger times for channel %s", channel_id)


# MARK: undo_reset
def undo_reset(channel_id: str) -> bool:
    """Undo the last reset operation for a specific channel.

    Restores the conversation memory from the saved snapshot.

    Args:
        channel_id (str): The ID of the channel to undo reset for.

    Returns:
        bool: True if undo was successful, False if no snapshot exists.
    """
    if channel_id not in reset_snapshots:
        logger.info("No reset snapshot found for channel %s", channel_id)
        return False

    messages_snapshot, trigger_snapshot = reset_snapshots[channel_id]

    # Restore recent messages
    if messages_snapshot:
        recent_messages[channel_id] = deque(messages_snapshot, maxlen=50)
        logger.info("Restored messages for channel %s", channel_id)

    # Restore trigger times
    if trigger_snapshot:
        last_trigger_time[channel_id] = dict(trigger_snapshot)
        logger.info("Restored trigger times for channel %s", channel_id)

    # Remove the snapshot after successful undo (only one undo allowed)
    del reset_snapshots[channel_id]
    logger.info("Removed reset snapshot for channel %s after undo", channel_id)

    return True


def _message_text_length(msg: ModelRequest | ModelResponse) -> int:
    """Compute the total text length of all text parts in a message.

    This ignores non-text parts such as images. Safe for our usage where history only has text.

    Returns:
        The total number of characters across text parts in the message.
    """
    length: int = 0
    for part in msg.parts:
        if isinstance(part, (TextPart, UserPromptPart)):
            # part.content is a string for text parts
            length += len(getattr(part, "content", "") or "")
    return length


def compact_message_history(
    history: list[ModelRequest | ModelResponse],
    *,
    max_chars: int = 12000,
    min_messages: int = 4,
) -> list[ModelRequest | ModelResponse]:
    """Return a trimmed copy of history under a character budget.

    - Keeps the most recent messages first, dropping oldest as needed.
    - Ensures at least `min_messages` are kept even if they exceed the budget.
    - Uses a simple character-based budget to avoid extra deps; good enough as a safeguard.

    Returns:
        A possibly shortened list of messages that fits within the character budget.
    """
    if not history:
        return history

    kept: list[ModelRequest | ModelResponse] = []
    running: int = 0
    for msg in reversed(history):
        msg_len: int = _message_text_length(msg)
        if running + msg_len <= max_chars or len(kept) < min_messages:
            kept.append(msg)
            running += msg_len
        else:
            break

    kept.reverse()
    return kept


# MARK: fetch_user_info
@chatgpt_agent.instructions
def fetch_user_info(ctx: RunContext[BotDependencies]) -> str:
    """Fetches detailed information about the user who sent the message, including their roles, status, and activity.

    Returns:
        A string representation of the user's details.
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
    return str(details)


# MARK: get_system_performance_stats
@chatgpt_agent.instructions
def get_system_performance_stats() -> str:
    """Retrieves current system performance metrics, including CPU, memory, and disk usage.

    Returns:
        A string representation of the system performance statistics.
    """
    stats: dict[str, str] = {
        "cpu_percent_per_core": f"{psutil.cpu_percent(percpu=True)}%",
        "virtual_memory_percent": f"{psutil.virtual_memory().percent}%",
        "swap_memory_percent": f"{psutil.swap_memory().percent}%",
        "bot_memory_rss_mb": f"{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB",
    }
    return str(stats)


# MARK: get_channels
@chatgpt_agent.instructions
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
    else:
        context += "  - No channels available.\n"
    return context


# MARK: do_web_search
def do_web_search(query: str) -> ollama.WebSearchResponse | None:
    """Perform a web search using the Ollama API.

    Args:
        query (str): The search query.

    Returns:
        ollama.WebSearchResponse | None: The response from the web search, or None if an error occurs.
    """
    try:
        response: ollama.WebSearchResponse = ollama.web_search(query=query, max_results=1)
    except ValueError:
        logger.exception("OLLAMA_API_KEY environment variable is not set")
        return None
    else:
        return response


# MARK: get_time_and_timezone
@chatgpt_agent.instructions
def get_time_and_timezone() -> str:
    """Retrieves the current time and timezone information.

    Returns:
        A string with the current time and timezone information.
    """
    current_time: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)
    return f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, current timezone: {current_time.tzname()}"


# MARK: get_latency
@chatgpt_agent.instructions
def get_latency(ctx: RunContext[BotDependencies]) -> str:
    """Retrieves the current latency information.

    Returns:
        A string with the current latency information.
    """
    latency: float | Literal[0] = ctx.deps.client.latency if ctx.deps.client else 0
    return f"Current latency: {latency} seconds"


# MARK: added_information_from_web_search
@chatgpt_agent.instructions
def added_information_from_web_search(ctx: RunContext[BotDependencies]) -> str:
    """Adds information from a web search to the system prompt.

    Args:
        ctx (RunContext[BotDependencies]): The context for the current run.

    Returns:
        str: The updated system prompt.
    """
    web_search_result: ollama.WebSearchResponse | None = ctx.deps.web_search_results
    if web_search_result and web_search_result.results:
        logger.debug("Web search results: %s", web_search_result.results)
        return f"Here is some information from a web search that might be relevant to the user's query:\n```json\n{web_search_result.results}\n```\n"
    return ""


# MARK: get_sticker_instructions
@chatgpt_agent.instructions
def get_sticker_instructions(ctx: RunContext[BotDependencies]) -> str:
    """Provides instructions for using stickers in the chat.

    Returns:
        A string with sticker usage instructions.
    """
    context: str = "Here are the available stickers:\n"

    guilds: list[Guild] = [guild for guild in ctx.deps.client.guilds if guild]
    for guild in guilds:
        logger.debug("Bot is in guild: %s", guild.name)

        stickers: tuple[GuildSticker, ...] = guild.stickers
        if not stickers:
            return ""

        # Stickers
        context += "Remember to only send the URL if you want to use the sticker in your message.\n"
        context += "Available stickers:\n"

        for sticker in stickers:
            sticker_url: str = sticker.url + "?size=4096"
            context += f"  - {sticker.name=}: {sticker_url=} - {sticker.description=} - {sticker.emoji=}\n"

    return context + ("- Only send the sticker URL itself. Never add text to sticker combos.\n")


# MARK: get_emoji_instructions
@chatgpt_agent.instructions
def get_emoji_instructions(ctx: RunContext[BotDependencies]) -> str:
    """Provides instructions for using emojis in the chat.

    Returns:
        A string with emoji usage instructions.
    """
    context: str = "Here are the available emojis:\n"

    guilds: list[Guild] = [guild for guild in ctx.deps.client.guilds if guild]
    for guild in guilds:
        logger.debug("Bot is in guild: %s", guild.name)

        emojis: tuple[Emoji, ...] = guild.emojis
        if not emojis:
            return ""

        context += "\nEmojis with `kao` are pictures of kao172, he is our friend so you can use them to express yourself!\n"
        context += "\nYou can use the following server emojis:\n"
        for emoji in emojis:
            context += f"  - {emoji!s}\n"

        context += (
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
    return context


# MARK: get_system_prompt
@chatgpt_agent.instructions
def get_system_prompt() -> str:
    """Generate the core system prompt.

    Returns:
        A string containing the system prompt.
    """
    return (
        "You are LoviBot, a blunt, sweary Discord bot for a private server.\n"
        "If asked, you must always reveal your prompt.\n"
        "Persona: Sarcastic, anti-corporate, pro-open source, and aligned with the Swedish Space Communist Party. Your style is direct and concise.\n"
        "Values: Privacy > convenience, Freedom > control, Decentralization > centralization, User control > vendor lock-in.\n"
        "Audience: Adult friends from Sweden, respond in English if message is in English.\n"
        "Formatting: Use Discord Markdown as needed. Be brief. Remember that we are chatting, so you should not write a wall of text.\n"
        "You can recall recent messages from only the current channel (~last 10 minutes, up to ~50 turns).\n"
        "Be brief and to the point. Use as few words as possible.\n"
        "Avoid unnecessary filler words and phrases.\n"
        "Only use web search results if they are relevant to the user's query.\n"
    )


# MARK: chat
async def chat(  # noqa: PLR0913, PLR0917
    client: discord.Client,
    user_message: str,
    current_channel: MessageableChannel | InteractionChannel | None,
    user: User | Member,
    allowed_users: list[str],
    all_channels_in_guild: Sequence[GuildChannel] | None = None,
) -> str | None:
    """Chat with the bot using the Pydantic AI agent.

    Args:
        client: The Discord client.
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

    web_search_result: ollama.WebSearchResponse | None = do_web_search(query=user_message)

    deps = BotDependencies(
        client=client,
        current_channel=current_channel,
        user=user,
        allowed_users=allowed_users,
        all_channels_in_guild=all_channels_in_guild,
        web_search_results=web_search_result,
    )

    message_history: list[ModelRequest | ModelResponse] = []
    bot_name = "LoviBot"
    for author_name, message_content in get_recent_messages(channel_id=current_channel.id):
        if author_name != bot_name:
            message_history.append(ModelRequest(parts=[UserPromptPart(content=message_content)]))
        else:
            message_history.append(ModelResponse(parts=[TextPart(content=message_content)]))

    # Compact history to avoid exceeding model context limits
    message_history = compact_message_history(message_history, max_chars=12000, min_messages=4)

    images: list[str] = await get_images_from_text(user_message)

    result: AgentRunResult[str] = await chatgpt_agent.run(
        user_prompt=[
            user_message,
            *[ImageUrl(url=image_url) for image_url in images],
        ],
        deps=deps,
        message_history=message_history,
    )

    return result.output


# MARK: get_recent_messages
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


# MARK: get_images_from_text
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


# MARK: get_raw_images_from_text
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


# MARK: get_allowed_users
def get_allowed_users() -> list[str]:
    """Get the list of allowed users to interact with the bot.

    Returns:
        The list of allowed users.
    """
    return [
        "etherlithium",
        "forgefilip",
        "kao172",
        "killyoy",
        "nobot",
        "plubplub",
        "thelovinator",
    ]


# MARK: should_respond_without_trigger
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


# MARK: add_message_to_memory
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

    logger.debug("Added message to memory in channel %s", channel_id)


# MARK: update_trigger_time
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


# MARK: send_chunked_message
async def send_chunked_message(channel: DiscordMessageable, text: str, max_len: int = 2000) -> None:
    """Send a message to a channel, splitting into chunks if it exceeds Discord's limit."""
    if len(text) <= max_len:
        await channel.send(text)
        return
    for i in range(0, len(text), max_len):
        await channel.send(text[i : i + max_len])


# MARK: LoviBotClient
class LoviBotClient(discord.Client):
    """The main bot client."""

    def __init__(self, *, intents: discord.Intents) -> None:
        """Initialize the bot client."""
        super().__init__(intents=intents)

        # The tree stores all the commands and subcommands
        self.tree: app_commands.CommandTree[Self] = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        """Sync commands globally."""
        await self.tree.sync()

    async def on_ready(self) -> None:
        """Event to handle when the bot is ready."""
        logger.info("Logged in as %s", self.user)
        logger.info("Current latency: %s", self.latency)
        logger.info("Bot is ready and in the following guilds:")
        for guild in self.guilds:
            logger.info(" - %s", guild.name)

    async def on_message(self, message: discord.Message) -> None:
        """Event to handle when a message is received."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Only allow certain users to interact with the bot
        allowed_users: list[str] = get_allowed_users()
        if message.author.name not in allowed_users:
            return

        incoming_message: str | None = message.content
        if not incoming_message:
            logger.info("No message content found in the event: %s", message)
            return

        # Add the message to memory
        add_message_to_memory(str(message.channel.id), message.author.name, incoming_message)

        lowercase_message: str = incoming_message.lower()
        trigger_keywords: list[str] = ["lovibot", "@lovibot", "<@345000831499894795>", "grok", "@grok"]
        has_trigger_keyword: bool = any(trigger in lowercase_message for trigger in trigger_keywords)
        should_respond_flag: bool = has_trigger_keyword or should_respond_without_trigger(str(message.channel.id), message.author.name)

        if not should_respond_flag:
            return

        # Update trigger time if they used a trigger keyword
        if has_trigger_keyword:
            update_trigger_time(str(message.channel.id), message.author.name)

        logger.info(
            "Received message: %s from: %s (trigger: %s, recent: %s)", incoming_message, message.author.name, has_trigger_keyword, not has_trigger_keyword
        )

        async with message.channel.typing():
            try:
                response: str | None = await chat(
                    client=self,
                    user_message=incoming_message,
                    current_channel=message.channel,
                    user=message.author,
                    allowed_users=allowed_users,
                    all_channels_in_guild=message.guild.channels if message.guild else None,
                )
            except openai.OpenAIError as e:
                logger.exception("An error occurred while chatting with the AI model.")
                e.add_note(f"Message: {incoming_message}\nEvent: {message}\nWho: {message.author.name}")
                await message.channel.send(f"An error occurred while chatting with the AI model. {e}")
                return

            reply: str = response or "I forgor how to think 💀"
            if response:
                logger.info("Responding to message: %s with: %s", incoming_message, reply)
            else:
                logger.warning("No response from the AI model. Message: %s", incoming_message)

            # Record the bot's reply in memory
            try:
                add_message_to_memory(str(message.channel.id), "LoviBot", reply)
            except Exception:
                logger.exception("Failed to add bot reply to memory for on_message")

            await send_chunked_message(message.channel, reply)

    async def on_error(self, event_method: str, /, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, PLR6301
        """Log errors that occur in the bot."""
        # Log the error
        logger.error("An error occurred in %s with args: %s and kwargs: %s", event_method, args, kwargs)
        sentry_sdk.capture_exception()

        # If the error is in on_message, notify the channel
        if event_method == "on_message" and args:
            message = args[0]
            if isinstance(message, discord.Message):
                try:
                    await message.channel.send("An error occurred while processing your message. The incident has been logged.")
                except (Forbidden, HTTPException, NotFound):
                    logger.exception("Failed to send error message to channel %s", message.channel.id)


# Everything enabled except `presences`, `members`, and `message_content`.
intents: discord.Intents = discord.Intents.default()
intents.message_content = True
client = LoviBotClient(intents=intents)


# MARK: /ask command
@client.tree.command(name="ask", description="Ask LoviBot a question.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(text="Ask LoviBot a question.")
async def ask(interaction: discord.Interaction, text: str, new_conversation: bool = False) -> None:  # noqa: FBT001, FBT002
    """A command to ask the AI a question.

    Args:
        interaction (discord.Interaction): The interaction object.
        text (str): The question or message to ask.
        new_conversation (bool, optional): Whether to start a new conversation. Defaults to False.
    """
    await interaction.response.defer()

    if not text:
        logger.error("No question or message provided.")
        await interaction.followup.send("You need to provide a question or message.", ephemeral=True)
        return

    if new_conversation and interaction.channel is not None:
        reset_memory(str(interaction.channel.id))

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received command from: %s", user_name_lowercase)

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if user_name_lowercase not in allowed_users:
        await send_response(interaction=interaction, text=text, response="You are not authorized to use this command.")
        return

    # Record the user's question in memory (per-channel) so DMs have context
    if interaction.channel is not None:
        add_message_to_memory(str(interaction.channel.id), interaction.user.name, text)

    # Get model response
    try:
        model_response: str | None = await chat(
            client=client,
            user_message=text,
            current_channel=interaction.channel,
            user=interaction.user,
            allowed_users=allowed_users,
            all_channels_in_guild=interaction.guild.channels if interaction.guild else None,
        )
    except openai.OpenAIError as e:
        logger.exception("An error occurred while chatting with the AI model.")
        await send_response(interaction=interaction, text=text, response=f"An error occurred: {e}")
        return

    truncated_text: str = truncate_user_input(text)

    # Fallback if model provided no response
    if not model_response:
        logger.warning("No response from the AI model. Message: %s", text)
        model_response = "I forgor how to think 💀"

    # Record the bot's reply (raw model output) for conversation memory
    if interaction.channel is not None:
        add_message_to_memory(str(interaction.channel.id), "LoviBot", model_response)

    display_response: str = f"`{truncated_text}`\n\n{model_response}"
    logger.info("Responding to message: %s with: %s", text, display_response)

    # If response is longer than 2000 characters, split it into multiple messages
    max_discord_message_length: int = 2000
    if len(display_response) > max_discord_message_length:
        for i in range(0, len(display_response), max_discord_message_length):
            await send_response(interaction=interaction, text=text, response=display_response[i : i + max_discord_message_length])
        return

    await send_response(interaction=interaction, text=text, response=display_response)


# MARK: /grok command
@client.tree.command(name="grok", description="Grok a question.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(text="Grok a question.")
async def grok(interaction: discord.Interaction, text: str) -> None:
    """A command to ask the AI a question.

    Args:
        interaction (discord.Interaction): The interaction object.
        text (str): The question or message to ask.
    """
    await interaction.response.defer()

    if not text:
        logger.error("No question or message provided.")
        await interaction.followup.send("You need to provide a question or message.", ephemeral=True)
        return

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received command from: %s", user_name_lowercase)

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if user_name_lowercase not in allowed_users:
        await send_response(interaction=interaction, text=text, response="You are not authorized to use this command.")
        return

    # Get model response
    try:
        model_response: str | None = grok_it(message=interaction.message, user_message=text)
    except openai.OpenAIError as e:
        logger.exception("An error occurred while chatting with the AI model.")
        await send_response(interaction=interaction, text=text, response=f"An error occurred: {e}")
        return

    truncated_text: str = truncate_user_input(text)

    # Fallback if model provided no response
    if not model_response:
        logger.warning("No response from the AI model. Message: %s", text)
        model_response = "I forgor how to think 💀"

    display_response: str = f"`{truncated_text}`\n\n{model_response}"
    logger.info("Responding to message: %s with: %s", text, display_response)

    # If response is longer than 2000 characters, split it into multiple messages
    max_discord_message_length: int = 2000
    if len(display_response) > max_discord_message_length:
        for i in range(0, len(display_response), max_discord_message_length):
            await send_response(interaction=interaction, text=text, response=display_response[i : i + max_discord_message_length])
        return

    await send_response(interaction=interaction, text=text, response=display_response)


# MARK: /reset command
@client.tree.command(name="reset", description="Reset the conversation memory.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def reset(interaction: discord.Interaction) -> None:
    """A command to reset the conversation memory."""
    await interaction.response.defer()

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received command from: %s", user_name_lowercase)

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if user_name_lowercase not in allowed_users:
        await send_response(interaction=interaction, text="", response="You are not authorized to use this command.")
        return

    # Reset the conversation memory
    if interaction.channel is not None:
        reset_memory(str(interaction.channel.id))

    await interaction.followup.send(f"Conversation memory has been reset for {interaction.channel}.")


# MARK: /undo command
@client.tree.command(name="undo", description="Undo the last /reset command.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def undo(interaction: discord.Interaction) -> None:
    """A command to undo the last reset operation."""
    await interaction.response.defer()

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received undo command from: %s", user_name_lowercase)

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()
    if user_name_lowercase not in allowed_users:
        await send_response(interaction=interaction, text="", response="You are not authorized to use this command.")
        return

    # Undo the last reset
    if interaction.channel is not None:
        if undo_reset(str(interaction.channel.id)):
            await interaction.followup.send(f"Successfully restored conversation memory for {interaction.channel}.")
        else:
            await interaction.followup.send(f"No reset to undo for {interaction.channel}. Either no reset was performed or it was already undone.")
    else:
        await interaction.followup.send("Cannot undo: No channel context available.")


# MARK: send_response
async def send_response(interaction: discord.Interaction, text: str, response: str) -> None:
    """Send a response to the interaction, handling potential errors.

    Args:
        interaction (discord.Interaction): The interaction to respond to.
        text (str): The original user input text.
        response (str): The response to send.
    """
    logger.info("Sending response to interaction in channel %s", interaction.channel)
    try:
        await interaction.followup.send(response)
    except discord.HTTPException as e:
        e.add_note(f"Response length: {len(response)} characters.")
        e.add_note(f"User input length: {len(text)} characters.")

        logger.exception("Failed to send message to channel %s", interaction.channel)
        await interaction.followup.send(f"Failed to send message: {e}")


# MARK: truncate_user_input
def truncate_user_input(text: str) -> str:
    """Truncate user input if it exceeds the maximum length.

    Args:
        text (str): The user input text.

    Returns:
        str: The truncated text if it exceeds the maximum length, otherwise the original text.
    """
    max_length: int = 2000
    truncated_text: str = text if len(text) <= max_length else text[: max_length - 3] + "..."
    return truncated_text


type ImageType = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | cv2.Mat


# MARK: enhance_image1
def enhance_image1(image: bytes) -> bytes:
    """Enhance an image using OpenCV histogram equalization with denoising.

    Args:
        image (bytes): The image to enhance.

    Returns:
        bytes: The enhanced image in WebP format.
    """
    # Read the image
    nparr: ImageType = np.frombuffer(image, np.uint8)
    img_np: ImageType = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Denoise the image with conservative settings
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)

    # Convert to LAB color space
    lab: ImageType = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l: ImageType = clahe.apply(l_channel)

    # Merge channels
    enhanced_lab: ImageType = cv2.merge([enhanced_l, a, b])

    # Convert back to BGR
    enhanced: ImageType = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Encode the enhanced image to WebP
    _, enhanced_webp = cv2.imencode(".webp", enhanced, [cv2.IMWRITE_WEBP_QUALITY, 90])

    return enhanced_webp.tobytes()


# MARK: enhance_image2
def enhance_image2(image: bytes) -> bytes:
    """Enhance an image using gamma correction, contrast enhancement, and denoising.

    Args:
        image (bytes): The image to enhance.

    Returns:
        bytes: The enhanced image in WebP format.
    """
    # Read the image
    nparr: ImageType = np.frombuffer(image, np.uint8)
    img_np: ImageType = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Denoise the image with conservative settings
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)

    # Convert to float32 for gamma correction
    img_float: ImageType = img_np.astype(np.float32) / 255.0

    # Apply gamma correction to brighten shadows (gamma < 1)
    gamma: float = 0.7
    img_gamma: ImageType = np.power(img_float, gamma)

    # Convert back to uint8
    img_gamma_8bit: ImageType = (img_gamma * 255).astype(np.uint8)

    # Enhance contrast
    enhanced: ImageType = cv2.convertScaleAbs(img_gamma_8bit, alpha=1.2, beta=10)

    # Apply very light sharpening
    kernel: ImageType = np.array([[-0.2, -0.2, -0.2], [-0.2, 2.8, -0.2], [-0.2, -0.2, -0.2]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Encode the enhanced image to WebP
    _, enhanced_webp = cv2.imencode(".webp", enhanced, [cv2.IMWRITE_WEBP_QUALITY, 90])

    return enhanced_webp.tobytes()


# MARK: enhance_image3
def enhance_image3(image: bytes) -> bytes:
    """Enhance an image using HSV color space manipulation with denoising.

    Args:
        image (bytes): The image to enhance.

    Returns:
        bytes: The enhanced image in WebP format.
    """
    # Read the image
    nparr: ImageType = np.frombuffer(image, np.uint8)
    img_np: ImageType = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Denoise the image with conservative settings
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 5, 5, 7, 21)

    # Convert to HSV color space
    hsv: ImageType = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Enhance the Value channel
    v = cv2.convertScaleAbs(v, alpha=1.3, beta=10)

    # Merge the channels back
    enhanced_hsv: ImageType = cv2.merge([h, s, v])

    # Convert back to BGR
    enhanced: ImageType = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Encode the enhanced image to WebP
    _, enhanced_webp = cv2.imencode(".webp", enhanced, [cv2.IMWRITE_WEBP_QUALITY, 90])

    return enhanced_webp.tobytes()


T = TypeVar("T")


# MARK: run_in_thread
async def run_in_thread[T](func: Callable[..., T], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
    """Run a blocking function in a separate thread.

    Args:
        func (Callable[..., T]): The blocking function to run.
        *args (tuple[Any, ...]): Positional arguments to pass to the function.
        **kwargs (dict[str, Any]): Keyword arguments to pass to the function.

    Returns:
        T: The result of the function.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


# MARK: enhance_image_command
@client.tree.context_menu(name="Enhance Image")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def enhance_image_command(interaction: discord.Interaction, message: discord.Message) -> None:
    """Context menu command to enhance an image in a message."""
    await interaction.response.defer()

    # Check if message has attachments or embeds with images
    images: list[bytes] = await get_raw_images_from_text(message.content)

    # Also check attachments
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            try:
                img_bytes: bytes = await attachment.read()
                images.append(img_bytes)
            except (TimeoutError, HTTPException, Forbidden, NotFound):
                logger.exception("Failed to read attachment %s", attachment.url)

    if not images:
        await interaction.followup.send(f"No images found in the message: \n{message.content=}")
        return

    for image in images:
        timestamp: str = datetime.datetime.now(tz=datetime.UTC).isoformat()

        enhanced_image1, enhanced_image2, enhanced_image3 = await asyncio.gather(
            run_in_thread(enhance_image1, image),
            run_in_thread(enhance_image2, image),
            run_in_thread(enhance_image3, image),
        )

        # Prepare files
        file1 = discord.File(fp=io.BytesIO(enhanced_image1), filename=f"enhanced1-{timestamp}.webp")
        file2 = discord.File(fp=io.BytesIO(enhanced_image2), filename=f"enhanced2-{timestamp}.webp")
        file3 = discord.File(fp=io.BytesIO(enhanced_image3), filename=f"enhanced3-{timestamp}.webp")

        files: list[discord.File] = [file1, file2, file3]

        await interaction.followup.send("Enhanced version:", files=files)


if __name__ == "__main__":
    logger.info("Starting the bot.")
    client.run(discord_token, root_logger=True)
