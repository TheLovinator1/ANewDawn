from __future__ import annotations

import asyncio
import datetime
import io
import logging
import os
from typing import TYPE_CHECKING, Any, TypeVar

import cv2
import discord
import numpy as np
import openai
import sentry_sdk
from discord import Forbidden, HTTPException, NotFound, app_commands
from dotenv import load_dotenv

from misc import add_message_to_memory, chat, get_allowed_users, get_raw_images_from_text, should_respond_without_trigger, update_trigger_time

if TYPE_CHECKING:
    from collections.abc import Callable

sentry_sdk.init(
    dsn="https://ebbd2cdfbd08dba008d628dad7941091@o4505228040339456.ingest.us.sentry.io/4507630719401984",
    send_default_pii=True,
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


load_dotenv(verbose=True)

discord_token: str = os.getenv("DISCORD_TOKEN", "")


class LoviBotClient(discord.Client):
    """The main bot client."""

    def __init__(self, *, intents: discord.Intents) -> None:
        """Initialize the bot client."""
        super().__init__(intents=intents)

        # The tree stores all the commands and subcommands
        self.tree = app_commands.CommandTree(self)

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

        lowercase_message: str = incoming_message.lower() if incoming_message else ""
        trigger_keywords: list[str] = ["lovibot", "@lovibot", "<@345000831499894795>", "grok", "@grok"]
        has_trigger_keyword: bool = any(trigger in lowercase_message for trigger in trigger_keywords)
        should_respond: bool = has_trigger_keyword or should_respond_without_trigger(str(message.channel.id), message.author.name)

        if should_respond:
            # Update trigger time if they used a trigger keyword
            if has_trigger_keyword:
                update_trigger_time(str(message.channel.id), message.author.name)

            logger.info(
                "Received message: %s from: %s (trigger: %s, recent: %s)", incoming_message, message.author.name, has_trigger_keyword, not has_trigger_keyword
            )

            async with message.channel.typing():
                try:
                    response: str | None = await chat(
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

                if response:
                    logger.info("Responding to message: %s with: %s", incoming_message, response)

                    await message.channel.send(response)
                else:
                    logger.warning("No response from the AI model. Message: %s", incoming_message)
                    await message.channel.send("I forgor how to think 💀")

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


@client.tree.command(name="ask", description="Ask LoviBot a question.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(text="Ask LoviBot a question.")
async def ask(interaction: discord.Interaction, text: str) -> None:
    """A command to ask the AI a question."""
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

    try:
        response: str | None = await chat(
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

    if response:
        response = f"`{truncated_text}`\n\n{response}"
        logger.info("Responding to message: %s with: %s", text, response)
    else:
        logger.warning("No response from the AI model. Message: %s", text)
        response = "I forgor how to think 💀"

    # If response is longer than 2000 characters, split it into multiple messages
    max_discord_message_length: int = 2000
    if len(response) > max_discord_message_length:
        for i in range(0, len(response), max_discord_message_length):
            await send_response(interaction=interaction, text=text, response=response[i : i + max_discord_message_length])

        return

    await send_response(interaction=interaction, text=text, response=response)


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
