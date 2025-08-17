from __future__ import annotations

import datetime
import io
import logging
import re
from typing import Any

import cv2
import discord
import httpx
import numpy as np
import openai
import sentry_sdk
from discord import app_commands
from openai import OpenAI

from misc import add_message_to_memory, chat, get_allowed_users
from settings import Settings

sentry_sdk.init(
    dsn="https://ebbd2cdfbd08dba008d628dad7941091@o4505228040339456.ingest.us.sentry.io/4507630719401984",
    send_default_pii=True,
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

settings: Settings = Settings.from_env()
discord_token: str = settings.discord_token
openai_api_key: str = settings.openai_api_key


openai_client = OpenAI(api_key=openai_api_key)


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
            logger.info("Ignoring message from: %s", message.author.name)
            return

        incoming_message: str | None = message.content
        if not incoming_message:
            logger.info("No message content found in the event: %s", message)
            return

        # Add the message to memory
        add_message_to_memory(str(message.channel.id), message.author.name, incoming_message)

        lowercase_message: str = incoming_message.lower() if incoming_message else ""
        trigger_keywords: list[str] = ["lovibot", "@lovibot", "<@345000831499894795>", "grok", "@grok"]
        if any(trigger in lowercase_message for trigger in trigger_keywords):
            logger.info("Received message: %s from: %s", incoming_message, message.author.name)

            async with message.channel.typing():
                try:
                    response: str | None = chat(
                        user_message=incoming_message,
                        openai_client=openai_client,
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

    async def on_error(self, event_method: str, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Log errors that occur in the bot."""
        # Log the error
        logger.error("An error occurred in %s with args: %s and kwargs: %s", event_method, args, kwargs)

        # Add context to Sentry
        with sentry_sdk.push_scope() as scope:
            # Add event details
            scope.set_tag("event_method", event_method)
            scope.set_extra("args", args)
            scope.set_extra("kwargs", kwargs)

            # Add bot state
            scope.set_tag("bot_user_id", self.user.id if self.user else "Unknown")
            scope.set_tag("bot_user_name", str(self.user) if self.user else "Unknown")
            scope.set_tag("bot_latency", self.latency)

            # If specific arguments are available, extract and add details
            if args:
                interaction = next((arg for arg in args if isinstance(arg, discord.Interaction)), None)
                if interaction:
                    scope.set_extra("interaction_id", interaction.id)
                    scope.set_extra("interaction_user", interaction.user.id)
                    scope.set_extra("interaction_user_tag", str(interaction.user))
                    scope.set_extra("interaction_command", interaction.command.name if interaction.command else None)
                    scope.set_extra("interaction_channel", str(interaction.channel))
                    scope.set_extra("interaction_guild", str(interaction.guild) if interaction.guild else None)

                    # Add Sentry tags for interaction details
                    scope.set_tag("interaction_id", interaction.id)
                    scope.set_tag("interaction_user_id", interaction.user.id)
                    scope.set_tag("interaction_user_tag", str(interaction.user))
                    scope.set_tag("interaction_command", interaction.command.name if interaction.command else "None")
                    scope.set_tag("interaction_channel_id", interaction.channel.id if interaction.channel else "None")
                    scope.set_tag("interaction_channel_name", str(interaction.channel))
                    scope.set_tag("interaction_guild_id", interaction.guild.id if interaction.guild else "None")
                    scope.set_tag("interaction_guild_name", str(interaction.guild) if interaction.guild else "None")

            sentry_sdk.capture_exception()


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

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = get_allowed_users()

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received command from: %s", user_name_lowercase)

    if user_name_lowercase not in allowed_users:
        logger.info("Ignoring message from: %s", user_name_lowercase)
        await interaction.followup.send("You are not allowed to use this command.", ephemeral=True)
        return

    try:
        response: str | None = chat(
            user_message=text,
            openai_client=openai_client,
            current_channel=interaction.channel,
            user=interaction.user,
            allowed_users=allowed_users,
            all_channels_in_guild=interaction.guild.channels if interaction.guild else None,
        )
    except openai.OpenAIError as e:
        logger.exception("An error occurred while chatting with the AI model.")
        await interaction.followup.send(f"An error occurred: {e}")
        return

    if response:
        response = f"`{text}`\n\n{response}"

        logger.info("Responding to message: %s with: %s", text, response)

        await interaction.followup.send(response)
    else:
        await interaction.followup.send(f"I forgor how to think 💀\nText: {text}")


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


@client.tree.context_menu(name="Enhance Image")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def enhance_image_command(interaction: discord.Interaction, message: discord.Message) -> None:
    """Context menu command to enhance an image in a message."""
    await interaction.response.defer()

    # Check if message has attachments or embeds with images
    image_url: str | None = extract_image_url(message)
    if not image_url:
        await interaction.followup.send("No image found in the message.", ephemeral=True)
        return

    try:
        # Download the image
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.get(image_url)
            response.raise_for_status()
            image_bytes: bytes = response.content

        timestamp: str = datetime.datetime.now(tz=datetime.UTC).isoformat()

        enhanced_image1: bytes = enhance_image1(image_bytes)
        file1 = discord.File(fp=io.BytesIO(enhanced_image1), filename=f"enhanced1-{timestamp}.webp")

        enhanced_image2: bytes = enhance_image2(image_bytes)
        file2 = discord.File(fp=io.BytesIO(enhanced_image2), filename=f"enhanced2-{timestamp}.webp")

        enhanced_image3: bytes = enhance_image3(image_bytes)
        file3 = discord.File(fp=io.BytesIO(enhanced_image3), filename=f"enhanced3-{timestamp}.webp")

        files: list[discord.File] = [file1, file2, file3]
        logger.info("Enhanced image: %s", image_url)
        logger.info("Enhanced image files: %s", files)

        await interaction.followup.send("Enhanced version:", files=files)

    except (httpx.HTTPError, openai.OpenAIError) as e:
        logger.exception("Failed to enhance image")
        await interaction.followup.send(f"An error occurred: {e}")


def extract_image_url(message: discord.Message) -> str | None:
    """Extracts the first image URL from a given Discord message.

    This function checks the attachments of the provided message for any image
    attachments. If none are found, it then examines the message embeds to see if
    they include an image. Finally, if no images are found in attachments or embeds,
    the function searches the message content for any direct links ending in
    common image file extensions (e.g., .png, .jpg, .jpeg, .gif, .webp).

    Additionally, it handles Twitter image URLs and normalizes them to a standard format.

    Args:
        message (discord.Message): The message from which to extract the image URL.

    Returns:
        str | None: The URL of the first image found, or None if no image is found.
    """
    image_url: str | None = None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                image_url = attachment.url
                break

    elif message.embeds:
        for embed in message.embeds:
            if embed.image:
                image_url = embed.image.url
                break

    if not image_url:
        match: re.Match[str] | None = re.search(r"(https?://[^\s]+\.(png|jpg|jpeg|gif|webp)(\?[^\s]*)?)", message.content, re.IGNORECASE)
        if match:
            image_url = match.group(0)

    # Handle Twitter image URLs
    if image_url and "pbs.twimg.com/media/" in image_url:
        # Normalize Twitter image URLs to the highest quality format
        image_url = re.sub(r"\?format=[^&]+&name=[^&]+", "?format=jpg&name=orig", image_url)

    return image_url


if __name__ == "__main__":
    logger.info("Starting the bot.")
    client.run(discord_token, root_logger=True)
