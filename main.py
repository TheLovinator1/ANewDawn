from __future__ import annotations

import datetime
import io
import logging
from typing import TYPE_CHECKING

import discord
import httpx
import openai
from discord import app_commands
from openai import OpenAI

from misc import chat, get_allowed_users
from settings import Settings

if TYPE_CHECKING:
    from openai.types import ImagesResponse
    from openai.types.image import Image

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
        """Setup the bot client."""
        # Copy the global commands to all the guilds so we don't have to wait 1 hour for the commands to be available
        self.tree.copy_global_to(guild=discord.Object(id=98905546077241344))  # KillYoy's server
        self.tree.copy_global_to(guild=discord.Object(id=341001473661992962))  # TheLovinator's server

        # Sync commands globally
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
            logger.error("No message content found in the event: %s", message)
            return

        lowercase_message: str = incoming_message.lower() if incoming_message else ""
        trigger_keywords: list[str] = ["lovibot", "<@345000831499894795>"]
        if any(trigger in lowercase_message for trigger in trigger_keywords):
            logger.info("Received message: %s from: %s", incoming_message, message.author.name)

            async with message.channel.typing():
                try:
                    response: str | None = chat(incoming_message, openai_client)
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


# Everything enabled except `presences`, `members`, and `message_content`.
intents: discord.Intents = discord.Intents.default()
intents.message_content = True
client = LoviBotClient(intents=intents)


@client.tree.context_menu(name="Ask LoviBot")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def handle_ai_query(interaction: discord.Interaction, message: discord.Message) -> None:
    """A context menu command to ask the AI a question."""
    await interaction.response.defer()

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received command from: %s", user_name_lowercase)

    allowed_users: list[str] = get_allowed_users()
    if user_name_lowercase not in allowed_users:
        logger.info("Ignoring message from: %s", user_name_lowercase)
        await interaction.followup.send("You are not allowed to use this command.", ephemeral=True)
        return

    try:
        response: str | None = chat(message.content, openai_client)
    except openai.OpenAIError as e:
        logger.exception("An error occurred while chatting with the AI model.")
        await interaction.followup.send(f"An error occurred: {e}")
        return

    if response:
        await interaction.followup.send(response)
    else:
        await interaction.followup.send("I forgor how to think 💀")


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
        response: str | None = chat(text, openai_client)
    except openai.OpenAIError as e:
        logger.exception("An error occurred while chatting with the AI model.")
        await interaction.followup.send(f"An error occurred: {e}")
        return

    if response:
        await interaction.followup.send(response)
    else:
        await interaction.followup.send(f"I forgor how to think 💀\nText: {text}")


@client.tree.command(name="create_image", description="Create an image using a prompt.")
@app_commands.describe(prompt="The prompt to generate the image.")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def create_image(interaction: discord.Interaction, prompt: str) -> None:
    """A command to create an image using the AI."""
    await interaction.response.defer()

    if not prompt:
        logger.error("No prompt provided.")
        await interaction.followup.send("You need to provide a prompt.", ephemeral=True)
        return

    # Only allow certain users to interact with the bot
    allowed_users: list[str] = ["thelovinator", "killyoy"]

    user_name_lowercase: str = interaction.user.name.lower()
    logger.info("Received image creation command from: %s", user_name_lowercase)

    if user_name_lowercase not in allowed_users:
        logger.info("Ignoring image creation command from: %s", user_name_lowercase)
        await interaction.followup.send("You are not allowed to use this command.", ephemeral=True)
        return

    try:
        response: ImagesResponse = openai_client.images.generate(prompt=prompt, model="dall-e-3", quality="hd")
        data: list[Image] = response.data
        if not data:
            await interaction.followup.send("No image data found in the response.")
            return

        image_url: str | None = data[0].url
        if not image_url:
            await interaction.followup.send("No image URL found in the response.")
            return

        # Download the image with httpx
        async with httpx.AsyncClient() as client:
            image_response: httpx.Response = await client.get(image_url)
            image_response.raise_for_status()

        # Send the image as a file
        image_bytes: bytes = image_response.content

        iso8601_timestamp: str = datetime.datetime.now(tz=datetime.UTC).isoformat()

        file = discord.File(fp=io.BytesIO(image_bytes), filename=f"image-{iso8601_timestamp}.png")
        await interaction.followup.send(file=file)

    except openai.OpenAIError as e:
        logger.exception("An error occurred while creating the image.")
        await interaction.followup.send(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    logger.info("Starting the bot.")
    client.run(discord_token, root_logger=True)
