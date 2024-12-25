from __future__ import annotations

import logging
import os
import sys

import hikari
import lightbulb
import openai
from dotenv import load_dotenv
from openai import OpenAI

from misc import chat, get_allowed_users, get_trigger_keywords

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv(verbose=True)

discord_token: str | None = os.getenv("DISCORD_TOKEN")
openai_api_key: str | None = os.getenv("OPENAI_TOKEN")
if not discord_token or not openai_api_key:
    logger.error("You haven't configured the bot correctly. Please set the environment variables.")
    sys.exit(1)


bot = hikari.GatewayBot(
    token=discord_token,
    intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.GUILD_MESSAGE_TYPING,
    logs="INFO",
)
bot_client: lightbulb.GatewayEnabledClient = lightbulb.client_from_app(bot)
bot.subscribe(hikari.StartingEvent, bot_client.start)

openai_client = OpenAI(api_key=openai_api_key)


@bot_client.register()
class Ask(
    lightbulb.SlashCommand,
    name="ask",
    description="Ask the AI a question.",
):
    """A command to ask the AI a question."""

    text: str = lightbulb.string("text", "The question or message to ask the AI.")

    @lightbulb.invoke
    async def invoke(self, ctx: lightbulb.Context) -> None:
        """Handle the /ask command."""
        # Only allow certain users to interact with the bot
        allowed_users: list[str] = get_allowed_users()
        if ctx.user.username not in allowed_users:
            logger.info("Ignoring message from: %s", ctx.user.username)
            await ctx.respond("You are not allowed to use this command.", ephemeral=True)
            return

        if not self.text:
            logger.error("No question or message provided.")
            await ctx.respond("You need to provide a question or message.")
            return

        try:
            response: str | None = chat(self.text, openai_client)
        except openai.OpenAIError as e:
            logger.exception("An error occurred while chatting with the AI model.")
            await ctx.respond(f"An error occurred: {e}")
            return

        if response:
            await ctx.respond(response)
        else:
            await ctx.respond("I forgor how to think 💀")


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
    trigger_keywords: list[str] = get_trigger_keywords(bot)
    if any(trigger in lowercase_message for trigger in trigger_keywords):
        logger.info("Received message: %s from: %s", incoming_message, event.author.username)

        async with bot.rest.trigger_typing(event.channel_id):
            try:
                response: str | None = chat(incoming_message, openai_client)
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


if __name__ == "__main__":
    logger.info("Starting the bot.")
    bot.run()
