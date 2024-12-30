from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(verbose=True)


@dataclass
class Settings:
    """Class to hold settings for the bot."""

    discord_token: str
    openai_api_key: str

    @classmethod
    @lru_cache(maxsize=1)
    def from_env(cls) -> Settings:
        """Create a new instance of the class from environment variables.

        Returns:
            A new instance of the class with the settings.
        """
        discord_token: str = os.getenv("DISCORD_TOKEN", "")
        openai_api_key: str = os.getenv("OPENAI_TOKEN", "")
        return cls(discord_token, openai_api_key)
