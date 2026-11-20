"""Application settings, loaded from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    llm_timeout_seconds: float = 60.0
    llm_max_output_tokens: int = 12000
    llm_temperature: float = 0.3
    # minimal | low | medium | high. Empty string = model default. Lower = faster and cheaper.
    llm_thinking_level: str = "low"

    request_timeout_seconds: float = 15.0
    max_response_bytes: int = 2_000_000
    max_content_chars: int = 24_000
    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
