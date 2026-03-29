import os

from dotenv import load_dotenv

load_dotenv(".env")


class Settings:
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "openai")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5.4-mini")
    STREAM_VERSION: str = "v2"


settings = Settings()
