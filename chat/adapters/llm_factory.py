import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import settings

_GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class GoogleVertexAISettings:
    credential_path: Path | None
    project: str | None
    location: str


def _read_google_vertexai_settings(
    env: Mapping[str, str] | None = None,
) -> GoogleVertexAISettings:
    source = os.environ if env is None else env
    credential_path = source.get("GOOGLE_APPLICATION_CREDENTIALS")

    return GoogleVertexAISettings(
        credential_path=Path(credential_path).expanduser() if credential_path else None,
        project=source.get("GOOGLE_CLOUD_PROJECT"),
        location=source.get("GOOGLE_CLOUD_LOCATION", "global"),
    )


def _load_google_credentials(
    google_settings: GoogleVertexAISettings,
    credentials_loader: Callable[..., Any] | None = None,
) -> tuple[Any | None, str | None]:
    if google_settings.credential_path is None:
        return None, google_settings.project

    if credentials_loader is None:
        from google.oauth2.service_account import Credentials

        credentials_loader = Credentials.from_service_account_file

    credentials = credentials_loader(
        str(google_settings.credential_path),
        scopes=[_GOOGLE_CLOUD_PLATFORM_SCOPE],
    )
    project = google_settings.project or getattr(credentials, "project_id", None)

    return credentials, project


def _build_google_llm_kwargs(
    model_name: str,
    env: Mapping[str, str] | None = None,
    credentials_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    google_settings = _read_google_vertexai_settings(env)
    credentials, project = _load_google_credentials(
        google_settings, credentials_loader=credentials_loader
    )

    google_kwargs: dict[str, Any] = {
        "model": model_name,
        "streaming": True,
        "vertexai": True,
        "location": google_settings.location,
    }
    if project:
        google_kwargs["project"] = project
    if credentials is not None:
        google_kwargs["credentials"] = credentials

    return google_kwargs


def _build_google_llm(model_name: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(**_build_google_llm_kwargs(model_name))


def _build_openai_llm(model_name: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model_name=model_name,
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "high", "summary": "detailed"},
        streaming=True,
    )


def build_llm(provider: str, model_name: str) -> BaseChatModel:
    normalized_provider = provider.lower()

    if normalized_provider == "google":
        return _build_google_llm(model_name)
    if normalized_provider == "openai":
        return _build_openai_llm(model_name)

    raise ValueError(f"Unsupported model provider: {provider}")


MODEL = build_llm(settings.MODEL_PROVIDER, settings.MODEL_NAME)
