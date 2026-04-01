from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROMPT = (
    "In one sentence, say hello and mention the configured thinking setting."
)
CONFIG_PREFIX = "gemini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Gemini 3 Flash preview runtime configuration via "
            "langchain.chat_models.init_chat_model."
        )
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Gemini 3 thinking level to pass at runtime.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Gemini 2.5 thinking budget to pass at runtime.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Runtime temperature override.",
    )
    parser.add_argument(
        "--include-thoughts",
        action="store_true",
        help="Request thought content in the response when supported.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without sending a model request.",
    )
    return parser.parse_args()


def is_gemini_25_model(model_name: str) -> bool:
    return model_name.startswith("gemini-2.5")


def is_gemini_3_model(model_name: str) -> bool:
    return model_name.startswith("gemini-3")


def build_thinking_config(args: argparse.Namespace) -> dict[str, Any]:
    if is_gemini_25_model(args.model):
        thinking_budget = args.thinking_budget
        if thinking_budget is None:
            raise ValueError(
                "--thinking-budget is required when probing Gemini 2.5 models."
            )
        if args.thinking_level is not None:
            raise ValueError(
                "--thinking-level is only supported for Gemini 3 model families."
            )
        return {f"{CONFIG_PREFIX}_thinking_budget": thinking_budget}

    if is_gemini_3_model(args.model):
        return {
            f"{CONFIG_PREFIX}_thinking_level": args.thinking_level or "medium",
        }

    raise ValueError(
        "Unsupported Gemini model family for this probe. "
        "Use a gemini-3* model with --thinking-level or a gemini-2.5* model "
        "with --thinking-budget."
    )


def load_vertex_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "vertexai": True,
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        "temperature": 1.0,
        "include_thoughts": True,
    }

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credential_path:
        if project:
            defaults["project"] = project
        return defaults

    from google.oauth2.service_account import Credentials

    credentials = Credentials.from_service_account_file(
        str(Path(credential_path).expanduser()),
        scopes=[GOOGLE_CLOUD_PLATFORM_SCOPE],
    )
    defaults["credentials"] = credentials

    resolved_project = project or getattr(credentials, "project_id", None)
    if resolved_project:
        defaults["project"] = resolved_project

    return defaults


def build_runtime_config(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    configurable = {
        f"{CONFIG_PREFIX}_model": f"google_genai:{args.model}",
        f"{CONFIG_PREFIX}_temperature": args.temperature,
        f"{CONFIG_PREFIX}_include_thoughts": args.include_thoughts,
    }
    configurable.update(build_thinking_config(args))
    return {"configurable": configurable}


def build_model() -> Any:
    return init_chat_model(
        f"google_genai:{DEFAULT_MODEL}",
        configurable_fields=(
            "model",
            "thinking_level",
            "thinking_budget",
            "temperature",
            "include_thoughts",
        ),
        config_prefix=CONFIG_PREFIX,
        **load_vertex_defaults(),
    )


def serialize_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def main() -> None:
    load_dotenv(".env")
    args = parse_args()

    model = build_model()
    runtime_config = build_runtime_config(args)

    print("Resolved runtime config:")
    print(json.dumps(runtime_config, indent=2))

    if args.dry_run:
        return

    response = model.invoke(args.prompt, config=runtime_config)

    print("\nResponse:")
    print(serialize_response(response))

    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata:
        print("\nResponse metadata:")
        print(json.dumps(response_metadata, default=str, indent=2))


if __name__ == "__main__":
    main()
