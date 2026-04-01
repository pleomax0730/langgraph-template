import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import sentinel

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("MODEL_PROVIDER", "openai")

from unittest.mock import patch

from chat.adapters.llm_factory import (  # noqa: E402
    _GOOGLE_CLOUD_PLATFORM_SCOPE,
    _build_google_llm,
    _build_google_llm_kwargs,
    _build_openai_llm,
    _read_google_vertexai_settings,
    build_llm,
)


class LlmFactoryTests(unittest.TestCase):
    def test_read_google_vertexai_settings_expands_paths_and_defaults_location(
        self,
    ) -> None:
        settings = _read_google_vertexai_settings(
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "~/service-account.json",
                "GOOGLE_CLOUD_PROJECT": "demo-project",
            }
        )

        self.assertEqual(
            settings.credential_path, Path("~/service-account.json").expanduser()
        )
        self.assertEqual(settings.project, "demo-project")
        self.assertEqual(settings.location, "global")

    def test_build_google_llm_kwargs_infers_project_from_credentials(self) -> None:
        captured: dict[str, object] = {}

        def fake_loader(path: str, *, scopes: list[str]) -> SimpleNamespace:
            captured["path"] = path
            captured["scopes"] = scopes
            return SimpleNamespace(project_id="credential-project")

        kwargs = _build_google_llm_kwargs(
            env={"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/service-account.json"},
            credentials_loader=fake_loader,
        )

        self.assertEqual(captured["path"], "/tmp/service-account.json")
        self.assertEqual(captured["scopes"], [_GOOGLE_CLOUD_PLATFORM_SCOPE])
        self.assertEqual(kwargs["project"], "credential-project")
        self.assertEqual(kwargs["location"], "global")
        self.assertTrue(kwargs["vertexai"])
        self.assertTrue(kwargs["streaming"])
        self.assertIn("credentials", kwargs)

    def test_build_google_llm_kwargs_prefers_explicit_project(self) -> None:
        def fake_loader(path: str, *, scopes: list[str]) -> SimpleNamespace:
            return SimpleNamespace(project_id="credential-project")

        kwargs = _build_google_llm_kwargs(
            env={
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/service-account.json",
                "GOOGLE_CLOUD_PROJECT": "explicit-project",
                "GOOGLE_CLOUD_LOCATION": "asia-east1",
            },
            credentials_loader=fake_loader,
        )

        self.assertEqual(kwargs["project"], "explicit-project")
        self.assertEqual(kwargs["location"], "asia-east1")

    def test_build_google_llm_uses_init_chat_model(self) -> None:
        with (
            patch(
                "chat.adapters.llm_factory._build_google_llm_kwargs",
                return_value={"vertexai": True, "streaming": True},
            ) as google_kwargs_mock,
            patch(
                "chat.adapters.llm_factory.init_chat_model",
                return_value=sentinel.google_llm,
            ) as init_chat_model_mock,
        ):
            result = _build_google_llm("gemini-3-flash-preview")

        google_kwargs_mock.assert_called_once_with()
        init_chat_model_mock.assert_called_once_with(
            "google_genai:gemini-3-flash-preview",
            vertexai=True,
            streaming=True,
        )
        self.assertIs(result, sentinel.google_llm)

    def test_build_openai_llm_uses_init_chat_model(self) -> None:
        with patch(
            "chat.adapters.llm_factory.init_chat_model",
            return_value=sentinel.openai_llm,
        ) as init_chat_model_mock:
            result = _build_openai_llm("gpt-5.4-mini")

        init_chat_model_mock.assert_called_once_with(
            "openai:gpt-5.4-mini",
            use_responses_api=True,
            output_version="responses/v1",
            reasoning={"effort": "high", "summary": "detailed"},
            streaming=True,
        )
        self.assertIs(result, sentinel.openai_llm)

    def test_build_llm_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            build_llm("anthropic", "claude-sonnet")


if __name__ == "__main__":
    unittest.main()
