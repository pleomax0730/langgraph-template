import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("MODEL_PROVIDER", "openai")

from chat.use_cases.chat_streamer import _state_value  # noqa: E402


class ChatStreamerTests(unittest.TestCase):
    def test_state_value_reads_dict_state(self) -> None:
        state = {"final_response": "ok"}

        self.assertEqual(_state_value(state, "final_response"), "ok")

    def test_state_value_reads_object_state(self) -> None:
        state = SimpleNamespace(final_response="ok")

        self.assertEqual(_state_value(state, "final_response"), "ok")

    def test_state_value_reads_langgraph_output_value(self) -> None:
        state = SimpleNamespace(value=SimpleNamespace(final_response="ok"))

        self.assertEqual(_state_value(state, "final_response"), "ok")

    def test_state_value_returns_default_when_missing(self) -> None:
        state = SimpleNamespace()

        self.assertEqual(_state_value(state, "final_usage", {}), {})


if __name__ == "__main__":
    unittest.main()
