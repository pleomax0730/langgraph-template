import os
import unittest
from unittest.mock import Mock, sentinel

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("MODEL_PROVIDER", "openai")

from chat.adapters.graph_nodes import _bind_tools_for_provider  # noqa: E402


class GraphNodesTests(unittest.TestCase):
    def test_bind_tools_for_openai_disables_parallel_tool_calls(self) -> None:
        model = Mock()
        model.bind_tools.return_value = sentinel.bound_model
        tools = [sentinel.tool]

        result = _bind_tools_for_provider(model, tools, "openai")

        model.bind_tools.assert_called_once_with(tools, parallel_tool_calls=False)
        self.assertIs(result, sentinel.bound_model)

    def test_bind_tools_for_google_does_not_pass_openai_kwargs(self) -> None:
        model = Mock()
        model.bind_tools.return_value = sentinel.bound_model
        tools = [sentinel.tool]

        result = _bind_tools_for_provider(model, tools, "google")

        model.bind_tools.assert_called_once_with(tools)
        self.assertIs(result, sentinel.bound_model)

    def test_bind_tools_without_tools_returns_original_model(self) -> None:
        model = Mock()

        result = _bind_tools_for_provider(model, [], "openai")

        model.bind_tools.assert_not_called()
        self.assertIs(result, model)


if __name__ == "__main__":
    unittest.main()
