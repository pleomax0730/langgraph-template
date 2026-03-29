from collections.abc import AsyncGenerator, Callable
from typing import Any

from .use_cases.chat_streamer import stream_execute


def get_stream_use_case() -> Callable[..., AsyncGenerator[dict[str, Any]]]:
    """DI Provider for the streaming chat use case.
    In the future, DB sessions or Auth users can be injected here.
    """
    return stream_execute
