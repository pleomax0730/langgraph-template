import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..adapters.graph_builder import GLOBAL_APP
from ..config import settings
from ..domain import (
    SYSTEM_PROMPT,
    THOUGHT_TITLE_PATTERN,
    ChatContext,
    ChatWorkflowState,
    extract_reasoning_blocks,
    extract_text_blocks,
)

logger = logging.getLogger("chat.use_cases")


class StreamEventHandler:
    def __init__(self):
        self.reasoning_buffer = ""
        self.emitted_titles = set()

    def handle_messages(self, data: tuple[BaseMessage, dict[str, Any]]) -> Any:
        token, _metadata = data
        if not isinstance(token, AIMessageChunk):
            return
        for delta in extract_reasoning_blocks(token):
            self.reasoning_buffer += delta
            if "**" not in self.reasoning_buffer:
                yield {"event": "thought", "delta": delta}
                continue
            matches = list(THOUGHT_TITLE_PATTERN.finditer(self.reasoning_buffer))
            for m in matches:
                t = m.group(1).strip()
                if t not in self.emitted_titles:
                    yield {"event": "thought", "title": t}
                    self.emitted_titles.add(t)
            if matches:
                self.reasoning_buffer = self.reasoning_buffer[matches[-1].end() :]
            yield {"event": "thought", "delta": delta}
        for delta in extract_text_blocks(token):
            yield {"event": "llm_progress", "delta": delta}

    def handle_custom(self, data: Any) -> Any:
        yield data

    def handle_updates(self, data: dict[str, Any]) -> Any:
        agent_update = data.get("agent")
        if not agent_update or not agent_update.get("final_response"):
            return
        resp = {"event": "final_response", "response": agent_update["final_response"]}
        if agent_update.get("final_usage"):
            resp["usage"] = agent_update["final_usage"]
        if agent_update.get("last_error"):
            resp["error"] = agent_update["last_error"]
        yield resp


def _build_input_messages(
    user_input: str, chat_history_dicts: list | None = None
) -> list[AnyMessage]:
    input_messages: list[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    if chat_history_dicts:
        for msg in chat_history_dicts:
            if msg.role == "user":
                input_messages.append(
                    HumanMessage(content=msg.content or "", name=msg.name)
                )
            elif msg.role == "assistant":
                kwargs = {"content": msg.content or ""}
                if msg.name:
                    kwargs["name"] = msg.name
                if msg.tool_calls:
                    kwargs["tool_calls"] = msg.tool_calls
                input_messages.append(AIMessage(**kwargs))
            elif msg.role == "tool":
                input_messages.append(
                    ToolMessage(
                        content=msg.content or "",
                        name=msg.name or "",
                        tool_call_id=msg.tool_call_id or "",
                    )
                )

    if user_input:
        input_messages.append(HumanMessage(content=user_input))

    return input_messages


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    if hasattr(state, "value"):
        return _state_value(state.value, key, default)
    return getattr(state, key, default)


async def stream_execute(
    user_input: str, chat_mode: str, chat_history_dicts: list | None = None
) -> AsyncGenerator[dict[str, Any]]:
    app = GLOBAL_APP
    input_messages = _build_input_messages(user_input, chat_history_dicts)
    handler = StreamEventHandler()
    dispatch_map = {
        "messages": handler.handle_messages,
        "custom": handler.handle_custom,
        "updates": handler.handle_updates,
    }

    try:
        async for chunk in app.astream(
            ChatWorkflowState(messages=input_messages),
            context=ChatContext(chat_mode=chat_mode),
            stream_mode=["messages", "custom", "updates"],
            version=settings.STREAM_VERSION,
        ):
            if process_func := dispatch_map.get(chunk["type"]):
                for event in process_func(chunk["data"]):
                    yield event
    except Exception as exc:
        logger.exception("Stream execution failed")
        yield {"event": "error", "error": str(exc)}


async def invoke_execute(
    user_input: str, chat_mode: str, chat_history_dicts: list | None = None
) -> dict[str, Any]:
    app = GLOBAL_APP
    input_messages = _build_input_messages(user_input, chat_history_dicts)

    try:
        final_state = await app.ainvoke(
            ChatWorkflowState(messages=input_messages),
            context=ChatContext(chat_mode=chat_mode),
            version=settings.STREAM_VERSION,
        )
        return {
            "response": _state_value(final_state, "final_response"),
            "usage": _state_value(final_state, "final_usage", {}),
        }
    except Exception as exc:
        logger.exception("Invoke execution failed")
        return {"error": str(exc)}
