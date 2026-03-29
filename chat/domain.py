import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessageChunk, AnyMessage

THOUGHT_TITLE_PATTERN = re.compile(r"\*\*([^\n*][^\n]*?)\*\*\s*(?:\n|$)")
SYSTEM_PROMPT = """You are a demo assistant for a streaming chat workflow.
Follow these rules:
- When the user explicitly asks to use a named tool, call it.
- After tool results arrive, answer briefly and clearly.
- If a tool fails, acknowledge the failure and continue with a helpful answer."""


@dataclass(frozen=True, slots=True)
class ChatWorkflowState:
    messages: list[AnyMessage] = field(default_factory=list)
    final_response: str | None = None
    final_usage: dict[str, Any] | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class ChatContext:
    chat_mode: str = "fast"


def extract_text_blocks(chunk: AIMessageChunk) -> list[str]:
    deltas: list[str] = []
    for block in chunk.content if isinstance(chunk.content, list) else []:
        if isinstance(block, dict) and block.get("type") == "text":
            if text := block.get("text", ""):
                deltas.append(text)
    if isinstance(chunk.content, str) and chunk.content:
        deltas.append(chunk.content)
    return deltas


def extract_reasoning_blocks(chunk: AIMessageChunk) -> list[str]:
    deltas: list[str] = []
    for block in chunk.content if isinstance(chunk.content, list) else []:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            for summary in block.get("summary", []):
                if isinstance(summary, dict) and summary.get("type") == "summary_text":
                    if text := summary.get("text", ""):
                        deltas.append(text)
    return deltas


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return {"value": str(usage)}


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
