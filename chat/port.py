from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="Must be 'user', 'assistant' or 'tool'")
    content: str | None = Field(default="", description="Message content")
    name: str | None = Field(default=None, description="Optional name")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls made by the assistant"
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool call ID if role is 'tool'"
    )


class ChatCommandRequest(BaseModel):
    user_input: str = Field(description="Latest prompt from the user")
    chat_history: list[ChatMessage] | None = Field(
        default_factory=list, description="Historical exchange"
    )
    chat_mode: str = Field(
        default="plan", description="The strategy mode for graph behavior"
    )
