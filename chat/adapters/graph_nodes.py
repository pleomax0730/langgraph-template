from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.runtime import Runtime

from ..config import settings
from ..domain import ChatContext, ChatWorkflowState, extract_text_blocks, usage_to_dict
from .llm_factory import MODEL
from .tools_registry import LANGCHAIN_TOOLS, TOOL_CATALOG, _execute_tool_call


def _bind_tools_for_provider(model, tools: list, provider: str):
    if not tools:
        return model
    if provider.lower() == "openai":
        return model.bind_tools(tools, parallel_tool_calls=False)
    return model.bind_tools(tools)


async def agent_node(
    state: ChatWorkflowState, config: RunnableConfig, runtime: Runtime[ChatContext]
):
    mode = runtime.context.chat_mode
    current_tools = list(LANGCHAIN_TOOLS) if mode == "plan" else []
    llm_with_tools = _bind_tools_for_provider(
        MODEL,
        current_tools,
        settings.MODEL_PROVIDER,
    )
    final_message = await llm_with_tools.ainvoke(state.messages, config)
    text_blocks = extract_text_blocks(final_message)
    final_response = (
        "".join(text_blocks) if text_blocks and not final_message.tool_calls else None
    )
    return {
        "messages": state.messages + [final_message],
        "final_response": final_response,
        "final_usage": (
            usage_to_dict(final_message.usage_metadata) if final_response else None
        ),
        "last_error": None,
    }


async def tools_node(state: ChatWorkflowState):
    writer = get_stream_writer()
    last_message = state.messages[-1]
    tool_messages: list[ToolMessage] = []

    if not isinstance(last_message, AIMessage):
        raise TypeError("Expected AIMessage")
    for call in last_message.tool_calls:
        writer(
            {
                "event": "tool_call",
                "tool_name": call["name"],
                "tool_kwargs": call.get("args", {}),
            }
        )
        writer(
            {"event": "tool_progress", "tool_name": call["name"], "status": "progress"}
        )
        result = await _execute_tool_call(call, TOOL_CATALOG)  # type: ignore
        writer(
            {
                "event": "tool_result",
                "tool_name": result.tool_name,
                "tool_id": result.tool_id,
                "result": result.result,
                "error": result.error,
            }
        )
        tool_messages.append(
            ToolMessage(
                content=(
                    result.llm_context if result.llm_context else (result.error or "")
                ),
                tool_call_id=result.tool_id,
                name=result.tool_name,
                status="error" if result.error else "success",
            )
        )
    return {"messages": state.messages + tool_messages}


def should_continue(state: ChatWorkflowState) -> str:
    last_message = state.messages[-1]
    return (
        "tools"
        if isinstance(last_message, AIMessage) and last_message.tool_calls
        else END
    )
