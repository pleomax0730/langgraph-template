from langgraph.graph import END, START, StateGraph

from ..domain import ChatContext, ChatWorkflowState
from .graph_nodes import agent_node, should_continue, tools_node


def build_graph():
    graph = StateGraph(ChatWorkflowState, context_schema=ChatContext)
    graph.add_node("agent", agent_node)  # type: ignore
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(name="fastapi_chat_graph")


GLOBAL_APP = build_graph()
