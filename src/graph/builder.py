from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    note_agent_node,
    slides_analysis_node,
    video_analysis_node, make_save_state_node,
)
from src.graph.state import AgentState
from src.log import get_logger
logger = get_logger(__name__)


def _build_base_graph(store=None):
    """Build and return the base state graph with all nodes and edges.
    
    Args:
        store: Optional store instance for state persistence. If provided,
            a save_state node will be added to the graph.
    
    Returns:
        StateGraph: The graph builder instance with nodes and edges configured.
    """
    builder = StateGraph(AgentState)

    builder.add_node("slides_analysis", slides_analysis_node)
    builder.add_node("video_analysis", video_analysis_node)
    builder.add_node("note_agent", note_agent_node)

    # Add save_state node only if store is provided
    if store is not None:
        builder.add_node("save_state", make_save_state_node(store))

    builder.add_edge(START, "slides_analysis")
    builder.add_edge(START, "video_analysis")

    builder.add_edge("slides_analysis", "note_agent")
    builder.add_edge("video_analysis", "note_agent")

    # After note_agent: save state then end if store exists, otherwise end directly
    if store is not None:
        builder.add_edge("note_agent", "save_state")
        builder.add_edge("save_state", END)
    else:
        builder.add_edge("note_agent", END)

    return builder
def build_graph(store=None, checkpointer=None):
    """Build and return the agent workflow graph with optional memory/store.
    
    Args:
        store: Optional store instance (e.g., InMemoryStore).
            If provided, uses the new store API.
        checkpointer: Optional checkpointer instance (e.g., PostgresSaver).
            Used if store is not provided. This is the recommended approach for LangGraph.
    
    Returns:
        Compiled graph instance.
    """
    builder = _build_base_graph(store=store)
    
    if store is not None and checkpointer is not None:
        return builder.compile(store=store, checkpointer=checkpointer)
    elif checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    elif store is not None:
        # Direct store instance (e.g., InMemoryStore)
        if hasattr(store, 'setup'):
            store.setup()
        return builder.compile(store=store)
    else:
        return builder.compile()

graph = build_graph()