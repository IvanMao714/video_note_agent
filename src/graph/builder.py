
import sys
from pathlib import Path
from langgraph.graph import END, START, StateGraph

# Add src directory to Python path to ensure imports work
# This is needed when the module is loaded directly by langgraph
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent.parent  # Go up from graph/ to src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from graph.nodes import (
    note_agent_node,
    slides_analysis_node,
    video_analysis_node,
)
from graph.state import AgentState
from log import get_logger


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges.
    
    Implements parallel execution following the example pattern:
    1. START -> slides_analysis (fan-out 1)
    2. START -> video_analysis (fan-out 2)
    3. slides_analysis -> note_agent (fan-in)
    4. video_analysis -> note_agent (fan-in)
    5. note_agent -> END
    
    By using multiple direct edges from START to slides_analysis and video_analysis,
    these two nodes will execute in parallel. Nodes internally check if execution is needed,
    returning immediately if not. Both nodes complete and enter note_agent, implementing fan-in.
    
    Returns:
        StateGraph: The graph builder instance.
    """
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("slides_analysis", slides_analysis_node)
    builder.add_node("video_analysis", video_analysis_node)
    builder.add_node("note_agent", note_agent_node)
    
    # From START to two analysis nodes (fan-out, parallel execution)
    builder.add_edge(START, "slides_analysis")
    builder.add_edge(START, "video_analysis")
    
    # From two analysis nodes to note_agent (fan-in)
    builder.add_edge("slides_analysis", "note_agent")
    builder.add_edge("video_analysis", "note_agent")
    
    # End after note_agent
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
    builder = _build_base_graph()
    
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


logger = get_logger(__name__)

# store = get_store_by_type("postgres")
graph = build_graph()