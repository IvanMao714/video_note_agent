
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    note_agent_node,
    slides_analysis_node,
    video_analysis_node,
)
from graph.state import AgentState


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges.
    
    按照示例的方式实现并行执行：
    1. START -> slides_analysis (扇出 1)
    2. START -> video_analysis (扇出 2)
    3. slides_analysis -> note_agent (扇入)
    4. video_analysis -> note_agent (扇入)
    5. note_agent -> END
    
    通过使用多个直接边从 START 到 slides_analysis 和 video_analysis，
    这两个节点会真正并行执行。节点内部会检查是否需要执行，如果不需要则立即返回。
    两个节点完成后都会进入 note_agent，实现扇入。
    """
    builder = StateGraph(AgentState)
    
    # 添加节点
    builder.add_node("slides_analysis", slides_analysis_node)
    builder.add_node("video_analysis", video_analysis_node)
    builder.add_node("note_agent", note_agent_node)
    
    # 从 START 到两个分析节点（扇出，实现并行执行）
    builder.add_edge(START, "slides_analysis")
    builder.add_edge(START, "video_analysis")
    
    # 从两个分析节点到 note_agent（扇入）
    builder.add_edge("slides_analysis", "note_agent")
    builder.add_edge("video_analysis", "note_agent")
    
    # note_agent 后结束
    builder.add_edge("note_agent", END)
    
    return builder


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
