#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import argparse
import inspect
from pathlib import Path

from src.memory import get_store_by_type

from src.graph.builder import build_graph
from src.log import get_logger

logger = get_logger(__name__)

import asyncio

async def run_graph(slides_path: str | None, video_path: str | None, video_oss_suffix: str, user_query: str, use_memory: bool):
    """
    运行 graph
    
    注意：
    - Store (PostgresStore) 用于长期键值存储，不保存 graph 执行状态
    - Checkpointer (PostgresSaver) 用于保存 graph 的执行状态和检查点
    - 要保存 graph 执行状态到数据库，必须使用 checkpointer
    """
    store = None
    checkpointer = None
    
    if use_memory:
        try:
            # from memory import get_store_by_type, get_checkpointer_by_type
            # # 获取 checkpointer 用于保存 graph 执行状态
            # checkpointer = await get_checkpointer_by_type("postgres")
            # if checkpointer:
            #     logger.info("✓ PostgreSQL checkpointer enabled (will save graph state)")
            #
            # Store 用于长期存储（可选）
            store = get_store_by_type("postgres")
        except Exception as e:
            logger.warning(f"PostgreSQL memory unavailable, fallback to no-memory mode: {e}")
            logger.warning(f"Error details: {e}", exc_info=True)


    # 构建 graph：优先使用 checkpointer（保存执行状态）
    # 如果同时提供 store 和 checkpointer，两者都会使用
    graph = build_graph(store=store, checkpointer=checkpointer)

    # 初始 state（按你已有 state 字段）
    initial_state = {
        "slides_input_path": slides_path or "",
        "video_input_path": video_path or "",
        "user_query": user_query or "",
        "video_oss_suffix": video_oss_suffix or "",
        "slides_list": [],
        "slides_summary": [],
        "video_transcript": "",
        "notes": "",
        "slides_images": [],
        "messages": [],
    }

    # config：使用 checkpointer 时必须提供 thread_id
    # thread_id 用于标识不同的对话/会话
    thread_id = "test-thread-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    if checkpointer:
        logger.info(f"Using thread_id: {thread_id} (state will be saved to database)")

    # 执行 graph
    result = await graph.ainvoke(initial_state, config)
    
    # 如果使用 checkpointer，数据会自动保存
    # 可以通过 thread_id 在后续调用中恢复状态
    if checkpointer:
        logger.info(f"✓ Graph state saved to database (thread_id: {thread_id})")
        logger.info("You can resume this conversation later using the same thread_id")
    
    return result

#
# def _check_file(path_str: str | None, label: str):
#     if not path_str:
#         return
#     p = Path(path_str)
#     if not p.exists():
#         raise FileNotFoundError(f"{label} file not found: {p}")


def main():
    parser = argparse.ArgumentParser(description="Test LangGraph without langgraph dev")
    parser.add_argument("--slides", type=str, default="", help="PDF slides path (optional)")
    parser.add_argument("--video", type=str, default="", help="Video path (optional)")
    parser.add_argument("--query", type=str, default="", help="User query (optional)")
    parser.add_argument("--memory", default=True, action="store_true", help="Use PostgreSQL memory")
    parser.add_argument("--video_oss_suffix", type=str, default="", help="Video OSS suffix (optional)")
    args = parser.parse_args()

    slides_path = args.slides.strip() or None
    video_path = args.video.strip() or "E:\\video_note_agent\\example\\cs336_01.mp4"
    video_oss_suffix = args.video_oss_suffix.strip() or "cs336/video"

    if not slides_path and not video_path:
        raise ValueError("You must provide at least one input: --slides or --video")

    # _check_file(slides_path, "Slides")
    # _check_file(video_path, "Video")

    logger.info("=" * 60)
    logger.info(f"Slides: {slides_path or '(none)'}")
    logger.info(f"Video : {video_path or '(none)'}")
    logger.info(f"Query : {args.query or '(none)'}")
    logger.info(f"Video OSS Suffix: {video_oss_suffix or '(none)'}")
    logger.info(f"Memory: {args.memory}")
    logger.info("=" * 60)

    result = asyncio.run(run_graph(slides_path, video_path, video_oss_suffix, args.query, args.memory))


    # 保存结果
    out = Path("test_output.md")
    with out.open("w", encoding="utf-8") as f:
        f.write("=== INPUT ===\n")
        f.write(f"slides: {slides_path or ''}\n")
        f.write(f"video : {video_path or ''}\n")
        f.write(f"query : {args.query or ''}\n\n")

        if result.get("slides_list"):
            f.write(f"=== SLIDES ({len(result['slides_list'])}) ===\n")
            for slide in result["slides_list"]:
                f.write(f"\n[Page {slide.get('page_number', '')}]\n")
                f.write(slide.get("content", "") + "\n")

        if result.get("video_transcript"):
            f.write("\n=== TRANSCRIPT ===\n")
            f.write(result["video_transcript"] + "\n")

        if result.get("notes"):
            f.write("\n=== NOTES ===\n")
            f.write(result["notes"] + "\n")



if __name__ == "__main__":
    main()
