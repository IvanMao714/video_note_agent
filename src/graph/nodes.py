import json
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig

from src.graph.slide_analysis.utils import analyze_presentation, get_slides_images, encode_image
from src.graph.state import AgentState
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template
from src.log import get_logger
from datetime import datetime, timezone
logger = get_logger(__name__)

async def slides_analysis_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Node for analyzing slides and generating summaries.
    
    If slides_input_path is not provided, returns immediately without modifying state.
    This allows the node to complete quickly even when execution is not needed,
    preventing workflow blocking.
    
    Args:
        state: The current agent state.
        config: Runnable configuration.
        
    Returns:
        Dictionary containing slides_list if analysis was performed, empty dict otherwise.
    """
    logger.trace("\n"+"-"*80+"\nSlides Analysis Node\n"+"-"*80)
    slides_input_path = state.get("slides_input_path")
    if not slides_input_path:
        logger.debug("No slides_input_path provided, skipping slides analysis")
        return {}
    
    logger.debug(f"Slides Analysis Node processing slides from: {slides_input_path}")
    
    try:
        slides_list = await analyze_presentation(slides_input_path)
        logger.debug(f"Successfully analyzed {len(slides_list)} slides")
        return {"slides_list": slides_list}
    except Exception as e:
        logger.error(f"Failed to analyze presentation: {e}", exc_info=True)
        return {"slides_list": []}


async def video_analysis_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Node for analyzing video and generating transcript.
    
    If video_input_path is not provided, returns immediately without modifying state.
    This allows the node to complete quickly even when execution is not needed,
    preventing workflow blocking.
    
    Args:
        state: The current agent state.
        config: Runnable configuration.
        
    Returns:
        Dictionary containing video_transcript if analysis was performed, empty dict otherwise.
    """
    logger.trace("\n" + "-" * 80 + "\nVideo Analysis Node\n" + "-" * 80)
    video_input_path = state.get("video_input_path")
    if not video_input_path:
        logger.debug("No video_input_path provided, skipping video analysis")
        return {}
    
    logger.debug(f"Video Analysis Node processing video from: {video_input_path}")
    
    try:
        # Get ASR instance for video transcription
        asr = get_llm_by_type("asr")
        
        # Get OSS file suffix from user-specified state, or use None (will use ASR instance default)
        oss_file_suffix = state.get("video_oss_suffix") or None
        
        # Call ASR for transcription (invoke is synchronous, needs to run in executor)
        import asyncio
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, asr.invoke, video_input_path, oss_file_suffix)
        
        logger.debug(f"Successfully transcribed video, transcript length: {len(transcript)}")
        return {"video_transcript": transcript}
    except Exception as e:
        logger.error(f"Failed to transcribe video: {e}", exc_info=True)
        return {"video_transcript": ""}


# def find_matching_slides(note_content: str, slides_list: list, slides_images: list) -> list:
#     """Find the most relevant slide images based on note content.
#
#     Args:
#         note_content: Generated note content.
#         slides_list: List of slide analysis results, each containing page_number and content.
#         slides_images: List of slide images, each containing page_number and image_base64.
#
#     Returns:
#         List of matching slide images in format: [{"page_number": int, "image_base64": str, "relevance": str}]
#     """
#     if not slides_list or not slides_images:
#         return []
#
#     # Use LLM to match the most relevant slides
#     # Simplified processing: uses basic text matching, can be improved with embedding or LLM for more precise matching
#     matching_slides = []
#
#     # Convert slides_list to dictionary for fast lookup
#     slides_dict = {slide.get("page_number"): slide.get("content", "") for slide in slides_list}
#     images_dict = {img.get("page_number"): img.get("image_base64", "") for img in slides_images}
#
#     # Simple keyword matching (can be improved with more complex semantic matching)
#     note_lower = note_content.lower()
#     for page_num, slide_content in slides_dict.items():
#         if page_num in images_dict:
#             slide_lower = slide_content.lower()
#             # Simple relevance check: if note mentions keywords from slide
#             # Can be improved using embedding similarity or LLM judgment
#             if any(keyword in note_lower for keyword in slide_lower.split()[:10]):  # Check first 10 keywords
#                 matching_slides.append({
#                     "page_number": page_num,
#                     "image_base64": images_dict[page_num],
#                     "relevance": "matched"
#                 })
#
#     # If no matches found, return first few slides as fallback
#     if not matching_slides and slides_images:
#         matching_slides = slides_images[:3]  # Return first 3 as default
#
#     return matching_slides


async def note_agent_node(
    state: AgentState, config: RunnableConfig
) -> dict:
    """Node for generating notes based on video transcript and slides analysis.
    
    Args:
        state: The current agent state containing video transcript and slides data.
        config: Runnable configuration.
        
    Returns:
        Dictionary containing generated notes and matched slide images.
    """
    
    # Get video transcript and slides analysis results
    video_transcript = state.get("video_transcript", "")
    slides_list = state.get("slides_list", [])
    slides_input_path = state.get("slides_input_path", "")
    user_query = state.get("user_query", "")
    feedback = state.get("notes_feedback", "")
    prev_notes = state.get("notes", "")
    
    # Prepare content for note generation
    has_video = bool(video_transcript)
    has_slides = bool(slides_list)
    
    if not has_video and not has_slides:
        logger.warning("No video transcript or slides available for note generation")
        state["notes"] = "Unable to generate notes: missing video transcript or slide content."
        return {"notes": state["notes"]}
    
    # Build input for note generation
    note_input = {
        "user_query": user_query,
        "has_video": has_video,
        "has_slides": has_slides,
    }

    messages_content = []
    if not feedback:
        if has_video:
            note_input["video_transcript"] = video_transcript

        if has_slides:
            # Convert slides_list to structured format
            # Each slide's content is already structured according to slide_analyzer.md format
            # Contains: Executive Summary, Visual Analysis, Textual Content, Key Insights
            slides_parts = []
            for slide in slides_list:
                slides_parts.append(f"## Slide {slide.get('page_number', '?')}\n\n{slide.get('content', '')}")
            slides_text = "\n\n".join(slides_parts)
            note_input["slides_content"] = slides_text

        # Build messages for note generation
        if user_query:
            messages_content.append({"type": "text", "text": f"User question: {user_query}\n\n"})

        messages_content.append({"type": "text", "text": "Please generate detailed notes based on the following content:\n\n"})

        if has_video:
            messages_content.append({"type": "text", "text": f"## Video Transcript Content\n\n{video_transcript}\n\n"})

        if has_slides:
            messages_content.append({
                "type": "text",
                "text": f"## Slide Content (Structured Analysis)\n\n"
                       f"The following slide content has been analyzed in structured format, each slide contains:\n"
                       f"- Executive Summary\n"
                       f"- Visual Analysis\n"
                       f"- Textual Content\n"
                       f"- Key Insights\n\n"
                       f"{slides_text}\n\n"
            })
    else:
        messages_content.append({
            "type": "text",
            "text": (
                "IMPORTANT: The previous notes were reviewed and need improvement.\n"
                "Please revise/rewrite the notes using the following review feedback.\n\n"
                f"## Review Feedback (rewrite instructions)\n{feedback}\n\n"
                "## Previous Notes (to revise)\n"
                f"{prev_notes}\n\n"
            )
        })

    note_state = AgentState(
        messages=[{
            "role": "user",
            "content": messages_content
        }]
    )
    
    try:
        # Apply note_generator prompt template
        # This prompt file explains how to use the structured format from slide_analyzer
        messages = apply_prompt_template("note_generator", note_state)

        
        # Call LLM to generate notes
        llm = get_llm_by_type("basic")
        response = await llm.ainvoke(messages)
        notes = response.content
        state["notes"] = notes
        # logger.info(f"Successfully generated notes, length: {len(notes)}")
        
        # # If slides exist, extract images and match relevant slide images
        # if has_slides and slides_input_path:
        #     try:
        #         slides_images = get_slides_images(slides_input_path)
        #         state["slides_images"] = slides_images
        #
        #         # Match most relevant slide images based on note content
        #         matching_slides = find_matching_slides(notes, slides_list, slides_images)
        #         # Can add matched slides to notes or store separately
        #         logger.info(f"Found {len(matching_slides)} matching slides")
        #     except Exception as e:
        #         logger.error(f"Failed to extract or match slide images: {e}", exc_info=True)
        #         state["slides_images"] = []
        
    except Exception as e:
        logger.error(f"Failed to generate notes: {e}", exc_info=True)
        state["notes"] = f"Error generating notes: {str(e)}"

    attempts = int(state.get("notes_attempts", 0)) + 1

    logger.trace("\nNote Agent Node\n"
                 + f"Generated Notes:{state.get('notes')[:200]}...\n"
                 + f"Attempts: {attempts}\n"
                 + "-" * 80)
    # Return updated state
    return {
        "notes_attempts": attempts,
        "notes": state.get("notes", ""),
        "slides_images": state.get("slides_images", []),
    }


async def note_review_node(state: AgentState, config: RunnableConfig) -> dict:
    notes = state.get("notes", "")
    if not notes.strip():
        return {
            "notes_ok": False,
            "notes_review": {"ok": False, "score": 0, "issues": ["Empty notes"], "rewrite_instructions": "Generate notes first."},
            "notes_feedback": "Notes are empty. Please generate complete notes.",
        }

    # 你也可以把 user_query / slides_list / transcript 作为上下文给 critic
    review_state = AgentState(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a strict study-notes reviewer.\n"},
            {"type": "text", "text": f"User question: {state.get('user_query','')}\n\n"},
            {"type": "text", "text": "Please review the following notes for student learning quality.\n\n"},
            {"type": "text", "text": notes},
        ]
    }])

    messages = apply_prompt_template("note_quality_checker", review_state)

    llm = get_llm_by_type("basic")
    resp = await llm.ainvoke(messages)
    raw = resp.content

    # 期望 prompt 输出 JSON；这里做一个稳健解析
    review: Dict[str, Any] = {}
    try:
        review = json.loads(raw)
    except Exception:
        # fallback：把纯文本当成 feedback
        review = {
            "ok": False,
            "score": 0,
            "issues": ["Non-JSON output from reviewer"],
            "rewrite_instructions": raw.strip()[:4000],
        }

    ok = bool(review.get("ok", False))
    rewrite_instructions = review.get("rewrite_instructions", "")

    # logger.trace(f"review raw head: {(raw or '')[:200]}")
    # logger.trace(
    #     f"review parsed: ok={review.get('ok')} score={review.get('score')} rewrite_head={(review.get('rewrite_instructions', '') or '')[:120]}")
    logger.trace("\nNote Review Node\n"
                 + f"Notes Feedback:{rewrite_instructions[:200]}...\n"
                 + "-" * 80)

    return {
        "notes_ok": ok,
        "notes_review": review,
        "notes_feedback": "" if ok else rewrite_instructions,
    }

def make_save_state_node(store):
    """Create a save state node function for the graph.
    
    Args:
        store: Store instance for persisting state data.
    
    Returns:
        Async function that saves the current state to the store.
    """
    logger.trace("\n" + "-" * 80 + "\nSave State Node\n" + "-" * 80)
    async def save_state_node(state: AgentState, config: RunnableConfig) -> dict:
        """Save the current agent state to the store.
        
        Args:
            state: The current agent state to save.
            config: Runnable configuration containing thread_id.
        
        Returns:
            Empty dictionary (state is saved to store, not returned).
        """
        if store is None:
            return {}

        # RunnableConfig is a TypedDict, can be accessed like a dict
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "no-thread") if configurable else "no-thread"

        payload = {
            "thread_id": thread_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "slides_input_path": state.get("slides_input_path", ""),
            "video_input_path": state.get("video_input_path", ""),
            "user_query": state.get("user_query", ""),
            "notes": state.get("notes", ""),
            "video_transcript": state.get("video_transcript", ""),
            "slides_list": state.get("slides_list", []),
        }

        # Extract filename from video_input_path and remove .mp4 extension
        video_input_path = state.get("video_input_path", "")
        video_oss_suffix = state.get("video_oss_suffix", "")
        
        # Get filename without .mp4 extension
        filename = ""
        if video_input_path:
            filename = video_input_path.split("/")[-1].split("\\")[-1]  # Handle both / and \ separators
            if filename.lower().endswith(".mp4"):
                filename = filename[:-4]  # Remove .mp4 extension
        
        # Convert video_oss_suffix to tuple for namespace (PostgresStore requires tuple for namespace)
        # If video_oss_suffix is "cs336/video", convert to ("cs336", "video")
        namespace = tuple(video_oss_suffix.split("/")) if video_oss_suffix else ("default",)
        
        # PostgresStore typically uses async aput; fallback to sync put for compatibility
        if hasattr(store, "aput"):
            await store.aput(namespace, filename, payload)
        else:
            store.put(namespace, filename, payload)

        return {}
    return save_state_node
