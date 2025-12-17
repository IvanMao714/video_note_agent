from typing import List, Dict

from langgraph.graph import MessagesState


class AgentState(MessagesState):

    plan_iterations: int = 0

    slides_input_path : str

    slides_list: List[Dict]
    slides_summary: List[str]
    
    video_input_path: str = ""  # Video input path
    video_oss_suffix: str = ""  # OSS file suffix for video storage (optional, user-specified)
    video_transcript: str = ""  # Video transcript text
    
    notes: str = ""  # Generated notes
    slides_images: List[Dict] = []  # Store slide images in base64 encoding, format: [{"page_number": int, "image_base64": str}]