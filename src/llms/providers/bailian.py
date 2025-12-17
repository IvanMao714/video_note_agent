
import json
import os
from http import HTTPStatus
# Standard library imports
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union, cast

import dashscope
from dashscope.audio.asr import Transcription
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from openai import base_url

# Third-party imports
import openai
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _create_usage_metadata,
    _handle_openai_bad_request,
    warnings,
)


from src.config.oss import OSSType
from src.llms.providers.utils import get_text_from_bailian_result
from src.oss.oss import get_oss_by_type
from src.log import get_logger

logger = get_logger(__name__)


def _convert_delta_to_message_chunk(
    delta_dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta dictionary to a message chunk.

    This function processes delta information from OpenAI streaming responses and
    converts it into the appropriate message chunk type based on the role and
    content type. It handles various message types including user, assistant,
    system, function, and tool messages, as well as function calls and tool calls.

    Args:
        delta_dict: Dictionary containing delta information from OpenAI response.
            Expected keys include 'id', 'role', 'content', 'function_call',
            'tool_calls', 'name', 'tool_call_id', and 'reasoning_content'.
        default_class: Default message chunk class to use if role is not
            specified in the delta dictionary.

    Returns:
        BaseMessageChunk: Appropriate message chunk instance based on role and
            content. Can be HumanMessageChunk, AIMessageChunk, SystemMessageChunk,
            FunctionMessageChunk, ToolMessageChunk, or ChatMessageChunk.

    Raises:
        KeyError: If required keys are missing from the delta dictionary.
    """
    message_id = delta_dict.get("id")
    role = cast(str, delta_dict.get("role", ""))
    content = cast(str, delta_dict.get("content") or "")
    additional_kwargs: Dict[str, Any] = {}

    # Handle function calls
    if function_call_data := delta_dict.get("function_call"):
        function_call = dict(function_call_data)
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    # Handle tool calls
    tool_call_chunks = []
    if raw_tool_calls := delta_dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc.get("function", {}).get("name"),
                    args=rtc.get("function", {}).get("arguments"),
                    id=rtc.get("id"),
                    index=rtc.get("index", 0),
                )
                for rtc in raw_tool_calls
                if rtc.get("function")  # Ensure function key exists
            ]
        except (KeyError, TypeError):
            # Log the error but continue processing
            pass

    # Return appropriate message chunk based on role
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=message_id)
    elif role == "assistant" or default_class == AIMessageChunk:
        # Handle reasoning content for OpenAI reasoning models
        if reasoning_content := delta_dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = reasoning_content
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=message_id,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    elif role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        return SystemMessageChunk(
            content=content, id=message_id, additional_kwargs=additional_kwargs
        )
    elif role == "function" or default_class == FunctionMessageChunk:
        function_name = delta_dict.get("name", "")
        return FunctionMessageChunk(content=content, name=function_name, id=message_id)
    elif role == "tool" or default_class == ToolMessageChunk:
        tool_call_id = delta_dict.get("tool_call_id", "")
        return ToolMessageChunk(
            content=content, tool_call_id=tool_call_id, id=message_id
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=message_id)
    else:
        return default_class(content=content, id=message_id)  # type: ignore


def _convert_chunk_to_generation_chunk(
    chunk: Dict[str, Any],
    default_chunk_class: Type[BaseMessageChunk],
    base_generation_info: Optional[Dict[str, Any]],
) -> Optional[ChatGenerationChunk]:
    """Convert a streaming chunk to a generation chunk.

    This function processes raw chunk data from OpenAI streaming responses and
    converts it into a ChatGenerationChunk object. It handles both standard
    streaming format and beta.chat.completions.stream format, extracts usage
    metadata, and includes generation information such as finish reasons and
    log probabilities.

    Args:
        chunk: Raw chunk data from OpenAI streaming response. Expected keys
            include 'type', 'usage', 'choices', 'chunk', 'model', and
            'system_fingerprint'.
        default_chunk_class: Default message chunk class to use when creating
            the message chunk from the delta.
        base_generation_info: Base generation information dictionary to include
            in the generation chunk. May contain headers or other metadata.

    Returns:
        Optional[ChatGenerationChunk]: Generated chunk with message and
            generation info, or None if the chunk should be skipped (e.g.,
            content.delta type chunks).
    """
    # Skip content.delta type chunks from beta.chat.completions.stream
    if chunk.get("type") == "content.delta":
        return None

    token_usage = chunk.get("usage")
    choices = (
        chunk.get("choices", [])
        # Handle chunks from beta.chat.completions.stream format
        or chunk.get("chunk", {}).get("choices", [])
    )

    usage_metadata: Optional[UsageMetadata] = (
        _create_usage_metadata(token_usage) if token_usage else None
    )

    # Handle empty choices
    if not choices:
        generation_chunk = ChatGenerationChunk(
            message=default_chunk_class(content="", usage_metadata=usage_metadata)
        )
        return generation_chunk

    choice = choices[0]
    if choice.get("delta") is None:
        return None

    message_chunk = _convert_delta_to_message_chunk(
        choice["delta"], default_chunk_class
    )
    generation_info = dict(base_generation_info) if base_generation_info else {}

    # Add finish reason and model info if available
    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
        if model_name := chunk.get("model"):
            generation_info["model_name"] = model_name
        if system_fingerprint := chunk.get("system_fingerprint"):
            generation_info["system_fingerprint"] = system_fingerprint

    # Add log probabilities if available
    if logprobs := choice.get("logprobs"):
        generation_info["logprobs"] = logprobs

    # Attach usage metadata to AI message chunks
    if usage_metadata and isinstance(message_chunk, AIMessageChunk):
        message_chunk.usage_metadata = usage_metadata

    generation_chunk = ChatGenerationChunk(
        message=message_chunk, generation_info=generation_info or None
    )
    return generation_chunk


class ChatDashscope(ChatOpenAI):
    """Extended ChatOpenAI model with reasoning capabilities.

    This class extends the base ChatOpenAI model to support OpenAI's reasoning models
    that include reasoning_content in their responses. It handles the extraction and
    preservation of reasoning content during both streaming and non-streaming operations.

    The class overrides key methods to ensure reasoning content is properly extracted
    from API responses and attached to message objects, making it available for
    downstream processing and analysis.
    """

    def _create_chat_result(
        self,
        response: Union[Dict[str, Any], openai.BaseModel],
        generation_info: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Create a chat result from the OpenAI response.

        This method processes the OpenAI API response and creates a ChatResult object.
        It extends the base implementation to extract and preserve reasoning content
        from reasoning model responses, attaching it to the message's additional_kwargs.

        Args:
            response: The response from OpenAI API. Can be either a dictionary or
                an openai.BaseModel instance. For reasoning models, should contain
                choices with messages that have reasoning_content.
            generation_info: Optional dictionary containing additional generation
                information to include in the result.

        Returns:
            ChatResult: The formatted chat result containing generations with
                reasoning content attached if available. The reasoning content
                is stored in the message's additional_kwargs dictionary.

        Note:
            If the response is not a BaseModel instance or if reasoning content
            extraction fails, the method falls back to the parent implementation
            without reasoning content.
        """
        chat_result = super()._create_chat_result(response, generation_info)

        # Only process BaseModel responses (not raw dict responses)
        if not isinstance(response, openai.BaseModel):
            return chat_result

        # Extract reasoning content if available
        try:
            if (
                hasattr(response, "choices")
                and response.choices
                and hasattr(response.choices[0], "message")
                and hasattr(response.choices[0].message, "reasoning_content")
            ):
                reasoning_content = response.choices[0].message.reasoning_content
                if reasoning_content and chat_result.generations:
                    chat_result.generations[0].message.additional_kwargs[
                        "reasoning_content"
                    ] = reasoning_content
        except (IndexError, AttributeError):
            # If reasoning content extraction fails, continue without it
            pass

        return chat_result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Create a streaming generator for chat completions.

        This method creates a streaming generator that yields ChatGenerationChunk
        objects as they arrive from the OpenAI API. It handles both standard
        streaming and beta.chat.completions.stream formats, supports response
        headers inclusion, and properly manages callbacks for token generation.

        Args:
            messages: List of BaseMessage objects representing the conversation
                history to send to the model.
            stop: Optional list of stop sequences that will cause the model to
                stop generating further tokens.
            run_manager: Optional CallbackManagerForLLMRun instance for handling
                callbacks during token generation. If provided, on_llm_new_token
                will be called for each generated chunk.
            **kwargs: Additional keyword arguments to pass to the API call.
                Common options include 'temperature', 'max_tokens', 'response_format',
                etc. The 'stream' parameter is automatically set to True.

        Yields:
            ChatGenerationChunk: Individual chunks from the streaming response,
                each containing a message chunk and optional generation information
                such as finish reasons, log probabilities, and usage metadata.

        Raises:
            openai.BadRequestError: If the API request is invalid or malformed.
                This exception is handled and re-raised with additional context
                via _handle_openai_bad_request.

        Note:
            For response_format requests, the method also handles final completion
            chunks if the response object supports get_final_completion().
        """
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        base_generation_info: Dict[str, Any] = {}

        # Handle response format for beta completions
        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_client.beta.chat.completions.stream(**payload)
            context_manager = response_stream
        else:
            # Handle regular streaming with optional response headers
            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = self.client.create(**payload)
            context_manager = response

        try:
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    # Convert chunk to dict if it's a model object
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()

                    generation_chunk = _convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )

                    if generation_chunk is None:
                        continue

                    # Update default chunk class for subsequent chunks
                    default_chunk_class = generation_chunk.message.__class__

                    # Handle log probabilities for callback
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )

                    is_first_chunk = False
                    yield generation_chunk

        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)

        # Handle final completion for response_format requests
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            try:
                final_completion = response.get_final_completion()
                generation_chunk = self._get_generation_chunk_from_completion(
                    final_completion
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk
                    )
                yield generation_chunk
            except AttributeError:
                # If get_final_completion method doesn't exist, continue without it
                pass


class ASRDashscope:
    """ASR (Automatic Speech Recognition) client for Alibaba DashScope.

    This class provides an interface to Alibaba DashScope's ASR transcription
    service, similar to LangChain's invoke pattern. It handles the submission
    of audio/video files for transcription and retrieves the resulting text.

    Attributes:
        model_name: Name of the ASR model to use (e.g., 'fun-asr').
        api_key: API key for authenticating with DashScope service.
    """

    def __init__(self, model: str, api_key: str, oss: OSSType, cache=True, oss_file_suffix = "video") -> None:
        """Initialize the ASRDashscope client.

        Args:
            model: Name of the ASR model to use for transcription.
            api_key: API key for DashScope authentication. Must be provided.
            base_url: Base URL for the DashScope API (currently not used).

        Raises:
            ValueError: If api_key is not provided or is empty.
        """
        self.model_name = model
        self.api_key = api_key
        self.cache = cache
        self.oss = get_oss_by_type(oss)
        self.oss_file_suffix = oss_file_suffix
        if not api_key:
            raise ValueError(
                "API key not provided. Please set environment variable DASHSCOPE_API_KEY, "
                "or set it in code via dashscope.api_key = your_api_key."
            )
        # dashscope.base_http_api_url = base_url
        dashscope.api_key = api_key

    def invoke(self, file_path: str, oss_file_suffix: str = None) -> str:
        """Transcribe audio/video file and return the text content.

        This method follows LangChain's invoke pattern. It submits a file URL
        for transcription, waits for the job to complete, and returns the
        transcribed text. Supports caching mechanism to avoid redundant API calls.

        Args:
            file_path: URL or path to the audio/video file to transcribe.
                Can be a local file path or a remote URL.
            oss_file_suffix: Optional suffix for OSS object path. If provided,
                overrides the instance-level oss_file_suffix setting.

        Returns:
            str: The transcribed text content from the audio/video file.
                Multiple channels are concatenated with "Channel N:" prefix.
                Returns empty string if transcription fails or data is invalid.

        Raises:
            Exception: If the transcription API call fails or returns non-200 status.

        Note:
            The method uses async_call to submit the transcription job and
            wait() to retrieve the results. The transcription is performed
            asynchronously by the DashScope service. Results are cached in OSS
            to improve performance for repeated requests.
            :param file_path:
            :param oss_file_suffix:
        """
        # Extract filename from path, handling both Unix and Windows paths
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)

        if oss_file_suffix:
            # User-specified suffix controls the OSS path.
            # Supported forms:
            # - "cs336/video"            -> bucket="notes", prefix="cs336/video"
            # - "notes/cs336/video"      -> bucket="notes", prefix="cs336/video" (leading "notes/" is ignored)
            self.oss_file_suffix = oss_file_suffix.strip().strip("/")  # Remove leading/trailing slashes

        # Do NOT read bucket name from YAML or client defaults here.
        # Bucket is fixed to "notes". The user-provided suffix is treated as object prefix only.
        bucket_name = "notes"
        raw_suffix = (self.oss_file_suffix or "").strip().strip("/")
        if raw_suffix.lower().startswith("notes/"):
            raw_suffix = raw_suffix[6:]
        object_prefix = raw_suffix.strip("/")

        # Path for JSON cache file (downloaded)
        # Ensure no leading slash, format: "video/cs336_02.json" or "cs336_02.json"
        oss_json_path = f"{object_prefix}/{base_name}.json" if object_prefix else f"{base_name}.json"
        # Path for MP4 file (uploaded)
        # Ensure no leading slash, format: "video/cs336_02.mp4" or "cs336_02.mp4"
        oss_mp4_path = f"{object_prefix}/{base_name}{ext}" if object_prefix else f"{base_name}{ext}"

        # URL for ASR service (DashScope will download this URL).
        # MinIO endpoints are often configured without scheme (host:port).
        endpoint = str(getattr(self.oss, "endpoint", "")).rstrip("/")
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            scheme = "https" if getattr(self.oss, "secure", False) else "http"
            endpoint = f"{scheme}://{endpoint}"
        oss_file_url = f"{endpoint}/{bucket_name}/{oss_mp4_path.lstrip('/')}"
        data = None

        if self.cache:
            logger.debug("Checking cache for JSON file: %s", oss_json_path)
            try:
                data_bytes = self.oss.download_file(bucket_name=bucket_name, object_name=oss_json_path)
                if data_bytes:
                    data_str = data_bytes.decode('utf-8')
                    data = json.loads(data_str)
                else:
                    logger.warning("Cache miss: file not found or empty")
            except Exception as e:
                logger.warning(f"Failed to download cache file: {e}")
                data = None

        if not self.cache or not data:
            # Upload MP4 file to OSS
            logger.debug("Uploading MP4 file to OSS: %s", oss_mp4_path)
            self.oss.upload_file(bucket_name=bucket_name, object_name=oss_mp4_path, file_path=file_path)

            task_id = Transcription.async_call(
                api_key=self.api_key,
                model=self.model_name,
                file_urls=[oss_file_url]
            )
            status_response = Transcription.wait(task=task_id)
            if status_response.status_code != 200:
                raise RuntimeError(
                    f"ASR task failed with status_code={status_response.status_code}. "
                    f"Response={status_response}"
                )

            # DashScope may put results in different places depending on SDK/version.
            # Prefer status_response.output, then fall back to task_id.output, then to mapping access.
            results = None
            output = getattr(status_response, "output", None)
            if isinstance(output, dict):
                results = output.get("results")

            if results is None:
                output2 = getattr(task_id, "output", None)
                if isinstance(output2, dict):
                    results = output2.get("results")

            if results is None:
                try:
                    # Some SDK responses behave like mappings
                    output3 = status_response.get("output") if hasattr(status_response, "get") else None
                    if isinstance(output3, dict):
                        results = output3.get("results")
                except Exception:
                    results = None

            if not results:
                out_keys = list(output.keys()) if isinstance(output, dict) else None
                out2 = getattr(task_id, "output", None)
                out2_keys = list(out2.keys()) if isinstance(out2, dict) else None
                raise RuntimeError(
                    "ASR task completed but no results were returned by DashScope. "
                    f"status_code={status_response.status_code}, "
                    f"status_response.output_keys={out_keys}, "
                    f"task_id.output_keys={out2_keys}"
                )

            total = len(results)
            for idx, entry in enumerate(results, start=1):
                if not isinstance(entry, dict) or "transcription_url" not in entry:
                    logger.warning("Invalid ASR result entry at index %s: %s", idx, entry)
                    continue

                transcription_url = entry.get("transcription_url")
                if not transcription_url:
                    logger.warning("Empty transcription_url at index %s", idx)
                    continue

                resp = requests.get(transcription_url)
                resp.raise_for_status()
                data = resp.json()

                json_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")

                if total == 1:
                    # Single result: xxx.json
                    json_file_name = f"{base_name}.json"
                else:
                    # Multiple results: xxx_1.json, xxx_2.json, ...
                    json_file_name = f"{base_name}_{idx}.json"

                # Use forward slash for OSS object names
                oss_json_path = (
                    f"{self.oss_file_suffix}/{json_file_name}"
                    if self.oss_file_suffix
                    else json_file_name
                )

                self.oss.upload_bytes(
                    bucket_name=bucket_name,
                    object_name=oss_json_path,
                    data=json_bytes,
                )

        text = ""
        if data and isinstance(data, dict) and 'transcripts' in data:
            for idx, trans in enumerate(data['transcripts']):
                text += "Channel " + str(idx) + ":\n"
                text += trans.get('text', '') + "\n"
        else:
            logger.warning("No transcripts data available or data format is invalid")
        return text
