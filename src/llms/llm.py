import logging
import os
from pathlib import Path
from typing import Any, Dict, get_args, Type

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.config.agents import LLMType
from src.llms.providers.bailian import ChatDashscope, ASRDashscope
from src.log import get_logger
from src.config import load_yaml_config

logger = get_logger(__name__)

# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}

# Cache for config file path to avoid blocking calls to os.getcwd()
# Initialize cache at module load time to avoid blocking calls in async contexts
_config_file_path_cache: str = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())

# Allowed LLM configuration keys to prevent unexpected parameters from being passed
# to LLM constructors (Issue #411 - SEARCH_ENGINE warning fix)
ALLOWED_LLM_CONFIG_KEYS = {
    # Common LLM configuration keys
    "model",
    "api_key",
    "base_url",
    "api_base",
    "max_retries",
    "timeout",
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "stream",
    "logprobs",
    "echo",
    "best_of",
    "logit_bias",
    "user",
    "seed",
    # SSL and HTTP client settings
    "verify_ssl",
    "http_client",
    "http_async_client",
    # Platform-specific keys
    "platform",
    "google_api_key",
    # Azure-specific keys
    "azure_endpoint",
    "azure_deployment",
    "api_version",
    "azure_ad_token",
    "azure_ad_token_provider",
    # Dashscope/Doubao specific keys
    "extra_body",
    # Token limit for context compression (removed before passing to LLM)
    "token_limit",
    # Default headers
    "default_headers",
    "default_query",
    "oss"
}


def _get_config_file_path() -> str:
    """Get the path to the configuration file.
    
    Returns:
        The absolute path to the conf.yaml configuration file located in the
        project root directory.
    
    Note:
        The path is cached at module load time to avoid blocking calls to
        os.getcwd() in async contexts.
    """
    return _config_file_path_cache


def _get_llm_type_config_keys() -> dict[str, str]:
    """Get mapping of LLM types to their configuration keys.
    
    Returns:
        A dictionary mapping LLM type names to their corresponding configuration
        key names used in the configuration file. The keys are: "reasoning",
        "basic", "vision", and "code".
    """
    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
        "asr": "ASR_MODEL",
    }


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """Get LLM configuration from environment variables.
    
    Retrieves LLM configuration from environment variables that follow the
    naming convention: {LLM_TYPE}_MODEL__{KEY}. For example, BASIC_MODEL__api_key
    or BASIC_MODEL__base_url.
    
    Args:
        llm_type: The type of LLM (e.g., "basic", "reasoning", "vision", "code").
            This will be converted to uppercase and used as a prefix.
    
    Returns:
        A dictionary containing the LLM configuration parsed from environment
        variables. Keys are converted to lowercase. Returns an empty dictionary
        if no matching environment variables are found.
    
    Example:
        If llm_type is "basic" and environment contains BASIC_MODEL__api_key=xxx,
        the returned dictionary will be {"api_key": "xxx"}.
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    return conf


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> ChatOpenAI | Type[ASRDashscope] | ChatDashscope:
    """Create an LLM instance using the provided configuration.
    
    Creates an appropriate LLM instance based on the LLM type and configuration.
    The function handles multiple LLM providers including OpenAI, Azure OpenAI,
    Google AI Studio, Dashscope, and DeepSeek. It merges configuration from
    YAML files and environment variables, filters out invalid parameters, and
    sets up SSL verification and HTTP clients as needed.
    
    Args:
        llm_type: The type of LLM to create (e.g., "basic", "reasoning", "vision", "code").
        conf: The complete configuration dictionary loaded from the YAML file.
    
    Returns:
        An instance of BaseChatModel corresponding to the specified LLM type.
        The specific implementation depends on the configuration (e.g., ChatOpenAI,
        AzureChatOpenAI, ChatGoogleGenerativeAI, ChatDashscope, or ChatDeepSeek).
    
    Raises:
        ValueError: If the LLM type is unknown, the configuration is invalid,
            or no configuration is found for the specified LLM type.
    
    Note:
        - Environment variables take precedence over YAML configuration
        - Unexpected configuration keys are filtered out to prevent warnings
        - SSL verification can be disabled via verify_ssl configuration
        - For reasoning LLM type, special handling is applied for DeepSeek and Dashscope
    """
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    if not config_key:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    llm_conf = conf.get(config_key, {})
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM configuration for {llm_type}: {llm_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_llm_conf(llm_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**llm_conf, **env_conf}

    # Filter out unexpected parameters to prevent LangChain warnings (Issue #411)
    # This prevents configuration keys like SEARCH_ENGINE from being passed to LLM constructors
    allowed_keys_lower = {k.lower() for k in ALLOWED_LLM_CONFIG_KEYS}
    unexpected_keys = [key for key in merged_conf.keys() if key.lower() not in allowed_keys_lower]
    for key in unexpected_keys:
        removed_value = merged_conf.pop(key)
        logger.warning(
            f"Removed unexpected LLM configuration key '{key}'. "
            f"This key is not a valid LLM parameter and may have been placed in the wrong section of conf.yaml. "
            f"Valid LLM config keys include: model, api_key, base_url, max_retries, temperature, etc."
        )

    # Remove unnecessary parameters when initializing the client
    if "token_limit" in merged_conf:
        merged_conf.pop("token_limit")

    if not merged_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Add max_retries to handle rate limit errors
    if "max_retries" not in merged_conf:
        merged_conf["max_retries"] = 3

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client
    # Check if it's Google AI Studio platform based on configuration
    platform = merged_conf.pop("platform", "").lower()  # Remove platform from config as it's not a valid LLM parameter
    is_google_aistudio = platform == "google_aistudio" or platform == "google-aistudio"

    if platform == "openrouter":
        api_key = os.getenv("OPENROUTER_KEY", None)
        base_url = os.getenv("OPENROUTER_URL", None)
        # Handle OpenRouter specific configuration
        merged_conf["base_url"] = base_url
        merged_conf["api_key"] = api_key
        return ChatOpenAI(**merged_conf)

    if platform == "alibailian":
        api_key = os.getenv("ALIBAILIN_KEY", None)
        base_url = os.getenv("ALIBAILIN_URL", None)
        merged_conf["base_url"] = base_url
        merged_conf["api_key"] = api_key
        if llm_type == "reasoning":
            merged_conf["extra_body"] = {"enable_thinking": True}
        elif llm_type == "asr":

            merged_conf.pop("max_tokens", None)  # ASR does not need max_tokens
            merged_conf.pop("max_retries", None)  # ASR does not need max_retries
            merged_conf.pop("base_url", None)  # ASR uses api_base instead of base_url
            return ASRDashscope(**merged_conf)
        else:
            merged_conf["extra_body"] = {"enable_thinking": False}
        return ChatDashscope(**merged_conf)
    else:
        return ChatOpenAI(**merged_conf)


def get_llm_by_type(llm_type: LLMType) -> BaseChatModel | Type[ASRDashscope] | ChatDashscope:
    """Get an LLM instance by type.
    
    Retrieves an LLM instance for the specified type. If an instance has already
    been created and cached, returns the cached instance. Otherwise, loads the
    configuration, creates a new instance, caches it, and returns it.
    
    Args:
        llm_type: The type of LLM to retrieve (e.g., "basic", "reasoning", "vision", "code").
    
    Returns:
        A cached or newly created BaseChatModel instance for the specified LLM type.
    
    Note:
        LLM instances are cached in memory after the first creation. Subsequent
        calls with the same llm_type will return the cached instance without
        reloading the configuration.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(_get_config_file_path())
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


def get_configured_llm_models() -> dict[str, list[str]]:
    """Get all configured LLM models grouped by type.
    
    Scans the configuration file and environment variables to find all configured
    LLM models. Groups them by LLM type (basic, reasoning, vision, code) and
    returns a dictionary mapping each type to a list of configured model names.
    
    Returns:
        A dictionary mapping LLM type names to lists of configured model names.
        For example: {"basic": ["gpt-4"], "reasoning": ["gpt-4-turbo"]}.
        Returns an empty dictionary if configuration loading fails or no models
        are configured.
    
    Note:
        This function handles errors gracefully. If configuration loading fails,
        it prints a warning and returns an empty dictionary instead of raising
        an exception.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        llm_type_config_keys = _get_llm_type_config_keys()

        configured_models: dict[str, list[str]] = {}

        for llm_type in get_args(LLMType):
            # Get configuration from YAML file
            config_key = llm_type_config_keys.get(llm_type, "")
            yaml_conf = conf.get(config_key, {}) if config_key else {}

            # Get configuration from environment variables
            env_conf = _get_env_llm_conf(llm_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**yaml_conf, **env_conf}

            # Check if model is configured
            model_name = merged_conf.get("model")
            if model_name:
                configured_models.setdefault(llm_type, []).append(model_name)

        return configured_models
    except Exception as e:
        logger.warning(f"Failed to load LLM configuration: {e}")
        return {}


def _get_model_token_limit_defaults() -> dict[str, int]:
    """Get default token limits for common LLM models.
    
    Returns a dictionary mapping model names (or prefixes) to their default
    token limits. These are conservative limits designed to prevent token
    overflow errors. Users can override these defaults by setting token_limit
    in their configuration file.
    
    Returns:
        A dictionary mapping model name patterns to their default token limits.
        Includes entries for OpenAI models (gpt-4o, gpt-4-turbo, etc.),
        Anthropic Claude models, Google Gemini models, Bytedance Doubao,
        DeepSeek, and other common models. Also includes a "default" entry
        as a fallback for unknown models.
    
    Note:
        These limits are conservative estimates. The actual token limits may
        vary by model version and provider. Users should refer to the official
        documentation for their specific model.
    """
    return {
        # OpenAI models
        "gpt-4o": 120000,
        "gpt-4-turbo": 120000,
        "gpt-4": 8000,
        "gpt-3.5-turbo": 4000,
        # Anthropic Claude
        "claude-3": 180000,
        "claude-2": 100000,
        # Google Gemini
        "gemini-2": 180000,
        "gemini-1.5-pro": 180000,
        "gemini-1.5-flash": 180000,
        "gemini-pro": 30000,
        # Bytedance Doubao
        "doubao": 200000,
        # DeepSeek
        "deepseek": 100000,
        # Ollama/local
        "qwen": 30000,
        "llama": 4000,
        # Default fallback for unknown models
        "default": 100000,
    }


def _infer_token_limit_from_model(model_name: str) -> int:
    """Infer a reasonable token limit from the model name.
    
    Analyzes the model name to determine an appropriate token limit based on
    known model capabilities. This helps protect against token overflow errors
    when token_limit is not explicitly configured in the configuration file.
    
    Args:
        model_name: The model name from configuration (e.g., "gpt-4o", "claude-3-opus").
            Can be None or empty string.
    
    Returns:
        A conservative token limit (integer) based on known model capabilities.
        Returns 100,000 as a safe default if the model name is empty or no
        matching model pattern is found.
    
    Note:
        The function performs case-insensitive matching and checks for partial
        matches (e.g., "gpt-4" matches "gpt-4o" and "gpt-4-turbo").
    """
    if not model_name:
        return 100000  # Safe default
    
    model_name_lower = model_name.lower()
    defaults = _get_model_token_limit_defaults()
    
    # Try exact or prefix matches
    for key, limit in defaults.items():
        if key in model_name_lower:
            return limit
    
    return defaults["default"]


def get_llm_token_limit_by_type(llm_type: str) -> int:
    """Get the maximum token limit for a given LLM type.
    
    Determines the token limit for a specific LLM type using a priority-based
    approach. This helps prevent token overflow errors even when token_limit
    is not explicitly configured in the configuration file.
    
    Priority order:
    1. Explicitly configured token_limit in conf.yaml
    2. Inferred from model name based on known model capabilities
    3. Safe default (100,000 tokens)
    
    Args:
        llm_type: The type of LLM (e.g., 'basic', 'reasoning', 'vision', 'code').
            Must match one of the keys returned by _get_llm_type_config_keys().
    
    Returns:
        The maximum token limit (integer) for the specified LLM type. This is
        a conservative estimate designed to prevent token overflow errors.
        Returns 100,000 tokens as a safe default if no configuration is found
        and the model name cannot be inferred.
    
    Note:
        This function does not raise exceptions if the LLM type is not found
        in the configuration. It will return the default limit instead.
    """
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    conf = load_yaml_config(_get_config_file_path())
    model_config = conf.get(config_key, {})
    
    # First priority: explicitly configured token_limit
    if "token_limit" in model_config:
        configured_limit = model_config["token_limit"]
        if configured_limit is not None:
            return configured_limit
    
    # Second priority: infer from model name
    model_name = model_config.get("model")
    if model_name:
        inferred_limit = _infer_token_limit_from_model(model_name)
        return inferred_limit
    
    return _get_model_token_limit_defaults()["default"]

if __name__ == '__main__':
    get_llm_by_type("vision")
    logger.info(get_configured_llm_models())