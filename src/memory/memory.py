import os
import sys
from pathlib import Path
from typing import Any, Dict, get_args

from dotenv import load_dotenv

from config.memory import MemoryType
from memory.providers.base import BaseMemoryClient
from memory.providers.postgres_memory import PostgresMemoryClient
from memory.providers.inmemory_memory import InMemoryMemoryClient
from config import load_yaml_config

from log import get_logger

logger = get_logger(__name__)

# Memory client cache
_memory_cache: dict[MemoryType, BaseMemoryClient] = {}

# Cache for config file path to avoid blocking calls to os.getcwd()
# Initialize cache at module load time to avoid blocking calls in async contexts
_config_file_path_cache: str = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())

# Allowed memory configuration keys
ALLOWED_MEMORY_CONFIG_KEYS = {
    # Common configuration
    "connection_string",
    "async_mode",
    "table_name",
    "host",
    "port",
    "database",
    "user",
    "password",
    "platform",
}


def _get_config_file_path() -> str:
    """Get the path to the configuration file.

    Returns:
        Absolute path to the conf.yaml configuration file in the project root directory.

    Note:
        The path is cached at module load time to avoid blocking calls to
        os.getcwd() in async contexts.
    """
    return _config_file_path_cache


def _get_memory_type_config_keys() -> dict[str, str]:
    """Get mapping of memory types to their configuration keys.

    Returns:
        Dictionary mapping memory type names to their corresponding configuration key names.
    """
    return {
        "postgres": "POSTGRES_MEMORY",
        "inmemory": "INMEMORY_MEMORY",
        "default": "MEMORY_DEFAULT",
    }


def _get_env_memory_conf(memory_type: str) -> Dict[str, Any]:
    """Get memory configuration from environment variables.

    Retrieves memory configuration from environment variables following the naming
    convention: {MEMORY_TYPE}__{KEY}. For example, POSTGRES__connection_string or POSTGRES__host.

    Args:
        memory_type: Memory type (e.g., "postgres", "inmemory").

    Returns:
        Dictionary containing memory configuration parsed from environment variables.
        Keys are converted to lowercase. Returns an empty dictionary if no matching
        environment variables are found.
    """
    prefix = f"{memory_type.upper()}__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix):].lower()
            conf[conf_key] = value
    return conf


def _build_connection_string(conf: Dict[str, Any]) -> str:
    """Build PostgreSQL connection string from configuration.

    Args:
        conf: Configuration dictionary containing database connection parameters.

    Returns:
        PostgreSQL connection string.
    """
    # If connection_string is provided directly, use it
    if "connection_string" in conf:
        return conf["connection_string"]

    # Otherwise, build from individual components
    host = conf.get("host", "localhost")
    port = conf.get("port", "5432")
    database = conf.get("database", "langgraph")
    user = conf.get("user", "postgres")
    password = conf.get("password", "")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _create_memory_use_conf(memory_type: MemoryType, conf: Dict[str, Any]) -> BaseMemoryClient:
    """Create a memory client instance using the provided configuration.

    Creates an appropriate memory client instance based on the memory type and configuration.
    This function handles multiple memory providers including PostgreSQL and in-memory.

    Args:
        memory_type: Memory type to create (e.g., "postgres", "inmemory").
        conf: Complete configuration dictionary loaded from the YAML file.

    Returns:
        BaseMemoryClient instance corresponding to the specified memory type.

    Raises:
        ValueError: If the memory type is unknown, configuration is invalid, or no
            configuration is found for the specified memory type.
    """
    memory_type_config_keys = _get_memory_type_config_keys()
    config_key = memory_type_config_keys.get(memory_type)

    if not config_key:
        raise ValueError(f"Unknown memory type: {memory_type}")

    memory_conf = conf.get(config_key, {})
    if not isinstance(memory_conf, dict):
        raise ValueError(f"Invalid memory configuration for {memory_type}: {memory_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_memory_conf(memory_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**memory_conf, **env_conf}

    # Filter out unexpected parameters
    allowed_keys_lower = {k.lower() for k in ALLOWED_MEMORY_CONFIG_KEYS}
    unexpected_keys = [
        key for key in merged_conf.keys() if key.lower() not in allowed_keys_lower
    ]
    for key in unexpected_keys:
        removed_value = merged_conf.pop(key)
        logger.warning(
            f"Removed unexpected memory configuration key '{key}'. "
            f"This key is not a valid memory parameter."
        )

    # Create client based on memory type
    if memory_type == "postgres":
        if not merged_conf:
            raise ValueError(f"No configuration found for memory type: {memory_type}")

        connection_string = _build_connection_string(merged_conf)
        async_mode = merged_conf.get("async_mode", False)
        table_name = merged_conf.get("table_name", "checkpoints")

        return PostgresMemoryClient(
            connection_string=connection_string,
            async_mode=async_mode,
            table_name=table_name,
        )
    elif memory_type == "inmemory":
        return InMemoryMemoryClient()
    else:
        # Default to in-memory
        logger.warning(f"Unknown memory type '{memory_type}', using in-memory as default")
        return InMemoryMemoryClient()


def get_memory_by_type(memory_type: MemoryType) -> BaseMemoryClient:
    """Get a memory client instance by type.

    Retrieves a memory client instance for the specified type. If an instance has already
    been created and cached, returns the cached instance. Otherwise, loads the
    configuration, creates a new instance, caches it, and returns it.

    Args:
        memory_type: Memory type to retrieve (e.g., "postgres", "inmemory").

    Returns:
        Cached or newly created BaseMemoryClient instance for the specified memory type.

    Note:
        Memory client instances are cached in memory after the first creation. Subsequent
        calls with the same memory_type will return the cached instance without reloading
        the configuration.
    """
    if memory_type in _memory_cache:
        return _memory_cache[memory_type]

    conf = load_yaml_config(_get_config_file_path())
    logger.debug(f"Loaded memory configuration: {conf}")
    memory_client = _create_memory_use_conf(memory_type, conf)
    _memory_cache[memory_type] = memory_client
    return memory_client


def get_memory_client(memory_type: MemoryType = "default") -> BaseMemoryClient:
    """Get a memory client (convenience function).

    Args:
        memory_type: Memory type, defaults to "default".

    Returns:
        Memory client instance.
    """
    return get_memory_by_type(memory_type)


def get_configured_memory_clients() -> dict[str, list[str]]:
    """Get all configured memory clients grouped by type.

    Scans the configuration file and environment variables to find all configured
    memory clients. Groups them by memory type (postgres, inmemory) and returns
    a dictionary mapping each type to a list of configured client names.

    Returns:
        Dictionary mapping memory type names to lists of configured client names.
        For example: {"postgres": ["postgres1"], "inmemory": ["inmemory1"]}.
        Returns an empty dictionary if configuration loading fails or no clients
        are configured.

    Note:
        This function handles errors gracefully. If configuration loading fails,
        it logs a warning and returns an empty dictionary instead of raising
        an exception.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        memory_type_config_keys = _get_memory_type_config_keys()

        configured_clients: dict[str, list[str]] = {}

        for memory_type in get_args(MemoryType):
            # Get configuration from YAML file
            config_key = memory_type_config_keys.get(memory_type, "")
            yaml_conf = conf.get(config_key, {}) if config_key else {}

            # Get configuration from environment variables
            env_conf = _get_env_memory_conf(memory_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**yaml_conf, **env_conf}

            # Check if client is configured
            if merged_conf or memory_type == "inmemory":
                # In-memory doesn't need configuration
                configured_clients.setdefault(memory_type, []).append(
                    merged_conf.get("host", "default") if memory_type == "postgres" else "default"
                )

        return configured_clients

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        logger.warning(f"Failed to load memory configuration: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    client = get_memory_by_type("inmemory")
    logger.info(f"Memory client: {client}")
    logger.info(get_configured_memory_clients())
