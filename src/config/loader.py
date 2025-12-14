import os
from typing import Any, Dict

import yaml


def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_str_env(name: str, default: str = "") -> str:
    """Get a string value from environment variables.
    
    Args:
        name: The name of the environment variable to retrieve.
        default: The default value to return if the environment variable is not set.
            Defaults to an empty string.
    
    Returns:
        A string value from the environment variable, stripped of leading and
        trailing whitespace. Returns the default value if the environment variable
        is not set.
    """
    val = os.getenv(name)
    return default if val is None else str(val).strip()


def get_int_env(name: str, default: int = 0) -> int:
    """Get an integer value from environment variables.
    
    Args:
        name: The name of the environment variable to retrieve.
        default: The default value to return if the environment variable is not set
            or cannot be parsed as an integer. Defaults to 0.
    
    Returns:
        An integer value parsed from the environment variable. Returns the default
        value if the environment variable is not set or cannot be converted to an
        integer.
    
    Note:
        Prints a warning message to stdout if the environment variable value
        cannot be converted to an integer.
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        from log import get_logger
        logger = get_logger(__name__)
        logger.warning(f"Invalid integer value for {name}: {val}. Using default {default}.")
        return default


def replace_env_vars(value: str) -> str:
    """Replace environment variable references in string values.
    
    If a string value starts with "$", it is treated as an environment variable
    reference and will be replaced with the corresponding environment variable value.
    
    Args:
        value: The string value that may contain an environment variable reference.
            If the value is not a string, it will be returned as-is.
    
    Returns:
        The replaced string value. If the value starts with "$", returns the
        environment variable value if it exists, otherwise returns the variable
        name without the "$" prefix. If the value is not a string or doesn't
        start with "$", returns the original value.
    """
    if not isinstance(value, str):
        return value
    if value.startswith("$"):
        env_var = value[1:]
        return os.getenv(env_var, env_var)
    return value


def process_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process dictionary to replace environment variable references.
    
    Traverses the dictionary structure recursively and replaces any string values
    that contain environment variable references (starting with "$") with their
    corresponding environment variable values.
    
    Args:
        config: The dictionary configuration to process. Can contain nested
            dictionaries and string values with environment variable references.
    
    Returns:
        A new dictionary with the same structure as the input, but with all
        environment variable references replaced. Returns an empty dictionary
        if the input is empty or None.
    """
    if not config:
        return {}
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = process_dict(value)
        elif isinstance(value, str):
            result[key] = replace_env_vars(value)
        else:
            result[key] = value
    return result


_config_cache: Dict[str, Dict[str, Any]] = {}


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load and process a YAML configuration file.
    
    Loads a YAML configuration file, processes it to replace environment variable
    references, and caches the result for subsequent calls. If the file does not
    exist, returns an empty dictionary.
    
    Args:
        file_path: The path to the YAML configuration file to load.
    
    Returns:
        A dictionary containing the parsed and processed configuration. Returns
        an empty dictionary if the file does not exist. The configuration is
        cached after the first load, so subsequent calls with the same file path
        will return the cached version.
    
    Note:
        The configuration is cached in memory. To reload a file, you may need
        to clear the cache or restart the application.
    """
    # Return empty dict if file doesn't exist
    if not os.path.exists(file_path):
        return {}

    # Check if configuration is already cached
    if file_path in _config_cache:
        return _config_cache[file_path]

    # Load and process configuration
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    processed_config = process_dict(config)

    # Cache the processed configuration
    _config_cache[file_path] = processed_config
    return processed_config