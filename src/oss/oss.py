import os
import sys
from pathlib import Path
from typing import Any, Dict, get_args

from dotenv import load_dotenv


from config.oss import OSSType
from oss.providers.minio_oss import MinIOClient
from oss.providers.base import BaseOSSClient
from config import load_yaml_config

from log import get_logger

logger = get_logger(__name__)

# OSS client cache
_oss_cache: dict[OSSType, BaseOSSClient] = {}

# Cache for config file path to avoid blocking calls to os.getcwd()
# Initialize cache at module load time to avoid blocking calls in async contexts
_config_file_path_cache: str = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())

# Allowed OSS configuration keys
ALLOWED_OSS_CONFIG_KEYS = {
    # Common configuration
    "platform",
    "endpoint",
    "access_key",
    "secret_key",
    "bucket_name",
    "region",
    "secure",
    # MinIO specific
    "access_key_id",
    # Aliyun OSS specific
    "access_key_secret",
    # Tencent COS specific
    "secret_id",
    "scheme",
    # AWS S3 specific
    "aws_access_key_id",
    "aws_secret_access_key",
    "region_name",
    "endpoint_url",
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


def _get_oss_type_config_keys() -> dict[str, str]:
    """Get mapping of OSS types to their configuration keys.

    Returns:
        Dictionary mapping OSS type names to their corresponding configuration key names.
    """
    return {
        "minio": "MINIO",
        "aliyun": "ALIYUN_OSS",
        "tencent": "TENCENT_COS",
        "aws": "AWS_S3",
        "default": "OSS_DEFAULT",
    }


def _get_env_oss_conf(oss_type: str) -> Dict[str, Any]:
    """Get OSS configuration from environment variables.

    Retrieves OSS configuration from environment variables following the naming
    convention: {OSS_TYPE}__{KEY}. For example, MINIO__endpoint or MINIO__access_key.

    Args:
        oss_type: OSS type (e.g., "minio", "aliyun", "tencent", "aws").

    Returns:
        Dictionary containing OSS configuration parsed from environment variables.
        Keys are converted to lowercase. Returns an empty dictionary if no matching
        environment variables are found.
    """
    prefix = f"{oss_type.upper()}__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix):].lower()
            conf[conf_key] = value
    return conf


def _create_oss_use_conf(oss_type: OSSType, conf: Dict[str, Any]) -> BaseOSSClient:
    """Create an OSS client instance using the provided configuration.

    Creates an appropriate OSS client instance based on the OSS type and configuration.
    This function handles multiple OSS providers including MinIO, Aliyun OSS, Tencent COS,
    and AWS S3. It merges configuration from YAML files and environment variables, and
    filters out invalid parameters.

    Args:
        oss_type: OSS type to create (e.g., "minio", "aliyun", "tencent", "aws").
        conf: Complete configuration dictionary loaded from the YAML file.

    Returns:
        BaseOSSClient instance corresponding to the specified OSS type.

    Raises:
        ValueError: If the OSS type is unknown, configuration is invalid, or no
            configuration is found for the specified OSS type.
    """
    oss_type_config_keys = _get_oss_type_config_keys()
    config_key = oss_type_config_keys.get(oss_type)

    if not config_key:
        raise ValueError(f"Unknown OSS type: {oss_type}")

    oss_conf = conf.get(config_key, {})
    if not isinstance(oss_conf, dict):
        raise ValueError(f"Invalid OSS configuration for {oss_type}: {oss_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_oss_conf(oss_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**oss_conf, **env_conf}

    # Filter out unexpected parameters
    allowed_keys_lower = {k.lower() for k in ALLOWED_OSS_CONFIG_KEYS}
    unexpected_keys = [key for key in merged_conf.keys() if key.lower() not in allowed_keys_lower]
    for key in unexpected_keys:
        removed_value = merged_conf.pop(key)
        logger.warning(
            f"Removed unexpected OSS configuration key '{key}'. "
            f"This key is not a valid OSS parameter."
        )

    if not merged_conf:
        raise ValueError(f"No configuration found for OSS type: {oss_type}")

    # Get platform type
    platform = merged_conf.pop("platform", "").lower()

    # Create client based on platform
    if platform == "minio":
        return MinIOClient(
            endpoint=merged_conf.get("endpoint", ""),
            access_key=merged_conf.get("access_key", ""),
            secret_key=merged_conf.get("secret_key", ""),
            secure=merged_conf.get("secure", False),
            region=merged_conf.get("region"),
            bucket_name=merged_conf.get("bucket_name", "test"),
        )
    else:
        # Default to MinIO
        logger.warning(f"Unknown platform '{platform}', using MinIO as default")
        return MinIOClient(
            endpoint=merged_conf.get("endpoint"),
            access_key=merged_conf.get("access_key", "minioadmin"),
            secret_key=merged_conf.get("secret_key", "minioadmin"),
            secure=merged_conf.get("secure", False),
            region=merged_conf.get("region"),
            bucket_name=merged_conf.get("bucket_name"),
        )


def get_oss_by_type(oss_type: OSSType) -> BaseOSSClient:
    """Get an OSS client instance by type.

    Retrieves an OSS client instance for the specified type. If an instance has already
    been created and cached, returns the cached instance. Otherwise, loads the
    configuration, creates a new instance, caches it, and returns it.

    Args:
        oss_type: OSS type to retrieve (e.g., "minio", "aliyun", "tencent", "aws").

    Returns:
        Cached or newly created BaseOSSClient instance for the specified OSS type.

    Note:
        OSS client instances are cached in memory after the first creation. Subsequent
        calls with the same oss_type will return the cached instance without reloading
        the configuration.
    """
    if oss_type in _oss_cache:
        return _oss_cache[oss_type]

    conf = load_yaml_config(_get_config_file_path())
    logger.debug(f"Loaded OSS configuration: {conf}")
    oss_client = _create_oss_use_conf(oss_type, conf)
    _oss_cache[oss_type] = oss_client
    return oss_client


def get_oss_client(oss_type: OSSType = "default") -> BaseOSSClient:
    """Get an OSS client (convenience function).

    Args:
        oss_type: OSS type, defaults to "default".

    Returns:
        OSS client instance.
    """
    return get_oss_by_type(oss_type)


def get_configured_oss_clients() -> dict[str, list[str]]:
    """Get all configured OSS clients grouped by type.

    Scans the configuration file and environment variables to find all configured
    OSS clients. Groups them by OSS type (minio, aliyun, tencent, aws) and returns
    a dictionary mapping each type to a list of configured client names.

    Returns:
        Dictionary mapping OSS type names to lists of configured client names.
        For example: {"minio": ["minio1"], "aliyun": ["oss1"]}.
        Returns an empty dictionary if configuration loading fails or no clients
        are configured.

    Note:
        This function handles errors gracefully. If configuration loading fails,
        it logs a warning and returns an empty dictionary instead of raising
        an exception.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        oss_type_config_keys = _get_oss_type_config_keys()

        configured_clients: dict[str, list[str]] = {}

        for oss_type in get_args(OSSType):
            # Get configuration from YAML file
            config_key = oss_type_config_keys.get(oss_type, "")
            yaml_conf = conf.get(config_key, {}) if config_key else {}

            # Get configuration from environment variables
            env_conf = _get_env_oss_conf(oss_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**yaml_conf, **env_conf}

            # Check if client is configured
            if merged_conf:
                configured_clients.setdefault(oss_type, []).append(merged_conf.get("endpoint", "default"))

        return configured_clients

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        logger.warning(f"Failed to load OSS configuration: {e}")
        return {}


if __name__ == '__main__':
    # Example usage
    client = get_oss_by_type("minio")
    logger.info(f"OSS client: {client}")
    logger.info(get_configured_oss_clients())

