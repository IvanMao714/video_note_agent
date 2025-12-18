import os
import logging
from pathlib import Path
from typing import Optional

from src.log.trace import TRACE
from src.log.colorformat import ColoredFormatter
from src.config import load_yaml_config

# Flag to track if logging has been initialized
_initialized = False

# Cache for config file path to avoid blocking calls to os.getcwd()
_config_file_path_cache: Optional[str] = None

class AllowOnly(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.prefixes)



def _get_config_file_path() -> str:
    """Get the path to the configuration file.
    
    Returns:
        Absolute path to the conf.yaml configuration file in the project root directory.
    
    Note:
        The path is cached at module load time to avoid blocking calls to
        os.getcwd() in async contexts.
    """
    global _config_file_path_cache
    if _config_file_path_cache is None:
        _config_file_path_cache = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    return _config_file_path_cache


def _get_log_level_from_config() -> int:
    """Get log level from YAML configuration file or use default.
    
    Reads LOG_LEVEL from conf.yaml configuration file. Valid values:
    - DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)
    - Or numeric values: 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
    
    Returns:
        int: Logging level constant. Defaults to logging.INFO (does not show DEBUG logs).
    """
    try:
        config = load_yaml_config(_get_config_file_path())
        log_level_str = config.get("LOG_LEVEL", "").strip().upper() if isinstance(config.get("LOG_LEVEL"), str) else ""
        
        if not log_level_str:
            return logging.INFO
        
        # Map string names to logging constants
        level_map = {
            "DEBUG": logging.DEBUG,
            "TRACE": TRACE,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        if log_level_str in level_map:
            return level_map[log_level_str]
        
        # Try parsing as integer
        try:
            level_int = int(log_level_str)
            if level_int in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
                return level_int
        except (ValueError, TypeError):
            pass
    except Exception:
        # If config loading fails, use default
        pass
    
    # Invalid value or config not found, use default (INFO - does not show DEBUG)
    return logging.INFO


def setup_colored_logging(log_level=None):
    """Configure the root logger to use colored output for all child loggers.
    
    This function should be called once at the application entry point.
    It sets up a StreamHandler with ColoredFormatter on the root logger,
    which will be inherited by all child loggers.
    
    Args:
        log_level: Minimum logging level. If None, reads from LOG_LEVEL in conf.yaml
            configuration file or defaults to logging.INFO (does not show DEBUG logs).
        
    Returns:
        None
    """
    global _initialized
    
    if _initialized:
        return
    
    # Use provided level, or get from YAML config, or use INFO as default
    if log_level is None:
        log_level = _get_log_level_from_config()
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Avoid adding duplicate handlers
    if not root_logger.handlers:
        # Create StreamHandler for console output
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create custom ColoredFormatter
        formatter = ColoredFormatter()
        
        # Add formatter to handler
        ch.setFormatter(formatter)

        ch.addFilter(AllowOnly(prefixes=("src.", "__main__")))
        
        # Add handler to root logger
        root_logger.addHandler(ch)
    
    _initialized = True


def get_logger(name=None):
    """Get a Logger instance with colored output support.
    
    This is a convenience function that automatically initializes colored
    logging if not already done, then returns a logger with the specified name.
    The log level can be controlled via the LOG_LEVEL setting in conf.yaml.
    
    Usage:
        from src.log import get_logger
        
        logger = get_logger(__name__)
        logger.info("This is an info message")
    
    Configuration:
        LOG_LEVEL: Set in conf.yaml to DEBUG, INFO, WARNING, ERROR, or CRITICAL
            to control the minimum log level. Defaults to INFO if not set
            (does not show DEBUG logs).
    
    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.
    
    Returns:
        logging.Logger: Logger instance with colored output configured.
    """
    if not _initialized:
        setup_colored_logging()
    
    return logging.getLogger(name)


# --- Usage Examples ---
if __name__ == "__main__":
    # Method 1: Configure once at entry point, then use standard logging
    setup_colored_logging(logging.DEBUG)
    
    logger1 = logging.getLogger("test_module")
    logger1.debug("This is a DEBUG message.")
    logger1.info("This is an INFO message.")
    logger1.warning("This is a WARNING message!")
    logger1.error("This is an ERROR message. Please check.")
    logger1.critical("This is a CRITICAL error! System will exit.")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Use the convenience function get_logger
    logger2 = get_logger("another_module")
    logger2.debug("This is a DEBUG message from another module.")
    logger2.info("This is an INFO message from another module.")
    logger2.warning("This is a WARNING message from another module.")