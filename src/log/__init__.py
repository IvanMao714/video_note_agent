import logging

from log.colorformat import ColoredFormatter

# Flag to track if logging has been initialized
_initialized = False


def setup_colored_logging(log_level=logging.DEBUG):
    """
    Configure the root logger to use colored output for all child loggers.
    
    This function should be called once at the application entry point.
    It sets up a StreamHandler with ColoredFormatter on the root logger,
    which will be inherited by all child loggers.
    
    Args:
        log_level: Minimum logging level. Defaults to logging.INFO.
        
    Returns:
        None
    """
    global _initialized
    
    if _initialized:
        return
    
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
        
        # Add handler to root logger
        root_logger.addHandler(ch)
    
    _initialized = True


def get_logger(name=None):
    """
    Get a Logger instance with colored output support.
    
    This is a convenience function that automatically initializes colored
    logging if not already done, then returns a logger with the specified name.
    
    Usage:
        from log import get_logger
        
        logger = get_logger(__name__)
        logger.info("This is an info message")
    
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