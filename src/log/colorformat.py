import logging
from colorama import init, Fore, Style

# Initialize colorama with autoreset to ensure colors reset after each print
# This prevents color codes from affecting subsequent terminal output
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to log messages based on their level.

    This formatter applies different colors to log messages according to their
    severity level, making it easier to distinguish between different types
    of log entries in the terminal.
    """

    # Color mapping for different log levels
    # Fore is foreground color, Style.BRIGHT makes colors more vivid
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN + Style.BRIGHT,  # Cyan for debug messages
        logging.INFO: Fore.GREEN,  # Green for info messages
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,  # Bright yellow for warnings
        logging.ERROR: Fore.RED + Style.BRIGHT,  # Bright red for errors
        logging.CRITICAL: Fore.RED + Style.BRIGHT + Fore.WHITE,  # Red + white for critical errors
    }

    # Log output format: [timestamp] [level] <module_name> message
    # FORMAT = "[%(asctime)s] [%(levelname)-8s] <%(name)s> %(message)s"
    FORMAT = "[%(asctime)s] [%(levelname)-1s](%(filename)s:%(lineno)d) %(message)s"

    # DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DATE_FORMAT = "%H:%M:%S"

    def __init__(self, fmt=FORMAT, datefmt=DATE_FORMAT):
        """
        Initialize the ColoredFormatter.

        Args:
            fmt: Format string for log messages. Defaults to FORMAT.
            datefmt: Date format string. Defaults to DATE_FORMAT.
        """
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        """
        Format the log record with color based on log level.

        Overrides the parent format method to add color codes to the entire
        log message based on the log level.

        Args:
            record: LogRecord instance containing all information about the log event.

        Returns:
            str: Formatted log message with color codes applied.
        """
        # Get the color prefix for this log level
        color_prefix = self.COLOR_MAP.get(record.levelno, Fore.RESET)

        # Get the formatted log message from parent class
        # Parent class handles formatting of timestamp, level, module name, etc.
        log_message = super().format(record)

        # Apply color to the entire log message
        # With init(autoreset=True), colors will automatically reset after printing
        colored_log_message = color_prefix + log_message + Style.RESET_ALL

        return colored_log_message

