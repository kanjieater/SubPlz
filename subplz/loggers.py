import sys
import os
from loguru import logger

class StreamToLogger:
    """
    A stream-like object that redirects writes to a loguru logger.
    This is used to capture stdout and stderr from any module.
    """
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        # Write each line from the buffer to the logger
        for line in buffer.rstrip().splitlines():
            logger.log(self._level, line.rstrip())

    def flush(self):
        # Required for the stream interface, but we don't need to do anything.
        pass

def configure_logging(config: dict):
    """
    Configures the global loguru logger based on the application config.
    This function should be called once at the very start of the application.

    Args:
        config: The loaded configuration dictionary from your config.yml file.
    """
    log_config = config.get('log', {})

    # 1. Remove the default handler to start with a clean slate
    logger.remove()

    # 2. Configure the console logger with nice colors and formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"  # Set the minimum level for console output
    )

    # 3. Configure the rotating file logger if a directory is specified
    log_dir = log_config.get('dir')
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            # Use a generic filename; the date will be added automatically by loguru
            log_file_path = os.path.join(log_dir, "subplz.log")

            logger.add(
                log_file_path,
                rotation="25 MB",      # Rotate the file when it reaches 25 MB
                retention="30 days",   # Keep logs for up to 30 days
                compression="zip",     # Compress old log files
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",         # Log everything from DEBUG level upwards to the file
                encoding="utf-8",
                backtrace=True,        # Show the full stack trace on exceptions
                diagnose=True          # Add exception variable values for easier debugging
            )
            logger.success(f"File logging is enabled. Log directory: {log_dir}")
        except Exception as e:
            # If file logging fails, we still have the console logger
            logger.error(f"Failed to configure file logging at '{log_dir}': {e}")
            logger.error("Logs will only be sent to the console.")

    # 4. Redirect stdout and stderr to the logger to capture all print() calls
    sys.stdout = StreamToLogger(level="INFO")
    sys.stderr = StreamToLogger(level="ERROR")

    logger.info("Logging is configured. Standard output is now being captured.")