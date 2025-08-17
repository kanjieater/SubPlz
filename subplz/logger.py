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

class TqdmToLogger:
    """
    A file-like object that redirects writes from tqdm to a custom loguru level.
    """
    def write(self, buffer):
        # We use a custom level "TQDM" to identify these messages.
        message = buffer.strip()
        if message: # Avoid logging empty lines
            logger.log("TQDM", message)

    def flush(self):
        # Required for the stream interface.
        pass

def configure_logging(config: dict):
    """
    Configures the global loguru logger based on the application config.
    This function should be called once at the very start of the application.
    """
    log_config = config.get('log', {})
    logger.remove()

    # Define a custom level for tqdm progress bars
    logger.level("TQDM", no=15, color="<yellow>")

    # 1. CONFIGURE THE MAIN CONSOLE HANDLER
    # This handler processes everything EXCEPT tqdm messages and formats them nicely.
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        # This filter ensures this handler ignores messages from tqdm.
        filter=lambda record: record["level"].name != "TQDM"
    )

    # 2. CONFIGURE THE TQDM CONSOLE HANDLER
    # This handler ONLY processes tqdm messages and leaves them unformatted.
    logger.add(
        sys.stderr,
        format="{message}",
        level="TQDM",
        # This filter ensures this handler ONLY processes messages with the "TQDM" level.
        filter=lambda record: record["level"].name == "TQDM"
    )

    # 3. CONFIGURE THE ROTATING FILE HANDLER
    log_dir = log_config.get('dir')
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, "subplz.log")

            logger.add(
                log_file_path,
                rotation="25 MB",
                retention="30 days",
                compression="zip",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                encoding="utf-8",
                backtrace=True,
                diagnose=True,
                # This filter ensures progress bars are NEVER written to the log file.
                filter=lambda record: record["level"].name != "TQDM"
            )
            logger.success(f"File logging enabled. Log directory: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to configure file logging at '{log_dir}': {e}")

    # 4. REDIRECT STDOUT AND STDERR TO CAPTURE `print`
    sys.stdout = StreamToLogger(level="INFO")
    sys.stderr = StreamToLogger(level="ERROR")

    logger.info("Logging configured. Standard output is now being captured.")