import sys
import os
from loguru import logger


class TqdmToLogger:
    """
    A custom stream for tqdm to write its progress bars to the logger
    without interfering with other console output.
    """

    def write(self, buffer):
        message = buffer.strip()
        if message:
            logger.log("TQDM", message)

    def flush(self):
        pass


def format_record(record):
    """
    Custom formatter that handles TQDM messages differently from regular log messages.
    """
    # if record["level"].name == "TQDM":
    #     return "{message}\n"
    # else:
    return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level: <8} | P:{process.id} | <level>{message}</level>\n{exception}"


def configure_logging(config: dict):
    logger.remove()
    logger.level("TQDM", no=15, color="<white>")
    logger.level("CMD", no=22, color="<blue>")
    logger.add(
        sys.stderr,
        format=format_record,
        level="DEBUG",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Configure file logging, which remains unchanged.

    # Configure file logging.
    log_dir = config.get("base_dirs", {}).get("logs")

    # This will raise an error if log_dir is not provided in the config.
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "subplz.log")

    logger.add(
        log_file_path,
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | P:{process.id} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        filter=lambda record: record["level"].name != "TQDM",
    )

    logger.debug("Logging configured.")
