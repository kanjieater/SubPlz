import sys
import os
from loguru import logger


class StreamToLogger:
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.log(self._level, line.rstrip())

    def flush(self):
        pass


class TqdmToLogger:
    def write(self, buffer):
        message = buffer.strip()
        if message:
            logger.log("TQDM", message)

    def flush(self):
        pass


def configure_logging(config: dict):
    log_config = config.get("log", {})
    logger.remove()
    logger.level("TQDM", no=15, color="<yellow>")

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        filter=lambda record: record["level"].name != "TQDM",
    )
    logger.add(
        sys.stderr,
        format="{message}",
        level="TQDM",
        filter=lambda record: record["level"].name == "TQDM",
    )
    logger.level("CMD", no=22, color="<blue>")

    # File handler gets an updated format
    log_dir = log_config.get("dir")
    if log_dir:
        try:
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
            logger.success(f"File logging enabled. Log directory: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to configure file logging at '{log_dir}': {e}")

    sys.stdout = StreamToLogger(level="INFO")
    sys.stderr = StreamToLogger(level="ERROR")
    logger.info(
        "Logging configured. Standard output and errors are now being captured."
    )
