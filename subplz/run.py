import sys

from .logger import configure_logging, logger
from .config import load_config
from .helpers import find, rename, copy, extract
from .batch import run_batch
from .sync import run_sync
from .gen import run_gen
from .cli import get_inputs
from .auto.watcher import run_watcher
from .auto.scanner import run_scanner


def execute_on_inputs():
    """
    Parses CLI arguments, configures logging, and dispatches to the correct handler function.
    """
    inputs = get_inputs()
    config_path = getattr(inputs, "config", None)
    config = load_config(config_path)
    configure_logging(config)
    inputs.config_data = config

    COMMAND_MAP = {
        "watch": run_watcher,
        "scanner": run_scanner,
        "find": lambda args: find(args.dirs),
        "rename": rename,
        "copy": copy,
        "extract": extract,
        "batch": run_batch,
        "sync": run_sync,
        "gen": run_gen,
    }

    handler_function = COMMAND_MAP.get(inputs.subcommand)

    if handler_function:
        try:
            handler_function(inputs)
        except Exception:
            logger.opt(exception=True).critical(
                f"A fatal error occurred during the '{inputs.subcommand}' command."
            )
            sys.exit(1)
    else:
        logger.error(f"Error: Unknown subcommand '{inputs.subcommand}'")
        sys.exit(1)
