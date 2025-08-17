import yaml
import os
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

def setup_logging_from_args(inputs):
    """
    Initializes the logging system based on command-line arguments.

    It looks for a config file path in the arguments. If found, it loads the
    config to set up the logger. If not found, or if the file is invalid,

    it sets up a default logger that writes to a 'logs' directory in the
    current working folder.
    """
    config = None
    # Check if the command-line arguments object has a 'config' attribute
    # and if the user provided a value for it.
    if hasattr(inputs, 'config') and inputs.config:
        try:
            with open(inputs.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # If the file is empty or invalid YAML, safe_load returns None
            if not config:
                 raise yaml.YAMLError("Config file is empty or invalid.")
        except Exception as e:
            # Use a raw print here, as this is a pre-logging fatal error
            print(f"FATAL: Could not read or parse config file at '{inputs.config}': {e}")
            sys.exit(1) # Exit because the user specified a config that can't be used.

    # If a valid config was loaded, use it. Otherwise, create a default config.
    if config:
        configure_logging(config)
    else:
        # This block runs for commands that don't take a --config flag,
        # or if the flag wasn't used.
        project_dir = os.getcwd()
        default_log_dir = os.path.join(project_dir, 'logs')
        default_config = {'log': {'dir': default_log_dir}}
        configure_logging(default_config)
        # Log a warning to inform the user about the default behavior.
        logger.warning("No config file provided. Using default file logging to ./logs")

def execute_on_inputs():
    """
    Parses CLI arguments, configures logging, and dispatches to the correct handler function.
    """
    inputs = get_inputs()


    config_path = getattr(inputs, 'config', None)
    config = load_config(config_path)

    configure_logging(config)

    inputs.config_data = config

    COMMAND_MAP = {
        "watch":   run_watcher,
        "scanner": run_scanner,
        "find":    lambda args: find(args.dirs),
        "rename":  rename,
        "copy":    copy,
        "extract": extract,
        "batch":   run_batch,
        "sync":    run_sync,
        "gen":     run_gen,
    }

    handler_function = COMMAND_MAP.get(inputs.subcommand)

    if handler_function:
        try:
            handler_function(inputs)
        except Exception:
            logger.opt(exception=True).critical(f"A fatal error occurred during the '{inputs.subcommand}' command.")
            sys.exit(1)
    else:
        logger.error(f"Error: Unknown subcommand '{inputs.subcommand}'")
        sys.exit(1)