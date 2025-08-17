import yaml
import os
from copy import deepcopy
from .logger import logger

# --- 1. Define the default configuration structure ---
# This serves as the base, ensuring no part of the app crashes if a key is missing.
DEFAULT_CONFIG = {
    "log": {"dir": os.path.join(os.getcwd(), "logs")},
    "watcher": {
        "jobs": None,
        "path_map": {},
        "error_directory": None,
        "polling_interval_seconds": None,
    },
    "scanner": {
        "target_sub_extensions": [],
        "blacklist_filenames": [],
        "blacklist_dirs": [],
    },
    "batch_pipeline": [],
}


def deep_merge(source, destination):
    """Recursively merge dictionary source into destination."""
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def load_config(config_path: str | None) -> dict:
    """
    Loads and validates the configuration.

    - Starts with a deep copy of the default settings.
    - If a config_path is provided, it loads the YAML file.
    - It deeply merges the user's config on top of the defaults.
    - Returns the final, complete configuration dictionary.
    """
    # Start with a fresh copy of the defaults
    final_config = deepcopy(DEFAULT_CONFIG)

    if not config_path:
        logger.warning("No config file path provided. Using default settings.")
        return final_config

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at '{config_path}'")

        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            if not user_config:
                raise yaml.YAMLError("Config file is empty or invalid.")

        # Deep merge the user's config over the defaults
        final_config = deep_merge(user_config, final_config)
        logger.info(f"Successfully loaded and merged configuration from {config_path}")

    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        # In case of a critical error, we can decide whether to exit or proceed with defaults.
        # For now, we will proceed with defaults but the app will likely fail later.
        logger.error("Proceeding with default configuration due to loading error.")

    # You can add validation logic here if needed, e.g.:
    # if not final_config.get('watcher', {}).get('jobs'):
    #     logger.warning("'watcher.jobs' is not defined in the config.")

    return final_config
