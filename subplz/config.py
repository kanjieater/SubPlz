import yaml
import os
from copy import deepcopy
from .logger import logger

# --- 1. Define the default configuration structure ---
# This serves as the base, ensuring no part of the app crashes if a key is missing.
DEFAULT_CONFIG = {
    "base_dirs": {
        "logs": "logs",
        "cache": "cache",
        "watcher_jobs": "jobs",
        "watcher_errors": "fails",
    },
    "watcher": {
        "path_map": {},
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


def resolve_based_paths(config: dict) -> dict:
    """
    Looks for a BASE_PATH env var, prepends it to all paths in the 'base_dirs' section,
    and ensures the directories exist. Defaults to a 'config' subdirectory in the current
    working directory if BASE_PATH is not set.
    """
    base_path = os.environ.get("BASE_PATH")
    if not base_path:
        base_path = os.path.join(os.getcwd(), "config")
        logger.warning(
            f"BASE_PATH environment variable not set. Defaulting to a 'config' subdirectory in the current working directory: '{base_path}'"
        )
    else:
        logger.info(
            f"Resolving paths in 'base_dirs' relative to BASE_PATH: '{base_path}'"
        )

    # Only proceed if the base_dirs key exists and is a dictionary
    if "base_dirs" in config and isinstance(config["base_dirs"], dict):
        for key, relative_path in config["base_dirs"].items():
            # Check if the path from the config is not empty and is a string
            if relative_path and isinstance(relative_path, str):
                # Construct the absolute path by joining the base and relative paths
                absolute_path = os.path.join(base_path, relative_path)

                # Update the value in the config dictionary with the new absolute path
                config["base_dirs"][key] = absolute_path
                # logger.debug(f"  Resolved path for '{key}': '{absolute_path}'")

                try:
                    if not os.path.exists(absolute_path):
                        os.makedirs(absolute_path, exist_ok=True)
                        logger.info(f"  Created missing directory: '{absolute_path}'")
                except OSError as e:
                    # Log an error if directory creation fails for reasons other than it already existing.
                    logger.error(f"  Failed to create directory '{absolute_path}': {e}")

    return config


def load_config(config_path: str | None) -> dict:
    """
    Loads and validates the configuration.

    - Starts with a deep copy of the default settings.
    - If a config_path is provided, it loads the YAML file.
    - If no config_path is provided, it checks for a 'config.yml' in the BASE_PATH.
    - It deeply merges the user's config on top of the defaults.
    - It resolves paths in 'base_dirs' using the BASE_PATH environment variable.
    - Returns the final, complete configuration dictionary.
    """
    # Start with a fresh copy of the defaults
    final_config = deepcopy(DEFAULT_CONFIG)

    if not config_path:
        logger.debug(
            "No config path provided. Checking for a default 'config.yml' in BASE_PATH."
        )
        base_path = os.environ.get("BASE_PATH")
        if not base_path:
            # --- CHANGED HERE ---
            base_path = os.path.join(os.getcwd(), "config")

        potential_path = os.path.join(base_path, "config.yml")

        if os.path.exists(potential_path):
            logger.info(f"Found default config file at '{potential_path}'. Loading it.")
            config_path = potential_path
        else:
            logger.warning(
                "No config file provided and no default found. Using default settings."
            )

    # (The rest of the function is unchanged)
    if config_path:
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at '{config_path}'")

            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if not user_config:
                    logger.warning(
                        f"Config file '{config_path}' is empty. Merging nothing."
                    )
                else:
                    final_config = deep_merge(user_config, final_config)
                    logger.info(
                        f"Successfully loaded and merged configuration from {config_path}"
                    )

        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            logger.error("Proceeding with default configuration due to loading error.")
            final_config = deepcopy(DEFAULT_CONFIG)

    final_config = resolve_based_paths(final_config)
    return final_config
