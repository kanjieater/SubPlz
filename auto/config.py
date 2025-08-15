# automation/config.py
import os
import yaml

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

def load_config(config_path):
    """
    Loads and validates the YAML configuration file from a given path.
    Returns the configuration dictionary if valid.
    """
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise ConfigError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Error parsing YAML file: {e}")

    if not config:
        raise ConfigError("Configuration file is empty or invalid.")

    # Validate that the main sections exist
    if 'job_consumer_settings' not in config:
        raise ConfigError("Missing required section in config file: 'job_consumer_settings'")

    if 'scanner_settings' not in config:
        raise ConfigError("Missing required section in config file: 'scanner_settings'")

    print("✅ Configuration loaded and validated successfully.")
    return config