"""
Configuration loader for API keys and settings.
"""

import json
import os


def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            "Please create it with your API keys."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Global config object
config = load_config()
