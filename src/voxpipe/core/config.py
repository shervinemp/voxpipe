"""
Configuration management for the Voice Control application.

This module provides a singleton `Config` class that handles loading
default and user-provided configurations from YAML files, and validates
them using Pydantic models.
"""

import os
import yaml
from typing import Any
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import AppConfig
from ..core.exceptions import ConfigError


class Config:
    """
    A singleton class to manage application configuration.

    It loads a default configuration, overrides with user settings,
    and validates the final configuration against a Pydantic model.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, default_config_path=None, user_config_path=None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        if default_config_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            default_config_path = os.path.join(
                base_dir, "data/config.defaults.yaml"
            )

        if user_config_path is None:
            user_config_path = os.environ.get(
                "VOXPIPE_CONFIG",
                os.path.join(os.getcwd(), "config.yaml"),
            )

        # Load project-local secrets before resolving env-backed YAML fields.
        load_dotenv()
        config_data = self._load_config(default_config_path)

        if os.path.exists(user_config_path):
            user_config = self._load_config(user_config_path) or {}
            self._deep_merge(config_data, user_config)

        config_data = self._recursive_resolve_env_vars(config_data)

        try:
            self.config = AppConfig(**config_data)
        except ValidationError as e:
            raise ConfigError(f"Configuration validation error: {e}")

        self._initialized = True

    def _recursive_resolve_env_vars(self, data: Any) -> Any:
        """
        Recursively traverses a config structure to resolve environment variables.
        """
        if isinstance(data, dict):
            if "env" in data and isinstance(data.get("env"), dict):
                env_block = data.pop("env")
                for key, env_var_name in env_block.items():
                    data[key] = os.getenv(env_var_name)

            for key, value in data.items():
                data[key] = self._recursive_resolve_env_vars(value)

        elif isinstance(data, list):
            return [self._recursive_resolve_env_vars(item) for item in data]

        return data

    def _load_config(self, path: str) -> dict:
        """Loads a YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # It's okay if the user config doesn't exist, but not the default.
            if "defaults" in path:
                raise ConfigError(
                    f"Default configuration file not found at {path}"
                )
            return {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file at {path}: {e}")

    def _deep_merge(self, source: dict, destination: dict) -> dict:
        """
        Deeply merges the destination dict into the source dict.
        Overwrites values in source with values from destination.
        """
        for key, value in destination.items():
            if (
                isinstance(value, dict)
                and key in source
                and isinstance(source[key], dict)
            ):
                self._deep_merge(source[key], value)
            else:
                source[key] = value
        return source

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using dot notation from the Pydantic model.

        Args:
            key: The key to retrieve, e.g., 'database.neo4j.uri'.
            default: The value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        value = self.config
        try:
            for k in key.split("."):
                value = getattr(value, k)
            return value
        except AttributeError:
            return default


config = Config()
