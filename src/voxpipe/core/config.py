"""Configuration loading.

Settings come from three layers, later ones winning: the defaults packaged
with voxpipe, an optional user YAML file, and environment variables named in
``env:`` blocks.  Nothing is read at import time: call :func:`get_config` for
the process-wide configuration, or :func:`load_config` to build one explicitly
and pass it to the components that need it.
"""

import os
import threading
from importlib import resources
from typing import Any, Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from .exceptions import ConfigError
from .models import AppConfig

CONFIG_ENV_VAR = "VOXPIPE_CONFIG"
DEFAULT_USER_CONFIG = "config.yaml"

_lock = threading.Lock()
_config: Optional[AppConfig] = None


def _parse_yaml(text: str, source: str) -> dict:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML from {source}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must contain a mapping at the top level.")
    return data


def _load_defaults() -> dict:
    resource = resources.files("voxpipe.data").joinpath("config.defaults.yaml")
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError("Packaged default configuration is missing.") from e
    return _parse_yaml(text, "the packaged defaults")


def _resolve_user_path(user_config_path: Optional[str]) -> Optional[str]:
    """Return the user config file to read, or None when there is none.

    A path given explicitly (argument or environment variable) must exist;
    only the conventional ``./config.yaml`` is optional.
    """
    explicit = user_config_path or os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        if not os.path.isfile(explicit):
            raise ConfigError(f"Configuration file not found: {explicit}")
        return explicit
    conventional = os.path.join(os.getcwd(), DEFAULT_USER_CONFIG)
    return conventional if os.path.isfile(conventional) else None


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` merged in; nested mappings merge key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env(data: Any) -> Any:
    """Replace ``env: {field: VAR_NAME}`` blocks with the variables' values.

    A set variable wins over a literal value for the same field; an unset one
    leaves the literal in place (or the field as None).
    """
    if isinstance(data, list):
        return [_resolve_env(item) for item in data]
    if not isinstance(data, dict):
        return data

    resolved = {key: _resolve_env(value) for key, value in data.items() if key != "env"}
    env_block = data.get("env")
    if isinstance(env_block, dict):
        for field, variable in env_block.items():
            value = os.environ.get(str(variable))
            if value is not None or field not in resolved:
                resolved[field] = value
    elif env_block is not None:
        resolved["env"] = env_block
    return resolved


def load_config(user_config_path: Optional[str] = None) -> AppConfig:
    """Build a validated configuration from the defaults, the user file and the environment."""
    # Secrets live in the working directory's .env, not next to the installed package.
    load_dotenv(find_dotenv(usecwd=True))

    data = _load_defaults()
    user_path = _resolve_user_path(user_config_path)
    if user_path:
        with open(user_path, "r", encoding="utf-8") as f:
            data = _deep_merge(data, _parse_yaml(f.read(), user_path))

    try:
        return AppConfig(**_resolve_env(data))
    except ValidationError as e:
        raise ConfigError(f"Configuration validation error: {e}") from e


def get_config() -> AppConfig:
    """Return the process-wide configuration, loading it on first use."""
    global _config
    with _lock:
        if _config is None:
            _config = load_config()
        return _config


def set_config(config: Optional[AppConfig]) -> None:
    """Install a process-wide configuration; ``None`` discards it so the next use reloads."""
    global _config
    with _lock:
        _config = config
