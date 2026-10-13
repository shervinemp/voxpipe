"""Logging helpers and small file loaders shared across voxpipe."""

import json
import logging
from typing import Any


def setup_logging(log_level: int | str = logging.INFO, log_format: str | None = None, stream=None) -> None:
    """Configure root logging once; later calls leave the existing handlers alone."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=log_level,
        format=log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_specs(specs_path: str) -> Any:
    """Load a JSON tool specification file."""
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)
