"""Logging helpers and small file loaders shared across voxpipe."""

import json
import logging
import os
import re
import tempfile
from typing import Any

_JSON_VALUE_RE = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


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


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """Parse the JSON in a model's output, ignoring prose or Markdown fences around it.

    Returns ``fallback`` (an empty list when that is None) if nothing parses.
    """
    if isinstance(text, str):
        candidates = [match.group(0) for match in _JSON_VALUE_RE.finditer(text.strip())]
        for candidate in (*reversed(candidates), text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return fallback if fallback is not None else []


def write_atomically(path: str, data: bytes) -> None:
    """Write via a temporary sibling so a crash or a full disk never leaves a truncated file behind."""
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp_path, path)
    except BaseException:
        os.unlink(temp_path)
        raise


def load_specs(specs_path: str) -> Any:
    """Load a JSON tool specification file."""
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)
