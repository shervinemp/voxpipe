"""Model weight downloading and file verification module."""

import os
from typing import Dict, Optional

import yaml

from .exceptions import ModelError
from .utils import download_file, download_hf_file, verify_file_sha256, get_logger

_MANIFEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(_MANIFEST_DIR, "data/manifest.yaml")


def _resolve_path(suggested: str, override: Optional[str] = None) -> str:
    path = override or suggested
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def ensure_downloaded(
    name: str,
    local_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Download model files if missing. Returns {logical_key: local_path}."""
    logger = get_logger(__name__)
    if not os.path.exists(_MANIFEST_PATH):
        raise ModelError(f"Manifest file not found at {_MANIFEST_PATH}")

    with open(_MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    entry = manifest.get(name)
    if not entry:
        raise ModelError(f"Unknown model: {name!r}")

    results = {}
    base = _resolve_path(entry.get("local_dir", ""), local_dir)

    if entry.get("backend") == "huggingface":
        dest = os.path.join(base, entry["file"])
        if os.path.exists(dest):
            verify_file_sha256(dest, entry["sha256"])
        else:
            os.makedirs(base, exist_ok=True)
            download_hf_file(
                repo_id=entry["repo"],
                filename=entry["file"],
                directory=base,
                revision=entry["revision"],
                expected_sha256=entry["sha256"],
            )
        results["model"] = dest
        return results

    if "files" in entry:
        allowed_hosts = set(entry.get("allowed_hosts", []))
        for key, info in entry["files"].items():
            dest = os.path.join(base, os.path.basename(info["url"]))
            if os.path.exists(dest):
                verify_file_sha256(dest, info["sha256"])
            else:
                os.makedirs(base, exist_ok=True)
                download_file(
                    url=info["url"],
                    destination=dest,
                    expected_sha256=info["sha256"],
                    allowed_hosts=allowed_hosts,
                    max_bytes=info["max_bytes"],
                )
            results[key] = dest
        return results

    raise ModelError(f"Model {name!r} has no files entry.")


def is_downloaded(
    name: str,
    local_dir: Optional[str] = None,
) -> bool:
    """Check whether all model files exist on disk and pass integrity checks without downloading."""
    try:
        if not os.path.exists(_MANIFEST_PATH):
            return False
        with open(_MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f)

        entry = manifest.get(name)
        if not entry:
            return False

        base = _resolve_path(entry.get("local_dir", ""), local_dir)

        if entry.get("backend") == "huggingface":
            dest = os.path.join(base, entry["file"])
            if not os.path.exists(dest):
                return False
            verify_file_sha256(dest, entry["sha256"])
            return True

        if "files" in entry:
            for key, info in entry["files"].items():
                dest = os.path.join(base, os.path.basename(info["url"]))
                if not os.path.exists(dest):
                    return False
                verify_file_sha256(dest, info["sha256"])
            return True

        return False
    except Exception:
        return False
