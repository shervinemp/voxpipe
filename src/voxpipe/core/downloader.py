"""Model weight download and integrity verification, driven by the packaged manifest."""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from importlib import resources
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
import yaml

from .exceptions import ModelError
from .utils import get_logger

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_MAX_REDIRECTS = 4
_CHUNK_SIZE = 1024 * 1024

logger = get_logger(__name__)


def load_manifest() -> Dict[str, dict]:
    """Return the packaged model manifest: {model name: entry}."""
    resource = resources.files("voxpipe.data").joinpath("manifest.yaml")
    try:
        return yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise ModelError("Packaged model manifest is missing.") from e


def verify_file_sha256(path: str, expected_sha256: str) -> None:
    """Raise if a file does not match its pinned SHA-256 digest."""
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hex digest.")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    # ASVS 11.4.1 / 11.4.3: SHA-256 provides collision-resistant integrity.
    if digest.hexdigest().lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 verification failed for {os.path.basename(path)}.")


def download_hf_file(repo_id: str, filename: str, directory: str, *, revision: str, expected_sha256: str) -> str:
    """Download one file from the Hugging Face Hub at a pinned revision and verify it."""
    logger.info("Downloading '%s' from '%s'...", filename, repo_id)
    os.makedirs(directory, exist_ok=True)

    # ASVS 15.2.4: pin the source repository to an immutable commit.
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=directory, revision=revision)
    verify_file_sha256(downloaded, expected_sha256)
    return downloaded


def _open_validated(url: str, allowed_hosts: set[str]) -> requests.Response:
    """GET ``url`` following redirects by hand, checking every hop against the allowlist."""
    current = url
    for _ in range(_MAX_REDIRECTS):
        parsed = urlparse(current)
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ValueError("Download URL is not an approved HTTPS origin.")
        # ASVS 13.2.4 / 15.3.2: redirects are followed only after each
        # target is checked against the explicit HTTPS origin allowlist.
        response = requests.get(current, stream=True, allow_redirects=False, timeout=(5, 60))
        if not (response.is_redirect or response.is_permanent_redirect):
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise ValueError("Download redirect omitted its destination.")
        current = urljoin(current, location)
    raise ValueError("Download exceeded the redirect limit.")


def download_file(url: str, destination: str, *, expected_sha256: str, allowed_hosts: set[str], max_bytes: int) -> None:
    """Download ``url`` to ``destination``, enforcing origin, size and SHA-256 checks.

    The destination is replaced only after the complete file has passed every check.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if not allowed_hosts:
        raise ValueError("allowed_hosts must not be empty.")
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hex digest.")

    name = os.path.basename(urlparse(url).path)
    destination_dir = os.path.dirname(os.path.abspath(destination))
    temp_path: Optional[str] = None
    try:
        with _open_validated(url, allowed_hosts) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("Download exceeds the configured size limit.")

            os.makedirs(destination_dir, exist_ok=True)
            digest = hashlib.sha256()
            total = 0
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination_dir, delete=False, prefix=".download-") as file:
                temp_path = file.name
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Download exceeds the configured size limit.")
                    digest.update(chunk)
                    file.write(chunk)

        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ValueError("Downloaded file failed SHA-256 verification.")

        # ASVS 15.4.2: replace the destination only after the complete artifact
        # has passed size and integrity checks.
        os.replace(temp_path, destination)
        temp_path = None
        logger.info("Downloaded and verified %s", os.path.basename(destination))
    except ValueError as e:
        logger.warning("Rejected download for %s: %s", name, e)
        raise
    except (requests.exceptions.RequestException, OSError) as e:
        logger.error("Failed to download %s: %s", name, e)
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@dataclass(frozen=True)
class _Artifact:
    """One file of a manifest entry: where it lives, its digest, and how to fetch it."""

    key: str
    path: str
    sha256: str
    fetch: Callable[[], object]


def _artifacts(name: str, local_dir: Optional[str]) -> List[_Artifact]:
    entry = load_manifest().get(name)
    if not entry:
        raise ModelError(f"Unknown model: {name!r}")
    base = os.path.abspath(local_dir or entry.get("local_dir", ""))

    if entry.get("backend") == "huggingface":
        return [
            _Artifact(
                key="model",
                path=os.path.join(base, entry["file"]),
                sha256=entry["sha256"],
                fetch=lambda: download_hf_file(
                    repo_id=entry["repo"],
                    filename=entry["file"],
                    directory=base,
                    revision=entry["revision"],
                    expected_sha256=entry["sha256"],
                ),
            )
        ]

    if "files" in entry:
        allowed_hosts = set(entry.get("allowed_hosts", []))

        def artifact(key: str, info: dict) -> _Artifact:
            path = os.path.join(base, os.path.basename(info["url"]))
            return _Artifact(
                key=key,
                path=path,
                sha256=info["sha256"],
                fetch=lambda: download_file(
                    url=info["url"],
                    destination=path,
                    expected_sha256=info["sha256"],
                    allowed_hosts=allowed_hosts,
                    max_bytes=info["max_bytes"],
                ),
            )

        return [artifact(key, info) for key, info in entry["files"].items()]

    raise ModelError(f"Model {name!r} has no files entry.")


def ensure_downloaded(name: str, local_dir: Optional[str] = None) -> Dict[str, str]:
    """Download a model's missing files and verify the rest. Returns {logical key: local path}.

    Raises ModelError if a file already on disk fails its integrity check (it
    is left in place, never replaced silently) or a download fails.
    """
    paths = {}
    for artifact in _artifacts(name, local_dir):
        if os.path.exists(artifact.path):
            try:
                verify_file_sha256(artifact.path, artifact.sha256)
            except ValueError as e:
                raise ModelError(
                    f"{artifact.path} is not the file the manifest expects for {name!r}; "
                    "delete it to download a fresh copy."
                ) from e
        else:
            try:
                artifact.fetch()
            except (ValueError, OSError) as e:
                raise ModelError(f"Could not download {name!r}: {e}") from e
        paths[artifact.key] = artifact.path
    return paths


def is_downloaded(name: str, local_dir: Optional[str] = None) -> bool:
    """Whether every file of a model is on disk and passes its integrity check. Never downloads."""
    try:
        for artifact in _artifacts(name, local_dir):
            if not os.path.exists(artifact.path):
                return False
            verify_file_sha256(artifact.path, artifact.sha256)
    except (ModelError, ValueError, OSError):
        return False
    return True
