import json
import hashlib
import logging
import os
import re
import tempfile
from urllib.parse import urljoin, urlparse
import requests
from typing import Dict, Any


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """Robustly parses JSON from LLM output, stripping Markdown/filler."""
    try:
        # Match array or object but parse aggressively to prevent early matching failures
        matches = re.finditer(r'(\[.*\]|\{.*\})', text.strip(), re.DOTALL)
        clean_texts = [match.group(0) for match in matches]

        if not clean_texts:
            return json.loads(text)

        # Iterate backwards through possible matches (inner objects first, or last objects)
        # to circumvent regex early capture
        for clean_text in reversed(clean_texts):
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                continue

        # If all regex captures fail, try the original text
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return fallback if fallback is not None else []


def setup_logging(log_level=logging.INFO, log_format=None, stream=None):
    """
    Configure the logging system with a standardized format and level.

    Args:
        log_level: The minimum severity level to log (default: INFO)
        log_format: A custom format string or None for default format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Ensure logging is only configured once to avoid duplicate handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=stream
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: The name of the logger

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)


def verify_file_sha256(path: str, expected_sha256: str) -> None:
    """Raise if a file does not match its pinned SHA-256 digest."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hex digest.")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    # ASVS 11.4.1 / 11.4.3: SHA-256 provides collision-resistant integrity.
    if digest.hexdigest().lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 verification failed for {os.path.basename(path)}.")


def download_hf_file(
    repo_id: str,
    filename: str,
    directory: str,
    *,
    revision: str,
    expected_sha256: str,
):
    """
    Downloads a single file from the Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repository identifier.
        filename (str): The specific file to download from the repo.
        directory (str): The local directory to save the model file.
    """
    logger = get_logger(__name__)
    logger.info(f"Preparing to download '{filename}' from '{repo_id}'...")

    destination = os.path.join(directory, filename)
    if os.path.exists(destination):
        verify_file_sha256(destination, expected_sha256)
        logger.info("Existing model file passed SHA-256 verification.")
        return destination

    logger.info("Starting download...")
    os.makedirs(directory, exist_ok=True)

    # ASVS 15.2.4: pin the source repository to an immutable commit.
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=directory,
        revision=revision,
    )
    verify_file_sha256(downloaded, expected_sha256)
    return downloaded


def download_file(
    url: str,
    destination: str,
    *,
    expected_sha256: str,
    allowed_hosts: set[str],
    max_bytes: int,
):
    """
    Download a file from a URL to a specified destination.

    Args:
        url: The URL of the file to download.
        destination: The local path to save the file.
    """
    logger = get_logger(__name__)  # Get logger inside function
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if not allowed_hosts:
        raise ValueError("allowed_hosts must not be empty.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hex digest.")

    def validate_url(candidate: str) -> None:
        parsed = urlparse(candidate)
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ValueError("Download URL is not an approved HTTPS origin.")

    temp_path = None
    try:
        current_url = url
        response = None
        for _ in range(4):
            validate_url(current_url)
            # ASVS 13.2.4 / 15.3.2: redirects are followed only after each
            # target is checked against the explicit HTTPS origin allowlist.
            response = requests.get(
                current_url,
                stream=True,
                allow_redirects=False,
                timeout=(5, 60),
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("Download redirect omitted its destination.")
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise ValueError("Download exceeded the redirect limit.")

        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("Download exceeds the configured size limit.")

        destination_dir = os.path.dirname(os.path.abspath(destination))
        os.makedirs(destination_dir, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination_dir, delete=False, prefix=".download-"
        ) as file:
            temp_path = file.name
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Download exceeds the configured size limit.")
                digest.update(chunk)
                file.write(chunk)
        response.close()

        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ValueError("Downloaded file failed SHA-256 verification.")

        # ASVS 15.4.2: replace the destination only after the complete artifact
        # has passed size and integrity checks.
        os.replace(temp_path, destination)
        temp_path = None
        logger.info("Downloaded and verified %s", os.path.basename(destination))
    except ValueError as e:
        logger.warning(
            "Rejected download for %s: %s",
            os.path.basename(urlparse(url).path),
            e,
        )
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}", exc_info=True)
        raise
    except IOError as e:
        logger.error(
            f"Failed to write file to {destination}: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during download of {url}: {e}",
            exc_info=True,
        )
        raise
    finally:
        if "response" in locals() and response is not None:
            response.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_specs(specs_path: str) -> Dict[str, Any]:
    with open(specs_path, "r") as f:
        return json.load(f)
