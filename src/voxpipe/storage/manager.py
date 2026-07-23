"""Backward compatibility forwarding module for downloader utilities."""

from voxpipe.core.downloader import ensure_downloaded, is_downloaded, _resolve_path, _MANIFEST_PATH

__all__ = ["ensure_downloaded", "is_downloaded", "_resolve_path", "_MANIFEST_PATH"]
