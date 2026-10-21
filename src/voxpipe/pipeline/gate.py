"""Compatibility import for the transcript filter.

The implementation lives in :mod:`voxpipe.pipeline.transcript`; this module
stays so existing imports keep working.
"""

from .transcript import qualify_transcript

__all__ = ["qualify_transcript"]
