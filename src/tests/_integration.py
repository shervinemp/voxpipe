"""Helpers for the integration tests, which need downloaded model weights to run."""

from functools import lru_cache
from unittest.mock import patch

import pytest

from voxpipe.core.config import get_config
from voxpipe.core.downloader import is_downloaded


def requires_model(name: str):
    """Marks for a test module that needs the manifest model ``name`` on disk."""
    return [
        pytest.mark.integration,
        pytest.mark.skipif(not is_downloaded(name), reason=f"{name} weights not downloaded"),
    ]


def requires_configured_llm():
    """Marks for a test module that needs the configured local LLM on disk."""
    return requires_model(get_config().llm.model)


@lru_cache(maxsize=None)
def tts():
    """The configured synthesiser, with playback disabled."""
    from voxpipe.tts.model import TTSProviders

    with patch("voxpipe.tts.model.AudioPlayer"):
        return TTSProviders.create()


@lru_cache(maxsize=None)
def recogniser():
    """The Parakeet model on its own, without a microphone or VAD around it."""
    from onnx_asr import load_model

    from voxpipe.asr.models import ParakeetV2

    return load_model(ParakeetV2.MODEL_NAME, quantization="int8")


def transcribe(samples, sample_rate) -> str:
    return recogniser().recognize(samples, sample_rate=sample_rate) or ""
