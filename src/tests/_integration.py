"""Helpers for the integration tests, which need optional packages and downloaded model weights."""

from functools import lru_cache
from importlib.util import find_spec
from unittest.mock import patch

import pytest

from voxpipe.core.config import get_config
from voxpipe.core.downloader import is_downloaded

LLAMA = ("llama_cpp",)
KOKORO = ("kokoro_onnx",)
PARAKEET = ("onnx_asr",)


def requires_model(name: str, packages: tuple[str, ...] = ()):
    """Marks for a test module that needs ``packages`` installed and the manifest model ``name`` on disk."""
    missing = [package for package in packages if find_spec(package) is None]
    if missing:
        reason = f"optional package(s) not installed: {', '.join(missing)}"
    elif not is_downloaded(name):
        reason = f"{name} weights not downloaded"
    else:
        reason = ""
    return [pytest.mark.integration, pytest.mark.skipif(bool(reason), reason=reason)]


def requires_configured_llm():
    """Marks for a test module that needs the configured local LLM."""
    return requires_model(get_config().llm.model, LLAMA)


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
