#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

import os
import re
import shutil
import numpy as np

# Dynamically locate espeak-ng for phonemization
_espeak_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
if not _espeak_lib:
    candidates = []
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
            r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
        ]
    espeak_exe = shutil.which("espeak-ng")
    if espeak_exe:
        candidates.append(espeak_exe)
    candidates = [c for c in candidates if c is not None]
    for path in candidates:
        if os.path.exists(path):
            _espeak_lib = path
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _espeak_lib)
            if os.name == "nt":
                parent = os.path.dirname(path)
                os.environ.setdefault("PATH", "")
                if parent not in os.environ["PATH"]:
                    os.environ["PATH"] = parent + os.pathsep + os.environ["PATH"]
            break

from .audio import AudioPlayer  # noqa: E402

from ..core.utils import get_logger  # noqa: E402


from ..core.config import config  # noqa: E402
from ..core.exceptions import TTSError  # noqa: E402


class Kokoro:
    sample_rate: int = 24_000

    def __init__(self):
        self.logger = get_logger(__name__)

        weights_dir = config.get("tts.weights_dir", "model_files/tts")
        from ..storage.manager import ensure_downloaded
        paths = ensure_downloaded("Kokoro", local_dir=weights_dir)

        from kokoro_onnx import Kokoro as KokoroONNX
        from kokoro_onnx.tokenizer import Tokenizer
        self.kokoro = KokoroONNX(
            model_path=paths["model"],
            voices_path=paths["voices"],
        )
        self.tokenizer = Tokenizer()
        self.audio_player = AudioPlayer()
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def download(cls):
        weights_dir = config.get("tts.weights_dir", "model_files/tts")
        from ..storage.manager import ensure_downloaded
        ensure_downloaded("Kokoro", local_dir=weights_dir)

    def _synthesize(self, text: str, voice: str, language: str, speed: float, interrupt: bool):
        try:
            text = re.sub(r'[*_~`´<>]', '', text)
            text = re.sub(r'[\U00002600-\U000027BF\U0001F300-\U0010FFFF]', '', text)
            text = text.strip()
            if not text:
                return np.array([], dtype=np.float32), 0

            phonemes = self.tokenizer.phonemize(text, lang=language)
            if not phonemes.strip():
                self.logger.warning("Empty phonemes. Skipping TTS.")
                return np.array([], dtype=np.float32), 0

            self.logger.debug("TTS synthesizing %d chars -> %d phonemes", len(text), len(phonemes))
            samples, sample_rate = self.kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True, trim=False)
            self.logger.debug("TTS generated %d samples (%.1fs)", len(samples), len(samples) / sample_rate if sample_rate else 0)
            self.audio_player(samples, sample_rate, interrupt)
            return samples, sample_rate
        except Exception as e:
            self.logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            return np.array([], dtype=np.float32), 0

    def __call__(
        self,
        text: str,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
        interrupt: bool = False,
    ):
        self._executor.submit(self._synthesize, text, voice, language, speed, interrupt)

    def start(self):
        self.audio_player.start()

    def stop(self):
        if self._executor:
            self._executor.shutdown(wait=False)
        self.audio_player.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ----------------------------------------------------------------------


class TTSProviders:
    Kokoro: type = Kokoro


# ----------------------------------------------------------------------
