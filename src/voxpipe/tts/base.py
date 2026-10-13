"""Shared speech queueing for text-to-speech providers."""

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

from ..core.models import TTSConfig
from ..core.utils import get_logger
from .audio import AudioPlayer


class TTSBase(ABC):
    """A synthesiser with a speech queue.

    Calling the instance queues a sentence and returns at once; sentences are
    synthesised and played in order on background threads, which start on
    first use.  :meth:`interrupt` silences the current sentence and discards
    the rest, including any still being synthesised.  Subclasses supply the
    model through :meth:`synthesize`.
    """

    def __init__(self, audio_player: AudioPlayer):
        self.logger = get_logger(type(self).__module__)
        self.audio_player = audio_player
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._run = 0  # counts start/stop cycles, so work left over from an earlier one is not counted
        self._pending = 0

    @classmethod
    @abstractmethod
    def from_config(cls, config: TTSConfig) -> "TTSBase":
        """Build the provider from the ``tts`` configuration section."""

    @classmethod
    def download(cls, config: TTSConfig) -> None:  # noqa: B027 - optional hook, not an abstract method
        """Fetch whatever the provider needs before first use; by default there is nothing to fetch."""

    @abstractmethod
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Turn text into audio: (samples, sample rate). Nothing is played.

        Text with nothing pronounceable in it yields an empty array.
        Providers may take keyword options (voice, speed, ...); callers pass
        them through ``__call__``.
        """

    @property
    def generation(self) -> int:
        """Identifies the current run of speech; it changes whenever speech is interrupted."""
        return self.audio_player.generation

    def __call__(self, text: str, *, interrupt: bool = False, generation: Optional[int] = None, **options) -> None:
        """Queue text to be spoken after whatever is already queued.

        With ``interrupt`` the current speech is cut off first.  ``generation``
        ties the text to a run of speech read earlier from :attr:`generation`:
        if speech has been interrupted since, the text is dropped instead of
        starting a new run.  ``options`` are passed to :meth:`synthesize`.
        """
        if interrupt:
            self.interrupt()
        if generation is None:
            # Fixed now: if speech is interrupted while this sentence is still being synthesised,
            # the player refuses the stale audio.
            generation = self.generation
        with self._lock:
            executor = self._started()
            executor.submit(self._speak, text, options, generation, self._run)
            self._pending += 1

    def _speak(self, text: str, options: dict, generation: int, run: int) -> None:
        try:
            if generation != self.audio_player.generation:
                return  # interrupted before synthesis began; don't waste the compute
            samples, sample_rate = self.synthesize(text, **options)
            if len(samples):
                self.audio_player.enqueue(samples, sample_rate, generation)
        except Exception:  # noqa: BLE001 - one bad sentence must not end the synthesis worker
            self.logger.error("TTS synthesis failed.", exc_info=True)
        finally:
            with self._lock:
                if run == self._run:
                    self._pending -= 1

    def interrupt(self) -> None:
        """Stop speaking now and forget everything queued."""
        self.audio_player.stop_playback()

    @property
    def is_speaking(self) -> bool:
        """Whether speech is audible, queued, or still being synthesised."""
        return self._pending > 0 or self.audio_player.is_playing

    def start(self) -> None:
        """Start the synthesis and playback threads; a no-op while they are running."""
        with self._lock:
            self._started()

    def _started(self) -> ThreadPoolExecutor:
        """The synthesis worker, started together with the player if need be. Caller holds the lock."""
        if self._executor is None:
            # One worker: sentences are synthesised, and so spoken, in the order they were queued.
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxpipe-tts")
            self.audio_player.start()
        return self._executor

    def stop(self) -> None:
        """Stop speaking and end the background threads. Speaking again starts them afresh."""
        with self._lock:
            executor, self._executor = self._executor, None
            self._run += 1
            self._pending = 0
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self.audio_player.stop()

    def __enter__(self) -> "TTSBase":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
