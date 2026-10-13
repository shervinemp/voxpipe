"""Queued audio playback that can be cut off mid-clip."""

import threading
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np
import sounddevice as sd

from ..core.utils import get_logger


class AudioPlayer:
    """Plays queued clips in order on a background thread.

    Every clip belongs to the *generation* that was current when it was
    requested.  :meth:`stop_playback` starts a new generation: the clip being
    played is cut off, queued clips are discarded, and any clip still being
    prepared for the old generation is rejected when it finally arrives.
    """

    def __init__(self, output_device: Optional[int] = None, max_queued: int = 32):
        self.logger = get_logger(__name__)
        self.output_device = sd.default.device[1] if output_device is None else output_device
        device_name = sd.query_devices(self.output_device)["name"]
        self.logger.info("AudioPlayer using device '%s' (ID: %s)", device_name, self.output_device)

        self._queue: Queue = Queue(maxsize=max_queued)
        self._lock = threading.Lock()  # orders "start a clip" against "stop playback"
        self._generation = 0
        self._playing = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def generation(self) -> int:
        """Identifies the current run of speech; changes whenever playback is stopped."""
        return self._generation

    @property
    def is_playing(self) -> bool:
        """Whether a clip is audible or waiting its turn."""
        return self._playing.is_set() or not self._queue.empty()

    def enqueue(self, audio_data: np.ndarray, sample_rate: int, generation: Optional[int] = None) -> bool:
        """Queue a clip for ``generation`` (the current one by default).

        Returns False when the clip was dropped: it belongs to a run of speech
        that has since been stopped, or the queue is full.
        """
        with self._lock:
            if generation is None:
                generation = self._generation
            elif generation != self._generation:
                return False
            try:
                self._queue.put_nowait((audio_data, sample_rate, generation))
            except Full:
                self.logger.warning("Audio queue full, dropping a clip to stay responsive.")
                return False
        return True

    def stop_playback(self) -> None:
        """Silence the current clip and discard everything queued behind it."""
        with self._lock:
            self._generation += 1
            self._drain()
            sd.stop()

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _run(self) -> None:
        while self._running:
            try:
                audio_data, sample_rate, generation = self._queue.get(timeout=0.2)
            except Empty:
                continue
            with self._lock:
                # Checked under the lock: a stop either precedes this (and the clip is
                # skipped) or follows it (and cuts the clip off). It cannot slip between.
                if generation != self._generation:
                    continue
                self._playing.set()
                try:
                    sd.play(audio_data, samplerate=sample_rate, device=self.output_device)
                except Exception:  # noqa: BLE001 - PortAudio failures must not end the player thread
                    self.logger.error("Audio playback failed.", exc_info=True)
                    self._playing.clear()
                    continue
            try:
                sd.wait()
            finally:
                self._playing.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="voxpipe-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and end the player thread."""
        if not self._running:
            return
        self._running = False
        self.stop_playback()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
