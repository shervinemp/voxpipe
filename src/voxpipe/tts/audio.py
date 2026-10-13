"""Queued audio playback that can be cut off mid-clip."""

import threading
import time
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np
import sounddevice as sd

from ..core.utils import get_logger

_FULL_QUEUE_POLL_SECONDS = 0.05


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
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def generation(self) -> int:
        """Identifies the current run of speech; changes whenever playback is stopped."""
        return self._generation

    @property
    def is_playing(self) -> bool:
        """Whether a clip is audible or waiting its turn."""
        # Counts every clip from the moment it is queued until it has finished (or been dropped),
        # so there is no instant between "taken off the queue" and "playing" that reads as silence.
        return self._queue.unfinished_tasks > 0

    def enqueue(self, audio_data: np.ndarray, sample_rate: int, generation: Optional[int] = None) -> bool:
        """Queue a clip for ``generation`` (the current one by default).

        Waits while the queue is full, so speech that is produced faster than
        it can be played is slowed down rather than losing sentences.  Returns
        False when the clip was dropped: its run of speech has since been
        stopped, or the queue is full and the player is not running.
        """
        while True:
            with self._lock:
                if generation is None:
                    generation = self._generation
                elif generation != self._generation:
                    return False
                try:
                    self._queue.put_nowait((audio_data, sample_rate, generation))
                    return True
                except Full:
                    if not self._running:
                        self.logger.warning("Audio queue is full and the player is not running; dropping a clip.")
                        return False
            time.sleep(_FULL_QUEUE_POLL_SECONDS)

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
            self._queue.task_done()

    def _run(self) -> None:
        while self._running:
            try:
                audio_data, sample_rate, generation = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._play(audio_data, sample_rate, generation)
            except Exception:  # noqa: BLE001 - PortAudio failures must not end the player thread
                self.logger.error("Audio playback failed.", exc_info=True)
            finally:
                self._queue.task_done()

    def _play(self, audio_data: np.ndarray, sample_rate: int, generation: int) -> None:
        with self._lock:
            # Checked under the lock: a stop either precedes this (and the clip is
            # skipped) or follows it (and cuts the clip off). It cannot slip between.
            if generation != self._generation:
                return
            sd.play(audio_data, samplerate=sample_rate, device=self.output_device)
        sd.wait()

    def start(self) -> None:
        """Start the player thread; a no-op while it is running."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="voxpipe-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and end the player thread. :meth:`start` brings it back."""
        if not self._running:
            return
        self._running = False
        self.stop_playback()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
