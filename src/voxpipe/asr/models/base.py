from collections import deque
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable, Optional

import sounddevice as sd

from ...streaming.splitter import ConsumerProducer
from ...core.utils import get_logger


class ModelBase(ConsumerProducer, ABC):

    MAX_RECONNECT_ATTEMPTS = 60  # ~1 minute at 1s intervals

    def __init__(
        self,
        sound_device: int | str,
        reconnect_timeout: float = 3.0,
    ):
        self.logger = get_logger(__name__)
        self._sound_device = sound_device
        self._sound_device_name = self._resolve_device_name(sound_device)
        self._reconnect_timeout = reconnect_timeout
        self.audio_queue = deque(maxlen=100)
        self._audio_event = threading.Event()
        self._is_muted = threading.Event()
        self._is_running = threading.Event()
        self._is_running.set()
        self._device_lost = threading.Event()
        self._lost_since: Optional[float] = None
        self._fail_count = 0
        self._last_chunk_time = float('inf')
        self._last_check = 0.0  # throttle watchdog to once per second

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()

        def sound_cb(in_data, frames, t, status):
            if status:
                self.logger.warning("Audio status: %s", status)
            self._last_chunk_time = time.monotonic()
            self.audio_queue.append(in_data.copy())
            if len(self.audio_queue) > 128:
                self.audio_queue.popleft()
            self._audio_event.set()

        self._original_cb = sound_cb
        self._input_stream = self._inputstream(sound_device, sound_cb)

    @staticmethod
    def _resolve_device_name(device: int | str) -> str:
        try:
            info = sd.query_devices(device)
            return info["name"]
        except Exception:
            return str(device)

    def _find_device_by_name(self) -> Optional[int]:
        for i, info in enumerate(sd.query_devices()):
            if self._sound_device_name.lower() in info["name"].lower():
                return i
        return None

    def _vad_worker(self):
        while self._is_running.is_set():
            now = time.monotonic()

            # Watchdog: check for device silence once per second
            if now - self._last_check > 1.0:
                self._last_check = now
                elapsed = now - self._last_chunk_time
                if elapsed > self._reconnect_timeout and not self._device_lost.is_set():
                    self._device_lost.set()
                    self._lost_since = now
                    self.logger.warning(
                        "Audio device lost (no data for %.1fs). Reconnecting...",
                        elapsed,
                    )
                    try:
                        self._input_stream.stop()
                    except Exception:
                        pass

            if self._device_lost.is_set():
                self._reconnect_blocking()
                continue

            try:
                chunk = self.audio_queue.popleft()
            except IndexError:
                self._audio_event.wait(timeout=0.05)
                self._audio_event.clear()
                continue

            try:
                self.__call__(chunk)
            except Exception:
                self.logger.warning(
                    "VAD worker exception processing chunk.", exc_info=True
                )
                self._audio_event.clear()

    def _reconnect_blocking(self):
        """Block until reconnected or max attempts reached. Called from _vad_worker."""
        while self._is_running.is_set() and self._device_lost.is_set():
            if self._fail_count >= self.MAX_RECONNECT_ATTEMPTS:
                self.logger.error(
                    "Giving up on audio device after %d attempts.",
                    self.MAX_RECONNECT_ATTEMPTS,
                )
                self._device_lost.clear()
                return

            device_idx = self._find_device_by_name()
            if device_idx is None:
                self._fail_count += 1
                time.sleep(1.0)
                continue

            try:
                stream = self._inputstream(device_idx, self._original_cb)
                stream.start()
                self._input_stream = stream
                self._last_chunk_time = time.monotonic()
                self._device_lost.clear()
                self._fail_count = 0
                elapsed = time.monotonic() - (self._lost_since or time.monotonic())
                self.logger.info("Audio device reconnected after %.1fs.", elapsed)
            except Exception as e:
                self._fail_count += 1
                self.logger.debug("Reconnect attempt %d failed: %s", self._fail_count, e)
                time.sleep(1.0)

    def enable(self):
        super().enable()
        self._is_muted.clear()

    def disable_w_passthrough(self, value=None):
        super().disable_w_passthrough(value)
        self._is_muted.set()

    @abstractmethod
    def _consume(self, chunk: Iterable[float]): ...

    @abstractmethod
    def _produce(self) -> Generator[str, None, None]: ...

    @abstractmethod
    def _inputstream(self, device: int | str, callback: Callable) -> sd.InputStream: ...

    def start(self):
        self._input_stream.start()

    def stop(self):
        self._input_stream.stop()
        self._is_running.clear()
        if hasattr(self, "_vad_thread"):
            self._vad_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
