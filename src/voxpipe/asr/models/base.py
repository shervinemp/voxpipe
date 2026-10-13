"""Shared microphone capture for speech recognisers: audio in, VAD segments, transcripts out."""

import threading
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union

import numpy as np
import sounddevice as sd

from ...core.exceptions import ASRError
from ...core.utils import get_logger
from ..vad import Silero

SoundDevice = Union[int, str, None]


class ModelBase(ABC):
    """A speech recogniser fed by a microphone.

    Iterating yields one transcript per utterance the VAD detects (an empty
    one when recognition failed), blocking between utterances, until
    :meth:`stop`.  Recognition runs on the iterating
    thread, so the consumer decides which thread pays for it.  Subclasses supply
    the model through :meth:`_transcribe`.
    """

    RECONNECT_INTERVAL = 1.0
    MAX_RECONNECT_ATTEMPTS = 60  # ~1 minute

    def __init__(self, vad: Silero, sound_device: SoundDevice = None, reconnect_timeout: float = 3.0):
        """``sound_device`` is an input device index, a fragment of its name, or None for the default."""
        self.logger = get_logger(type(self).__module__)
        self._vad = vad
        self._device = self._resolve_device(sound_device)
        try:
            self._device_name = sd.query_devices(self._device, kind="input")["name"]
        except (sd.PortAudioError, ValueError) as e:
            raise ASRError(f"No usable audio input device ({sound_device!r}): {e}") from e
        self._reconnect_timeout = reconnect_timeout

        self._muted = threading.Event()
        self._stopped = threading.Event()
        self._last_chunk_time: Optional[float] = None
        self._stream_lock = threading.Lock()  # stop() and a reconnect both replace or close the stream
        try:
            self._input_stream = self._open_stream(self._device)
        except (sd.PortAudioError, ValueError) as e:
            raise ASRError(f"Could not open audio input '{self._device_name}': {e}") from e
        self._watchdog: Optional[threading.Thread] = None

    @property
    def vad(self) -> Silero:
        return self._vad

    # --- device handling -------------------------------------------------

    def _resolve_device(self, device: SoundDevice) -> Optional[int]:
        if not isinstance(device, str):
            return device
        index = self._find_input_device(device)
        if index is None:
            self.logger.warning("Sound device '%s' not found. Using the default input.", device)
        return index

    @staticmethod
    def _find_input_device(name: str) -> Optional[int]:
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0 and name.lower() in info["name"].lower():
                return index
        return None

    def _open_stream(self, device: Optional[int]) -> sd.InputStream:
        return sd.InputStream(
            samplerate=self._vad.sample_rate,
            blocksize=self._vad.hop_size,
            device=device,
            channels=1,
            callback=self._on_audio,
        )

    def _on_audio(self, in_data: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback: hand the chunk to the VAD and return at once."""
        if status:
            self.logger.warning("Audio status: %s", status)
        self._last_chunk_time = time.monotonic()
        if not self._muted.is_set() and not self._stopped.is_set():
            self._vad.feed(in_data.copy())

    def _watch_device(self) -> None:
        """Reopen the input stream when the device stops delivering audio (unplugged, driver reset)."""
        while not self._stopped.wait(self.RECONNECT_INTERVAL):
            last = self._last_chunk_time
            if last is None or time.monotonic() - last <= self._reconnect_timeout:
                continue
            self.logger.warning("Audio device lost (no data for %.1fs). Reconnecting...", time.monotonic() - last)
            if not self._reconnect():
                self.logger.error("Giving up on audio device after %d attempts.", self.MAX_RECONNECT_ATTEMPTS)
                self._vad.close()
                return

    def _reconnect(self) -> bool:
        try:
            self._input_stream.close()
        except sd.PortAudioError:
            pass
        lost_at = time.monotonic()
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            if self._stopped.is_set():
                return True
            device = self._find_input_device(self._device_name)
            if device is not None:
                try:
                    stream = self._open_stream(device)
                    stream.start()
                except sd.PortAudioError as e:
                    self.logger.debug("Reconnect attempt %d failed: %s", attempt, e)
                else:
                    with self._stream_lock:
                        if self._stopped.is_set():  # stop() ran while this stream was being opened
                            self._close(stream)
                            return True
                        self._input_stream = stream
                    self._last_chunk_time = time.monotonic()
                    self.logger.info("Audio device reconnected after %.1fs.", time.monotonic() - lost_at)
                    return True
            self._stopped.wait(self.RECONNECT_INTERVAL)
        return False

    # --- control ---------------------------------------------------------

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    def mute(self) -> None:
        """Stop listening; whatever was being said up to now becomes a complete utterance."""
        self._muted.set()
        self._vad.flush()

    def unmute(self) -> None:
        self._muted.clear()

    def start(self) -> None:
        self._input_stream.start()
        # Counted from now: a stream that never delivers its first chunk is as lost as one that stops.
        self._last_chunk_time = time.monotonic()
        if self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watch_device, name="voxpipe-mic-watchdog", daemon=True)
            self._watchdog.start()

    def stop(self) -> None:
        """Stop capturing and end iteration."""
        self._stopped.set()
        with self._stream_lock:
            self._close(self._input_stream)
        self._vad.close()
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=2 * self.RECONNECT_INTERVAL)

    def _close(self, stream: sd.InputStream) -> None:
        try:
            stream.stop()
            stream.close()
        except sd.PortAudioError as e:
            self.logger.debug("Closing the input stream failed: %s", e)

    def __enter__(self) -> "ModelBase":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # --- recognition -----------------------------------------------------

    def __iter__(self) -> Iterator[str]:
        for segment in self._vad:
            try:
                text = self._transcribe(segment)
            except Exception:  # noqa: BLE001 - model backends raise assorted types; lose one utterance, not the mic
                self.logger.error("Speech recognition failed for an utterance.", exc_info=True)
                text = ""  # the consumer still has to learn that the user stopped speaking
            self.logger.debug("ASR returned %r", text[:40] if text else text)
            yield text

    @abstractmethod
    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one utterance of mono float32 audio sampled at ``self.vad.sample_rate``."""
