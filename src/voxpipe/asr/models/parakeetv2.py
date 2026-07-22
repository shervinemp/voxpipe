import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Generator, Iterable
from collections import deque
import numpy as np

from .base import ModelBase
from ...streaming.splitter import ConsumerProducer
from ...core.utils import get_logger
from ...core.exceptions import ASRError

_asr_lock = threading.Lock()


class ParakeetV2(ModelBase):

    def __init__(self, sound_device: int | str = 0):
        from onnx_asr import load_model

        # Resolve string device name to integer index
        if isinstance(sound_device, str):
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if sound_device.lower() in dev["name"].lower():
                    sound_device = i
                    break
            else:
                self.logger.warning(
                    f"Sound device '{sound_device}' not found. Using default (0)."
                )
                sound_device = 0

        self._model = load_model(
            "nemo-parakeet-tdt-0.6b-v2", quantization="int8"
        )

        # Read VAD settings from config before initializing Silero
        from voxpipe.core.config import config as _cfg
        vad_threshold = _cfg.get("asr.vad_threshold", 0.4)
        trailing_ms = _cfg.get("asr.trailing_silence_ms", 800)
        leading_ms = _cfg.get("asr.leading_silence_ms", 1000)
        max_segment = _cfg.get("asr.max_segment_duration", 0.0)
        self._vad = Silero(
            vad_threshold=vad_threshold,
            trailing_silence_duration=trailing_ms / 1000.0,
            leading_silence_duration=leading_ms / 1000.0,
            max_segment_duration=max_segment,
        )
        self._lock = _asr_lock

        super().__init__(sound_device)

        # Warm up ONNX model to avoid cold start latency on first utterance
        try:
            sr = self._vad._model.SAMPLE_RATE
            dummy = np.zeros(sr, dtype=np.float32)
            self._model.recognize(dummy, sample_rate=sr)
        except Exception:
            self.logger.warning("ONNX warmup failed (may affect first-utterance latency)")

    def _consume(self, chunk: Iterable[float]):
        if self._is_muted.is_set():
            return
        self._vad(chunk)

    def _produce(self) -> Generator[str, None, None]:
        for e in self._vad:
            r = None
            with self._lock:
                r = self._model.recognize(
                    e, sample_rate=self._vad._model.SAMPLE_RATE
                )
            self.logger.debug("ASR recognize returned %r (len=%d)", r[:40] if r else r, len(r) if r else 0)
            yield r

    def _inputstream(self, sound_device: int, callback: Callable):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=self._vad._model.SAMPLE_RATE,
            blocksize=self._vad._model.HOP_SIZE,
            device=sound_device,
            channels=1,
            callback=callback,
        )

    def disable_w_passthrough(self, value: Any = None):
        value = np.zeros(
            (self._input_stream.blocksize, self._input_stream.channels),
            dtype=np.float32,
        )
        super().disable_w_passthrough(value)
        self._vad.flush()


class Silero(ConsumerProducer):
    """Voice Activity Detection using Silero VAD.

    All ONNX inference runs in _produce (main thread). _consume (VAD worker
    thread) only pushes raw audio chunks — no ONNX calls, no lock contention.
    """

    def __init__(
        self,
        vad_threshold: float = 0.4,
        leading_silence_duration: float = 1.0,
        trailing_silence_duration: float = 0.8,
        trailing_buffer_duration: float = 1.2,
        max_segment_duration: float = 0.0,
        audio_gate=None,
        on_speech_onset: Callable | None = None,
    ):
        from onnx_asr import load_vad

        self.logger = get_logger(__name__)

        self._model = load_vad("silero")
        self._queue = Queue(maxsize=1000)

        self.vad_threshold = vad_threshold
        self.pre_speech_dur = leading_silence_duration
        self.post_speech_dur = trailing_silence_duration
        self.post_speech_keep = trailing_buffer_duration
        self.max_segment_duration = max_segment_duration
        self.on_speech_onset = on_speech_onset
        self.on_audio_level: Callable | None = None
        self._gate = audio_gate
        self._gate_active = False
        self._min_speech_chunks = 5  # 5 × 32ms = 160ms — unified: onset gate + ASR yield

        self.reset()

    def reset(self):
        self._is_speech_segment = False
        self._silence_counter = 0
        self._segment_start_time = time.monotonic()

        coeff_ = self._model.SAMPLE_RATE / self._model.HOP_SIZE
        pre_speech_chunks = int(self.pre_speech_dur * coeff_)
        self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)
        self._trailing_silent_chunks = int(self.post_speech_dur * coeff_ + 1)
        self._trailing_buffer_chunks = int(self.post_speech_keep * coeff_ + 1)
        self._model_input_frame = np.zeros(
            self._model.CONTEXT_SIZE + self._model.HOP_SIZE, dtype=np.float32
        )

    def flush(self):
        with _asr_lock:
            if self._is_speech_segment:
                self._is_speech_segment = False
                self._silence_counter = 0
                self._queue.put(None)

    def _produce(self) -> Generator[np.ndarray, None, None]:
        """Main thread: run VAD ONNX, accumulate speech, yield segments."""
        buffer = deque()
        while True:
            try:
                c = self._queue.get(timeout=1)
            except Empty:
                continue

            if c is None:
                if len(buffer) >= self._min_speech_chunks:
                    try:
                        seg = self._to_mono(buffer)
                    except (ValueError, Exception) as e:
                        self.logger.warning("VAD buffer error, dropping segment: %s", e)
                        buffer.clear()
                        continue
                    yield seg
                    buffer.clear()
                continue

            try:
                speech_prob = self._vad_encode(c)
            except Exception as e:
                self.logger.warning("VAD encode error: %s", e)
                continue

            is_loud = speech_prob > self.vad_threshold

            if self.on_audio_level:
                try:
                    rms = float(np.sqrt(np.mean(np.asarray(c, dtype=np.float64) ** 2)))
                    self.on_audio_level(rms, float(speech_prob))
                except Exception:
                    pass

            if not self._is_speech_segment:
                self._pre_speech_buffer.append(c)
                if is_loud:
                    self._onset_counter = getattr(self, "_onset_counter", 0) + 1
                    if self._onset_counter >= self._min_speech_chunks:
                        self._is_speech_segment = True
                        self._segment_start_time = time.monotonic()
                        self.logger.debug("VAD speech onset (prob=%.3f)", speech_prob)
                        if self.on_speech_onset:
                            self.on_speech_onset()
                        buffer.extend(self._pre_speech_buffer)
                        buffer.append(c)
                        self._silence_counter = 0
                else:
                    self._onset_counter = 0
                continue

            buffer.append(c)
            if is_loud:
                self._silence_counter = 0
                if self.max_segment_duration > 0:
                    elapsed = time.monotonic() - self._segment_start_time
                    if elapsed > self.max_segment_duration:
                        self._is_speech_segment = False
                        self._silence_counter = 0
                        self._onset_counter = 0
                        self._segment_start_time = time.monotonic()
                        self._queue.put(None)
            else:
                self._silence_counter += 1
                if self._silence_counter >= self._trailing_silent_chunks:
                    self._is_speech_segment = False
                    self._silence_counter = 0
                    self._onset_counter = 0
                    self._gate_active = False
                    self.logger.debug("VAD speech offset after %d silent chunks", self._trailing_silent_chunks)
                    self._queue.put(None)

    @staticmethod
    def _to_mono(buffer):
        chunks = []
        for c in buffer:
            if c.ndim > 1 and c.shape[1] > 1:
                chunks.append(np.mean(c, axis=1))
            elif c.ndim > 1:
                chunks.append(c[:, 0])
            else:
                chunks.append(c)
        return np.concatenate(chunks, dtype=np.float32)

    def _vad_encode(self, chunk: np.ndarray) -> float:
        """Run Silero VAD model. Called from _produce (main thread)."""
        if len(chunk.shape) > 1 and chunk.shape[1] > 1:
            chunk = np.mean(chunk, axis=1)
        elif len(chunk.shape) > 1:
            chunk = chunk[:, 0]
        self._model_input_frame = np.concatenate(
            [self._model_input_frame[-self._model.CONTEXT_SIZE:], chunk]
        )
        speech_prob, *_ = self._model._encode(
            self._model_input_frame[np.newaxis, :]
        )
        return speech_prob[0]

    def _consume(self, chunk: Iterable[np.ndarray]):
        """VAD worker thread: push raw audio to queue. No ONNX calls."""
        if self._gate and not self._gate_active:
            state = self._gate.process(np.asarray(chunk))
            if state == "active":
                self._gate_active = True
                self.reset()
            return
        self._queue.put(chunk)
