"""Voice activity detection: turns a stream of microphone chunks into speech segments."""

import queue
from collections import deque
from typing import Any, Callable, Deque, Iterator, List, Optional, Protocol

import numpy as np

from ..core.exceptions import ASRError
from ..core.models import ASRConfig
from ..core.utils import get_logger

_FLUSH = object()
_CLOSE = object()


class AudioGate(Protocol):
    """A pre-filter deciding whether audio reaches the VAD at all.

    ``process`` returns ``"active"`` to open the gate for the utterance that
    follows; any other value keeps dropping audio.  Typical gates are a
    wake-word detector or a noise gate.
    """

    def process(self, chunk: np.ndarray) -> str: ...


class Silero:
    """Silero VAD wrapped as a producer of speech segments.

    Audio is pushed in with :meth:`feed` from any thread (it never blocks) and
    segments are pulled out by iterating, on the one thread that runs the VAD
    model.  The model is recurrent: each chunk is judged with the audio context
    and internal state left by the chunks before it, which is what lets it hear
    quiet speech and speech over noise.  A segment opens after
    ``MIN_SPEECH_CHUNKS`` consecutive speech
    chunks, carries up to ``leading_silence_duration`` of audio from before
    that, and closes after ``trailing_silence_duration`` of silence, on
    :meth:`flush`, or once it exceeds ``max_segment_duration``.
    """

    MIN_SPEECH_CHUNKS = 5  # 5 x 32 ms = 160 ms of sustained speech before a segment opens
    MAX_BACKLOG_CHUNKS = 1000  # ~32 s; beyond this, new audio is dropped rather than queued

    def __init__(
        self,
        vad_threshold: float = 0.4,
        leading_silence_duration: float = 1.0,
        trailing_silence_duration: float = 0.8,
        max_segment_duration: float = 0.0,
        audio_gate: Optional[AudioGate] = None,
        on_speech_onset: Optional[Callable[[], None]] = None,
        model: Any = None,
    ):
        self.logger = get_logger(__name__)
        if model is None:
            from onnx_asr import load_vad

            model = load_vad("silero")
        self._model = model
        # onnx-asr only offers whole-file inference, which starts from a blank state on every call.
        # Streaming needs the state carried from chunk to chunk, so the ONNX session is driven directly.
        self._session: Any = getattr(model, "_model", None)
        if self._session is None:
            raise ASRError("This onnx-asr version does not expose Silero's ONNX session; voxpipe needs onnx-asr 0.6.")

        self.vad_threshold = vad_threshold
        self.on_speech_onset = on_speech_onset
        self.on_audio_level: Optional[Callable[[float, float], None]] = None

        chunks_per_second = self.sample_rate / self.hop_size
        self._leading_chunks = int(leading_silence_duration * chunks_per_second)
        self._trailing_chunks = int(trailing_silence_duration * chunks_per_second + 1)
        self._max_segment_chunks = int(max_segment_duration * chunks_per_second)

        self._gate = audio_gate
        self._gate_open = False
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._dropped = 0
        self._context: np.ndarray = np.zeros(0, dtype=np.float32)
        self._state: np.ndarray = np.zeros(0, dtype=np.float32)
        self._forget()

    @classmethod
    def from_config(cls, config: ASRConfig, **kwargs) -> "Silero":
        return cls(
            vad_threshold=config.vad_threshold,
            leading_silence_duration=config.leading_silence_ms / 1000.0,
            trailing_silence_duration=config.trailing_silence_ms / 1000.0,
            max_segment_duration=config.max_segment_duration,
            **kwargs,
        )

    @property
    def sample_rate(self) -> int:
        return self._model.SAMPLE_RATE

    @property
    def hop_size(self) -> int:
        """Samples per chunk the model expects."""
        return self._model.HOP_SIZE

    # --- producer side (any thread; never blocks) -------------------------

    def feed(self, chunk: np.ndarray) -> None:
        """Offer one chunk of microphone audio."""
        if self._gate is not None and not self._gate_open:
            if self._gate.process(np.asarray(chunk)) != "active":
                return
            self._gate_open = True
            self._queue.put(_FLUSH)  # start the gated utterance from a clean slate
            return
        if self._queue.qsize() >= self.MAX_BACKLOG_CHUNKS:
            self._dropped += 1
            if self._dropped % self.MAX_BACKLOG_CHUNKS == 1:
                self.logger.warning("VAD is not keeping up; dropping audio (%d chunks so far).", self._dropped)
            return
        self._queue.put(chunk)

    def flush(self) -> None:
        """End the segment in progress, if any, at this point in the audio."""
        self._queue.put(_FLUSH)

    def close(self) -> None:
        """Finish iteration once the audio already fed has been processed."""
        self._queue.put(_CLOSE)

    # --- consumer side (one thread) --------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.segments()

    def segments(self) -> Iterator[np.ndarray]:
        """Yield each completed speech segment as mono float32 samples, until :meth:`close`."""
        # Holds the lead-in plus the chunks that are confirming an onset, so no speech is clipped.
        leading: Deque[np.ndarray] = deque(maxlen=self._leading_chunks + self.MIN_SPEECH_CHUNKS)
        speech: List[np.ndarray] = []
        in_speech = False
        speech_run = 0
        silence_run = 0

        while True:
            item = self._queue.get()
            if item is _CLOSE:
                return
            if item is _FLUSH:
                if in_speech:
                    yield self._close_segment(speech)
                leading.clear()
                speech, in_speech, speech_run = [], False, 0
                self._forget()  # what follows is not a continuation of what came before
                continue

            try:
                chunk = self._to_mono(item)
                probability = self._speech_probability(chunk)
            except Exception:  # noqa: BLE001 - one unreadable chunk must not end recognition
                self.logger.warning("VAD could not process a chunk.", exc_info=True)
                continue
            is_speech = probability > self.vad_threshold
            self._report_level(chunk, probability)

            if not in_speech:
                leading.append(chunk)
                speech_run = speech_run + 1 if is_speech else 0
                if speech_run >= self.MIN_SPEECH_CHUNKS:
                    in_speech, silence_run = True, 0
                    speech = list(leading)
                    leading.clear()
                    self.logger.debug("VAD speech onset (prob=%.3f)", probability)
                    self._announce_onset()
                continue

            speech.append(chunk)
            silence_run = 0 if is_speech else silence_run + 1
            too_long = self._max_segment_chunks and len(speech) >= self._max_segment_chunks
            if silence_run >= self._trailing_chunks or too_long:
                self.logger.debug("VAD speech offset after %d chunks", len(speech))
                # A segment cut for length is the same utterance carrying on: an audio gate stays open.
                yield self._close_segment(speech, utterance_over=silence_run >= self._trailing_chunks)
                speech, in_speech, speech_run = [], False, 0

    def _close_segment(self, speech: List[np.ndarray], *, utterance_over: bool = True) -> np.ndarray:
        if utterance_over:
            self._gate_open = False
        return np.concatenate(speech, dtype=np.float32)

    def _announce_onset(self) -> None:
        if self.on_speech_onset is None:
            return
        try:
            self.on_speech_onset()
        except Exception:  # noqa: BLE001 - a listener must never end recognition
            self.logger.error("Speech-onset callback failed.", exc_info=True)

    def _forget(self) -> None:
        """Start the model from a blank slate: no audio context, no recurrent state."""
        self._context = np.zeros(self._model.CONTEXT_SIZE, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def _speech_probability(self, chunk: np.ndarray) -> float:
        """Run the model on one chunk, carrying over the audio context and recurrent state it needs."""
        samples: np.ndarray = np.concatenate([self._context, chunk])
        self._context = samples[-self._model.CONTEXT_SIZE :]
        output, self._state = self._session.run(
            ["output", "stateN"],
            {"input": samples[np.newaxis, :], "state": self._state, "sr": [self.sample_rate]},
        )
        return float(output[0, 0])

    def _report_level(self, chunk: np.ndarray, probability: float) -> None:
        if self.on_audio_level is None:
            return
        try:
            self.on_audio_level(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))), probability)
        except Exception:  # noqa: BLE001 - a level meter must never interfere with recognition
            self.logger.debug("Audio level callback failed.", exc_info=True)

    @staticmethod
    def _to_mono(chunk: Any) -> np.ndarray:
        samples: np.ndarray = np.asarray(chunk, dtype=np.float32)
        if samples.ndim == 1:
            return samples
        return samples[:, 0] if samples.shape[1] == 1 else samples.mean(axis=1, dtype=np.float32)
