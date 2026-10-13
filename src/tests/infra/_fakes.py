"""Test doubles shared by the audio-layer tests."""

import numpy as np

HOP = 512
CONTEXT = 64
SAMPLE_RATE = 16_000


class FakeVadSession:
    """Stands in for Silero's ONNX session: a chunk is speech when it is loud.

    It also holds callers to the streaming contract the real model needs:
    every frame is the audio context plus one chunk, and every call brings the
    state the previous call returned, or a blank one to start afresh.
    """

    def __init__(self):
        self.frames = []
        self.fresh_starts = 0
        self._state = None

    def run(self, outputs, feeds):
        frame, state = feeds["input"], feeds["state"]
        assert outputs == ["output", "stateN"], outputs
        assert feeds["sr"] == [SAMPLE_RATE], feeds["sr"]
        assert frame.shape == (1, CONTEXT + HOP), frame.shape
        assert state.shape == (2, 1, 128), state.shape
        if not state.any():
            self.fresh_starts += 1
        else:
            assert self._state is not None and np.array_equal(state, self._state), "the model's state was not carried"

        self.frames.append(frame)
        loud = np.abs(frame[0, CONTEXT:]).max() > 0.1
        self._state = np.full((2, 1, 128), len(self.frames), dtype=np.float32)
        return np.array([[0.9 if loud else 0.05]], dtype=np.float32), self._state


class FakeVadModel:
    """Mirrors what voxpipe uses of onnx-asr's Silero wrapper: its constants and its ONNX session."""

    SAMPLE_RATE = SAMPLE_RATE
    HOP_SIZE = HOP
    CONTEXT_SIZE = CONTEXT

    def __init__(self):
        self._model = FakeVadSession()

    @property
    def frames(self):
        return self._model.frames


def loud(n=1, value=0.5):
    """``n`` microphone chunks of speech-level audio."""
    return [np.full((HOP, 1), value, dtype=np.float32) for _ in range(n)]


def quiet(n=1):
    """``n`` microphone chunks of silence."""
    return [np.zeros((HOP, 1), dtype=np.float32) for _ in range(n)]
