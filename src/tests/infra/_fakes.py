"""Test doubles shared by the audio-layer tests."""

import numpy as np

HOP = 512


class FakeVadModel:
    """Calls a chunk speech when it is loud; mirrors the attributes of onnx-asr's Silero wrapper."""

    SAMPLE_RATE = 16_000
    HOP_SIZE = HOP
    CONTEXT_SIZE = 64

    def __init__(self):
        self.frames = []

    def _encode(self, frame):
        self.frames.append(frame)
        loud = np.abs(frame[0, self.CONTEXT_SIZE :]).max() > 0.1
        return (np.array([0.9 if loud else 0.05], dtype=np.float32),)


def loud(n=1, value=0.5):
    """``n`` microphone chunks of speech-level audio."""
    return [np.full((HOP, 1), value, dtype=np.float32) for _ in range(n)]


def quiet(n=1):
    """``n`` microphone chunks of silence."""
    return [np.zeros((HOP, 1), dtype=np.float32) for _ in range(n)]
