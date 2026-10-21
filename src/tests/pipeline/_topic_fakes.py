"""Test doubles for topic-tracking tests."""

from typing import List, Sequence

from voxpipe.pipeline.topics import Utterance


class ScriptedScorer:
    """Returns prepared score lists, recording what it was asked to compare.

    A batch may be a list of numbers, one number (used for every candidate),
    or a callable taking ``(text, candidates)``.  When the script runs out,
    every candidate scores zero.
    """

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def scores(self, text: str, candidates: Sequence[Sequence[Utterance]]) -> List[float]:
        self.calls.append((text, [[utterance.text for utterance in c] for c in candidates]))
        if not self.batches:
            return [0.0] * len(candidates)
        batch = self.batches.pop(0)
        if callable(batch):
            batch = batch(text, candidates)
        if isinstance(batch, (int, float)):
            return [float(batch)] * len(candidates)
        return [float(value) for value in batch]
