"""Gate implementations that are not tied to topic tracking."""

from .models import GateDecision


class PassThroughGate:
    """A null gate: everything is answered and nothing is buffered.

    Useful as an explicit default, in tests, and as a base for decorators.
    """

    def evaluate(self, text: str, *, explicit: bool = False) -> GateDecision:
        return GateDecision(answer=True, confidence=1.0, reason="pass-through")

    def record_user(self, text: str) -> None:
        pass

    def record_reply(self, text: str) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
