"""Append-only decision logging for shadow mode and offline calibration."""

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from ...core.utils import get_logger
from .models import FloorState, GateDecision


class DecisionLog:
    """Write one JSON line per gate decision, plus correction records.

    Logging is best-effort: a full disk or read-only path must never stop the
    assistant from answering.
    """

    def __init__(self, path: str):
        self.path = path
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()

    def write(self, text: str, decision: GateDecision, state: FloorState) -> None:
        self._append(
            {
                "type": "decision",
                "ts": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "answer": decision.answer,
                "score": decision.confidence,
                "reason": decision.reason,
                "state": asdict(state),
            }
        )

    def correction(self, text: str, addressed: bool) -> None:
        self._append(
            {
                "type": "correction",
                "ts": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "addressed": addressed,
            }
        )

    def _append(self, record: dict) -> None:
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as e:
            self.logger.warning("Could not write attention decision log %s: %s", self.path, e)
