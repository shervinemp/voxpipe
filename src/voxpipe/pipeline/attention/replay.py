"""Offline analysis for shadow-mode attention decisions.

The gate logs decisions and corrections as JSON lines.  This module pairs a
correction with the decision it corrects, then sweeps thresholds to show the
false-positive/false-negative trade-off on your own data.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Metrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def false_positive_rate(self) -> float:
        actual_negative = self.false_positive + self.true_negative
        return self.false_positive / actual_negative if actual_negative else 0.0

    @property
    def false_negative_rate(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        return self.false_negative / actual_positive if actual_positive else 0.0


def load_records(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL decision log, skipping blank or malformed lines."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def paired_labels(records: Sequence[Dict[str, Any]]) -> List[Tuple[float, bool]]:
    """Pair each correction with the most recent decision for the same text."""
    decisions: Dict[str, float] = {}
    labels: List[Tuple[float, bool]] = []
    for record in records:
        kind = record.get("type")
        text = str(record.get("text", ""))
        if kind == "decision":
            try:
                decisions[text] = float(record["score"])
            except (KeyError, TypeError, ValueError):
                continue
        elif kind == "correction" and text in decisions and isinstance(record.get("addressed"), bool):
            labels.append((decisions[text], record["addressed"]))
    return labels


def metrics_at(threshold: float, labels: Sequence[Tuple[float, bool]]) -> Metrics:
    tp = fp = tn = fn = 0
    for score, addressed in labels:
        predicted = score >= threshold
        if predicted and addressed:
            tp += 1
        elif predicted and not addressed:
            fp += 1
        elif not predicted and addressed:
            fn += 1
        else:
            tn += 1
    return Metrics(threshold, tp, fp, tn, fn)


def sweep(labels: Sequence[Tuple[float, bool]], thresholds: Optional[Sequence[float]] = None) -> List[Metrics]:
    """Evaluate every threshold from 0.05 to 0.95 in 0.05 steps."""
    if not labels:
        return []
    steps = thresholds if thresholds is not None else [index / 100 for index in range(5, 100, 5)]
    return [metrics_at(threshold, labels) for threshold in steps]


def best_under_false_positive_rate(
    labels: Sequence[Tuple[float, bool]], max_false_positive_rate: float = 0.01
) -> Optional[Metrics]:
    """The most accurate threshold that stays within the FP budget."""
    candidates = [m for m in sweep(labels) if m.false_positive_rate <= max_false_positive_rate]
    if not candidates:
        return None
    return max(candidates, key=lambda m: (m.accuracy, m.recall, m.threshold))


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cheap counts for a shadow log."""
    decisions = [record for record in records if record.get("type") == "decision"]
    corrections = [record for record in records if record.get("type") == "correction"]
    reasons: Dict[str, int] = {}
    for record in decisions:
        reason = str(record.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    answers = sum(1 for record in decisions if record.get("answer"))
    return {
        "decisions": len(decisions),
        "answers": answers,
        "corrections": len(corrections),
        "reasons": reasons,
        "paired_labels": len(paired_labels(records)),
    }
