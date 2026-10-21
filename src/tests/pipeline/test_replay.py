"""Tests for shadow decision replay and threshold calibration."""

import json
import os
import tempfile
import unittest

from voxpipe.pipeline.attention.replay import (
    best_under_false_positive_rate,
    load_records,
    metrics_at,
    paired_labels,
    summarize,
)


class TestReplay(unittest.TestCase):
    def records(self):
        return [
            {"type": "decision", "text": "add two burgers", "score": 0.9, "answer": True, "reason": "addressed"},
            {"type": "correction", "text": "add two burgers", "addressed": True},
            {"type": "decision", "text": "would you like fries", "score": 0.2, "answer": False, "reason": "other"},
            {"type": "correction", "text": "would you like fries", "addressed": False},
            {"type": "decision", "text": "the sky is blue", "score": 0.6, "answer": False, "reason": "background"},
            {"type": "correction", "text": "the sky is blue", "addressed": False},
            {"type": "decision", "text": "what's the total", "score": 0.8, "answer": True, "reason": "addressed"},
        ]

    def test_load_records_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "decisions.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self.records()[0]) + "\n")
                handle.write("not json\n")
                handle.write("\n")
            self.assertEqual(len(load_records(path)), 1)

    def test_pairs_corrections_with_the_most_recent_decision(self):
        labels = paired_labels(self.records())
        self.assertEqual(labels, [(0.9, True), (0.2, False), (0.6, False)])

    def test_metrics_at_a_threshold(self):
        metrics = metrics_at(0.5, paired_labels(self.records()))
        self.assertEqual((metrics.true_positive, metrics.false_positive), (1, 1))
        self.assertEqual((metrics.true_negative, metrics.false_negative), (1, 0))
        self.assertAlmostEqual(metrics.accuracy, 2 / 3)
        self.assertAlmostEqual(metrics.false_positive_rate, 0.5)

    def test_best_threshold_respects_the_false_positive_budget(self):
        labels = paired_labels(self.records())
        best = best_under_false_positive_rate(labels, max_false_positive_rate=0.0)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best.threshold, 0.65)
        self.assertEqual(best.false_positive_rate, 0.0)

    def test_summary_counts_reasons_and_corrections(self):
        summary = summarize(self.records())
        self.assertEqual(summary["decisions"], 4)
        self.assertEqual(summary["answers"], 2)
        self.assertEqual(summary["corrections"], 3)
        self.assertEqual(summary["paired_labels"], 3)
        self.assertEqual(summary["reasons"]["addressed"], 2)


if __name__ == "__main__":
    unittest.main()
