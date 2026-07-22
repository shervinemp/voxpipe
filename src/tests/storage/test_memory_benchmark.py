"""
Integration benchmarks for Memory using an LLM as a verifier.

Requires a real GGUF model to be downloaded. Skipped if absent.

The verifier classifies facts from a response against the ground-truth
history and reports recall, precision, and F1.
"""

import json
import logging
import os
import tempfile
import unittest

import pytest


# ---------------------------------------------------------------------------
# Model detection (lazy -- runs on first test, not at module import)
# ---------------------------------------------------------------------------


def _model_key() -> str:
    from voxpipe.core.config import config
    return config.get("llm.model", "Gemma4E4B")


def _has_model() -> bool:
    try:
        from voxpipe.storage.manager import is_downloaded
        return is_downloaded(_model_key())
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_model(),
        reason="GGUF model not downloaded -- run python -m voxpipe.llm.download first",
    ),
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict | None:
    for variant in (raw, raw.strip()):
        if variant.startswith("```"):
            variant = variant.split("\n", 1)[-1]
        if variant.endswith("```"):
            variant = variant.rsplit("```", 1)[0]
        try:
            return json.loads(variant.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _classify_facts(llm, truth: list[str], response: str) -> dict:
    from voxpipe.llm.conversation import Conversation
    facts_json = json.dumps(truth)
    conv = Conversation()
    conv.set_system_message(
        "You are a strict fact-checker. Compare the RESPONSE against the "
        "GROUND-TRUTH FACTS list.\n\n"
        "1. present  -- facts from the list that are clearly stated in the response\n"
        "2. absent   -- facts from the list that are NOT in the response\n"
        "3. hallucinated -- claims in the response that are NOT supported by the facts list\n\n"
        'Output ONLY valid JSON: {"present": [...], "absent": [...], "hallucinated": [...]}'
    )
    conv.add_user_message(f"GROUND-TRUTH FACTS: {facts_json}\n\nRESPONSE: {response}")
    raw = "".join(llm(conv))
    result = _parse_json(raw)
    return {
        "present": result.get("present", []) if result else [],
        "absent": result.get("absent", []) if result else [],
        "hallucinated": result.get("hallucinated", []) if result else [],
    }


def _f1_from_classification(cls: dict) -> dict:
    tp = len(cls.get("present", []))
    fn = len(cls.get("absent", []))
    fp = len(cls.get("hallucinated", []))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class MemoryBenchmark:
    """Compares LLM responses with and without Memory context."""

    def __init__(self, llm, verifier_llm=None):
        self.llm = llm
        self.verifier_llm = verifier_llm or llm

    def run(self, history: list[str], question: str, threshold: float = 0.3) -> dict:
        from voxpipe.llm.conversation import Conversation
        from voxpipe.storage.memory import Memory

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            memory = Memory(db_path=db_path, max_entries=50, ttl_days=1)
            for turn in history:
                memory.store(turn, role="user")
                memory.store(turn, role="assistant")

            conv_a = Conversation()
            for turn in memory.retrieve(question, top_k=5):
                if content := turn.get("content"):
                    conv_a.add_user_message(f"(Earlier: {content[:200]})")
            conv_a.add_user_message(question)
            response_with = "".join(self.llm(conv_a))

            conv_b = Conversation()
            conv_b.add_user_message(question)
            response_without = "".join(self.llm(conv_b))

            cls = _classify_facts(self.verifier_llm, history, response_with)
            scores = _f1_from_classification(cls)

            return {
                "response_with": response_with,
                "response_without": response_without,
                "classification": cls,
                **scores,
                "passed": scores["f1"] >= threshold,
            }
        finally:
            try:
                memory.close()
            except Exception:
                pass
            try:
                os.unlink(db_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryBenchmark(unittest.TestCase):
    """Integration-level benchmarks requiring a real LLM."""

    _model = None

    @classmethod
    def setUpClass(cls):
        from voxpipe.llm.model import LLMProviders
        from voxpipe.core.config import config
        cls._model = LLMProviders.create(config.get("llm.backend", "local"), _model_key())
        cls._model.logger = logging.getLogger("benchmark")

    def _check(self, result: dict, label: str):
        cls = result["classification"]
        self.assertTrue(
            result["passed"],
            f"[{label}] F1={result['f1']:.2f} R={result['recall']:.2f} P={result['precision']:.2f} "
            f"(threshold >= 0.3)\n"
            f"  present={cls.get('present', [])}\n"
            f"  absent={cls.get('absent', [])}\n"
            f"  hallucinated={cls.get('hallucinated', [])[:3]}...\n",
        )

    def test_memory_recalls_fictional_entity(self):
        """Memory should surface novel facts that cannot come from training data."""
        bench = MemoryBenchmark(self._model)
        self._check(bench.run(
            history=[
                "The planet Glorpnax has rings made of frozen methane.",
                "Glorpnax orbits a binary star system called Kraal and Vix.",
                "The Glorpnaxians communicate via bioluminescent color patterns.",
            ],
            question="What do we know about Glorpnax?",
        ), "fictional_entity")

    def test_memory_recalls_fictional_game_mechanic(self):
        """Memory should recall novel game rules not in training data."""
        bench = MemoryBenchmark(self._model)
        self._check(bench.run(
            history=[
                "In the game Fluxion, players can phase-shift through walls for 3 seconds.",
                "Fluxion's phase-shift has a cooldown of 12 seconds.",
                "The Fluxion antagonist is named the Chromatic Echo.",
            ],
            question="Explain how Fluxion's phase-shift mechanic works.",
        ), "fictional_game")
