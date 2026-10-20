"""Tests for the addressee gate, personal model and busy-mode preset."""

import os
import tempfile
import unittest
from unittest.mock import patch

from voxpipe.core.models import AttentionConfig
from voxpipe.pipeline.attention.embeddings import HashingEmbedder
from voxpipe.pipeline.topics import (
    AddresseeGate,
    AddresseeModel,
    FloorState,
    create_addressee_gate,
)


class TestAddresseeModel(unittest.TestCase):
    def make(self, path=None):
        return AddresseeModel(HashingEmbedder(), examples_path=path)

    def test_command_like_speech_scores_high(self):
        model = self.make()
        self.assertGreater(model.score("add two burgers for table four", FloorState()), 0.7)
        self.assertGreater(model.score("what's the total?", FloorState()), 0.7)

    def test_customer_script_scores_low(self):
        model = self.make()
        self.assertLess(model.score("would you like fries with that?", FloorState()), 0.35)
        self.assertLess(model.score("what can I get you?", FloorState()), 0.35)

    def test_a_short_answer_after_a_question_scores_high(self):
        model = self.make()
        self.assertGreater(model.score("Paris", FloorState(awaiting_reply=True)), 0.7)

    def test_learning_a_positive_example_raises_the_score(self):
        model = self.make()
        text = "alpha beta"
        before = model.score(text, FloorState())
        model.learn(text, True)
        self.assertGreater(model.score(text, FloorState()), before)
        self.assertEqual(model.examples[0].text, text)
        self.assertTrue(model.examples[0].addressed)

    def test_learning_a_negative_example_lowers_the_score(self):
        model = self.make()
        text = "alpha beta"
        model.learn(text, True)
        positive = model.score(text, FloorState())
        model.learn(text, False)
        self.assertLess(model.score(text, FloorState()), positive)
        self.assertEqual(len(model.examples), 1, "the same text updates rather than duplicating")

    def test_examples_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "examples.jsonl")
            first = self.make(path)
            first.learn("alpha beta", True)
            second = self.make(path)
            self.assertEqual(len(second.examples), 1)
            self.assertEqual(second.examples[0].text, "alpha beta")

    def test_a_corrupt_example_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "examples.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not json\n")
            with self.assertLogs("voxpipe.pipeline.attention.addressee", level="WARNING"):
                model = self.make(path)
            self.assertEqual(model.examples, [])


class TestAddresseeGate(unittest.TestCase):
    def make(self, **kwargs):
        gate = AddresseeGate(
            AddresseeModel(HashingEmbedder(), examples_path=None),
            speak_threshold=0.7,
            silence_threshold=0.35,
            defer_ms=250,
            **kwargs,
        )
        self.addCleanup(gate.close)
        return gate

    def test_explicit_input_is_always_answered(self):
        gate = self.make()
        decision = gate.evaluate("anything at all", explicit=True)
        self.assertTrue(decision.answer)
        self.assertEqual(decision.defer_ms, 0)
        self.assertEqual([u.text for u in gate.tracker.main.utterances], ["anything at all"])

    def test_addressed_speech_is_answered_with_a_deferral(self):
        gate = self.make()
        decision = gate.evaluate("add two burgers for table four")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.defer_ms, 250)
        self.assertEqual(decision.reason, "addressed")

    def test_customer_speech_goes_to_the_customer_buffer(self):
        gate = self.make()
        decision = gate.evaluate("would you like fries with that?")
        self.assertFalse(decision.answer)
        self.assertEqual(decision.reason, "customer")
        customer = gate.buffer("customer")
        self.assertIsNotNone(customer)
        self.assertEqual([u.text for u in customer.utterances], ["would you like fries with that?"])
        self.assertEqual([u.text for u in gate.tracker.main.utterances], [])

    def test_ambiguous_speech_goes_to_background_and_stays_silent(self):
        gate = self.make()
        decision = gate.evaluate("the sky is blue")
        self.assertFalse(decision.answer)
        self.assertEqual(decision.reason, "background")

    def test_a_short_answer_after_a_question_is_answered(self):
        gate = self.make()
        gate.record_reply("Which city?")
        decision = gate.evaluate("Paris")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "addressed")

    def test_shadow_mode_records_without_speaking(self):
        gate = self.make(shadow=True)
        decision = gate.evaluate("add two burgers")
        self.assertFalse(decision.answer)
        self.assertEqual(decision.reason, "shadow")
        self.assertEqual([u.text for u in gate.tracker.main.utterances], ["add two burgers"])

    def test_not_you_learns_the_previous_utterance_as_negative(self):
        gate = self.make()
        gate.evaluate("the sky is blue")
        correction = gate.evaluate("not you")
        self.assertFalse(correction.answer)
        self.assertEqual(correction.reason, "correction")
        self.assertEqual(gate.examples[0].text, "the sky is blue")
        self.assertFalse(gate.examples[0].addressed)

    def test_that_was_me_learns_the_previous_utterance_as_positive(self):
        gate = self.make()
        gate.evaluate("the sky is blue")
        correction = gate.evaluate("that was me")
        self.assertEqual(correction.reason, "correction")
        self.assertTrue(gate.examples[0].addressed)

    def test_customer_presence_and_order_phase_are_settable(self):
        gate = self.make()
        gate.set_customer_present(True)
        gate.set_order_phase("order")
        self.assertTrue(gate.state.customer_present)
        self.assertEqual(gate.state.order_phase, "order")


class TestBusyPreset(unittest.TestCase):
    def test_busy_mode_fills_in_deferral_and_thresholds(self):
        with patch(
            "voxpipe.pipeline.attention.factory.create_embedder",
            return_value=HashingEmbedder(),
        ):
            gate = create_addressee_gate(AttentionConfig(busy_mode=True, examples_path=None))
        self.addCleanup(gate.close)
        self.assertEqual(gate.defer_ms, 400)
        self.assertAlmostEqual(gate.speak_threshold, 0.70)
        self.assertAlmostEqual(gate.silence_threshold, 0.35)

    def test_explicit_values_override_the_busy_preset(self):
        with patch(
            "voxpipe.pipeline.attention.factory.create_embedder",
            return_value=HashingEmbedder(),
        ):
            gate = create_addressee_gate(
                AttentionConfig(busy_mode=True, defer_ms=100, speak_threshold=0.8, examples_path=None)
            )
        self.addCleanup(gate.close)
        self.assertEqual(gate.defer_ms, 100)
        self.assertAlmostEqual(gate.speak_threshold, 0.8)

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(Exception):
            AttentionConfig(speak_threshold=0.4, silence_threshold=0.6)


if __name__ == "__main__":
    unittest.main()
