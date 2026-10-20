"""Tests for topic tracking, relevance scorers and buffer lifecycle."""

import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

from voxpipe.core.exceptions import ConfigError
from voxpipe.core.models import TopicTrackingConfig
from voxpipe.pipeline.topics import (
    EmbeddingScorer,
    LayaScorer,
    LexicalScorer,
    NoFollowUpPolicy,
    PassThroughGate,
    RouteKind,
    TopicTracker,
    Utterance,
    create_tracker,
)

from ._topic_fakes import ScriptedScorer


def make_tracker(*scores, **kwargs):
    scorer = ScriptedScorer(*scores)
    return TopicTracker(scorer, **kwargs), scorer


class TestBufferLifecycle(unittest.TestCase):
    def test_first_utterance_opens_the_main_topic(self):
        tracker, scorer = make_tracker()
        decision = tracker.route("hello there")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "start")
        self.assertEqual([u.text for u in tracker.main.utterances], ["hello there"])
        self.assertEqual(scorer.calls, [], "nothing to compare with yet")

    def test_a_related_utterance_extends_the_main_topic(self):
        tracker, scorer = make_tracker([0.91])
        tracker.route("hello there")
        decision = tracker.route("open the kitchen door")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "main")
        self.assertEqual(decision.buffer_id, tracker.main.id)
        self.assertEqual(
            [u.text for u in tracker.main.utterances], ["hello there", "open the kitchen door"]
        )
        self.assertEqual(scorer.calls[0][1], [["hello there"]])

    def test_an_unrelated_utterance_starts_a_side_buffer_and_is_not_answered(self):
        tracker, _ = make_tracker([0.08])
        tracker.route("hello there")
        decision = tracker.route("the stock market fell today")
        self.assertFalse(decision.answer)
        self.assertTrue(decision.new_topic)
        self.assertFalse(decision.related)
        self.assertEqual(len(tracker.unrelated), 1)
        self.assertEqual([u.text for u in tracker.unrelated[0].utterances], ["the stock market fell today"])

    def test_a_later_sentence_can_join_the_side_buffer(self):
        tracker, _ = make_tracker([0.05], [0.08, 0.84])
        tracker.route("hello there")
        first = tracker.route("the stock market fell today")
        second = tracker.route("interest rates are rising again")
        self.assertFalse(second.answer)
        self.assertFalse(second.new_topic)
        self.assertTrue(second.related)
        self.assertEqual(second.buffer_id, first.buffer_id)
        self.assertEqual(len(tracker.unrelated), 1)
        self.assertEqual(
            [u.text for u in tracker.unrelated[0].utterances],
            ["the stock market fell today", "interest rates are rising again"],
        )

    def test_a_third_thread_gets_its_own_buffer(self):
        tracker, _ = make_tracker([0.05], [0.04, 0.07])
        tracker.route("hello there")
        tracker.route("the stock market fell today")
        decision = tracker.route("my cat has been sick")
        self.assertTrue(decision.new_topic)
        self.assertEqual(len(tracker.unrelated), 2)
        self.assertNotEqual(decision.buffer_id, tracker.unrelated[0].id)

    def test_unused_side_buffers_expire_after_the_ttl(self):
        tracker, _ = make_tracker([0.05], [0.8, 0.05], [0.8, 0.05], [0.8], ttl_sentences=2)
        tracker.route("hello there")
        tracker.route("the stock market fell today")  # index 2, side buffer
        tracker.route("back to the main topic")  # index 3
        tracker.route("still on the main topic")  # index 4
        self.assertEqual(len(tracker.unrelated), 1, "two idle utterances is still within a ttl of 2")
        tracker.route("and again")  # index 5: the side buffer has now been idle for three
        self.assertEqual(tracker.unrelated, [])

    def test_the_least_recently_used_side_buffer_is_dropped(self):
        tracker, _ = make_tracker([0.05], [0.04, 0.07], ttl_sentences=20, max_unrelated=1)
        tracker.route("hello there")
        tracker.route("the stock market fell today")
        tracker.route("my cat has been sick")
        self.assertEqual(len(tracker.unrelated), 1)
        self.assertEqual([u.text for u in tracker.unrelated[0].utterances], ["my cat has been sick"])

    def test_a_side_topic_is_promoted_after_a_run_of_related_sentences(self):
        tracker, _ = make_tracker([0.05], [0.04, 0.9], promote_after=2)
        tracker.route("hello there")
        tracker.record_reply("Hello. How can I help?")
        first = tracker.route("the stock market fell today")
        second = tracker.route("interest rates are rising again")
        self.assertFalse(first.answer)
        self.assertTrue(second.answer)
        self.assertTrue(second.promoted)
        self.assertIn("the stock market fell today", second.context or "")
        self.assertEqual(tracker.main.id, first.buffer_id)
        self.assertEqual([buffer.id for buffer in tracker.unrelated], [0], "the old topic is kept aside")

    def test_promote_after_one_answers_a_new_thread_immediately(self):
        tracker, _ = make_tracker([0.0], promote_after=1)
        tracker.route("hello there")
        decision = tracker.route("a completely different subject")
        self.assertTrue(decision.answer)
        self.assertTrue(decision.promoted)
        self.assertEqual(tracker.main.id, decision.buffer_id)

    def test_a_short_answer_to_a_question_needs_no_topic_match(self):
        tracker, scorer = make_tracker()
        tracker.route("what should I pack?")
        tracker.record_reply("Where are you going?")
        decision = tracker.route("Paris")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "follow-up")
        self.assertEqual(scorer.calls, [], "the reply heuristic wins before scoring")

    def test_evaluate_implements_the_generic_gate_contract(self):
        tracker, _ = make_tracker([0.0])
        first = tracker.evaluate("hello")
        self.assertTrue(first.answer)
        self.assertIsNotNone(first.route)
        second = tracker.evaluate("the stock market fell")
        self.assertFalse(second.answer)
        self.assertIsNotNone(second.route)
        explicit = tracker.evaluate("typed by the user", explicit=True)
        self.assertTrue(explicit.answer)
        self.assertIsNone(explicit.route, "explicit input is not routed by similarity")

    def test_named_buffers_can_be_routed_explicitly(self):
        tracker, _ = make_tracker()
        tracker.route("hello")
        decision = tracker.route_to("table_4", "two burgers")
        self.assertFalse(decision.answer)
        self.assertEqual(decision.reason, "explicit-buffer")
        named = tracker.buffer("table_4")
        self.assertIsNotNone(named)
        self.assertEqual(named.name, "table_4")
        tracker.route_to("table_4", "one coke")
        self.assertEqual([u.text for u in named.utterances], ["two burgers", "one coke"])
        self.assertIs(tracker.buffer("main"), tracker.main)

    def test_named_buffers_are_not_evicted_by_inferred_side_buffers(self):
        tracker, _ = make_tracker([0.0], [0.0, 0.0], max_unrelated=1)
        tracker.route("hello")
        tracker.route_to("table_4", "two burgers")
        tracker.route("the stock market fell")
        tracker.route("my cat is sick")
        self.assertIsNotNone(tracker.buffer("table_4"))
        self.assertEqual([buffer.name for buffer in tracker.named], ["table_4"])
        self.assertEqual(len(tracker.side), 1, "only inferred side buffers are capped")

    def test_named_buffers_can_have_their_own_ttl(self):
        tracker, _ = make_tracker([0.9], [0.9])
        tracker.route("hello")
        tracker.route_to("table_4", "two burgers", ttl_sentences=1)
        tracker.route("main again")
        self.assertIsNotNone(tracker.buffer("table_4"), "one idle utterance is within the TTL")
        tracker.route("main again")
        self.assertIsNone(tracker.buffer("table_4"))

    def test_a_follow_up_policy_can_be_replaced(self):
        tracker = TopicTracker(ScriptedScorer([0.0]), follow_up_policy=NoFollowUpPolicy())
        tracker.route("hello")
        tracker.record_reply("Which city?")
        decision = tracker.route("Paris")
        self.assertFalse(decision.answer, "without the policy, the scorer decides")

    def test_reset_during_scoring_drops_the_stale_decision(self):
        class BlockingScorer:
            default_threshold = 0.4

            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()

            def scores(self, text, candidates):
                self.entered.set()
                self.release.wait(timeout=3)
                return [0.0] * len(candidates)

        scorer = BlockingScorer()
        tracker = TopicTracker(scorer)
        tracker.route("hello")
        results = []
        worker = threading.Thread(target=lambda: results.append(tracker.route("something new")))
        worker.start()
        self.assertTrue(scorer.entered.wait(timeout=3))
        tracker.reset()
        scorer.release.set()
        worker.join(timeout=3)
        self.assertEqual(results[0].kind, RouteKind.RESET)
        self.assertFalse(results[0].answer)

    def test_routing_after_close_is_declined(self):
        tracker, _ = make_tracker()
        tracker.close()
        decision = tracker.route("anything")
        self.assertEqual(decision.kind, RouteKind.RESET)
        self.assertFalse(decision.answer)
        explicit = tracker.evaluate("typed", explicit=True)
        self.assertFalse(explicit.answer)

    def test_route_to_rejects_an_empty_buffer_name(self):
        tracker, _ = make_tracker()
        with self.assertRaises(ValueError):
            tracker.route_to("", "anything")

    def test_pass_through_gate_answers_everything(self):
        gate = PassThroughGate()
        decision = gate.evaluate("anything")
        self.assertTrue(decision.answer)
        self.assertIsNone(decision.route)

    def test_replies_are_part_of_the_main_topic(self):
        tracker, _ = make_tracker()
        tracker.route("hello")
        tracker.record_reply("Hi! How can I help?")
        self.assertEqual(
            [(u.speaker, u.text) for u in tracker.main.utterances],
            [("user", "hello"), ("assistant", "Hi! How can I help?")],
        )

    def test_reset_clears_every_buffer(self):
        tracker, _ = make_tracker([0.05])
        tracker.route("hello")
        tracker.route("the stock market fell")
        tracker.reset()
        self.assertEqual([u.text for u in tracker.main.utterances], [])
        self.assertEqual(tracker.unrelated, [])
        self.assertEqual(tracker.route("a fresh start").reason, "start")

    def test_an_empty_utterance_does_not_change_anything(self):
        tracker, _ = make_tracker()
        decision = tracker.route("   ")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "empty")
        self.assertEqual([u.text for u in tracker.main.utterances], [])

    def test_a_failing_scorer_falls_back_instead_of_silencing_the_assistant(self):
        class Broken:
            def scores(self, text, candidates):
                raise RuntimeError("model died")

        tracker = TopicTracker(Broken(), threshold=0.12)
        tracker.route("we should plan the Paris trip")
        decision = tracker.route("I booked a flight to Paris")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "main")

    def test_the_fallback_uses_its_own_threshold(self):
        class Broken:
            def scores(self, text, candidates):
                raise RuntimeError("model died")

        # A 0.9 primary threshold must not make the lexical fallback useless.
        tracker = TopicTracker(Broken(), threshold=0.9, fallback_threshold=0.12)
        tracker.route("we should plan the Paris trip")
        decision = tracker.route("I booked a flight to Paris")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "main")

    def test_close_is_idempotent_and_closes_the_scorer_once(self):
        class CountingScorer:
            def __init__(self):
                self.closed = 0

            def scores(self, text, candidates):
                return [0.0] * len(candidates)

            def close(self):
                self.closed += 1

        scorer = CountingScorer()
        tracker = TopicTracker(scorer)
        tracker.close()
        tracker.close()
        self.assertEqual(scorer.closed, 1)

    def test_rejects_impossible_settings(self):
        for kwargs in (
            {"threshold": -0.1},
            {"threshold": 1.1},
            {"ttl_sentences": 0},
            {"max_unrelated": 0},
            {"history_sentences": 0},
            {"promote_after": -1},
        ):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                TopicTracker(LexicalScorer(), **kwargs)


class TestLexicalScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = LexicalScorer()

    def test_shared_names_and_words_score_above_unrelated_text(self):
        candidates = [
            [Utterance("user", "We should plan the Paris trip")],
            [Utterance("user", "The stock market fell today")],
        ]
        scores = self.scorer.scores("I booked a flight to Paris", candidates)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], 0.1)
        self.assertEqual(scores[1], 0.0)

    def test_case_and_punctuation_do_not_matter(self):
        upper = self.scorer.scores("PARIS!", [[Utterance("user", "paris")]])
        lower = self.scorer.scores("paris", [[Utterance("user", "Paris.")]])
        self.assertEqual(upper, lower)

    def test_a_long_reply_is_compared_clause_by_clause(self):
        reply = Utterance(
            "assistant",
            "The weather is fine. Your flight to Paris leaves tomorrow. Anything else?",
        )
        scores = self.scorer.scores("When does my flight leave?", [[reply]])
        self.assertGreater(scores[0], 0.1)

    def test_empty_text_and_empty_buffers_score_zero(self):
        self.assertEqual(self.scorer.scores("", [[Utterance("user", "hello")]]), [0.0])
        self.assertEqual(self.scorer.scores("hello", [[]]), [0.0])


class FakeRouter:
    """A stand-in for laya.Router that records requests."""

    def __init__(self, probabilities, *, batch=True):
        self.probabilities = list(probabilities)
        self.requests = []
        self._batch = batch
        self.unloaded = False

    def unload(self):
        self.unloaded = True

    def predict(self, state, questions, **options):
        self.requests.append({"state": state, "questions": questions, **options})
        return {"answers": {"related": {"noul": self.probabilities.pop(0)}}}

    def predict_batch(self, requests):
        if not self._batch:
            raise RuntimeError("no batch support")
        self.requests.extend(requests)
        return [{"answers": {"related": {"noul": p}}} for p in self.probabilities]


class TestLayaScorer(unittest.TestCase):
    def test_scores_every_buffer_in_one_batch_call(self):
        router = FakeRouter([0.9, 0.1])
        scorer = LayaScorer(router=router, max_len=512)
        scores = scorer.scores(
            "I booked a flight to Paris",
            [
                [Utterance("user", "We should plan the Paris trip"), Utterance("assistant", "When?")],
                [Utterance("user", "The stock market fell today")],
            ],
        )
        self.assertEqual(scores, [0.9, 0.1])
        self.assertEqual(len(router.requests), 2)
        request = router.requests[0]
        self.assertEqual(request["state"]["new_utterance"], "I booked a flight to Paris")
        self.assertIn("User: We should plan the Paris trip", request["state"]["conversation"])
        self.assertIn("Assistant: When?", request["state"]["conversation"])
        self.assertEqual(request["questions"]["related"]["type"], "noul")
        self.assertEqual(request["max_len"], 512)

    def test_falls_back_to_one_request_at_a_time_when_batching_fails(self):
        router = FakeRouter([0.2, 0.8], batch=False)
        scorer = LayaScorer(router=router)
        scores = scorer.scores("hello", [[Utterance("user", "a")], [Utterance("user", "b")]])
        self.assertEqual(scores, [0.2, 0.8])
        self.assertEqual(len(router.requests), 2)

    def test_a_pinned_checkpoint_is_named_on_every_request(self):
        router = FakeRouter([0.5])
        LayaScorer(router=router, model="multilingual").scores("hello", [[Utterance("user", "a")]])
        self.assertEqual(router.requests[0]["model"], "multilingual")

    def test_probabilities_are_clamped(self):
        self.assertEqual(LayaScorer._probability({"answers": {"related": {"noul": 1.4}}}), 1.0)
        self.assertEqual(LayaScorer._probability({"answers": {"related": {"noul": -0.2}}}), 0.0)

    def test_a_malformed_laya_response_raises_so_the_tracker_can_fall_back(self):
        for result in ({"answers": {}}, None):
            with self.subTest(result=result), self.assertRaises((KeyError, TypeError)):
                LayaScorer._probability(result)

    def test_missing_laya_is_a_config_error(self):
        with patch.dict(sys.modules, {"laya": None}):
            with self.assertRaisesRegex(ConfigError, "voxpipe\\[laya\\]"):
                LayaScorer()

    def test_close_unloads_the_router(self):
        router = FakeRouter([0.5])
        LayaScorer(router=router).close()
        self.assertTrue(router.unloaded)

    def test_a_malformed_laya_response_falls_back_instead_of_starting_a_new_topic(self):
        class BadRouter:
            def predict_batch(self, requests):
                return [{} for _ in requests]

        tracker = TopicTracker(
            LayaScorer(router=BadRouter()),
            threshold=0.5,
            fallback_threshold=0.12,
        )
        tracker.route("we should plan the Paris trip")
        decision = tracker.route("I booked a flight to Paris")
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "main")

    def test_laya_circuit_breaker_stops_repeated_failed_calls(self):
        class FailingRouter:
            def __init__(self):
                self.calls = 0

            def predict_batch(self, requests):
                self.calls += 1
                raise RuntimeError("model down")

        router = FailingRouter()
        scorer = LayaScorer(router=router, failure_limit=2, cooldown_seconds=60)
        candidates = [[Utterance("user", "hello")]]
        for _ in range(2):
            with self.assertRaises((RuntimeError, AttributeError)):
                scorer.scores("hello", candidates)
        with self.assertRaisesRegex(RuntimeError, "cooling down"):
            scorer.scores("hello", candidates)
        self.assertEqual(router.calls, 2, "the open circuit should not call the model again")


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        for text in texts:
            yield np.asarray(self.vectors[text], dtype=np.float32)


class TestEmbeddingScorer(unittest.TestCase):
    def test_uses_the_closest_sentence_in_each_buffer(self):
        embedder = FakeEmbedder({"cat": [1, 0], "kitten": [1, 0], "stock": [0, 1]})
        scorer = EmbeddingScorer(model="fake", embedder=embedder)
        scores = scorer.scores(
            "kitten",
            [[Utterance("user", "cat")], [Utterance("user", "stock")]],
        )
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)

    def test_repeated_sentences_are_embedded_once(self):
        embedder = FakeEmbedder({"cat": [1, 0], "kitten": [1, 0], "dog": [1, 0]})
        scorer = EmbeddingScorer(model="fake", embedder=embedder)
        candidates = [[Utterance("user", "cat")]]
        scorer.scores("kitten", candidates)
        scorer.scores("dog", candidates)
        self.assertEqual(embedder.calls[0], ["cat", "kitten"])
        self.assertEqual(embedder.calls[1], ["dog"], "the buffer sentence came from the cache")

    def test_empty_buffers_score_zero(self):
        scorer = EmbeddingScorer(model="fake", embedder=FakeEmbedder({"hello": [1, 0]}))
        self.assertEqual(scorer.scores("hello", [[]]), [0.0])

    def test_a_tiny_cache_does_not_drop_vectors_from_the_current_batch(self):
        embedder = FakeEmbedder({"cat": [1, 0], "dog": [0, 1], "kitten": [1, 0]})
        scorer = EmbeddingScorer(model="fake", embedder=embedder, max_cache=1)
        scores = scorer.scores(
            "kitten",
            [[Utterance("user", "cat")], [Utterance("user", "dog")]],
        )
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)

    def test_missing_fastembed_is_a_config_error(self):
        with patch.dict(sys.modules, {"fastembed": None}):
            with self.assertRaisesRegex(ConfigError, "voxpipe\\[embed\\]"):
                EmbeddingScorer(model="BAAI/bge-small-en-v1.5")


class TestFactory(unittest.TestCase):
    def test_lexical_is_the_zero_dependency_default(self):
        tracker = create_tracker(TopicTrackingConfig())
        self.assertIsInstance(tracker.scorer, LexicalScorer)
        self.assertAlmostEqual(tracker.threshold, 0.12)

    def test_explicit_threshold_and_buffer_settings_are_used(self):
        tracker = create_tracker(
            TopicTrackingConfig(threshold=0.42, ttl_sentences=7, max_unrelated=2, promote_after=3)
        )
        self.assertEqual(tracker.threshold, 0.42)
        self.assertEqual(tracker.ttl_sentences, 7)
        self.assertEqual(tracker.max_unrelated, 2)
        self.assertEqual(tracker.promote_after, 3)

    def test_invalid_threshold_is_rejected_by_the_schema(self):
        with self.assertRaises(Exception):
            TopicTrackingConfig(threshold=2.0)


if __name__ == "__main__":
    unittest.main()
