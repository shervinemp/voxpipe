"""Synthetic restaurant scenario: generated voices, mixed audio, VAD, ASR and the gate.

This is an integration test.  It synthesises a worker talking to the assistant,
the worker talking to a customer, a customer reply, and background chatter with
different Kokoro voices; mixes them onto one track; segments the mix with the
real VAD; transcribes each segment with the real ASR; and checks that only the
AI-directed utterance is answered.
"""

import os
import tempfile
import unittest

import numpy as np

from voxpipe.asr.vad import Silero
from voxpipe.pipeline.attention.addressee import AddresseeGate, AddresseeModel
from voxpipe.pipeline.attention.embeddings import HashingEmbedder

from .._integration import KOKORO, PARAKEET, requires_model, transcribe, tts

pytestmark = requires_model("Kokoro", KOKORO + PARAKEET)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32)
    duration = len(samples) / source_rate
    target_count = max(1, int(round(duration * target_rate)))
    source_times = np.linspace(0.0, duration, len(samples), endpoint=False)
    target_times = np.linspace(0.0, duration, target_count, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def _place(track: np.ndarray, samples: np.ndarray, at_seconds: float, rate: int, gain: float = 1.0) -> None:
    start = int(at_seconds * rate)
    end = start + len(samples)
    if end > len(track):
        raise ValueError("track too short")
    track[start:end] += samples.astype(np.float32) * gain


class TestRestaurantScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts = tts()
        cls.rate = cls.tts.sample_rate

        cls.audio = {
            "background_1": cls.tts.synthesize("Did you see the game last night?", voice="am_michael")[0],
            "worker_ai_1": cls.tts.synthesize("Add two burgers for table four.", voice="af_heart")[0],
            "worker_other_1": cls.tts.synthesize("Would you like fries with that?", voice="af_heart")[0],
            "customer_1": cls.tts.synthesize("Yes, and a coke.", voice="am_adam")[0],
            "worker_other_2": cls.tts.synthesize("Would you like the salmon?", voice="af_heart")[0],
            "customer_2": cls.tts.synthesize("No, just water.", voice="bf_emma")[0],
            "background_2": cls.tts.synthesize("The kitchen is slammed tonight.", voice="bm_george")[0],
            "worker_ai_2": cls.tts.synthesize("What's the total for table four?", voice="af_heart")[0],
        }

    def _mixed_track(self) -> np.ndarray:
        # A 24-second track with five voices and several switches:
        # background staff -> AI command -> customer 1 (overlapping reply) ->
        # customer 2 (overlapping reply) -> background staff -> AI follow-up.
        duration = 24.0
        track = np.zeros(int(duration * self.rate), dtype=np.float32)
        _place(track, self.audio["background_1"], 0.0, self.rate, gain=0.30)
        _place(track, self.audio["worker_ai_1"], 3.2, self.rate, gain=1.0)
        _place(track, self.audio["worker_other_1"], 6.7, self.rate, gain=1.0)
        _place(track, self.audio["customer_1"], 8.2, self.rate, gain=0.85)
        _place(track, self.audio["worker_other_2"], 11.5, self.rate, gain=1.0)
        _place(track, self.audio["customer_2"], 13.0, self.rate, gain=0.85)
        _place(track, self.audio["background_2"], 16.0, self.rate, gain=0.28)
        _place(track, self.audio["worker_ai_2"], 19.5, self.rate, gain=1.0)
        noise = np.random.default_rng(7).normal(0.0, 0.004, len(track)).astype(np.float32)
        track += noise
        peak = float(np.max(np.abs(track))) or 1.0
        return track / peak * 0.9

    def _segments(self) -> list[np.ndarray]:
        vad = Silero(vad_threshold=0.4, leading_silence_duration=0.2, trailing_silence_duration=0.5)
        self.vad_rate = vad.sample_rate
        mixed = _resample(self._mixed_track(), self.rate, vad.sample_rate)
        hop = vad.hop_size
        for start in range(0, len(mixed) - hop + 1, hop):
            vad.feed(mixed[start : start + hop])
        vad.flush()
        vad.close()
        return [segment for segment in vad if len(segment)]

    def test_only_the_ai_directed_utterance_is_answered(self):
        segments = self._segments()
        self.assertTrue(segments, "VAD found no speech in the mixed track")

        transcripts = []
        for segment in segments:
            text = transcribe(segment, self.vad_rate)
            if text.strip():
                transcripts.append(text.strip())
        print("\nscenario transcripts:")
        for text in transcripts:
            print(f"  {text}")

        with tempfile.TemporaryDirectory() as tmp:
            model = AddresseeModel(
                HashingEmbedder(),
                examples_path=os.path.join(tmp, "examples.jsonl"),
            )
            gate = AddresseeGate(model, defer_ms=400, speak_threshold=0.70, silence_threshold=0.35)
            decisions = []
            for text in transcripts:
                decision = gate.evaluate(text)
                decisions.append((text, decision))
                if decision.answer:
                    gate.record_reply("Okay.")

            main_user = [u.text for u in gate.tracker.main.utterances if u.speaker == "user"]
            other_buffer = gate.buffer("other")
            background_buffer = gate.buffer("background")
            other_texts = [u.text for u in other_buffer.utterances] if other_buffer else []
            background_texts = [u.text for u in background_buffer.utterances] if background_buffer else []
            gate.close()

        print("\nscenario decisions:")
        for text, decision in decisions:
            print(f"  answer={decision.answer!s:<5} p={decision.confidence:.2f} reason={decision.reason:<10} {text}")

        answered = [(text, decision) for text, decision in decisions if decision.answer]
        self.assertEqual(len(answered), 2, f"expected two AI-directed answers, got {answered}")
        answered_text = " ".join(text.lower() for text, _ in answered)
        self.assertIn("add", answered_text)
        self.assertIn("burger", answered_text)
        self.assertIn("total", answered_text)
        for _, decision in answered:
            self.assertEqual(decision.defer_ms, 400)

        fries = [(text, decision) for text, decision in decisions if "fries" in text.lower()]
        self.assertTrue(fries, "the worker→customer utterance was not transcribed")
        self.assertTrue(any(decision.reason == "other" for _, decision in fries))
        for _, decision in fries:
            self.assertFalse(decision.answer)

        salmon = [(text, decision) for text, decision in decisions if "salmon" in text.lower()]
        self.assertTrue(salmon, "the second customer thread was not transcribed")
        self.assertTrue(any(decision.reason == "other" for _, decision in salmon))
        for _, decision in salmon:
            self.assertFalse(decision.answer)

        self.assertTrue(
            any("game" in text.lower() or "night" in text.lower() or "kitchen" in text.lower() for text in transcripts),
            "background chatter was not transcribed",
        )
        for text, decision in decisions:
            lower = text.lower()
            if "game" in lower or "night" in lower or "kitchen" in lower:
                self.assertFalse(decision.answer)

        # The typed buffers preserve the three threads instead of mixing them.
        self.assertEqual(len(main_user), 2, main_user)
        for text in main_user:
            lower = text.lower()
            self.assertTrue("add" in lower or "total" in lower, lower)
        self.assertTrue(any("fries" in text.lower() for text in other_texts), other_texts)
        self.assertTrue(any("salmon" in text.lower() for text in other_texts), other_texts)
        self.assertTrue(
            any("game" in text.lower() or "kitchen" in text.lower() for text in background_texts),
            background_texts,
        )


if __name__ == "__main__":
    unittest.main()
