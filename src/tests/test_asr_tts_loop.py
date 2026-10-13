"""Closed-loop ASR ↔ TTS tests using WER and edit distance.

Closed-loop tests: TTS → audio → ASR → text → compare to original.
Uses Word Error Rate (WER) as primary metric, character edit distance
as secondary.  Low WER means both TTS clarity and ASR accuracy are good.
"""
import re
import unittest

from ._integration import requires_model, transcribe, tts

pytestmark = requires_model("Kokoro")


# ---------------------------------------------------------------------------
# Edit-distance helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    n, m = len(a), len(b)
    if n < m:
        a, b, n, m = b, a, m, n
    prev = list(range(m + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[m]


def _wer(ref: str, hyp: str) -> tuple[int, int, float]:
    """Word Error Rate: (insertions + deletions + substitutions) / ref_word_count.

    Returns (errors, ref_words, wer_ratio).
    """
    r = ref.strip().split()
    h = hyp.strip().split()
    n, m = len(r), len(h)
    # DP matrix for word-level edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    errors = dp[n][m]
    return errors, n, errors / max(n, 1)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())


# ---------------------------------------------------------------------------
# Test sentences grouped by difficulty
# ---------------------------------------------------------------------------

SIMPLE = [
    "this is a test sentence for the voice detection system",
    "hello how are you doing today",
    "yes",
    "no",
    "go back",
]

COMPLEX = [
    "navigate to the main menu and select options",
    "save the current game state before quitting",
    "what is the capital of france",
    "defeat the dragon lord in the northern mountains",
    "cast fireball at the goblin camp",
]

ALL_SENTENCES = SIMPLE + COMPLEX


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTTSASRLoop(unittest.TestCase):
    """Closed-loop: TTS → audio → ASR → text, measure WER and edit distance."""

    def test_wer_simple(self):
        """Simple phrases should have low WER."""
        self._run_sentences(SIMPLE, max_wer=0.4)

    def test_wer_complex(self):
        """Longer sentences can tolerate higher WER."""
        self._run_sentences(COMPLEX, max_wer=0.5)

    def test_wer_overall_average(self):
        """Average WER across all sentences must be below threshold."""
        wers = self._run_sentences(ALL_SENTENCES, max_wer=1.0)
        avg = sum(wers) / len(wers)
        print(f"  Average WER: {avg:.2%}")
        self.assertLess(avg, 0.35, f"Average WER {avg:.2%} too high")

    def _run_sentences(self, sentences, max_wer):
        wers = []
        for sentence in sentences:
            with self.subTest(sentence=sentence[:30]):
                samples, sr = tts().synthesize(sentence)
                orig_norm = _normalize(sentence)
                trans_norm = _normalize(transcribe(samples, sr))
                err, n, ratio = _wer(orig_norm, trans_norm)
                char_dist = _levenshtein(orig_norm, trans_norm)
                wers.append(ratio)

                print(f"  orig: {orig_norm}")
                print(f"  asr:  {trans_norm}")
                print(f"  WER:  {err}/{n} = {ratio:.2%}  char_edit: {char_dist}")
                self.assertLess(ratio, max_wer, f"WER {ratio:.2%} exceeds {max_wer:.0%} for '{sentence}'")
        return wers


class TestASRConsistency(unittest.TestCase):
    """Determinism and self-consistency checks."""

    def test_asr_deterministic(self):
        """Same audio → same transcription (WER = 0 between runs)."""
        samples, sr = tts().synthesize("the quick brown fox jumps over the lazy dog")
        t1 = _normalize(transcribe(samples, sr))
        t2 = _normalize(transcribe(samples, sr))
        err, _, _ = _wer(t1, t2)
        self.assertEqual(err, 0, f"ASR not deterministic: '{t1}' vs '{t2}'")

    def test_asr_not_empty(self):
        """ASR should produce non-empty output for speech audio."""
        for text in SIMPLE:
            samples, sr = tts().synthesize(text)
            self.assertGreater(len(transcribe(samples, sr).strip()), 0, f"ASR empty for '{text}'")


class TestTTSAudioQuality(unittest.TestCase):
    """Basic audio quality heuristics."""

    def test_duration_proportional_to_length(self):
        """Longer text → proportionally longer audio."""
        short = "hello"
        long_ = "the quick brown fox jumps over the lazy dog near the riverbank"

        s_short, sr = tts().synthesize(short)
        s_long, _ = tts().synthesize(long_)

        d_short = len(s_short) / sr
        d_long = len(s_long) / sr

        self.assertGreater(d_short, 0.1, f"Short audio too brief: {d_short:.2f}s")
        self.assertGreater(d_long, d_short, f"Long ({d_long:.2f}s) should be longer than short ({d_short:.2f}s)")


if __name__ == "__main__":
    unittest.main()

