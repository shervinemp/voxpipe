"""Tests for stream sentence splitter."""
import unittest


class TestStreamSplitter(unittest.TestCase):
    def _split(self, chunks, min_len=0):
        from voxpipe.streaming.splitter import stream_splitter
        return list(stream_splitter(iter(chunks), min_len=min_len))

    def test_single_sentence(self):
        self.assertEqual(self._split(["Hello world."]), ["Hello world."])

    def test_sentences_across_chunks(self):
        self.assertEqual(self._split(["Hello. How ", "are ", "you? Fine."]),
                         ["Hello.", "How are you?", "Fine."])

    def test_abbreviation_not_split(self):
        self.assertEqual(self._split(["Dr. Smith is here. He came."]),
                         ["Dr. Smith is here.", "He came."])

    def test_no_trailing_punctuation(self):
        self.assertEqual(self._split(["Hello without punctuation"]),
                         ["Hello without punctuation"])

    def test_empty_chunks(self):
        self.assertEqual(self._split([]), [])

    def test_empty_string_chunk(self):
        self.assertEqual(self._split([""]), [])

    def test_min_len_triggers_scan(self):
        r = self._split(["Hi. Bye."], min_len=8)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], "Hi.")

    def test_one_sentence_per_iteration(self):
        r = self._split(["Hello. How are you? I'm fine."])
        self.assertEqual(r[0], "Hello.")
