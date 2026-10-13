"""Tests for the stream sentence splitter."""
import unittest

from voxpipe.streaming.splitter import stream_splitter


class TestStreamSplitter(unittest.TestCase):
    def _split(self, chunks, min_len=0):
        return list(stream_splitter(iter(chunks), min_len=min_len))

    def test_single_sentence(self):
        self.assertEqual(self._split(["Hello world."]), ["Hello world."])

    def test_sentences_across_chunks(self):
        self.assertEqual(self._split(["Hello. How ", "are ", "you? Fine."]), ["Hello.", "How are you?", "Fine."])

    def test_every_sentence_in_one_chunk(self):
        text = "One is here. Two is here! Three is here? Four is here. Five is the last one."
        self.assertEqual(
            self._split([text]),
            ["One is here.", "Two is here!", "Three is here?", "Four is here.", "Five is the last one."],
        )

    def test_long_chunk_is_split_throughout(self):
        text = " ".join(f"Sentence number {i} goes here." for i in range(40))
        self.assertEqual(len(self._split([text])), 40)

    def test_token_stream_matches_whole_text(self):
        text = 'She said "Stop!" Then she left. Was that it? Yes... it was.'
        tokens = [text[i : i + 3] for i in range(0, len(text), 3)]
        self.assertEqual(self._split(tokens), self._split([text]))
        self.assertEqual(self._split([text]), ['She said "Stop!"', "Then she left.", "Was that it?", "Yes...", "it was."])

    def test_boundary_split_between_chunks(self):
        self.assertEqual(self._split(["Wow!!!!!!!!!!!!", " Next."]), ["Wow!!!!!!!!!!!!", "Next."])

    def test_abbreviation_not_split(self):
        self.assertEqual(self._split(["Dr. Smith is here. He came."]), ["Dr. Smith is here.", "He came."])

    def test_initialisms_not_split(self):
        self.assertEqual(self._split(["Use a tool, e.g. a hammer. Done."]), ["Use a tool, e.g. a hammer.", "Done."])

    def test_list_numbers_stay_with_their_item(self):
        self.assertEqual(
            self._split(["Here are the steps. 1. Mix it.\n2. Bake it."]),
            ["Here are the steps.", "1. Mix it.", "2. Bake it."],
        )

    def test_sentence_ending_in_a_number_is_split(self):
        self.assertEqual(self._split(["The answer is 42. Next question."]), ["The answer is 42.", "Next question."])

    def test_decimal_numbers_not_split(self):
        self.assertEqual(self._split(["Pi is 3.14 roughly. Yes."]), ["Pi is 3.14 roughly.", "Yes."])

    def test_no_trailing_punctuation(self):
        self.assertEqual(self._split(["Hello without punctuation"]), ["Hello without punctuation"])

    def test_empty_input(self):
        self.assertEqual(self._split([]), [])
        self.assertEqual(self._split([""]), [])
        self.assertEqual(self._split(["   ", "\n"]), [])

    def test_min_len_delays_the_first_sentence(self):
        stream = stream_splitter(iter(["Hi. ", "Bye now. ", "End."]), min_len=8)
        self.assertEqual(next(stream), "Hi.", "released once 8 characters are buffered")
        self.assertEqual(list(stream), ["Bye now.", "End."])
        self.assertEqual(self._split(["Hi. Bye."], min_len=8), ["Hi.", "Bye."])

    def test_sentences_are_yielded_before_the_stream_ends(self):
        pulled = []

        def source():
            for chunk in ["First one. ", "Second ", "one. ", "Third."]:
                pulled.append(chunk)
                yield chunk

        stream = stream_splitter(source())
        self.assertEqual(next(stream), "First one.")
        self.assertEqual(pulled, ["First one. "])

    def test_closing_the_splitter_closes_its_source(self):
        closed = []

        def source():
            try:
                yield "First one. "
                yield "Second one. "
            finally:
                closed.append(True)

        stream = stream_splitter(source())
        next(stream)
        stream.close()
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
