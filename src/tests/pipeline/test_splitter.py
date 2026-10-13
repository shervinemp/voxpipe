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


class TestBoundaries(unittest.TestCase):
    @staticmethod
    def split(text):
        from voxpipe.streaming.splitter import stream_splitter

        return list(stream_splitter([text]))

    def test_an_initial_does_not_split_a_name(self):
        self.assertEqual(
            self.split("Samuel L. Jackson is an actor. He is famous."),
            ["Samuel L. Jackson is an actor.", "He is famous."],
        )

    def test_a_number_or_a_domain_can_end_a_sentence(self):
        self.assertEqual(self.split("That comes to 3.50. Shall I order it?"), ["That comes to 3.50.", "Shall I order it?"])
        self.assertEqual(self.split("Try example.com. It is free."), ["Try example.com.", "It is free."])

    def test_names_that_look_like_abbreviations_end_a_sentence(self):
        self.assertEqual(self.split("My dog is called Max. He is five."), ["My dog is called Max.", "He is five."])
        self.assertEqual(self.split("Her name is Jan. She is here."), ["Her name is Jan.", "She is here."])

    def test_the_words_i_and_a_can_end_a_sentence(self):
        self.assertEqual(self.split("So do I. Shall we go?"), ["So do I.", "Shall we go?"])

    def test_list_items_separated_by_line_breaks_are_spoken_one_by_one(self):
        self.assertEqual(
            self.split("Here are three options:\n- Paris\n- Rome\n- Oslo"),
            ["Here are three options:", "- Paris", "- Rome", "- Oslo"],
        )

    def test_a_numbered_list_keeps_each_number_with_its_item(self):
        self.assertEqual(self.split("1. Paris\n2. Rome\n"), ["1. Paris", "2. Rome"])


if __name__ == "__main__":
    unittest.main()
