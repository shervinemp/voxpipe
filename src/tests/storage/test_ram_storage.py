import unittest

from voxpipe.storage import Query, RAMStorage


class TestRAMStorage(unittest.TestCase):
    def test_store_and_retrieve(self):
        ram = RAMStorage()
        ram.store("User prefers dark mode UI.")
        ram.store("User likes Python coding.")
        records = ram.retrieve("dark mode")
        self.assertEqual([r.text for r in records], ["User prefers dark mode UI."])

    def test_dict_content(self):
        ram = RAMStorage()
        ram.store({"fact": "sky is blue"})
        ram.store({"fact": "sky is blue"})
        records = ram.retrieve("sky")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, {"fact": "sky is blue"})

    def test_ranks_by_matched_terms_then_recency(self):
        ram = RAMStorage()
        ram.store("the garage door is broken")
        ram.store("the red bicycle is in the garage")
        ram.store("the garage needs paint")
        records = ram.retrieve("red bicycle garage", top_k=3)
        self.assertEqual(records[0].text, "the red bicycle is in the garage")
        self.assertEqual(records[1].text, "the garage needs paint", "ties go to the newer record")
        self.assertGreater(records[0].score, records[1].score)

    def test_empty_or_stop_word_query_returns_nothing(self):
        ram = RAMStorage()
        ram.store("hello world")
        self.assertEqual(ram.retrieve(""), [])
        self.assertEqual(ram.retrieve("a an the"), [])

    def test_role_filter(self):
        ram = RAMStorage()
        ram.store("question about cats", role="user")
        ram.store("answer about cats", role="assistant")
        self.assertEqual([r["role"] for r in ram.retrieve("cats", role="assistant")], ["assistant"])

    def test_session_and_global_banks(self):
        ram = RAMStorage()
        ram.store("global coding rules")
        ram.store("alice likes dark mode", session_id="alice")
        ram.store("bob likes light mode", session_id="bob")

        alice = [r.text for r in ram.retrieve("coding mode", session_id="alice", top_k=5)]
        self.assertCountEqual(alice, ["global coding rules", "alice likes dark mode"])
        alice_only = [r.text for r in ram.retrieve("coding mode", session_id="alice", include_global=False)]
        self.assertEqual(alice_only, ["alice likes dark mode"])
        self.assertEqual([r.text for r in ram.retrieve("coding mode")], ["global coding rules"])

    def test_max_entries_drops_oldest(self):
        ram = RAMStorage(max_entries=2)
        for word in ("alpha", "bravo", "charlie"):
            ram.store(f"codeword {word}")
        self.assertEqual(ram.retrieve("alpha"), [])
        self.assertEqual(len(ram.retrieve("codeword", top_k=5)), 2)

    def test_max_entries_must_be_positive(self):
        for max_entries in (0, -1):
            with self.subTest(max_entries=max_entries), self.assertRaises(ValueError):
                RAMStorage(max_entries=max_entries)

    def test_query_top_k_overrides_argument(self):
        ram = RAMStorage()
        for i in range(5):
            ram.store(f"note number {i}")
        self.assertEqual(len(ram.retrieve(Query(text="note", top_k=4), top_k=1)), 4)
        self.assertEqual(len(ram.retrieve(Query(text="note"), top_k=1)), 1)


if __name__ == "__main__":
    unittest.main()
