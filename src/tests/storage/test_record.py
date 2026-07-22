import unittest
from voxpipe.storage.record import Record


class TestRecord(unittest.TestCase):
    def test_record_creation_and_attributes(self):
        rec = Record(content="Hello world", source="test", score=0.95, meta={"author": "alice"})
        self.assertEqual(rec.content, "Hello world")
        self.assertEqual(rec.source, "test")
        self.assertEqual(rec.score, 0.95)
        self.assertEqual(rec.meta["author"], "alice")

    def test_record_str_and_text(self):
        rec = Record(content="Text content")
        self.assertEqual(str(rec), "Text content")
        self.assertEqual(rec.text, "Text content")

        dict_rec = Record(content={"key": "val"})
        self.assertIn('"key": "val"', str(dict_rec))
        self.assertIn('"key": "val"', dict_rec.text)

    def test_dual_access_indexing(self):
        rec = Record(content="Content val", source="src1", meta={"extra": 123})
        self.assertEqual(rec["content"], "Content val")
        self.assertEqual(rec["source"], "src1")
        self.assertEqual(rec["extra"], 123)


if __name__ == "__main__":
    unittest.main()
