import unittest
from voxpipe.storage.protocols import Retriever, Storer, Storage, Query
from voxpipe.storage.ram import RAMStorage
from voxpipe.storage.sqlite import SQLiteStorage
import tempfile
import os


class TestProtocols(unittest.TestCase):
    def test_protocol_conformance(self):
        ram = RAMStorage()
        self.assertTrue(isinstance(ram, Retriever))
        self.assertTrue(isinstance(ram, Storer))
        self.assertTrue(isinstance(ram, Storage))

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        sqlite_store = SQLiteStorage(path)
        try:
            self.assertTrue(isinstance(sqlite_store, Retriever))
            self.assertTrue(isinstance(sqlite_store, Storer))
            self.assertTrue(isinstance(sqlite_store, Storage))
        finally:
            sqlite_store.close()
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_query_dataclass(self):
        q = Query(text="test query", top_k=5)
        self.assertEqual(str(q), "test query")
        self.assertEqual(q.top_k, 5)


if __name__ == "__main__":
    unittest.main()
