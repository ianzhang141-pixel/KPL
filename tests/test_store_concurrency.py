import tempfile
import threading
import unittest
from pathlib import Path

from kplab import store


class ConcurrentAppendTests(unittest.TestCase):
    def test_parallel_gzip_appends_do_not_corrupt_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl.gz"
            threads = [
                threading.Thread(target=store.append_jsonl_gz, args=(path, {"index": i}))
                for i in range(40)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            records = list(store.read_jsonl_gz(path))
            self.assertEqual(len(records), 40)
            self.assertEqual({record["index"] for record in records}, set(range(40)))


if __name__ == "__main__":
    unittest.main()
