"""Failed numerical consumers retain their independently decoded evidence."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('reader', Path(__file__).resolve().parents[2] / 'tools/event_track_v2x/compact_teacher_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class RetentionTests(unittest.TestCase):
    def test_consumer_failure_retains_and_success_releases_own_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / 'source.sqlite'
            with sqlite3.connect(source) as db:
                db.execute('CREATE TABLE priority_probe_witnesses(sha TEXT PRIMARY KEY, raw_bytes INTEGER, payload BLOB)')
            raw = source.read_bytes()
            packed = root / 'source.sqlite.gz'
            packed.write_bytes(gzip.compress(raw))
            digest = lambda b: hashlib.sha256(b).hexdigest()
            kwargs = dict(stored_sha256=digest(packed.read_bytes()), decoded_sha256=digest(raw),
                          decoded_bytes=len(raw), scratch_parent=root, max_scratch_bytes=1024**2, min_free_bytes=0)
            with self.assertRaisesRegex(RuntimeError, 'oracle failed'):
                with reader.open_compressed(packed, **kwargs) as db:
                    self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
                    raise RuntimeError('oracle failed')
            retained = list(root.glob('rbf-one-database-*'))
            self.assertEqual(len(retained), 1)
            self.assertEqual((retained[0] / 'state.sqlite').read_bytes(), raw)
            with reader.open_compressed(packed, **kwargs) as db:
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(list(root.glob('rbf-one-database-*')), retained)
            self.assertEqual(gzip.decompress(packed.read_bytes()), raw)


if __name__ == '__main__':
    unittest.main()
