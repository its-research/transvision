"""Run existing frozen-oracle controls plus shared-cache lifecycle checks."""
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/event_track_v2x'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_compact_teacher_oracle_adapter as original_tests
from compact_teacher_cached_oracle_adapter_candidate import bind_compact_sqlite


class FrozenOracleControls(original_tests.AdapterTests):
    def setUp(self):
        self.override=patch.object(original_tests,'bind_compact_sqlite',bind_compact_sqlite)
        self.override.start();self.addCleanup(self.override.stop)


class SharedCacheControls(unittest.TestCase):
    def test_all_oracle_connections_share_one_bound_and_close_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();path=root/'compact.sqlite'
            values=[b'{"value":'+str(i).encode()+b'}' for i in range(3)]
            keys=[hashlib.sha256(raw).hexdigest() for raw in values]
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE priority_probe_witnesses(sha TEXT PRIMARY KEY,raw_bytes INT,payload BLOB)')
                db.executemany('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',[(key,len(raw),zlib.compress(raw)) for key,raw in zip(keys,values)])
            sha=original_tests.file_sha(path)
            modules=[types.SimpleNamespace(__file__=__file__,sqlite3=sqlite3) for _ in range(6)]
            pins=[(m,original_tests.file_sha(__file__)) for m in modules]
            with bind_compact_sqlite(path,sha,pins,max_cache_bytes=512,max_cache_entries=2) as proxy:
                for m in modules:
                    db=m.sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
                    self.assertEqual(zlib.decompress(db.execute('SELECT payload FROM priority_probe_witnesses WHERE sha=?',(keys[0],)).fetchone()[0]),values[0])
                    db.close()
                info=proxy.cache_info();self.assertEqual(info['misses'],1);self.assertEqual(info['hits'],5)
                db=modules[0].sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
                for key in keys[1:]:db.execute('SELECT payload FROM priority_probe_witnesses WHERE sha=?',(key,)).fetchone()
                self.assertEqual(proxy.cache_info()['entries'],2)
                self.assertLessEqual(proxy.cache_info()['peak_payload_bytes'],512)
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute('SELECT rbf_legacy_witness_payload(?)',('0'*64,)).fetchone()
            self.assertEqual(proxy.cache_info()['payload_bytes'],0)
            self.assertTrue(all(m.sqlite3 is sqlite3 for m in modules))
            self.assertEqual(original_tests.file_sha(path),sha)


if __name__=='__main__':unittest.main()
