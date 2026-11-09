"""Transport rejection controls and unchanged frozen trajectory equivalence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/event_track_v2x'))
from compact_teacher_oracle_adapter import bind_compact_sqlite, file_sha
from compact_teacher_database import compact


class AdapterTests(unittest.TestCase):
    def test_bound_path_read_only_restoration_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); path = root/'compact.sqlite'
            raw = b'{"x":1}'; key = hashlib.sha256(raw).hexdigest()
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE priority_probe_witnesses(sha TEXT,raw_bytes INT,payload BLOB)')
            db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)', (key,len(raw),zlib.compress(raw)))
            db.commit(); db.close()
            module = types.SimpleNamespace(__file__=__file__, sqlite3=sqlite3)
            pins = [(module, file_sha(__file__))]; sha = file_sha(path)
            original_connect = sqlite3.connect
            with self.assertRaisesRegex(RuntimeError, 'oracle failure'):
                with bind_compact_sqlite(path, sha, pins) as proxy:
                    db = module.sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
                    self.assertEqual(zlib.decompress(db.execute('SELECT payload FROM priority_probe_witnesses').fetchone()[0]),raw)
                    with self.assertRaises(sqlite3.OperationalError):
                        db.execute('DELETE FROM priority_probe_witnesses')
                    for uri in (path.as_uri()+'?mode=rw', (root/'other.sqlite').as_uri()+'?mode=ro', path.as_uri()+'?mode=ro&mode=rw'):
                        with self.assertRaises(ValueError): proxy.connect(uri, uri=True)
                    with self.assertRaises(ValueError): proxy.connect(str(path))
                    self.assertIs(sqlite3.connect, original_connect)
                    raise RuntimeError('oracle failure')
            self.assertIs(module.sqlite3, sqlite3)
            with self.assertRaises(sqlite3.ProgrammingError): db.execute('SELECT 1')
            with self.assertRaisesRegex(ValueError, 'source SHA'):
                with bind_compact_sqlite(path, sha, [(module,'0'*64)]): pass
            with self.assertRaisesRegex(ValueError, 'physical compact SHA'):
                with bind_compact_sqlite(path, '0'*64, pins): pass

    def test_frozen_full_trajectory_oracle_legacy_vs_compacted(self):
        from transvision.models.event_track_v2x.exclusive_teacher_probe_witness import ExclusiveWitnessTeacher
        from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
        from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
        from transvision.models.event_track_v2x.identity_forest import IdentityNode
        frozen = Path('/Volumes/Data/test/recover-before-fuse')
        oracle_path = frozen/'source-freezes/rbf-independent-exclusive-teacher-capacity-model-bound-v1-20261002/trajectory.py'
        oracle_sha = '37ddf0a7437064d41f6df36a0879eb67b19a430e306c2fb1c8d6e48827c894a9'
        self.assertEqual(file_sha(oracle_path), oracle_sha)
        spec = importlib.util.spec_from_file_location('storage_control_frozen_trajectory', oracle_path)
        oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
        matrix_path = frozen/'artifacts/rbf-exclusive-frontier-CPU-interface-v1-20261002/interface-matrix.json'
        self.assertEqual(file_sha(matrix_path),'0662e77b541fcfd88fefa9d7df3266d69f555468e1c824251505dbdff5f1484a')
        matrix = json.loads(matrix_path.read_bytes())
        for regime,width,budget in [('late',1,4),('late',4,16),('conflicts',1,1),('conflicts',4,4),('extreme',1,16),('disconnected',4,4)]:
            base = next(c for c in matrix['cases'] if c['variant']=='bound' and c['seed']==1337 and c['regime']==regime and c['width']==width and c['budget']==budget)
            raw = tuple(RawIdentityDetection(**dict(v,node=IdentityNode(**v['node']))) for v in base['raw'])
            rows = tuple(tuple((p,float(w)) for p,w in r) for r in base['rows'])
            config = copy.deepcopy(base['configuration']); state = config.pop('state')
            cfg = PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(**state),**config)
            with self.subTest(regime=regime,width=width,budget=budget), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); source = root/'legacy.sqlite'; output = root/'compact.sqlite'
                teacher = ExclusiveWitnessTeacher(source,sequence_id='fixture',config=cfg)
                try:
                    for ordinal,(lo,hi,clock) in enumerate(((0,2,1100000),(2,4,1300000),(4,5,1500000),(5,5,1700000))):
                        teacher.step(raw[lo:hi],rows[lo:hi],frame_id=str(ordinal),event_id=str(ordinal),reference_us=clock,decision_us=clock)
                finally: teacher.close()
                original = oracle.verify_database(source,file_sha(source))
                report = compact(source,output,file_sha(source),max_output_bytes=64*1024**2,min_free_bytes=0)
                with bind_compact_sqlite(output, report['output_sha256'], [(oracle,oracle_sha)]) as adapter:
                    restored = oracle.verify_database(output,report['output_sha256'])
                    self.assertEqual(adapter.open_count,1)
                for result in (original,restored):
                    result.pop('database_sha256'); result.pop('elapsed_seconds')
                self.assertEqual(original,restored)
                self.assertEqual(restored['events'],4)
                self.assertEqual(restored['atol'],1e-8)
                self.assertEqual(restored['rtol'],1e-8)
                self.assertFalse(restored['full_real_teacher_target_admission'])
                self.assertGreater(restored['labels'],0)
                # Even a caller-supplied new physical SHA cannot make altered
                # snapshot bytes pass the unchanged frozen numeric reader.
                broken = sqlite3.connect(output)
                broken.execute("UPDATE priority_probe_witnesses SET payload=x'00'")
                broken.commit(); broken.close()
                with bind_compact_sqlite(output,file_sha(output),[(oracle,oracle_sha)]):
                    with self.assertRaises(sqlite3.OperationalError):
                        oracle.verify_database(output,file_sha(output))
                self.assertIs(oracle.sqlite3,sqlite3)


if __name__ == '__main__': unittest.main()
