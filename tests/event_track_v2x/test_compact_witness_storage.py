"""Storage controls: exact bytes, corruption, rollback and real solver parity."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('compact_database_controls',ROOT/'tools/event_track_v2x/compact_teacher_database.py')
packing=importlib.util.module_from_spec(spec); spec.loader.exec_module(packing)
codec=packing.codec


def snapshot():
    return {'kind':'exclusive_raw_kernel_search_snapshot_v1','sequence_id':'fixture',
            'prefixes':[[i,i-1,8,'choice',i//7,None,'[]',hashlib.sha256(str(i).encode()).hexdigest()] for i in range(512)],
            'slots':[[0,str(i)] for i in range(128)],
            'factors':[[[-1,-.123],[0,-1.2345678901234567]] for _ in range(128)],
            'active':[510,511],'frontier':[200,300], 'revision':0, 'nodes':128,'prefix_count':512}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:'); codec.create_tables(self.db); self.db.commit()
    def tearDown(self): self.db.close()

    def test_roundtrip_dedup_and_shared_content(self):
        value=snapshot(); old_bytes=0
        for index in range(20):
            value['frontier']=[200,index]
            raw=codec.canonical(value);old_bytes+=len(zlib.compress(raw,1))
            key=codec.store(self.db,value)
            self.assertEqual(codec.read(self.db,key),raw)
            self.assertEqual(codec.store(self.db,value),key)
        self.assertEqual(self.db.execute('SELECT count(*) FROM priority_probe_witnesses').fetchone()[0],20)
        stored=sum(self.db.execute('SELECT sum(length(payload)) FROM '+t).fetchone()[0]
                   for t in ('priority_probe_witnesses','priority_witness_chunks'))
        self.assertLess(stored,old_bytes*.15)

    def test_legacy_reader(self):
        raw=codec.canonical(snapshot());key=codec.digest(raw)
        self.db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',(key,len(raw),zlib.compress(raw,1)))
        self.assertEqual(codec.read(self.db,key),raw)

    def test_rollback_leaves_no_orphans(self):
        self.db.execute('BEGIN');codec.store(self.db,snapshot());self.db.rollback()
        for t in ('priority_probe_witnesses','priority_witness_chunks'):
            self.assertEqual(self.db.execute('SELECT count(*) FROM '+t).fetchone()[0],0)

    def test_corrupt_or_missing_chunk_is_refused(self):
        key=codec.store(self.db,snapshot())
        self.db.execute("UPDATE priority_witness_chunks SET payload=x'00' WHERE sha=(SELECT sha FROM priority_witness_chunks LIMIT 1)")
        with self.assertRaises((ValueError,zlib.error)):codec.read(self.db,key)
        self.db.execute('DELETE FROM priority_witness_chunks')
        with self.assertRaises(ValueError):codec.read(self.db,key)

    def test_wrong_snapshot_identity_and_trailing_bytes_refused(self):
        key=codec.store(self.db,snapshot())
        self.db.execute('UPDATE priority_probe_witnesses SET raw_bytes=raw_bytes+1')
        with self.assertRaises(ValueError):codec.read(self.db,key)
        with self.assertRaises(ValueError):codec.inflate(zlib.compress(b'raw')+b'trailing',3)
        with self.assertRaises(ValueError):codec.inflate(zlib.compress(b'raw'),2)

    def test_content_length_budget_covers_multiple_chunked_lists(self):
        value = {str(i): list(range(257 + i)) for i in range(25)}
        key = codec.store(self.db, value)
        self.assertEqual(codec.read(self.db, key), codec.canonical(value))
        reader_path = ROOT/'tools/event_track_v2x/compact_teacher_live_reader.py'
        spec = importlib.util.spec_from_file_location('independent_length_reader', reader_path)
        reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
        self.assertEqual(reader.decode(self.db, key), codec.canonical(value))
        self.db.execute('UPDATE priority_probe_witnesses SET raw_bytes=1 WHERE sha=?', (key,))
        with self.assertRaisesRegex(ValueError, 'chunk lengths'):
            codec.read(self.db, key)
        with self.assertRaisesRegex(ValueError, 'chunk lengths'):
            reader.decode(self.db, key)

    def test_migration_preserves_all_other_rows_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp).resolve()/'source.sqlite'; output=Path(tmp).resolve()/'compact.sqlite'
            db=sqlite3.connect(source)
            db.executescript('CREATE TABLE priority_probe_witnesses(sha TEXT PRIMARY KEY,raw_bytes INTEGER NOT NULL,payload BLOB NOT NULL);'
                'CREATE TABLE events(ordinal INTEGER PRIMARY KEY,audit BLOB,prediction BLOB);'
                'CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB) WITHOUT ROWID;'
                'CREATE VIEW event_view AS SELECT * FROM events;'
                "CREATE TRIGGER event_touch AFTER INSERT ON events BEGIN INSERT OR REPLACE INTO meta VALUES('touch',x'31'); END;")
            for i in range(20):
                value=snapshot();value['frontier']=[i];raw=codec.canonical(value)
                db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',(codec.digest(raw),len(raw),zlib.compress(raw,1)))
            db.execute('INSERT INTO events VALUES(?,?,?)',(7,b'audit\x00',b'prediction'))
            db.execute('INSERT INTO meta VALUES(?,?)',('schema',b'original'))
            db.commit();db.close()
            sha=packing.sha_file(source)
            report=packing.compact(source,output,sha,max_output_bytes=4*1024**2,min_free_bytes=0)
            self.assertTrue(report['all_witness_raw_bytes_equal'])
            self.assertFalse(report['scientific_acceptance'])
            self.assertEqual(packing.sha_file(source),sha)
            import sys
            sys.path.insert(0,str(ROOT/'tools/event_track_v2x'))
            from compact_teacher_reader import connect
            readback=connect(output,report['output_sha256'])
            original=sqlite3.connect(source)
            try:
                for key,size,payload in original.execute('SELECT sha,raw_bytes,payload FROM priority_probe_witnesses'):
                    recovered=readback.execute('SELECT raw_bytes,payload FROM priority_probe_witnesses WHERE sha=?',(key,)).fetchone()
                    self.assertEqual(recovered[0],size)
                    self.assertEqual(zlib.decompress(recovered[1]),zlib.decompress(payload))
                with self.assertRaises(sqlite3.OperationalError):readback.execute('DELETE FROM events')
            finally:original.close();readback.close()
            self.assertLess(report['output_bytes'],report['source_bytes']*.25)
            with self.assertRaises(ValueError):packing.compact(source,output,sha,max_output_bytes=4*1024**2,min_free_bytes=0)
            with self.assertRaises(ValueError):packing.compact(source,Path(tmp)/'bad.sqlite','0'*64,max_output_bytes=4*1024**2,min_free_bytes=0)


class SolverTests(unittest.TestCase):
    def test_new_runtime_default_explicit_legacy_and_reject_unknown(self):
        from transvision.models.event_track_v2x.exclusive_witness_paper_runtime import default_configuration,make_backend
        cfg=default_configuration()
        self.assertEqual(cfg['teacher_witness_storage'],codec.FORMAT)
        with tempfile.TemporaryDirectory() as tmp:
            t=make_backend(Path(tmp).resolve()/'new.sqlite','fixture',cfg)
            self.assertTrue(t.SCHEMA.endswith('_v2'));t.close()
            cfg.pop('teacher_witness_storage')
            t=make_backend(Path(tmp).resolve()/'old.sqlite','fixture',cfg)
            self.assertTrue(t.SCHEMA.endswith('_v1'));t.close()
            cfg['teacher_witness_storage']='unknown'
            with self.assertRaises(ValueError):make_backend(Path(tmp)/'bad.sqlite','fixture',cfg)

    def test_frozen_six_case_solver_predictions_labels_budgets_and_reopen(self):
        from transvision.models.event_track_v2x.exclusive_teacher_probe_witness import ExclusiveWitnessTeacher
        from transvision.models.event_track_v2x.compact_teacher_probe_witness import CompactWitnessTeacher
        from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
        from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection,PaperForestTrackingConfig
        from transvision.models.event_track_v2x.identity_forest import IdentityNode
        path=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-exclusive-frontier-CPU-interface-v1-20261002/interface-matrix.json')
        if not path.exists(): self.skipTest('frozen qualification matrix unavailable')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'0662e77b541fcfd88fefa9d7df3266d69f555468e1c824251505dbdff5f1484a')
        matrix=json.loads(path.read_bytes())
        for regime,width,budget in [('late',1,4),('late',4,16),('conflicts',1,1),('conflicts',4,4),('extreme',1,16),('disconnected',4,4)]:
            base=next(c for c in matrix['cases'] if c['variant']=='bound' and c['seed']==1337 and c['regime']==regime and c['width']==width and c['budget']==budget)
            raw=tuple(RawIdentityDetection(**dict(v,node=IdentityNode(**v['node']))) for v in base['raw'])
            rows=tuple(tuple((p,float(w)) for p,w in r) for r in base['rows'])
            config=copy.deepcopy(base['configuration']);state=config.pop('state')
            cfg=PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(**state),**config)
            with self.subTest(regime=regime,width=width,budget=budget),tempfile.TemporaryDirectory() as tmp:
                a=ExclusiveWitnessTeacher(Path(tmp).resolve()/'legacy.sqlite',sequence_id='fixture',config=cfg)
                b=CompactWitnessTeacher(Path(tmp).resolve()/'compact.sqlite',sequence_id='fixture',config=cfg)
                try:
                    for ordinal,(lo,hi,clock) in enumerate(((0,2,1100000),(2,4,1300000),(4,5,1500000),(5,5,1700000))):
                        kw=dict(frame_id=str(ordinal),event_id=str(ordinal),reference_us=clock,decision_us=clock)
                        x=a.step(raw[lo:hi],rows[lo:hi],**kw);y=b.step(raw[lo:hi],rows[lo:hi],**kw)
                        self.assertEqual(x.prediction_json,y.prediction_json)
                        for key in ('allocation_trace','search_steps','factor_rows_sha256','raw_probe_witness_uncompressed_bytes'):
                            self.assertEqual(x.audit[key],y.audit[key])
                    for key, in a.db.execute('SELECT sha FROM priority_probe_witnesses'):
                        self.assertEqual(codec.read(a.db,key),codec.read(b.db,key))
                    head=y.prediction['commit_sha256'];sha=b.close();b=None
                    b=CompactWitnessTeacher.open(Path(tmp).resolve()/'compact.sqlite',expected_prediction_sha256=head,expected_database_sha256=sha)
                    kw=dict(frame_id='4',event_id='4',reference_us=1900000,decision_us=1900000)
                    self.assertEqual(a.step((),(),**kw).prediction_json,b.step((),(),**kw).prediction_json)
                finally:
                    a.close()
                    if b is not None:b.close()


if __name__=='__main__':unittest.main()
