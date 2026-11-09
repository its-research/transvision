import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/event_track_v2x'))
from stream_compact_teacher_archive import collect
from verify_compact_teacher_collection import verify


class CollectionReadbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        source = self.root/'source.sqlite'
        db = sqlite3.connect(source)
        db.executescript('CREATE TABLE priority_probe_witnesses(sha TEXT PRIMARY KEY,raw_bytes INTEGER,payload BLOB);'
                         'CREATE TABLE events(ordinal INTEGER PRIMARY KEY,audit BLOB);')
        for i in range(3):
            raw = json.dumps({'factors':[[j,j+1] for j in range(70)],'event':i},sort_keys=True,separators=(',',':')).encode()
            db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',
                       (hashlib.sha256(raw).hexdigest(),len(raw),zlib.compress(raw,1)))
        db.execute('INSERT INTO events VALUES(?,?)',(0,b'original\x00audit'));db.commit();db.close()
        files = {'state.sqlite':source.read_bytes(),'plan.json':b'{"fixture":true}'}
        sha = lambda b:hashlib.sha256(b).hexdigest()
        receipt = {'kind':'rbf_paper_replay_receipt_v1','status':'software_replay_completed',
                   'files':{'plan.json':sha(files['plan.json'])},
                   'databases':{'fixture':{'path':'state.sqlite','sha256':sha(files['state.sqlite'])}}}
        files['receipt.json'] = json.dumps(receipt).encode()
        wire = io.BytesIO()
        with tarfile.open(fileobj=wire,mode='w:gz') as tar:
            for name, data in files.items():
                member = tarfile.TarInfo('rank-6/fixture/'+name);member.size=len(data)
                tar.addfile(member,io.BytesIO(data))
        self.wire=wire.getvalue()
        self.inventory={'archive_sha256':sha(self.wire),'archive_bytes':len(self.wire),
                        'files':[{'name':'rank-6/fixture/'+k,'bytes':len(v)} for k,v in files.items()]}
        self.collected=self.root/'collected'
        self.proof=collect(io.BytesIO(self.wire),self.collected,expected_bytes=len(self.wire),
                           expected_sha256=sha(self.wire),max_live_bytes=16*1024**2,
                           max_database_bytes=2*1024**2,max_compact_bytes=2*1024**2,min_free_bytes=0)
        self.manifest_sha=sha((self.collected/'compact-storage.json').read_bytes())

    def tearDown(self): self.tmp.cleanup()

    def run_verify(self, inventory=None):
        return verify(self.collected,manifest_sha256=self.manifest_sha,
                      inventory=inventory or self.inventory,scratch_parent=self.root,
                      max_scratch_bytes=2*1024**2,min_free_bytes=0)

    def test_full_readback_preserves_storage_scope_and_releases_scratch(self):
        result=self.run_verify()
        self.assertEqual(result['witnesses'],3)
        self.assertEqual(result['members'],3)
        self.assertEqual(len(result['databases']),1)
        self.assertFalse(result['scientific_acceptance'])
        self.assertFalse(result['full_teacher_target_acceptance'])
        self.assertFalse(list(self.root.glob('rbf-one-database-*')))
        self.assertTrue((self.root/'source.sqlite').is_file())

    def test_archive_binding_and_missing_member_rejected(self):
        bad=copy.deepcopy(self.inventory);bad['archive_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'archive binding'): self.run_verify(bad)
        bad=copy.deepcopy(self.inventory);bad['files'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'): self.run_verify(bad)

    def test_table_fidelity_receipt_cannot_be_skipped(self):
        entry=next(x for x in self.proof['files'] if 'storage_receipt' in x)
        path=self.collected/entry['storage_receipt'];report=json.loads(path.read_bytes())
        report['tables']['events']['sha256']='0'*64;path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'table digest'): self.run_verify()

    def test_corrupt_stored_database_rejected(self):
        entry=next(x for x in self.proof['files'] if 'storage_receipt' in x)
        path=self.collected/entry['stored_name'];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'compressed artifact SHA'): self.run_verify()

    def test_native_receipt_original_identity_required(self):
        entry=next(x for x in self.proof['files'] if x['original_name'].endswith('/receipt.json'))
        path=self.collected/entry['stored_name'];receipt=json.loads(path.read_bytes())
        receipt['databases']['fixture']['sha256']='0'*64
        raw=json.dumps(receipt).encode();path.write_bytes(raw)
        entry['original_sha256']=hashlib.sha256(raw).hexdigest()
        # Same-length corruption updates the untrusted storage manifest, but
        # cannot forge the database identity bound inside the native receipt.
        entry['original_bytes']=entry['stored_bytes']=len(raw)
        manifest=self.collected/'compact-storage.json';manifest.write_text(json.dumps(self.proof))
        self.manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'database receipt binding'): self.run_verify()


if __name__=='__main__':unittest.main()
