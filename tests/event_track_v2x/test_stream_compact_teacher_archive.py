import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import unittest
import zlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/event_track_v2x'))
from stream_compact_teacher_archive import collect
from compact_teacher_database import codec


def archive(entries):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as tar:
        for name,data,kind in entries:
            m=tarfile.TarInfo(name);m.type=kind;m.linkname='/tmp/escape'
            m.size=len(data) if kind==tarfile.REGTYPE else 0
            tar.addfile(m,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
    return out.getvalue()


class StreamStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        dbpath=self.root/'source.sqlite';db=sqlite3.connect(dbpath)
        db.execute('CREATE TABLE priority_probe_witnesses(sha TEXT PRIMARY KEY,raw_bytes INTEGER,payload BLOB)')
        raw=codec.canonical({'prefixes':[[1,2,3]],'factors':[[[-1,-.5]]],'sequence':'fixture'})
        db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',(codec.digest(raw),len(raw),zlib.compress(raw,1)))
        db.commit();db.close()
        self.entries=[('rank-0/s/audit.jsonl',b'{"audit":1}\n'*1000,tarfile.REGTYPE),
                      ('rank-0/s/state.sqlite',dbpath.read_bytes(),tarfile.REGTYPE),
                      ('rank-0/s/receipt.json',b'{"fixture":true}',tarfile.REGTYPE)]
        self.data=archive(self.entries)
    def tearDown(self):self.tmp.cleanup()
    def run_collect(self,**changes):
        args=dict(expected_bytes=len(self.data),expected_sha256=hashlib.sha256(self.data).hexdigest(),
                  max_live_bytes=16*1024**2,max_database_bytes=2*1024**2,max_compact_bytes=2*1024**2,
                  min_free_bytes=0)
        args.update(changes)
        return collect(io.BytesIO(self.data),self.root/'output',**args)
    def test_stream_preserves_members_and_releases_only_own_scratch(self):
        result=self.run_collect();root=self.root/'output'
        self.assertEqual(result['persistent_compressed_archive_bytes'],0)
        self.assertFalse(result['scientific_acceptance'])
        self.assertFalse((root/'.scratch').exists())
        self.assertFalse(list(root.rglob('*.tar.gz')))
        self.assertEqual(gzip.decompress((root/'rank-0/s/audit.jsonl.gz').read_bytes()),self.entries[0][1])
        self.assertTrue((self.root/'source.sqlite').exists())
        self.assertEqual(len(list(root.rglob('*.compact.sqlite.gz'))),1)
        self.assertEqual(len(result['files']),3)
        self.assertEqual(result['stored_bytes'], sum(
            f.stat().st_size for f in root.rglob('*')
            if f.is_file() and f.name != 'compact-storage.json'))
        from compact_teacher_reader import open_compressed
        spec=next(x for x in result['files'] if x['encoding']=='shared_witness_sqlite_v2+gzip')
        with open_compressed(root/spec['stored_name'],stored_sha256=spec['stored_sha256'],
            decoded_sha256=spec['decoded_sha256'],decoded_bytes=spec['decoded_bytes'],
            scratch_parent=self.root,max_scratch_bytes=2*1024**2,min_free_bytes=0) as db:
            raw=zlib.decompress(db.execute('SELECT payload FROM priority_probe_witnesses').fetchone()[0])
            self.assertEqual(json.loads(raw)['sequence'],'fixture')
        self.assertFalse(list(self.root.glob('rbf-one-database-*')))
    def test_already_compact_writer_output_is_not_recopied(self):
        db=sqlite3.connect(self.root/'native-v2.sqlite');codec.create_tables(db)
        codec.store(db,{'prefixes':[[1]],'sequence':'native-v2'});db.commit();db.close()
        self.data=archive([('rank-0/s/state.sqlite',(self.root/'native-v2.sqlite').read_bytes(),tarfile.REGTYPE)])
        result=self.run_collect();spec=result['files'][0]
        receipt=json.loads((self.root/'output'/spec['storage_receipt']).read_text())
        self.assertTrue(receipt['already_compact_no_database_copy'])
        self.assertEqual(receipt['source_sha256'],receipt['output_sha256'])
        self.assertEqual(spec['decoded_sha256'],spec['original_sha256'])

    def test_bad_archive_never_promotes(self):
        with self.assertRaises(ValueError):self.run_collect(expected_sha256='0'*64)
        self.assertFalse((self.root/'output').exists())
        self.assertTrue((self.root/'output.partial').exists())
    def test_storage_limit_fails_before_large_scratch(self):
        with self.assertRaises(ValueError):self.run_collect(max_database_bytes=1)
        self.assertFalse((self.root/'output.partial/.scratch/source.sqlite').exists())
    def test_duplicate_link_traversal_and_existing_refused(self):
        for item in [('x',b'',tarfile.SYMTYPE),('../x',b'',tarfile.REGTYPE)]:
            data=archive([item])
            with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
                collect(io.BytesIO(data),Path(tmp).resolve()/'out',expected_bytes=len(data),expected_sha256=hashlib.sha256(data).hexdigest(),max_live_bytes=1024**2,max_database_bytes=1024**2,max_compact_bytes=1024**2,min_free_bytes=0)
        self.data=archive([self.entries[0],self.entries[0]])
        with self.assertRaises(ValueError):self.run_collect()
        with self.assertRaises(ValueError):self.run_collect()

if __name__=='__main__':unittest.main()
