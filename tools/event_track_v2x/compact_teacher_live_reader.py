"""Read-only SQLite adapter for live compact teacher evidence, without a fixed witness size cap.

Independent storage decoder: no producer/codec imports. Existing numeric oracle
queries can consume a TEMP view of legacy zlib witnesses without materializing
another full database. Physical compact SHA and source storage proof remain
separate; this adapter never changes numerical thresholds or provenance gates.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

MAGIC=b'RBFW2\x00'


def decode(db,key):
    record=db.execute('SELECT raw_bytes,payload FROM main.priority_probe_witnesses WHERE sha=?',(key,)).fetchone()
    if record is None:raise ValueError('missing witness')
    expected,data=record
    if type(expected) is not int or expected<0:raise ValueError('invalid witness length')
    def expand(data,n):
        if type(n) is not int or n<0:raise ValueError('invalid compressed record length')
        decoder=zlib.decompressobj();raw=decoder.decompress(data,n+1)
        if len(raw)!=n or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('compressed record incomplete or extended')
        return raw
    if data.startswith(MAGIC):
        if len(data)<10:raise ValueError('truncated header')
        manifest=json.loads(expand(data[10:],int.from_bytes(data[6:10],'big')))
        if set(manifest)!= {'format','fields','lists'} or manifest['format']!='canonical_json_shared_list_chunks_v2':
            raise ValueError('unknown storage manifest')
        obj=dict(manifest['fields'])
        # Each extra list chunk contributes at most one byte of array syntax
        # compared with the reconstructed list; use content-derived lengths.
        budget=expected+sum(len(spec['chunks']) for spec in manifest['lists'].values())
        for name,spec in manifest['lists'].items():
            if name in obj or set(spec)!= {'rows','chunks'} or type(spec['rows']) is not int or spec['rows']<0:
                raise ValueError('malformed list manifest')
            values=[]
            for sha in spec['chunks']:
                part=db.execute('SELECT raw_bytes,payload FROM main.priority_witness_chunks WHERE sha=?',(sha,)).fetchone()
                if part is None:raise ValueError('missing shared chunk')
                budget-=part[0]
                if budget<0:raise ValueError('chunk lengths exceed recorded witness size')
                raw=expand(part[1],part[0])
                if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('chunk SHA differs')
                rows=json.loads(raw)
                if not isinstance(rows,list) or not 1<=len(rows)<=64:raise ValueError('chunk shape differs')
                values.extend(rows)
                if len(values)>spec['rows']:raise ValueError('extra list rows')
            if len(values)!=spec['rows']:raise ValueError('list truncated')
            obj[name]=values
        raw=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    else:raw=expand(data,expected)
    if len(raw)!=expected or hashlib.sha256(raw).hexdigest()!=key:raise ValueError('snapshot bytes/SHA differ')
    return raw


class _ReadOnly(sqlite3.Connection):
    def __init__(self,database,**kwargs):
        super().__init__(database,**kwargs)
        self._witness_source=None
        try:
            self._witness_source=sqlite3.connect(database,uri=True)
            self.create_function('rbf_legacy_witness_payload',1,
                lambda key:zlib.compress(decode(self._witness_source,key),1),deterministic=True)
            self.execute('CREATE TEMP VIEW priority_probe_witnesses AS '
                         'SELECT sha,raw_bytes,rbf_legacy_witness_payload(sha) AS payload '
                         'FROM main.priority_probe_witnesses')
        except BaseException:
            self.close();raise
    def close(self):
        if self._witness_source is not None:
            self._witness_source.close();self._witness_source=None
        super().close()


def connect(path,expected_sha256):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():raise ValueError('ordinary sealed database required')
    if any(Path(str(path)+s).exists() for s in ('-wal','-shm','-journal')):raise ValueError('unsealed database')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
    if h.hexdigest()!=expected_sha256:raise ValueError('physical compact SHA differs')
    return sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True,factory=_ReadOnly)


from contextlib import contextmanager
import gzip
import shutil
import tempfile


@contextmanager
def open_compressed(path, *, stored_sha256, decoded_sha256, decoded_bytes,
                    scratch_parent, max_scratch_bytes, min_free_bytes=5*1024**3):
    """Materialize exactly one bounded compact DB, then remove own scratch.

    No persistent decompressed cohort and no mutation of the sealed store.
    Verification failure leaves its named scratch directory for diagnosis.
    """
    path=Path(path).absolute();parent=Path(scratch_parent).absolute()
    if (not path.is_file() or not parent.is_dir() or
            any(p.is_symlink() for p in (path,*path.parents,parent,*parent.parents))):
        raise ValueError('ordinary compressed input and scratch directory required')
    if type(decoded_bytes) is not int or not 0<decoded_bytes<=max_scratch_bytes or min_free_bytes<0:
        raise ValueError('decoded database exceeds scratch contract')
    if shutil.disk_usage(parent).free<decoded_bytes+min_free_bytes:raise ValueError('insufficient scratch space')
    h=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(1024**2),b''):h.update(block)
    if h.hexdigest()!=stored_sha256:raise ValueError('compressed artifact SHA differs')
    root=Path(tempfile.mkdtemp(prefix='rbf-one-database-',dir=parent))
    target=root/'state.sqlite'
    h=hashlib.sha256();n=0
    with gzip.open(path,'rb') as src,target.open('xb') as dst:
        for block in iter(lambda:src.read(1024**2),b''):
            n+=len(block)
            if n>decoded_bytes:raise ValueError('expanded database exceeds registered size')
            if shutil.disk_usage(parent).free<len(block)+min_free_bytes:raise ValueError('scratch free-space reserve exhausted')
            h.update(block);dst.write(block)
    if n!=decoded_bytes or h.hexdigest()!=decoded_sha256:raise ValueError('decoded SQLite identity differs')
    db=connect(target,decoded_sha256)
    succeeded = False
    try:
        yield db
        succeeded = True
    finally:
        db.close()
        # Preserve failed consumer verification for diagnosis, even when the
        # compressed transport itself passed its identity checks.
        if succeeded:
            target.unlink();root.rmdir()
