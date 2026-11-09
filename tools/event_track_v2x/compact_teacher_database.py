#!/usr/bin/env python3
"""Copy a sealed teacher into shared-witness storage with full logical readback.

The source is never modified/deleted. The output has a NEW physical SHA and a
storage-binding receipt; it does not pass itself off as the original database
or as numerical teacher acceptance. Run original target oracles separately via
read_witness after qualifying the new storage reader.
"""
import argparse
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
CODEC_PATH = ROOT / 'transvision/models/event_track_v2x/compact_witness_storage.py'
spec = importlib.util.spec_from_file_location('compact_witness_codec', CODEC_PATH)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def quoted(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def row_digest(db, table):
    """Canonical typed rows in rowid or declared PK order; no producer imports."""
    sql = db.execute('SELECT sql FROM sqlite_master WHERE type=\'table\' AND name=?', (table,)).fetchone()[0]
    if 'WITHOUT ROWID' in sql.upper():
        columns = db.execute('PRAGMA table_info(' + quoted(table) + ')').fetchall()
        keys = [x[1] for x in sorted(columns, key=lambda x: x[5]) if x[5]]
        order = ','.join(map(quoted, keys))
    else:
        order = 'rowid'
    h = hashlib.sha256(); count = 0
    for row in db.execute('SELECT * FROM ' + quoted(table) + ' ORDER BY ' + order):
        count += 1
        h.update(len(row).to_bytes(4, 'big'))
        for v in row:
            if v is None: kind, data = b'n', b''
            elif isinstance(v, bytes): kind, data = b'b', v
            elif isinstance(v, str): kind, data = b's', v.encode()
            elif isinstance(v, int): kind, data = b'i', str(v).encode()
            elif isinstance(v, float): kind, data = b'f', struct.pack('>d', v)
            else: raise TypeError(type(v))
            h.update(kind + len(data).to_bytes(8, 'big')); h.update(data)
    return {'rows': count, 'sha256': h.hexdigest()}


def compact(source, destination, expected_sha256, *, max_output_bytes, min_free_bytes=5*1024**3,
            progress=lambda report: None):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    receipt_path = destination.with_suffix(destination.suffix + '.storage.json')
    if (not source.is_file() or any(p.is_symlink() for p in (source, *source.parents, destination, *destination.parents))
            or destination.exists() or receipt_path.exists()):
        raise ValueError('sealed ordinary source and new destination required')
    if any(Path(str(source)+suffix).exists() for suffix in ('-journal', '-wal', '-shm')):
        raise ValueError('source must be closed and checkpointed')
    if sha_file(source) != expected_sha256:
        raise ValueError('source SHA mismatch')
    if type(max_output_bytes) is not int or max_output_bytes < 4096 or min_free_bytes < 0:
        raise ValueError('explicit positive capacity contract required')
    if shutil.disk_usage(destination.parent).free < max_output_bytes + min_free_bytes:
        raise ValueError('insufficient free space for bounded output and reserve')
    started = time.monotonic()
    src = sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1', uri=True)
    dst = None
    try:
        if src.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('source SQLite integrity failure')
        schemas = src.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
        if any(kind not in ('table', 'index', 'view', 'trigger') for kind, _, _ in schemas):
            raise ValueError('unsupported source schema object; preserve source')
        if any(name in ('priority_witness_chunks','teacher_storage_binding') for _, name, _ in schemas):
            raise ValueError('already compact or reserved table present')
        if not any(name == 'priority_probe_witnesses' for _, name, _ in schemas):
            raise ValueError('not a teacher witness database')
        destination.touch(exist_ok=False)
        dst = sqlite3.connect(destination)
        dst.execute('PRAGMA page_size=4096')
        dst.execute('PRAGMA max_page_count='+str(max_output_bytes//4096))
        dst.execute('PRAGMA cache_size=-8192')
        for kind, _, sql in schemas:
            if kind == 'table': dst.execute(sql)
        codec.create_tables(dst)
        report = {'format': codec.FORMAT, 'source_sha256': expected_sha256,
                  'source_bytes': source.stat().st_size, 'tables': {}, 'witnesses': 0,
                  'scientific_acceptance': False, 'original_database_byte_identity': False,
                  'codec_sha256': sha_file(CODEC_PATH), 'compactor_sha256': sha_file(__file__)}
        for kind, table, sql in schemas:
            if kind != 'table': continue
            if table == 'priority_probe_witnesses':
                for sha, size, payload in src.execute('SELECT sha,raw_bytes,payload FROM priority_probe_witnesses ORDER BY rowid'):
                    raw = codec.inflate(payload, size)
                    if codec.digest(raw) != sha or codec.canonical(json.loads(raw)) != raw:
                        raise ValueError('source witness identity/canonical bytes differ')
                    if codec.store(dst, json.loads(raw)) != sha or codec.read(dst, sha) != raw:
                        raise ValueError('witness byte readback differs')
                    report['witnesses'] += 1
                    if report['witnesses'] % 100 == 0:
                        dst.commit()
                        if shutil.disk_usage(destination.parent).free < min_free_bytes:
                            raise ValueError('free-space reserve exhausted')
                        progress({'stage':'compact_witnesses','rows':report['witnesses'],
                                  'output_bytes':destination.stat().st_size})
            else:
                info = src.execute('PRAGMA table_info('+quoted(table)+')').fetchall()
                without_rowid = 'WITHOUT ROWID' in sql.upper()
                # Explicit rowid retains row identity and deterministic row order.
                columns = ','.join(quoted(c[1]) for c in info)
                if not without_rowid: columns = 'rowid,' + columns
                query = 'SELECT '+columns+' FROM '+quoted(table)
                for i, row in enumerate(src.execute(query)):
                    dst.execute('INSERT INTO '+quoted(table)+' ('+columns+') VALUES ('+','.join('?' for _ in row)+')', row)
                    if i % 100 == 0: dst.commit()
                dst.commit()
                left, right = row_digest(src, table), row_digest(dst, table)
                if left != right: raise ValueError('table readback differs: '+table)
                report['tables'][table] = left
                progress({'stage':'table_verified','table':table,'rows':left['rows']})
        for kind, _, sql in schemas:
            if kind in ('index', 'view', 'trigger'): dst.execute(sql)
        dst.commit()
        copied_schema = dst.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' AND name!='priority_witness_chunks' ORDER BY type,name").fetchall()
        if copied_schema != schemas:
            raise ValueError('source schema not preserved')
        if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('compact SQLite integrity failure')
        # Independent second pass after all writes: verify every snapshot, even
        # chunks first written by a different snapshot or transaction.
        n = 0
        for sha, size, payload in src.execute('SELECT sha,raw_bytes,payload FROM priority_probe_witnesses'):
            if codec.read(dst, sha) != codec.inflate(payload, size):
                raise ValueError('final witness readback differs')
            n += 1
            if n % 500 == 0:
                progress({'stage':'final_witness_readback','rows':n,'total':report['witnesses']})
        if n != report['witnesses'] or n != dst.execute('SELECT count(*) FROM priority_probe_witnesses').fetchone()[0]:
            raise ValueError('witness coverage differs')
        report['all_witness_raw_bytes_equal'] = True
        report['source_witness_payload_bytes'] = src.execute('SELECT sum(length(payload)) FROM priority_probe_witnesses').fetchone()[0] or 0
        report['compact_witness_payload_bytes'] = sum(dst.execute('SELECT coalesce(sum(length(payload)),0) FROM '+table).fetchone()[0]
            for table in ('priority_probe_witnesses','priority_witness_chunks'))
        dst.close(); dst = None
        if sha_file(source) != expected_sha256:
            raise ValueError('source changed during compaction')
        report.update(output_sha256=sha_file(destination), output_bytes=destination.stat().st_size,
                      elapsed_seconds=time.monotonic()-started)
        with receipt_path.open('x') as out: json.dump(report,out,indent=2)
        return report
    finally:
        if dst is not None: dst.close()
        src.close()


def verify_compact(source, expected_sha256):
    """Admit storage bytes already emitted by the v2 writer without recopying."""
    source=Path(source).absolute()
    if any(p.is_symlink() for p in (source,*source.parents)) or sha_file(source)!=expected_sha256:
        raise ValueError('compact source identity differs')
    if any(Path(str(source)+s).exists() for s in ('-journal','-wal','-shm')):
        raise ValueError('unsealed compact source')
    db=sqlite3.connect(source.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('compact integrity failure')
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='priority_witness_chunks'").fetchone():
            return None
        count=0
        for key, in db.execute('SELECT sha FROM priority_probe_witnesses'):
            codec.read(db,key);count+=1
        return {'format':codec.FORMAT,'source_sha256':expected_sha256,'output_sha256':expected_sha256,
                'source_bytes':source.stat().st_size,'output_bytes':source.stat().st_size,
                'witnesses':count,'all_witness_raw_bytes_equal':True,'original_database_byte_identity':True,
                'scientific_acceptance':False,'already_compact_no_database_copy':True}
    finally:db.close()


def seal_compressed(source, destination, expected_sha256, *, max_output_bytes, min_free_bytes=5*1024**3):
    """Compress a verified compact DB; decode/hash all bytes before returning.

    Never deletes either file. Callers may release ONLY their own verified
    scratch copy after this succeeds. A read failure preserves partial output.
    """
    source,destination=Path(source).absolute(),Path(destination).absolute()
    if any(p.is_symlink() for p in (source,*source.parents,destination,*destination.parents)) or destination.exists():
        raise ValueError('ordinary source and exclusive compressed destination required')
    if max_output_bytes<=0 or min_free_bytes<0:raise ValueError('positive storage contract required')
    if sha_file(source)!=expected_sha256:raise ValueError('compact source SHA differs')
    before=source.stat().st_size
    with source.open('rb') as incoming,destination.open('xb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',compresslevel=6,mtime=0) as encoded:
            for block in iter(lambda:incoming.read(1024**2),b''):
                if shutil.disk_usage(destination.parent).free<min_free_bytes+2*1024**2:
                    raise ValueError('compressed storage free-space reserve exhausted')
                encoded.write(block);encoded.flush()
                if out.tell()>max_output_bytes:raise ValueError('compressed storage byte cap exceeded')
    if destination.stat().st_size>max_output_bytes:raise ValueError('compressed storage byte cap exceeded')
    h=hashlib.sha256();n=0
    with gzip.open(destination,'rb') as decoded:
        for block in iter(lambda:decoded.read(1024**2),b''):
            n+=len(block)
            if n>before:raise ValueError('expanded compact database too large')
            h.update(block)
    if n!=before or h.hexdigest()!=expected_sha256 or sha_file(source)!=expected_sha256:
        raise ValueError('compressed database full-byte readback differs')
    return dict(encoding='gzip',decoded_bytes=n,decoded_sha256=h.hexdigest(),
                stored_bytes=destination.stat().st_size,stored_sha256=sha_file(destination))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--max-output-bytes',type=int,required=True)
    p.add_argument('--min-free-bytes',type=int,default=5*1024**3)
    a=p.parse_args()
    print(json.dumps(compact(a.source,a.destination,a.source_sha256,max_output_bytes=a.max_output_bytes,
        min_free_bytes=a.min_free_bytes,progress=lambda x:print(json.dumps(x),flush=True))),flush=True)
