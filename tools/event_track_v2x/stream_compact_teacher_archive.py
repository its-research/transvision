#!/usr/bin/env python3
"""Bounded, lossless logical teacher collection; no .tar.gz cache or cohort copy.

Stream registered compressed bytes once. Compact one SQLite at a time, retaining
all other members (large JSONL streams are gzip encoded). Promote only after the
FULL original archive SHA and every SQLite witness/table readback pass. This is
a storage receipt, not frozen v1 byte admission or teacher target acceptance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from compact_teacher_database import compact, seal_compressed, verify_compact


class CheckedReader:
    def __init__(self, source, size):
        self.source=source;self.expected=size;self.count=0;self.sha=hashlib.sha256()
    def read(self, size=-1):
        data=self.source.read(min(1024**2, size if size>=0 else 1024**2))
        self.count+=len(data);self.sha.update(data)
        if self.count>self.expected:raise ValueError('archive exceeds registered size')
        return data


def collect(source,destination,*,expected_bytes,expected_sha256,max_live_bytes,
            max_database_bytes,max_compact_bytes,min_free_bytes=5*1024**3,max_members=10000,
            progress=lambda r:None):
    destination=Path(destination).absolute()
    partial=destination.with_name(destination.name+'.partial')
    if any(p.is_symlink() for p in (destination,*destination.parents)) or destination.exists() or partial.exists():
        raise ValueError('new collection destination required')
    if min(expected_bytes,max_live_bytes,max_database_bytes,max_compact_bytes,max_members)<=0 or min_free_bytes<0:
        raise ValueError('positive explicit storage limits required')
    if len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('registered SHA256 required')
    partial.mkdir()
    scratch=partial/'.scratch';scratch.mkdir()
    seen=set();files=[];live=0
    def reserve(amount):
        if live+amount>max_live_bytes:raise ValueError('collection storage budget exhausted')
        if shutil.disk_usage(partial).free<amount+min_free_bytes:raise ValueError('free-space reserve exhausted')
    reader=CheckedReader(source,expected_bytes)
    with tarfile.open(fileobj=reader,mode='r|gz') as archive:
        for member in archive:
            path=PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name or
                    path.as_posix() in ('.','') or path.parts[0]=='.scratch' or path.as_posix() in seen):
                raise ValueError('unsafe or duplicate member')
            seen.add(path.as_posix())
            if len(seen)>max_members:raise ValueError('member limit exceeded')
            if not (member.isfile() or member.isdir()) or member.issparse() or member.size<0:
                raise ValueError('links/special/sparse members forbidden')
            if member.isdir():continue
            db=member.name.endswith('.sqlite')
            zipped=member.name.endswith('.jsonl')
            if db and member.size>max_database_bytes:raise ValueError('database exceeds scratch budget')
            target=partial.joinpath(*path.parts)
            if db:target=scratch/'source.sqlite'
            elif zipped:target=target.with_name(target.name+'.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            # Bound temporary SQLite + destination before creating either one.
            if db:reserve(member.size+2*max_compact_bytes)
            with archive.extractfile(member) as incoming, target.open('xb') as out:
                # GzipFile writes its ten-byte header during construction,
                # before the per-block tell() deltas below begin.
                if zipped:reserve(10)
                before=out.tell()
                stream=gzip.GzipFile(filename='',fileobj=out,mode='wb',compresslevel=6,mtime=0) if zipped else out
                live+=out.tell()-before
                try:
                    while True:
                        block=incoming.read(1024**2)
                        if not block:break
                        reserve(2*len(block)+1024)
                        h.update(block);count+=len(block)
                        before=out.tell();stream.write(block);stream.flush()
                        live+=out.tell()-before
                finally:
                    if zipped:
                        before=out.tell();stream.close();live+=out.tell()-before
            if count!=member.size:raise ValueError('truncated member')
            spec={'original_name':member.name,'original_bytes':count,'original_sha256':h.hexdigest()}
            if db:
                packed=partial.joinpath(*path.parts).with_suffix('.compact.sqlite')
                packed.parent.mkdir(parents=True,exist_ok=True)
                result=verify_compact(target,h.hexdigest())
                receipt=packed.with_suffix(packed.suffix+'.storage.json')
                if result is None:
                    result=compact(target,packed,h.hexdigest(),max_output_bytes=max_compact_bytes,
                                   min_free_bytes=min_free_bytes,progress=progress)
                    live+=packed.stat().st_size+receipt.stat().st_size
                    # Only our scratch file; all original data was read back.
                    target.unlink();live-=count
                else:
                    if count>max_compact_bytes:raise ValueError('existing compact database exceeds storage cap')
                    target.rename(packed)
                    receipt.write_text(json.dumps(result,indent=2));live+=receipt.stat().st_size
                compressed=packed.with_name(packed.name+'.gz')
                wire=seal_compressed(packed,compressed,result['output_sha256'],
                                     max_output_bytes=max_compact_bytes,min_free_bytes=min_free_bytes)
                live+=compressed.stat().st_size
                # Second temporary file, released only after full decompressed SHA.
                packed.unlink();live-=result['output_bytes']
                spec.update(stored_name=compressed.relative_to(partial).as_posix(),encoding='shared_witness_sqlite_v2+gzip',
                            storage_receipt=receipt.relative_to(partial).as_posix(),**{k:v for k,v in wire.items() if k!='encoding'})
            else:
                spec.update(stored_name=target.relative_to(partial).as_posix(),encoding='gzip' if zipped else 'identity',
                            stored_bytes=target.stat().st_size)
                if zipped:
                    decoded=hashlib.sha256();decoded_bytes=0
                    with gzip.open(target,'rb') as check:
                        for block in iter(lambda:check.read(1024**2),b''):decoded.update(block);decoded_bytes+=len(block)
                    if decoded.hexdigest()!=h.hexdigest() or decoded_bytes!=count:raise ValueError('JSONL readback mismatch')
            files.append(spec)
            progress({'stage':'member_storage_verified','member':member.name,'live_bytes':live})
    while reader.read(1024**2):pass
    if reader.count!=expected_bytes or reader.sha.hexdigest()!=expected_sha256:
        raise ValueError('full archive size/SHA mismatch; quarantine preserved')
    proof={'kind':'registered_teacher_compact_storage_v2','archive_sha256':expected_sha256,
           'archive_bytes':expected_bytes,'files':files,'persistent_compressed_archive_bytes':0,
           'max_live_bytes':max_live_bytes,'stored_bytes':live,'scientific_acceptance':False,
           'original_SQLite_byte_identity':False,'all_original_witness_raw_bytes_preserved':True}
    encoded_proof=json.dumps(proof,indent=2).encode()
    reserve(len(encoded_proof))
    (partial/'compact-storage.json').write_bytes(encoded_proof)
    scratch.rmdir();partial.rename(destination)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--destination',type=Path,required=True)
    for name in ('max-live-bytes','max-database-bytes','max-compact-bytes'):p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    from clearml import Task
    from clearml.backend_api.session import Session
    import urllib.request
    from urllib.parse import urlsplit
    t=Task.get_task(task_id=a.task)
    if str(t.status)!='completed':raise ValueError('completed producer required')
    artifact=t.artifacts['replay-rank'+str(a.rank)];url=urlsplit(artifact.url)
    if url.scheme!='http' or url.netloc not in ('10.100.34.118:8081','10.100.35.118:8081') or url.username or url.password:
        raise ValueError('unapproved registered artifact origin')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kw):return None
    req=urllib.request.Request(artifact.url,headers={'Authorization':'Bearer '+Session().token,'Accept-Encoding':'identity'})
    with urllib.request.build_opener(NoRedirect).open(req,timeout=120) as response:
        if response.status!=200 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('unexpected HTTP representation')
        proof=collect(response,a.destination,expected_bytes=artifact.size,expected_sha256=artifact.hash,
            max_live_bytes=a.max_live_bytes,max_database_bytes=a.max_database_bytes,max_compact_bytes=a.max_compact_bytes,
            progress=lambda r:print(json.dumps(r),flush=True))
    print(json.dumps({'stage':'compact_archive_complete','stored_bytes':proof['stored_bytes'],'scientific_acceptance':False}))
