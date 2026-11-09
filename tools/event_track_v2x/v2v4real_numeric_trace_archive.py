"""Create-once transport of complete numeric traces, with strict byte inventory.

Packaging and byte readback are distinct from independent numerical checking.
No cloud task, prediction, detector, GT or training is invoked here.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import digest, member, rows


def pack(root, output, *, expected_rows, raw_index_sha, max_archive_bytes):
    root=Path(root).resolve();output=Path(output).absolute()
    if expected_rows!=3986 or max_archive_bytes<=0:
        raise ValueError('full official source cohort and explicit archive budget required')
    if output.exists() or output.is_relative_to(root) or any(p.is_symlink() for p in output.parents):
        raise ValueError('fresh external packaging directory required')
    index=rows(root/'traces.jsonl')
    if len(index)!=expected_rows:
        raise ValueError('complete trace cohort required')
    names=['capacity-preflight.json','traces.jsonl']
    for row in index.values():
        if row.get('raw_rows_sha256')!=raw_index_sha or row.get('TF32_enabled') is not False:
            raise ValueError('trace raw binding or TF32 contract differs')
        if row.get('trace_layout')!='compact-cells-v1':
            raise ValueError('qualified compact trace layout required')
        member(root,row['payload'],row['payload_sha256']);names.append(row['payload'])
    if len(set(names))!=len(names):raise ValueError('duplicate trace payload')
    actual=[]
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('symlink in trace candidate')
        if p.is_file():actual.append(p.relative_to(root).as_posix())
    if set(actual)!=set(names):raise ValueError('trace file inventory differs')
    capacity=json.loads((root/'capacity-preflight.json').read_bytes())
    if capacity.get('within_budget') is not True or capacity.get('rows')!=expected_rows:
        raise ValueError('passed full-cohort capacity proof required')
    inventory=[dict(path=name,bytes=(root/name).stat().st_size,sha256=digest(root/name)) for name in sorted(names)]
    # TAR headers, per-member 512-byte padding and final 10240-byte record.
    upper=sum(512+((r['bytes']+511)//512)*512 for r in inventory)+10240
    if upper>max_archive_bytes:raise ValueError('trace archive budget exceeded before packing')
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'numeric-traces.tar'
    with tarfile.open(archive,'x',format=tarfile.USTAR_FORMAT) as stream:
        for record in inventory:
            p=root/record['path']
            info=tarfile.TarInfo(record['path']);info.size=record['bytes'];info.mode=0o600
            with p.open('rb') as f:stream.addfile(info,f)
            if digest(p)!=record['sha256']:raise ValueError('trace changed while packing')
    if archive.stat().st_size>max_archive_bytes:raise ValueError('actual archive exceeds budget')
    manifest=dict(kind='v2v4real_compact_numeric_trace_transport_v1',rows=expected_rows,
        raw_index_sha256=raw_index_sha,trace_index_sha256=digest(root/'traces.jsonl'),
        archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),inventory=inventory,
        independent_numeric_acceptance=False,network_forward_verified=False,
        nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)
    with (output/'trace-manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,allow_nan=False)
    return manifest


def readback(archive, manifest, output, *, manifest_sha, max_total_bytes):
    manifest=Path(manifest)
    if digest(manifest)!=manifest_sha:raise ValueError('registered trace manifest SHA differs')
    envelope=json.loads(manifest.read_bytes());archive=Path(archive)
    if (envelope.get('kind')!='v2v4real_compact_numeric_trace_transport_v1'
            or envelope.get('rows')!=3986 or envelope.get('independent_numeric_acceptance') is not False
            or archive.stat().st_size!=envelope['archive_bytes']
            or digest(archive)!=envelope['archive_sha256']):
        raise ValueError('full trace archive identity differs')
    inventory=envelope['inventory'];names=[r['path'] for r in inventory]
    if len(names)!=len(set(names)) or sum(r['bytes'] for r in inventory)>max_total_bytes:
        raise ValueError('duplicate members or extraction budget exceeded')
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in output.parents):raise ValueError('fresh readback directory required')
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        if [m.name for m in members]!=names:raise ValueError('registered trace inventory differs')
        for m,r in zip(members,inventory):
            name=PurePosixPath(m.name)
            if not m.isfile() or name.is_absolute() or '..' in name.parts or m.size!=r['bytes']:
                raise ValueError('unsafe trace member')
            p=output/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with stream.extractfile(m) as src,p.open('xb') as dst:
                for block in iter(lambda:src.read(1024**2),b''):
                    dst.write(block);h.update(block);count+=len(block)
            if count!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('trace member bytes differ')
    if digest(output/'traces.jsonl')!=envelope['trace_index_sha256']:
        raise ValueError('trace index differs after readback')
    return dict(rows=envelope['rows'],independent_transport_bytes_verified=True,
                independent_numeric_acceptance=False,network_forward_verified=False)
