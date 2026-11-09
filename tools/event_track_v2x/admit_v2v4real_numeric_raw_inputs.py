"""Transport existing accepted raw bytes into a fresh numerical capture runtime.

The proof hash must come from the frozen three-seed acceptance index. This does
not repeat raw schema acceptance or assert any new numerical acceptance.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def admit(*, proof, proof_sha, task_id, seed, archive, manifest, index, output,
          max_unpacked_bytes):
    if sha(proof)!=proof_sha:raise ValueError('frozen acceptance proof differs')
    receipt=json.loads(Path(proof).read_bytes())
    if (receipt.get('task_id')!=task_id or receipt.get('checkpoint_seed')!=seed
            or receipt.get('independent_byte_schema_coverage_accepted') is not True):
        raise ValueError('accepted original seed/task proof required')
    pins=receipt['input_sha256']
    for name,path in [('native-features',archive),('native-feature-manifest',manifest),('native-feature-index',index)]:
        if sha(path)!=pins[name]:raise ValueError('accepted raw transport bytes differ')
    envelope=json.loads(Path(manifest).read_bytes());index_bytes=Path(index).read_bytes()
    if (hashlib.sha256(index_bytes).hexdigest()!=envelope['rows_sha256']
            or envelope['checkpoint_seed']!=seed or envelope.get('gt_read') is not False):
        raise ValueError('raw envelope binding differs')
    rows=[json.loads(line) for line in index_bytes.splitlines()]
    if len(rows)!=3986:raise ValueError('full original source rows required')
    expected=['anchors.npy','manifest.json','predictions.jsonl',*[r['payload'] for r in rows]]
    if len(set(expected))!=len(expected):raise ValueError('duplicate raw member')
    hashes={r['payload']:r['payload_sha256'] for r in rows}
    hashes.update({'anchors.npy':envelope['anchor_file_sha256'],
        'manifest.json':pins['native-feature-manifest'],'predictions.jsonl':pins['native-feature-index']})
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in out.parents):raise ValueError('fresh nonsymlink input directory required')
    if type(max_unpacked_bytes) is not int or max_unpacked_bytes<=0:raise ValueError('positive transport capacity required')
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        if [m.name for m in members]!=expected or any(not m.isfile() for m in members):
            raise ValueError('exact accepted raw ordinary-member inventory required')
        for m in members:
            path=PurePosixPath(m.name)
            if path.is_absolute() or '..' in path.parts or str(path)!=m.name:
                raise ValueError('unsafe raw member')
        total=sum(m.size for m in members)
        if total>max_unpacked_bytes:raise ValueError('raw transport capacity exceeded')
        out.mkdir(parents=True,exist_ok=False)
        for m in members:
            target=out/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();written=0
            with stream.extractfile(m) as src,target.open('xb') as dst:
                for block in iter(lambda:src.read(1024*1024),b''):
                    written+=len(block);h.update(block);dst.write(block)
            if written!=m.size or h.hexdigest()!=hashes[m.name]:raise ValueError('accepted raw member bytes differ')
    for name,path in [('native-features',archive),('native-feature-manifest',manifest),('native-feature-index',index)]:
        if sha(path)!=pins[name]:raise ValueError('input changed during transport')
    return dict(task_id=task_id,checkpoint_seed=seed,source_rows=3986,
        acceptance_proof_sha256=proof_sha,unpacked_bytes=total,
        raw_index_sha256=pins['native-feature-index'],transport_bound_to_existing_acceptance=True,
        independent_numeric_acceptance=False,new_raw_acceptance=False)
