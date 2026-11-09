"""GT-free full raw archive IO for causal motion, preserving every source row."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np

from tools.event_track_v2x.v2v4real_motion_fit_inputs import read

RAW_ARRAYS={'states_source_legacy7','raw_scores','class_indices','appearance','appearance_valid'}


def load_raw(root, manifest_sha, archive_sha, pins, *, progress=lambda _: None):
    root=Path(root).absolute()
    manifest_bytes=read(root/'native-feature-manifest',manifest_sha,pins)
    m=json.loads(manifest_bytes)
    split=m.get('split')
    if (split not in ('train','official_test') or m.get('gt_read') is not False
            or m.get('kind')!='v2v4real_'+('train' if split=='train' else 'official_test')+'_raw_native_features_v1'
            or m.get('official_test_read') is not (split=='official_test')
            or m.get('protocol_id')!='v2v4real-nominal-10hz-formal-v1'
            or m.get('candidate_protocol')!='rbf-all-class-top64-v1'
            or m.get('matching_candidate_rule')!='raw_score_ge_0.05_stable_top64_before_nms'):
        raise ValueError('explicit GT-free native raw split required')
    index_bytes=read(root/'native-feature-index',m['rows_sha256'],pins)
    index=[json.loads(x) for x in index_bytes.splitlines()]
    fields={'sequence_id','frame_id','frame_ordinal','side','source_to_world','pcd_sha256','payload','payload_sha256','candidates'}
    payloads, identities={},set()
    for r in index:
        key=r['sequence_id'],r['frame_id'],r['side'];p=PurePosixPath(r['payload'])
        if (set(r)!=fields or p.parts[:1]!=('payloads',) or len(p.parts)!=2 or p.suffix!='.npz'
                or '..' in p.parts or p.as_posix()!=r['payload'] or key in identities or r['payload'] in payloads):
            raise ValueError('unique contained GT-free raw row required')
        identities.add(key);payloads[r['payload']]=r
    if len(index)!=m['rows'] or sum(r['candidates'] for r in index)!=m['total_candidates']:
        raise ValueError('complete raw coverage differs')
    for sid in m['sequence_ids']:
        source=[r for r in index if r['sequence_id']==sid]
        ordinals={s:{r['frame_ordinal'] for r in source if r['side']==s} for s in ('vehicle-side','infrastructure-side')}
        if (ordinals['vehicle-side']!=ordinals['infrastructure-side']
                or sorted(ordinals['vehicle-side'])!=list(range(len(ordinals['vehicle-side'])))
                or len(source)!=2*len(ordinals['vehicle-side'])):
            raise ValueError('full paired nominal ordinal schedule required')
    # Streaming hash avoids retaining the potentially large original archive in memory.
    archive=root/'native-features'
    if not archive.is_file() or any(p.is_symlink() for p in (archive,*archive.parents)):
        raise ValueError('ordinary original raw archive required')
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    if h.hexdigest()!=archive_sha:raise ValueError('original raw archive SHA differs')
    pins[str(archive)]=archive_sha
    expected=set(payloads)|{'manifest.json','predictions.jsonl','anchors.npy'}
    seen=set();data={}
    with tarfile.open(archive,'r:*') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError('exact ordinary original raw archive members required')
            seen.add(member.name)
            limit=(len(index_bytes) if member.name=='predictions.jsonl' else
                   len(manifest_bytes) if member.name=='manifest.json' else
                   64*1024**2 if member.name=='anchors.npy' else 4*1024**2)
            if member.size>limit:raise ValueError('oversized native raw member')
            b=tar.extractfile(member).read()
            if len(b)!=member.size:raise ValueError('truncated original raw member')
            if member.name=='manifest.json':
                if b!=manifest_bytes:raise ValueError('original raw manifest bytes differ')
            elif member.name=='predictions.jsonl':
                if b!=index_bytes:raise ValueError('original raw index bytes differ')
            elif member.name=='anchors.npy':
                if hashlib.sha256(b).hexdigest()!=m['anchor_file_sha256']:raise ValueError('anchor SHA differs')
            else:
                row=payloads[member.name];n=row['candidates']
                if hashlib.sha256(b).hexdigest()!=row['payload_sha256']:raise ValueError('original NPZ SHA differs')
                with np.load(io.BytesIO(b),allow_pickle=False) as z:
                    if set(z.files)!=RAW_ARRAYS:raise ValueError('GT-free raw array inventory differs')
                    s,c=z['states_source_legacy7'].copy(),z['class_indices'].copy()
                if (s.shape!=(n,7) or s.dtype not in (np.dtype('float32'),np.dtype('float64'))
                        or not np.isfinite(s).all() or np.any(s[:,3:6]<=0) or c.shape!=(n,)
                        or c.dtype.kind not in 'iu' or np.any(c!=0)):
                    raise ValueError('native vehicle geometry/class schema differs')
                row_sha=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                data[member.name]=dict(sequence_id=row['sequence_id'],side=row['side'],frame_ordinal=row['frame_ordinal'],
                    source_to_world=row['source_to_world'],states_source_legacy7=s,class_indices=c,row_sha256=row_sha)
                if len(data)%1000==0:progress(len(data))
    if seen!=expected or len(data)!=len(index):raise ValueError('complete raw archive coverage differs')
    ordered=sorted(index,key=lambda r:(r['sequence_id'],r['side'],r['frame_ordinal']))
    return m,ordered,[data[r['payload']] for r in ordered]
