"""Merge disjoint original frame-pair trace shards without rerunning inference."""
import json
from pathlib import Path
import shutil

from tools.event_track_v2x.run_v2v4real_official_test_raw_features import shard_records
from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import digest,identity,member,rows


def merge(shards, *, full_frames, frames_sha, raw_index, raw_sha, output):
    if len(shards) not in (4,8):raise ValueError('qualified four/eight-card shards required')
    if digest(full_frames)!=frames_sha or digest(raw_index)!=raw_sha:
        raise ValueError('original full frame/raw index binding differs')
    frames=[json.loads(line) for line in Path(full_frames).read_text().splitlines()]
    raw=rows(raw_index)
    if len(frames)!=3986 or len(raw)!=3986:raise ValueError('complete original source cohort required')
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in out.parents):raise ValueError('fresh merged directory required')
    indexed=[];seen=set();capacities=[]
    for rank,path in enumerate(shards):
        root=Path(path);index=rows(root/'traces.jsonl')
        assigned=shard_records(frames,rank,len(shards))
        expected={(r['sequence_id'],r['frame_ordinal'],
            'vehicle-side' if r['is_ego'] else 'infrastructure-side',r['frame_key']) for r in assigned}
        if index.keys()!=expected or seen & index.keys():raise ValueError('exact disjoint frame-pair partition required')
        seen.update(index)
        capacity=json.loads((root/'capacity-preflight.json').read_bytes())
        if capacity.get('within_budget') is not True or capacity.get('rows')!=len(expected):
            raise ValueError('qualified shard capacity proof required')
        capacities.append(capacity)
        for key,row in index.items():
            if (row.get('raw_rows_sha256')!=raw_sha or row.get('pcd_sha256')!=raw[key]['pcd_sha256']
                    or row.get('shard')!=dict(rank=rank,world_size=len(shards),full_source_rows=3986,
                        rule='sorted_original_frame_pair_ordinal_mod_world_size')):
                raise ValueError('trace original row/shard identity differs')
            path=member(root,row['payload'],row['payload_sha256'])
            indexed.append((rank,row,path))
    if seen!=raw.keys():raise ValueError('all original raw rows including empty predictions required')
    out.mkdir(parents=True,exist_ok=False);merged=[]
    for rank,row,source in sorted(indexed,key=lambda x:identity(x[1])):
        relative=f'rank{rank}/'+row['payload'];target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if digest(target)!=row['payload_sha256']:raise ValueError('trace changed while merging')
        merged.append(dict(row,payload=relative))
    with (out/'traces.jsonl').open('x') as f:
        for row in merged:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    capacity=dict(rows=3986,within_budget=True,trace_layout='compact-cells-v1',
        projected_cohort_uncompressed_upper_bound_bytes=sum(c['projected_cohort_uncompressed_upper_bound_bytes'] for c in capacities),
        budget_bytes=sum(c['budget_bytes'] for c in capacities),
        scope='sum of original shard array-byte bounds; container overhead additional')
    with (out/'capacity-preflight.json').open('x') as f:json.dump(capacity,f,sort_keys=True)
    return dict(rows=3986,world_size=len(shards),trace_index_sha256=digest(out/'traces.jsonl'),
        raw_index_sha256=raw_sha,independent_numeric_acceptance=False,network_forward_verified=False)
