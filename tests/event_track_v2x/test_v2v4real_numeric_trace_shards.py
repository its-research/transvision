"""Synthetic complete schedule transport; no numerical/GPU acceptance."""
import json
from pathlib import Path

import pytest

from tools.event_track_v2x.merge_v2v4real_numeric_trace_shards import merge
from tools.event_track_v2x.run_v2v4real_official_test_raw_features import shard_records
from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import digest,rows


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp('trace-shards');frames=[];raw=[]
    for n in range(1993):
        for ego in (True,False):
            frames.append(dict(sequence_id='synthetic',frame_ordinal=n,frame_key=str(n),is_ego=ego))
            raw.append(dict(sequence_id='synthetic',frame_ordinal=n,frame_id=str(n),
                side='vehicle-side' if ego else 'infrastructure-side',pcd_sha256='a'*64))
    frame_path=root/'frames.jsonl';raw_path=root/'raw.jsonl'
    frame_path.write_text(''.join(json.dumps(r)+'\n' for r in frames))
    raw_path.write_text(''.join(json.dumps(r)+'\n' for r in raw));raw_sha=digest(raw_path)
    lookup=rows(raw_path);paths=[]
    for rank in range(4):
        path=root/f'rank{rank}';path.mkdir();paths.append(path);assigned=shard_records(frames,rank,4)
        with (path/'traces.jsonl').open('w') as f:
            for i,r in enumerate(assigned):
                key=(r['sequence_id'],r['frame_ordinal'],'vehicle-side' if r['is_ego'] else 'infrastructure-side',r['frame_key'])
                payload=path/f'payload-{i}.npz';payload.write_bytes(b'synthetic only')
                row=dict(lookup[key],payload=payload.name,payload_sha256=digest(payload),raw_rows_sha256=raw_sha,
                    trace_layout='compact-cells-v1',TF32_enabled=False,
                    shard=dict(rank=rank,world_size=4,full_source_rows=3986,
                        rule='sorted_original_frame_pair_ordinal_mod_world_size'))
                f.write(json.dumps(row)+'\n')
        (path/'capacity-preflight.json').write_text(json.dumps(dict(within_budget=True,rows=len(assigned),
            projected_cohort_uncompressed_upper_bound_bytes=100000,budget_bytes=200000)))
    return paths,dict(full_frames=frame_path,frames_sha=digest(frame_path),raw_index=raw_path,raw_sha=raw_sha)


def test_merge_preserves_complete_original_identity_and_bytes(cohort,tmp_path):
    paths,args=cohort;result=merge(paths,output=tmp_path/'merged',**args)
    assert result['rows']==3986 and not result['independent_numeric_acceptance']
    assert rows(tmp_path/'merged/traces.jsonl').keys()==rows(args['raw_index']).keys()


def test_duplicate_or_wrong_rank_partition_is_rejected(cohort,tmp_path):
    paths,args=cohort
    with pytest.raises(ValueError):merge([paths[0],paths[0],*paths[2:]],output=tmp_path/'merged',**args)
    assert not (tmp_path/'merged').exists()
