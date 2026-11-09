"""Whole-schedule sharding and complete-envelope merge, including empty frames."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as e
from tools.event_track_v2x import merge_v2v4real_official_test_raw_shards as merge


def records(count=1993):
    return [dict(sequence_id=f's{i%9}',frame_ordinal=i,frame_key=str(i),is_ego=ego,
                 source_to_world=np.eye(4).tolist(),pcd_sha256='f'*64)
            for i in range(count) for ego in (True,False)]


def test_standalone_merge_schema_matches_complete_cache_consumer():
    from tools.event_track_v2x.build_v2v4real_raw_native_cache import RAW_MANIFEST_FIELDS
    assert merge.RAW_MANIFEST_FIELDS==RAW_MANIFEST_FIELDS


def test_frame_pairs_are_disjoint_exhaustive_and_order_independent():
    rows=records();parts=[e.shard_records(list(reversed(rows)),r,4) for r in range(4)]
    identities=[{(x['sequence_id'],x['frame_ordinal'],x['is_ego']) for x in p} for p in parts]
    assert sum(map(len,identities))==3986 and len(set.union(*identities))==3986
    assert [len(x) for x in parts]==[998,996,996,996]
    for rank,part in enumerate(parts):
        assert part==e.shard_records(rows,rank,4)
        assert all(part[i]['frame_key']==part[i+1]['frame_key'] for i in range(0,len(part),2))


@pytest.mark.parametrize('rank,world',[(-1,4),(4,4),(0,0),(False,4),(0,True)])
def test_invalid_shard_rejected(rank,world):
    with pytest.raises(ValueError):e.shard_records(records(3),rank,world)


def test_broken_pairs_rejected_even_if_on_another_rank():
    rows=records(3);rows.pop()
    with pytest.raises(ValueError):e.shard_records(rows,0,2)


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    original=records();frames=tmp_path/'frames.jsonl'
    frames.write_bytes(b''.join(e.canonical(x)+b'\n' for x in original))
    monkeypatch.setattr(e,'FRAME_SHA',e.sha(frames))
    projection=dict(dataset_split='test',kind='v2v4real_pose_lidar_projection_v2',gt_in_projection=False,
                    frames_sha256=e.FRAME_SHA,sequence_count=9,paired_frame_count=1993,source_frame_count=3986,
                    ego_agents={f's{i}':'ego' for i in range(9)})
    p=tmp_path/'projection';p.write_bytes(e.canonical(projection));monkeypatch.setattr(e,'PROJECTION_SHA',e.sha(p))
    a=tmp_path/'admission';a.write_bytes(e.canonical(dict(kind='v2v4real_official_split_admission_v1',split='official_test',
        projection_manifest_sha256=e.PROJECTION_SHA,protocol_id=e.PROTOCOL_ID,official_split_membership_verified=True,
        official_original_split_preserved=True,official_test_unchanged=True,overlap_disclosure_required=True,
        statistical_independence_claimed=False)));monkeypatch.setattr(e,'ADMISSION_SHA',e.sha(a))
    buf=io.BytesIO();np.savez_compressed(buf,states_source_legacy7=np.empty((0,7)),raw_scores=np.empty(0),
        class_indices=np.empty(0,dtype=np.int64),appearance=np.empty((0,128)),appearance_valid=np.empty(0,dtype=bool))
    raw=buf.getvalue();directories=[]
    for rank in range(2):
        d=tmp_path/f'rank{rank}';work=d/'raw';(work/'payloads').mkdir(parents=True)
        np.save(work/'anchors.npy',np.zeros((1,1,1,7)),allow_pickle=False)
        rows=[]
        for i,r in enumerate(e.shard_records(original,rank,2)):
            name=f'payloads/frame-{i:06d}.npz';(work/name).write_bytes(raw)
            rows.append(dict(sequence_id=r['sequence_id'],frame_ordinal=r['frame_ordinal'],frame_id=r['frame_key'],
                side='vehicle-side' if r['is_ego'] else 'infrastructure-side',source_to_world=r['source_to_world'],
                pcd_sha256=r['pcd_sha256'],payload=name,payload_sha256=hashlib.sha256(raw).hexdigest(),candidates=0))
        m={k:None for k in merge.RAW_MANIFEST_FIELDS}
        m.update(kind=e.SHARD_KIND,protocol_id=e.PROTOCOL_ID,split='official_test',gt_read=False,paper_metric=False,
            official_test_read=True,velocity_exported=False,covariance_exported=False,prepared_projection_sha256=e.PROJECTION_SHA,
            split_admission_sha256=e.ADMISSION_SHA,partition_sha256=e.PARTITION_SHA,
            matching_candidate_rule='raw_score_ge_0.05_stable_top64_before_nms',checkpoint_seed=1337,
            checkpoint_sha256=e.CHECKPOINTS[1337],anchor_file_sha256=e.sha(work/'anchors.npy'),
            rows=len(rows),rows_sha256=hashlib.sha256(b''.join(e.canonical(x)+b'\n' for x in rows)).hexdigest(),
            total_candidates=0,appearance_valid_candidates=0,projection_manifest={},split_admission={},
            shard=dict(rank=rank,world_size=2,rule='sorted_original_frame_pair_ordinal_mod_world_size',
                       full_source_rows=3986,projection_frames_sha256=e.FRAME_SHA))
        archive=e.pack(work,d,m,rows,p,a)
        receipt=dict(raw_manifest_sha256=e.sha(d/'native-feature-manifest'),raw_archive_sha256=archive,
            producer_sha256=e.sha(e.__file__),source_archive_sha256=e.SOURCE_ARCHIVE_SHA,inference_core_sha256=e.CORE_SHA,
            shard=dict(rank=rank,world_size=2),TF32_enabled=False,actual_device={'name':'software fixture'})
        (d/'raw-export-receipt.json').write_bytes(e.canonical(receipt));directories.append(d)
    return directories,frames,tmp_path


def test_merge_retains_all_3986_empty_rows_and_complete_schema(cohort):
    ds,frames,root=cohort;receipt=merge.merge(ds,frames,root/'merged')
    assert receipt['rows']==3986 and receipt['independent_raw_acceptance'] is False
    m=json.loads((root/'merged/native-feature-manifest').read_bytes())
    assert m['kind']==e.KIND and set(m)==merge.RAW_MANIFEST_FIELDS|{'projection_manifest','split_admission'}
    rows=[json.loads(x) for x in (root/'merged/native-feature-index').read_bytes().splitlines()]
    assert len({merge.identity(r) for r in rows})==3986 and all(r['candidates']==0 for r in rows)
    # Exercise the separate full-cohort readback on persisted merged bytes.
    from tools.event_track_v2x.readback_v2v4real_official_test_raw_gpu import verify
    import shutil
    shutil.copyfile(frames,root/'merged/projection-frames')
    (root/'merged/execution-receipt').write_bytes(e.canonical(receipt))
    accepted=verify(root/'merged')
    assert accepted['source_rows']==3986 and accepted['empty_source_rows']==3986
    assert accepted['independent_byte_schema_coverage_accepted'] is True
    assert accepted['independent_detector_numeric_accepted'] is False


def test_duplicate_rank_is_not_complete(cohort):
    ds,frames,root=cohort
    with pytest.raises(ValueError,match='rank'):merge.merge([ds[0],ds[0]],frames,root/'bad')


def test_tampered_archive_is_rejected_before_complete_output(cohort):
    ds,frames,root=cohort
    with (ds[1]/'native-features').open('ab') as f:f.write(b'mutation')
    with pytest.raises(ValueError,match='receipt'):merge.merge(ds,frames,root/'bad')
    assert not (root/'bad').exists()


@pytest.mark.parametrize('mutation',['checkpoint','frame','missing-row'])
def test_rehashed_metadata_cannot_replace_checkpoint_or_original_schedule(cohort,mutation):
    ds,frames,root=cohort;d=ds[1]
    m=json.loads((d/'native-feature-manifest').read_bytes())
    receipt=json.loads((d/'raw-export-receipt.json').read_bytes())
    rows=[json.loads(x) for x in (d/'native-feature-index').read_bytes().splitlines()]
    if mutation=='checkpoint':m['checkpoint_sha256']='0'*64
    else:
        if mutation=='frame':rows[0]['frame_id']='other-frame'
        else:rows.pop();m['rows']=len(rows)
        raw=b''.join(e.canonical(x)+b'\n' for x in rows)
        (d/'native-feature-index').write_bytes(raw);m['rows_sha256']=hashlib.sha256(raw).hexdigest()
    (d/'native-feature-manifest').write_bytes(e.canonical(m))
    receipt['raw_manifest_sha256']=e.sha(d/'native-feature-manifest')
    (d/'raw-export-receipt.json').write_bytes(e.canonical(receipt))
    with pytest.raises(ValueError):merge.merge(ds,frames,root/'bad')
    assert not (root/'bad').exists()
