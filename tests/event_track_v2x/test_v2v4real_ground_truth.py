import copy
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import yaml

from transvision.models.event_track_v2x.v2v4real_ground_truth import (
    native_id, local_car_boxes, prepare_frame, upright_corners, upright_lwh, load_train_ground_truth,
)
from transvision.models.event_track_v2x.v2v4real_inputs import pose_to_world
from tools.event_track_v2x import prepare_v2v4real_ground_truth as tool
from tools.event_track_v2x.extract_v2v4real_archive import extract_volume, digest_file
from tools.event_track_v2x.v2v4real_gt_oracle import native_oracle, verified_sources


def obj(x=10., *, aid=-1, kind='Car', angle=(0., 0., 0.)):
    return dict(location=[x, 0., 0.], center=[0., 0., 0.], angle=list(angle),
                extent=[2., 1., 1.], obj_type=kind, ass_id=aid)


def pair():
    return {'0': dict(lidar_pose=np.eye(4), vehicles={1: obj()}),
            '1': dict(lidar_pose=np.eye(4), vehicles={2: obj(20.)})}


def test_minus_one_is_local_and_shared_ids_are_not_renumbered():
    assert native_id(7, -1, 0) == 7
    assert native_id(7, -1, 1) == 107
    assert native_id(22, 7, 1) == 7
    assert native_id('22', '7', '1') == 7


@pytest.mark.parametrize('values', [(True,-1,0), (1,True,0), (1,-1,True),
    (-1,-1,0), (1,-2,0), (1,-1,2), ('01',-1,0), (1,'1.0',0), (2**53,-1,0)])
def test_unsupported_identity_is_rejected(values):
    with pytest.raises(ValueError): native_id(*values)


def test_raw_labels_are_source_local_not_world_and_inputs_stay_unchanged():
    frames = pair(); frames['0']['vehicles'] = {}
    frames['0']['lidar_pose'][:3,3] = [1000.,500.,10.]
    frames['1']['lidar_pose'][:3,3] = [1005.,500.,10.]
    before = copy.deepcopy(frames)
    result = prepare_frame(frames, ego_cav='0')
    assert len(result['objects']) == 1
    np.testing.assert_allclose(np.mean(result['objects'][0]['corners_ego'],axis=0), [25.,0.,0.],atol=1e-6)
    for cav in frames:
        np.testing.assert_array_equal(frames[cav]['lidar_pose'], before[cav]['lidar_pose'])
        assert frames[cav]['vehicles'] == before[cav]['vehicles']


def test_strict_car_does_not_merge_trucks_pedestrians_or_missing_classes():
    frames=pair()
    frames['0']['vehicles'].update({3:obj(kind='Truck'),4:obj(kind='ConcreteTruck'),5:obj(kind='Pedestrian')})
    result=prepare_frame(frames,ego_cav='0')
    assert {o['track_id'] for o in result['objects']} == {1,102}
    del frames['0']['vehicles'][1]['obj_type']
    with pytest.raises(ValueError): prepare_frame(frames,ego_cav='0')


def test_duplicate_same_cav_identity_fails_without_averaging():
    frames=pair();frames['0']['vehicles'][3]=obj(20.,aid=1)
    with pytest.raises(ValueError,match='same-CAV'):
        prepare_frame(frames,ego_cav='0')


def test_deduplication_is_before_global_roi_not_best_box_selection():
    frames=pair();frames['0']['vehicles']={7:obj(99.)};frames['1']['vehicles']={3:obj(10.,aid=7)}
    result=prepare_frame(frames,ego_cav='0')
    assert result['objects']==[] and result['audit']['ego_roi_rejected_ids']==[7]
    assert len(result['audit']['duplicate_ids'])==1
    assert prepare_frame(frames,ego_cav='1')['objects'][0]['selected_cav']=='1'


def test_local_xyz_mask_then_global_xy_mask_and_empty_frames():
    frames=pair();frames['0']['vehicles'][1]['location'][2]=10.
    assert {o['track_id'] for o in prepare_frame(frames,ego_cav='0')['objects']}=={102}
    frames['1']['vehicles']={}
    assert prepare_frame(frames,ego_cav='0')['objects']==[]


@pytest.mark.parametrize('bad',['simulation_pose','missing_cav','implicit_ego','negative_extent'])
def test_frame_contract(bad):
    frames=pair();ego='0'
    if bad=='simulation_pose':frames['0']['lidar_pose']=[0.]*6
    elif bad=='missing_cav':del frames['1']
    elif bad=='implicit_ego':ego=None
    else:frames['0']['vehicles'][1]['extent'][0]=-1
    with pytest.raises(ValueError):prepare_frame(frames,ego_cav=ego)


def make_volume(tmp_path, *, split='train'):
    archive=tmp_path/(split+'_01.zip')
    with zipfile.ZipFile(archive,'w') as z:
        for frame in range(2):
            metadata=pair()
            if frame==1:metadata['1']['vehicles'][2]['ass_id']=1
            for cav,m in metadata.items():
                m['lidar_pose']=m['lidar_pose'].tolist()
                z.writestr(f'scene/{cav}/{frame:06d}.yaml',yaml.safe_dump(m))
                z.writestr(f'scene/{cav}/{frame:06d}.pcd','fixture PCD bytes')
    release=tmp_path/'release.json'
    release.write_text(json.dumps(dict(kind='v2v4real_official_box_metadata_snapshot_v1',dataset='V2V4Real',files=[
        dict(name=archive.name,split=split,size_bytes=archive.stat().st_size,reported_sha1=digest_file(archive)[0])])) )
    volume=tmp_path/'volume';extract_volume(archive,release,digest_file(release)[1],volume)
    egos=tmp_path/'egos.json';egos.write_text('{"scene":"0"}')
    return volume,digest_file(volume/'receipt.json')[1],egos


def test_volume_output_is_hash_bound_separate_and_keeps_native_id_changes(tmp_path):
    args=make_volume(tmp_path);output=tmp_path/'gt'
    m=tool.prepare(*args,output)
    assert m['paired_frames']==2 and m['source_frames']==4 and m['gt_objects']==3
    assert m['local_tracks_with_multiple_mapped_ids']==1
    assert m['frames_sha256']==digest_file(output/'frames.jsonl')[1]
    assert m['audit_sha256']==digest_file(output/'audit.jsonl')[1]
    loaded,frames=load_train_ground_truth(output,expected_manifest_sha256=digest_file(output/'manifest.json')[1])
    assert loaded==m and len(frames)==2
    assert not any(m[k] for k in ('paper_eligible','inference_input','tracking_evaluation_performed',
        'full_official_split_verified','oracle_all_ids_and_roi_match','physical_identity_continuity_verified'))
    with pytest.raises(ValueError,match='fresh'):tool.prepare(*args,output)
    with pytest.raises(ValueError,match='outside'):tool.prepare(*args,args[0]/'inside')


@pytest.mark.parametrize('bad',['hash','stream','duplicate_id','class','nan','ego','ordinal','test'])
def test_gt_consumer_rejects_tampering_even_when_stream_hash_is_resealed(tmp_path,bad):
    args=make_volume(tmp_path);output=tmp_path/'gt';m=tool.prepare(*args,output)
    sha=digest_file(output/'manifest.json')[1]
    if bad=='hash':sha='0'*64
    elif bad=='stream':(output/'frames.jsonl').write_bytes(b'{}\n')
    else:
        rows=[json.loads(line) for line in (output/'frames.jsonl').read_bytes().splitlines()]
        if bad=='duplicate_id':rows[0]['objects'].append(rows[0]['objects'][0])
        elif bad=='class':rows[0]['objects'][0]['raw_class']='Truck'
        elif bad=='nan':rows[0]['objects'][0]['corners_ego'][0][0]=float('nan')
        elif bad=='ego':rows[0]['ego_cav']='1'
        elif bad=='ordinal':rows[0]['frame_ordinal']=1
        else:m['split']='test'
        (output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        m['frames_sha256']=digest_file(output/'frames.jsonl')[1]
        (output/'manifest.json').write_text(json.dumps(m));sha=digest_file(output/'manifest.json')[1]
    with pytest.raises(ValueError):load_train_ground_truth(output,expected_manifest_sha256=sha)


def test_test_volume_rejected_before_raw_labels_are_read(tmp_path,monkeypatch):
    args=make_volume(tmp_path,split='test')
    monkeypatch.setattr(tool,'audit_volume',lambda *a:pytest.fail('test labels opened'))
    with pytest.raises(ValueError,match='train only'):tool.prepare(*args,tmp_path/'gt')


def test_changed_yaml_after_initial_audit_has_no_success_output(tmp_path,monkeypatch):
    args=make_volume(tmp_path);audit=tool.audit_volume
    def mutate(*a):
        report=audit(*a);next((args[0]/'payload').rglob('*.yaml')).write_text('changed');return report
    monkeypatch.setattr(tool,'audit_volume',mutate)
    with pytest.raises(ValueError,match='label changed'):tool.prepare(*args,tmp_path/'gt')
    assert not (tmp_path/'gt').exists()


def test_pinned_reference_unchanged_numeric_functions_and_cleanup(tmp_path):
    root=os.environ.get('V2V4REAL_GT_ORACLE_ROOT')
    if not root:pytest.skip('separately acquired pinned research reference required')
    before=set(sys.modules);rng=np.random.default_rng(20270914)
    with native_oracle(root) as reference:
        for _ in range(100):
            frames=pair()
            for cav,m in frames.items():
                m['lidar_pose']=pose_to_world([*rng.uniform(-10,10,3),*rng.uniform(-10,10,3)])
                for item in m['vehicles'].values():
                    item['location']=rng.uniform([-100,-50,-6],[100,50,4]).tolist()
                    item['angle']=rng.uniform([-15,-180,-15],[15,180,15]).tolist()
            actual=prepare_frame(frames,ego_cav='0')
            expected=reference(frames,'0')
            assert {o['track_id'] for o in actual['objects']}==set(expected)
            for row in actual['objects']:
                np.testing.assert_allclose(row['corners_ego'],expected[row['track_id']],atol=tool.NUMERIC_ATOL_M,rtol=0)
    assert not (set(sys.modules)-before)&{'opencood','opencood.data_utils','opencood.data_utils.datasets','opencood.utils'}
    for name in tool.SOURCE_HASHES:
        (tmp_path/name).write_bytes((Path(root)/name).read_bytes())
    (tmp_path/'box_utils.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='reviewed pinned'):verified_sources(tmp_path)
