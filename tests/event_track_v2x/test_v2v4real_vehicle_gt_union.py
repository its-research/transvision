import copy
import hashlib
import json

import pytest

from tools.event_track_v2x import merge_v2v4real_vehicle_ground_truth as union
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.v2v4real_ground_truth import prepare_vehicle_frame, VEHICLE_GT_RECIPE
from transvision.models.event_track_v2x.paper_evaluation_policy import VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION
from test_v2v4real_ground_truth import pair, obj


def write_part(root, sequences, *, mutate=None):
    frames, audits = [], []
    for sequence in sequences:
        for i in range(2):
            metadata = pair()
            metadata['0']['vehicles'][1] = obj(kind='Truck')
            metadata['0']['vehicles'][3] = obj(kind='Pedestrian')
            if i == 1:
                metadata['0']['vehicles'] = {}; metadata['1']['vehicles'] = {}
            prepared = prepare_vehicle_frame(metadata, ego_cav='0')
            frames.append(dict(sequence_id=sequence, frame_key=f'{i:06}', frame_ordinal=i,
                ego_cav='0', objects=prepared['objects']))
            audits.append(dict(sequence_id=sequence, frame_key=f'{i:06}', **prepared['audit']))
    manifest = dict(kind='v2v4real_native_vehicle_gt_projection_v1', recipe=VEHICLE_GT_RECIPE,
        split='train', class_scope=['vehicle'], evaluation_class='vehicle', evaluation_protocol=VEHICLE_PROTOCOL,
        native_label_source=NATIVE_VEHICLE_SELECTION, time_basis='ordinal-only-no-clock',
        coordinate_frame='current_ego_lidar', box_representation='eight_ordered_corners_xyz_m', gt_range=[-100,-40,-5,100,40,3],
        source_local_roi='at_least_two_upright_corners_inside_xyz', final_ego_roi='all_eight_corners_inside_xy',
        deduplication='first_cav_in_explicit_ego_first_order_before_final_roi', identity_rule='native',
        reference_commit=union.COMMIT, reference_scope='unchanged_native_numeric_GT_definitions_and_native_Pedestrian_exclusion',
        reference_source_sha256=union.SOURCE_HASHES, oracle_corner_atol_m=1e-4, oracle_max_abs_corner_error_m=0.,
        oracle_compared_frames=len(frames), oracle_all_ids_and_roi_match=True,
        ego_agents={s:'0' for s in sequences}, source_sha256={'fixture':'a'*64}, volume_receipt_sha256='b'*64,
        paired_frames=len(frames), source_frames=2*len(frames), sequence_frames={s:2 for s in sequences},
        gt_objects=sum(len(f['objects']) for f in frames), GT_read=True, inference_input=False, test_payloads_read=False)
    if mutate:
        mutate(manifest, frames, audits)
    root.mkdir()
    for name, rows in (('frames.jsonl', frames), ('audit.jsonl', audits)):
        data=b''.join(canonical(r)+b'\n' for r in rows)
        (root/name).write_bytes(data); manifest[name.split('.')[0]+'_sha256']=hashlib.sha256(data).hexdigest()
    (root/'manifest.json').write_bytes(canonical(manifest))
    return sha_file(root/'manifest.json')


def inputs(tmp_path, bad=None):
    parts=[tmp_path/'part-a',tmp_path/'part-b']
    def mutate(m,f,a):
        if bad=='test': m.update(split='official_test', test_payloads_read=True)
        elif bad=='car': m.update(kind='v2v4real_native_train_gt_projection_v1',class_scope=['Car'])
        elif bad=='oracle_partial': m['oracle_compared_frames']-=1
        elif bad=='oracle_false': m['oracle_all_ids_and_roi_match']=False
        elif bad=='oracle_error': m['oracle_max_abs_corner_error_m']=.1
        elif bad=='oracle_source': m['reference_source_sha256']={}
        elif bad=='wrong_ego': f[0]['ego_cav']='1'
        elif bad=='missing_audit': a.pop()
        elif bad=='wrong_audit': a[0]['frame_key']='123456'
        elif bad=='wrong_roi': m['gt_range']=[-1]*6
    hashes=[write_part(parts[0], ['selected-a','not-selected']),
            write_part(parts[1], ['selected-a'] if bad=='overlap' else ['selected-b'], mutate=mutate)]
    partition=dict(kind='v2v4real_formal_nested_training_partition_v1',protocol_id='v2v4real-nominal-10hz-formal-v1',
        official_split='train',official_test_used_for_partition=False,gt_used_for_partition=False,
        detector_fit=['det'],calibration_fit=['cal'],identity_selection=['hold'],
        sequence_to_name_group={'selected-a':'cal','selected-b':'cal','not-selected':'det','holdout':'hold'})
    if bad=='missing_sequence':partition['sequence_to_name_group']['missing']='cal'
    if bad=='partition_test':partition['official_test_used_for_partition']=True
    if bad=='role_overlap':partition['detector_fit']=['cal']
    p=tmp_path/'partition.json';p.write_bytes(canonical(partition))
    return parts,hashes,p,sha_file(p),tmp_path/'union'


def test_selected_gt_union_preserves_empty_frames_geometry_and_precise_oracle_scope(tmp_path):
    args=inputs(tmp_path);before={str(p):p.read_bytes() for root in args[0] for p in root.iterdir()}
    manifest=union.merge(*args)
    assert manifest['paired_frames']==manifest['oracle_compared_frames']==4
    assert manifest['read_paired_frames']==6 and manifest['source_frames']==8
    assert manifest['calibration_fit_sequences']==['selected-a','selected-b']
    assert manifest['raw_train_sequences_read']==['not-selected','selected-a','selected-b']
    assert manifest['raw_annotation_counts']=={'Car':2,'Pedestrian':2,'Truck':2}
    assert manifest['selected_native_annotation_counts']=={'Car':2,'Truck':2}
    assert manifest['excluded_native_annotation_counts']=={'Pedestrian':2}
    assert manifest['gt_objects']==4
    frames=[json.loads(x) for x in (args[-1]/'frames.jsonl').read_bytes().splitlines()]
    audits=[json.loads(x) for x in (args[-1]/'audit.jsonl').read_bytes().splitlines()]
    assert [len(f['objects']) for f in frames]==[2,0,2,0]
    for frame,audit in zip(frames,audits):
        binding=audit.pop('parent_oracle_binding')
        assert binding['source_frame_sha256']==hashlib.sha256(canonical(frame)).hexdigest()
        assert binding['source_audit_sha256']==hashlib.sha256(canonical(audit)).hexdigest()
        assert binding['part_manifest_sha256'] in args[1]
    assert before=={str(p):p.read_bytes() for root in args[0] for p in root.iterdir()}
    assert not manifest['paper_eligible'] and not manifest['parameter_training_performed']
    with pytest.raises(ValueError,match='fresh'):union.merge(*args)


@pytest.mark.parametrize('bad',['test','car','oracle_partial','oracle_false','oracle_error','oracle_source',
    'wrong_ego','missing_audit','wrong_audit','wrong_roi','overlap','missing_sequence','partition_test','role_overlap'])
def test_gt_union_rejects_scope_identity_incomplete_oracle_and_coverage(tmp_path,bad):
    args=inputs(tmp_path,bad)
    with pytest.raises(ValueError):union.merge(*args)
    assert not args[-1].exists()


def test_gt_union_rejects_changed_part_and_wrong_explicit_hash(tmp_path):
    args=inputs(tmp_path)
    (args[0][0]/'audit.jsonl').write_text('changed')
    with pytest.raises(ValueError):union.merge(*args)
    assert not args[-1].exists()


def test_gt_union_detects_read_time_change_before_publication(tmp_path,monkeypatch):
    args=inputs(tmp_path);original=union.load_vehicle_ground_truth;calls=[]
    def read(root,**kwargs):
        result=original(root,**kwargs);calls.append(str(root))
        if len(calls)==2:(args[0][0]/'audit.jsonl').write_text('changed after first read')
        return result
    monkeypatch.setattr(union,'load_vehicle_ground_truth',read)
    with pytest.raises(ValueError):union.merge(*args)
    assert not args[-1].exists()
