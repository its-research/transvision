import copy
import hashlib
import json
import math
import pickle

import numpy as np
import pytest

from tools.event_track_v2x import evaluate_train_inference_diagnostic as tool


def inputs(frame='000001',time=1_000_000):
    info=dict(token=frame,scene_token='0000',timestamp=time,lidar2ego_rotation=[1,0,0,0],
        ego2global_rotation=[math.sqrt(.5),0,0,math.sqrt(.5)],lidar2ego_translation=[1,0,0],
        ego2global_translation=[10,20,0])
    row=dict(sequence_id='0000',vehicle_frame=frame,infrastructure_frame=frame,box_reference_timestamp_us=time)
    annotation=dict(track_id='1',token='gt-token',type='Car',veh_pointcloud_timestamp=str(time),
        **{'3d_location':dict(x=2,y=0,z=1),'3d_dimensions':dict(l=4,w=2,h=1.5)},rotation=0.)
    return info,row,annotation


def test_car_gt_uses_native_pose_and_explicitly_ignores_pedestrian():
    _,adapter=tool.evaluator()
    info,row,annotation=inputs()
    pedestrian=dict(annotation,type='Pedestrian',track_id='different')
    result=tool.car_frame(adapter,[annotation,pedestrian],info,row)
    assert len(result['objects'])==1 and result['objects'][0]['class_label']=='car'
    np.testing.assert_allclose(result['objects'][0]['mean'],[10,23,1,4,2,1.5,math.pi/2,0,0],atol=1e-12)
    assert result['ego_translation_world']==[10,20,0]
    assert result['objects'][0]['velocity_available'] is False


@pytest.mark.parametrize('error',['duplicate','time','fractional_time','dimensions','unknown_category','sequence'])
def test_car_gt_rejects_inconsistent_geometry_time_or_identity(error):
    _,adapter=tool.evaluator();info,row,annotation=inputs();annotations=[annotation]
    if error=='duplicate': annotations.append(copy.deepcopy(annotation))
    elif error=='time': annotation['veh_pointcloud_timestamp']='1000001'
    elif error=='fractional_time': info['timestamp']=1_000_000.5
    elif error=='dimensions': annotation['3d_dimensions']['l']=-1
    elif error=='unknown_category': annotation['type']='unknown'
    else: info['scene_token']='wrong'
    with pytest.raises(ValueError): tool.car_frame(adapter,annotations,info,row)


def test_preparation_seals_complete_selected_sequence_and_rejects_omission(tmp_path,monkeypatch):
    infos,rows,annotations=zip(*(inputs(f'{i:06d}',i*1_000_000) for i in (1,2)))
    projection=tmp_path/'projection';labels=projection/'cooperative/label';labels.mkdir(parents=True)
    inventory=[]
    for row,annotation in zip(rows,annotations):
        name='cooperative/label/'+row['vehicle_frame']+'.json';path=projection/name
        path.write_text(json.dumps([annotation]))
        inventory.append(dict(path=name,sha256=tool.sha(path),bytes=path.stat().st_size))
    pairs=[dict(vehicle_sequence='0000',infrastructure_sequence='0000',
        vehicle_frame=r['vehicle_frame'],infrastructure_frame=r['infrastructure_frame']) for r in rows]
    # The two-frame selected sequence is the test subject; other catalog rows
    # are deliberately metadata-only fixtures, never claimed as actual data.
    pairs.extend(dict(vehicle_sequence='0001',infrastructure_sequence='0001',
        vehicle_frame=f'{i:06d}',infrastructure_frame=f'{i:06d}') for i in range(3,7446))
    metadata=projection/'cooperative/data_info.json';metadata.write_text(json.dumps(pairs))
    poses=tmp_path/'fixture.pkl';poses.write_bytes(pickle.dumps(dict(infos=infos)))
    monkeypatch.setattr(tool,'VEHICLE_INFOS_SHA',tool.sha(poses))
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(converted_train_sha256={'vehicle-side':tool.sha(poses)},
        cooperative_metadata_sha256=tool.sha(metadata),cooperative_label_inventory=inventory)))
    plan=dict(kind='train_sequence_inference_diagnostic_plan_v1',cohort_mode='real-train-development',
        class_scope=['car'],complete_input_train_cohort_verified=True,cache_sha256=tool.TRAIN_CACHE_SHA,
        cooperative_metadata_sha256=tool.sha(metadata),selected_sequence='0000',selected_schedule=list(rows),scheduled_frames=2)
    plan_path=tmp_path/'fixture-plan.json';plan_path.write_text(json.dumps(plan))
    output=tmp_path/'gt'
    result=tool.prepare(plan_path,tool.sha(plan_path),projection,poses,audit,tool.sha(audit),output)
    assert result['frames']==2 and result['gt_objects']==2 and result['evaluator_only']
    assert not result['validation'] and not result['full_official_train'] and not result['paper_eligible']
    assert result['ground_truth_sha256']==tool.sha(output/'ground-truth.jsonl')
    plan['selected_schedule']=plan['selected_schedule'][:1];plan['scheduled_frames']=1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='not complete'):
        tool.prepare(plan_path,tool.sha(plan_path),projection,poses,audit,tool.sha(audit),tmp_path/'omitted')
    assert not (tmp_path/'omitted').exists()
    poses.write_bytes(b'not trusted pickle')
    monkeypatch.setattr(tool.pickle,'loads',lambda *a: pytest.fail('untrusted pickle decoded'))
    with pytest.raises(ValueError,match='refusing pickle'):
        tool.prepare(plan_path,tool.sha(plan_path),projection,poses,audit,tool.sha(audit),tmp_path/'untrusted')


def metric_fixture(tmp_path,case='perfect',*,bad=None):
    """Synthetic sealed receipts exercise the wrapper, not actual train results."""
    _,adapter=tool.evaluator()
    ground_truth=tmp_path/'fixture-gt';ground_truth.mkdir()
    replay=tmp_path/'fixture-replay';replay.mkdir()
    gt,predictions,schedule,audits=[],[],[],[];previous='0'*64;previous_audit='0'*64
    for i,time in enumerate((1_000_000,1_100_100,1_200_200,1_750_000)):
        row=dict(sequence_id='golden',vehicle_frame=str(i),infrastructure_frame=str(i),box_reference_timestamp_us=time)
        schedule.append(row)
        box=dict(track_id='gt1',class_label='car',mean=[i,0.,0.,4.,2.,1.5,0.,0.,0.])
        outside=dict(copy.deepcopy(box),track_id='outside-roi',mean=[400.,0.,0.,4.,2.,1.5,0.,0.,0.])
        gt.append(dict(sequence_id='golden',frame_id=str(i),box_reference_timestamp_us=time,
            ego_translation_world=[0.,0.,0.],objects=[box,outside]))
        objects=[] if case=='empty' else [dict(copy.deepcopy(box),
            track_id='pred2' if case=='identity_switch' and i>=2 else 'pred1',
            covariance=np.eye(9).tolist(),score=.9)]
        if case=='duplicate': objects.append(dict(copy.deepcopy(objects[0]),track_id='duplicate'))
        if bad=='noncar' and objects: objects[0]['class_label']='pedestrian'
        pred=dict(sequence_id='golden',frame_id=str(i),box_reference_timestamp_us=time,
            decision_timestamp_us=time+100_000,coordinate_frame='world',
            state_layout='gravity_xyz_length_width_height_yaw_vxy',previous_commit_sha256=previous,predictions=objects)
        pred['commit_sha256']=hashlib.sha256(adapter.canonical(pred)).hexdigest()
        previous=pred['commit_sha256'];predictions.append(pred)
        audit=dict(sequence_id='golden',event_id=str(i),prediction_sha256=previous,
            previous_audit_sha256=previous_audit,model_regret_upper=0.,global_fallback_used=False,
            components=[dict(recovery_events=[])])
        previous_audit=hashlib.sha256(adapter.canonical(audit)).hexdigest();audits.append(dict(tracking=audit))
    if bad=='commit': predictions[-1]['previous_commit_sha256']='0'*64
    if bad=='audit_commit': audits[-1]['tracking']['prediction_sha256']='a'*64
    if bad=='audit_chain': audits[-1]['tracking']['previous_audit_sha256']='b'*64
    gt_path=ground_truth/'ground-truth.jsonl';pred_path=replay/'predictions.jsonl'
    for path,records in ((gt_path,gt),(pred_path,predictions)):
        path.write_bytes(b''.join(adapter.canonical(record)+b'\n' for record in records))
    (replay/'tracking.jsonl').write_bytes(b''.join(adapter.canonical(record)+b'\n' for record in audits))
    manifest=dict(kind='spd_train_sequence_evaluator_ground_truth_v1',class_scope=['car'],
        evaluator_only=True,contains_train_payload=True,contains_test_payload=False,validation=False,
        selected_schedule=schedule,frames=4,ground_truth_sha256=tool.sha(gt_path))
    plan=dict(kind='train_sequence_inference_diagnostic_plan_v1',class_scope=['car'],
        cohort_mode='real-train-development',complete_input_train_cohort_verified=True,
        cache_sha256=tool.TRAIN_CACHE_SHA,selected_schedule=schedule,selected_sequence='golden',backend='component_completion')
    if bad=='schedule': plan['selected_schedule']=list(reversed(schedule))
    if bad=='cache': plan['cache_sha256']='a'*64
    (ground_truth/'manifest.json').write_bytes(adapter.canonical(manifest))
    (replay/'plan.json').write_bytes(adapter.canonical(plan))
    receipt=dict(status='complete',plan_sha256=tool.sha(replay/'plan.json'),cache_split='train',
        allocation_teacher=bad=='teacher',learned_identity_enabled=True,cache_sha256=tool.TRAIN_CACHE_SHA,
        completed_frames=4,scheduled_frames=4,predictions_sha256=tool.sha(pred_path),
        tracking_sha256=tool.sha(replay/'tracking.jsonl'))
    if bad=='incomplete': receipt['completed_frames']=3
    (replay/'receipt.json').write_bytes(adapter.canonical(receipt))
    final=dict(kind='train_sequence_inference_diagnostic_v1',status='complete',
        replay_receipt_sha256=tool.sha(replay/'receipt.json'),plan_sha256=receipt['plan_sha256'],
        cohort_mode='fixture-only' if bad=='fixture' else 'real-train-development',
        complete_selected_sequence_verified=True,training_performed=False,offline_teacher_probes=False,
        backend='component_covered_completion' if bad=='backend' else plan['backend'],
        selected_sequence='golden',completed_frames=4)
    (replay/'development-inference-receipt.json').write_bytes(adapter.canonical(final))
    return ground_truth,tool.sha(ground_truth/'manifest.json'),replay,tool.sha(replay/'development-inference-receipt.json')


@pytest.mark.parametrize('case',['perfect','empty','identity_switch'])
def test_native_engines_through_complete_evaluation_wrapper(tmp_path,case):
    inputs=metric_fixture(tmp_path,case)
    result=tool.evaluate(*inputs,tmp_path/'metrics')
    native=result['metrics']['nuscenes'];tracking=result['metrics']['trackeval']['car']
    expected={'perfect':(1.,1.,1.),'empty':(0.,0.,0.),'identity_switch':(math.sqrt(.5),.5,.5)}[case]
    assert tuple(tracking['summary'][key] for key in ('HOTA','AssA','IDF1'))==pytest.approx(expected)
    assert tracking['sequences']['golden']['gt_objects']==4  # 400 m objects excluded by native 50 m ROI.
    if case=='perfect':
        assert native['amota']==pytest.approx(1.) and native['amotp']==pytest.approx(0.,abs=1e-10)
    elif case=='empty':
        assert native['amota']==0. and native['amotp']==2.
    else: assert native['ids']==1
    assert result['counts']['frames']==4 and result['counts']['sequences']==1
    assert not result['validation'] and not result['paper_eligible'] and not result['full_official_train']
    assert result['protocol']['split']=='train_development'
    diagnostic=result['identity_event_diagnostics']
    assert diagnostic['counts']['roi_gt_frame_observations']==4
    if case=='empty':
        assert diagnostic['counts']['identity_unknown_gt_frames']==4 and diagnostic['episodes']==[]
        assert diagnostic['observed_error_span_seconds_quantiles'] is None
    elif case=='identity_switch':
        episode,=diagnostic['episodes']
        assert episode['observed_error_span_seconds']==pytest.approx(.5498)
        assert episode['right_censored'] and not episode['left_censored']
        assert diagnostic['counts']['anchor_disagreement_gt_frames']==2
    else:
        assert diagnostic['counts']['anchor_agreement_gt_frames']==4 and diagnostic['episodes']==[]
    assert diagnostic['model_bound_is_not_gt_error_probability']
    assert diagnostic['internal_recovery_is_not_gt_identity_recovery']
    assert json.loads((tmp_path/'metrics/metrics.json').read_bytes())==result


@pytest.mark.parametrize('bad',['incomplete','fixture','teacher','backend','cache','schedule','commit','noncar','audit_commit','audit_chain'])
def test_evaluation_rejects_unmatched_or_invalid_input_before_metrics(tmp_path,monkeypatch,bad):
    inputs=metric_fixture(tmp_path,bad=bad)
    module,adapter=tool.evaluator()
    monkeypatch.setattr(adapter,'compute_metrics',lambda *a,**kw:pytest.fail('invalid inputs reached metrics'))
    monkeypatch.setattr(tool,'evaluator',lambda:(module,adapter))
    with pytest.raises(ValueError): tool.evaluate(*inputs,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_duplicate_overlap_remains_unknown_not_a_forced_identity(tmp_path):
    ground_truth,_,replay,_=metric_fixture(tmp_path,'duplicate')
    _,adapter=tool.evaluator()
    gt=[json.loads(line) for line in (ground_truth/'ground-truth.jsonl').read_bytes().splitlines()]
    pred=[json.loads(line) for line in (replay/'predictions.jsonl').read_bytes().splitlines()]
    result=tool.identity_events(adapter,gt,pred,replay/'tracking.jsonl')
    assert result['counts']['duplicate_gt_frames']==4
    assert result['counts']['duplicate_excess_predictions']==4
    assert result['counts']['identity_unknown_gt_frames']==4
    assert result['counts']['prediction_ambiguous']==8
    assert not result['episodes']


def test_unreported_recovery_and_bounds_do_not_become_zero_evidence(tmp_path):
    ground_truth,_,replay,_=metric_fixture(tmp_path)
    _,adapter=tool.evaluator()
    gt=[json.loads(line) for line in (ground_truth/'ground-truth.jsonl').read_bytes().splitlines()]
    pred=[json.loads(line) for line in (replay/'predictions.jsonl').read_bytes().splitlines()]
    audits=[json.loads(line) for line in (replay/'tracking.jsonl').read_bytes().splitlines()]
    previous='0'*64
    for record in audits:
        audit=record['tracking']
        for key in ('components','model_regret_upper','global_fallback_used'): audit.pop(key)
        audit['previous_audit_sha256']=previous
        previous=hashlib.sha256(adapter.canonical(audit)).hexdigest()
    (replay/'tracking.jsonl').write_bytes(b''.join(adapter.canonical(r)+b'\n' for r in audits))
    result=tool.identity_events(adapter,gt,pred,replay/'tracking.jsonl')
    assert result['counts']['recovery_unreported_frames']==4
    assert result['counts']['model_bound_unreported_events']==4
    assert result['counts']['fallback_unreported_events']==4
    assert all(r['recovery_event_records'] is None and r['model_regret_upper'] is None for r in result['frames'])


def test_input_mutation_during_metric_computation_cannot_be_accepted(tmp_path,monkeypatch):
    inputs=metric_fixture(tmp_path);replay=inputs[2]
    module,adapter=tool.evaluator()
    def mutate(*args,**kwargs):
        (replay/'receipt.json').write_bytes(b'{}')
        return {}
    monkeypatch.setattr(adapter,'compute_metrics',mutate)
    monkeypatch.setattr(tool,'evaluator',lambda:(module,adapter))
    with pytest.raises(ValueError,match='inputs changed'):
        tool.evaluate(*inputs,tmp_path/'changed')
    assert not (tmp_path/'changed').exists()
