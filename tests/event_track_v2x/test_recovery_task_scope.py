from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from tools.event_track_v2x.prepare_recovery_task_poses import prepare, transform
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_configuration
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig, BeamRecoveryTracker
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
from transvision.models.event_track_v2x.recovery_task_scope import (
    SCOPE_MODE, VerifiedEgoPoseTable, allocation_weights, validate_scope,
)
from test_detection_cache_v2 import sources
from test_persistent_cache_stream import delivery, step
from test_forest_tracking import observation
from test_persistent_forest import brute
from test_persistent_component_tracking import joint_action


def make_poses(cache, root):
    for (seq,side,frame),(_,raw) in cache.index.items():
        if side!='vehicle-side':continue
        meta=json.loads(raw)
        # Nonzero lever arm deliberately distinguishes LiDAR from ego origin.
        r=np.array(meta['lidar_to_world_row_rotation']).T
        lt=np.array([.2,0.,0.])
        ego=np.array(meta['lidar_to_world_translation'])-r@lt
        values=dict(novatel_to_world=dict(rotation=r.tolist(),translation=ego[:,None].tolist()),
            lidar_to_novatel=dict(transform=dict(rotation=np.eye(3).tolist(),translation=lt[:,None].tolist())))
        for name,value in values.items():
            path=root/f'calib/{name}/{frame}.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(canonical(value))
    output=root/'poses.json'
    result=prepare(cache,root,output)
    return output,VerifiedEgoPoseTable(output,result['pose_table_sha256'],cache)


@pytest.fixture
def pose_inputs(sources,tmp_path):
    roots,calibration,inputs,output=sources
    checksum=build_cache(roots,calibration,sha_file(calibration),inputs,output)
    cache=VerifiedForestCache(output,checksum)
    path,table=make_poses(cache,tmp_path/'raw-navigation')
    return cache,path,table


def new_stream(cache,table,path,**options):
    t=BeamRecoveryTracker(path,sequence_id='0003',config=BeamRecoveryConfig(
        recovery_allocation_scope=SCOPE_MODE,**options))
    return PersistentForestCacheStream(cache,t,GeometryForestScorer(),origin_us=1_000_000,ego_poses=table)


def test_raw_pose_export_checks_composition_and_never_reads_GT(pose_inputs,monkeypatch):
    cache,path,table=pose_inputs
    data=json.loads(path.read_bytes());row=data['poses'][0]
    assert row['ego_translation_world']==[.8,2.,3.]
    assert row['information_us']==1_100_000 and row['state_us']==1_000_000
    assert len(data['poses'])==2 and not data['gt_model_inputs']
    with pytest.raises(FileExistsError):prepare(cache,path.parent,path)
    raw=path.parent/'calib/novatel_to_world/000120.json'
    value=json.loads(raw.read_bytes());value['translation'][0][0]+=1
    raw.write_bytes(canonical(value))
    with pytest.raises(ValueError,match='composition'):prepare(cache,path.parent,path.parent/'invalid.json')
    assert not (path.parent/'invalid.json').exists()


@pytest.mark.parametrize('bad',['hash','GT','missing','duplicate','pose_extra','state','information','cache','nan'])
def test_pose_table_refuses_changed_or_unbound_inputs(pose_inputs,tmp_path,bad):
    cache,path,_=pose_inputs;data=json.loads(path.read_bytes())
    if bad=='GT':data['gt_model_inputs']=True
    elif bad=='missing':data['poses'].pop()
    elif bad=='duplicate':data['poses'].append(data['poses'][0])
    elif bad=='pose_extra':data['poses'][0]['GT_track_id']='forbidden'
    elif bad=='state':data['poses'][0]['state_us']+=1
    elif bad=='information':data['poses'][0]['information_us']=0
    elif bad=='cache':data['poses'][0]['cache_frame_sha256']='f'*64
    elif bad=='nan':data['poses'][0]['ego_translation_world']=[True,2.,3.]
    changed=tmp_path/'bad.json';changed.write_bytes(canonical(data))
    with pytest.raises(ValueError):VerifiedEgoPoseTable(changed,'f'*64 if bad=='hash' else sha_file(changed),cache)


def test_receipt_gates_pose_with_missing_stale_duplicate_and_resume(pose_inputs,tmp_path,monkeypatch):
    cache,_,table=pose_inputs;s=new_stream(cache,table,tmp_path/'scope.db')
    first=step(s,[delivery(s,'infrastructure-side')])
    scope=first.tracking_audit['recovery_task_scope']
    assert scope['pose'] is None and scope['fallback']=='missing_arrived_pose'
    assert first.ingestion_audit['new_observations']==2
    vehicle=delivery(s,arrival=1_200_000)
    second=step(s,[vehicle],reference=1_200_000,event='vehicle')
    scope=second.tracking_audit['recovery_task_scope']
    assert scope['fallback'] is None and scope['pose']['arrival_us']==1_200_000
    assert scope['pose_age_us']==200_000 and scope['pose']['ego_translation_world']==[.8,2.,3.]
    assert not second.tracking_audit['recovery_scope_changes_decision_loss']
    assert len(second.tracking_audit['decision_indices'])==4 and s.tracker.n==4
    duplicate=step(s,[replace(vehicle,arrival_us=1_300_000)],reference=1_300_000,event='duplicate')
    assert duplicate.tracking_audit['recovery_task_scope']['pose']['arrival_us']==1_200_000
    old=s.tracker;checksum=old.close()
    t=BeamRecoveryTracker.open(old.path,expected_database_sha256=checksum,
        expected_prediction_sha256=duplicate.prediction['commit_sha256'])
    resumed=PersistentForestCacheStream(cache,t,GeometryForestScorer(),origin_us=1_000_000,ego_poses=table)
    assert step(resumed,[delivery(resumed,'infrastructure-side')])==first
    stale=step(resumed,reference=3_100_000,event='stale')
    assert stale.tracking_audit['recovery_task_scope']['fallback']=='stale_arrived_pose'
    assert t.n==4 and t.db.execute('SELECT count(*) FROM cache_receipts').fetchone()[0]==2
    t.close()


def test_future_vehicle_pose_cannot_enter_before_cache_payload(pose_inputs,tmp_path,monkeypatch):
    cache,_,table=pose_inputs;s=new_stream(cache,table,tmp_path/'future.db')
    monkeypatch.setattr(cache,'load_arrived',lambda *a:pytest.fail('future payload read'))
    with pytest.raises(ValueError):step(s,[delivery(s,arrival=1_200_000)])
    assert s.tracker.n==0
    s.tracker.close()


def test_scope_failure_rolls_back_receipt_and_all_raw_state(pose_inputs,tmp_path,monkeypatch):
    cache,_,table=pose_inputs;s=new_stream(cache,table,tmp_path/'rollback.db')
    before=tuple(s.tracker.db.iterdump());original=s.tracker._component_inference
    def fail(*a,**kw):
        original(*a,**kw);raise RuntimeError('scoped failure')
    monkeypatch.setattr(s.tracker,'_component_inference',fail)
    with pytest.raises(RuntimeError,match='scoped failure'):step(s,[delivery(s)])
    assert tuple(s.tracker.db.iterdump())==before and s.tracker._event_recovery_scope is None
    monkeypatch.setattr(s.tracker,'_component_inference',original)
    result=step(s,[delivery(s)])
    assert result.tracking_audit['recovery_task_scope']['pose'] is not None
    forged=dict(result.tracking_audit['recovery_task_scope'])
    forged['pose']=dict(forged['pose'],arrival_us=1_200_000)
    with pytest.raises(ValueError,match='future or invalid'):
        validate_scope(forged,s.tracker,result.ingestion_audit,1_100_000,1_100_000)
    s.tracker.close()


def test_disabled_recovery_remains_byte_identical_with_scope(pose_inputs,tmp_path):
    cache,_,table=pose_inputs
    scoped=new_stream(cache,table,tmp_path/'scoped.db',enable_recovery=False)
    base=BeamRecoveryTracker(tmp_path/'base.db',sequence_id='0003',config=BeamRecoveryConfig(enable_recovery=False))
    original=PersistentForestCacheStream(cache,base,GeometryForestScorer(),origin_us=1_000_000)
    for i,time in enumerate((1_100_000,1_200_000,3_100_000)):
        a,b=[step(s,[delivery(s)] if i==0 else (),reference=time,event=str(i)) for s in (original,scoped)]
        assert a.prediction_json==b.prediction_json
        assert a.tracking_audit['components']==b.tracking_audit['components']
        assert a.tracking_audit['decision_indices']==b.tracking_audit['decision_indices']
        assert a.tracking_audit['factor_rows_sha256']==b.tracking_audit['factor_rows_sha256']
    base.close();scoped.tracker.close()


def test_allocation_only_raw_scope_preserves_decoder_scope_and_uses_no_branch_state():
    obs=[observation('inside',49.),observation('boundary',50.),observation('outside',51.)]
    mean=list(obs[2].mean);mean[7]=-2.
    obs[2]=replace(obs[2],mean=mean)
    t=SimpleNamespace(store=SimpleNamespace(members=lambda c:{1:(0,1),2:(2,)}[c]),_observation=lambda i:obs[i])
    summaries={1:dict(weight=2/3,decision_indices=(0,1)),2:dict(weight=1/3,decision_indices=(0,))}
    scope=dict(fallback=None,pose=dict(ego_translation_world=[0.,0.,0.]))
    before=canonical(summaries)
    weights,counts=allocation_weights(t,summaries,scope,2_000_000)
    assert counts=={1:1,2:1} and weights=={1:.5,2:.5}
    assert canonical(summaries)==before
    scope['pose']['ego_translation_world']=[1000.,0.,0.]
    assert allocation_weights(t,summaries,scope,2_000_000)==({1:0.,2:0.},{1:0,2:0})
    scope['fallback']='missing_arrived_pose'
    assert allocation_weights(t,summaries,scope,2_000_000)==({1:2/3,2:1/3},None)


@pytest.mark.parametrize('center',[0.,1000.])
def test_actual_search_scope_keeps_all_detections_support_and_global_regret(tmp_path,center):
    t=BeamRecoveryTracker(tmp_path/'actual.db',sequence_id='0003',config=BeamRecoveryConfig(
        state=ForestTrackingConfig(active_limit=1),recovery_budget=10,recovery_allocation_scope=SCOPE_MODE))
    raw=tuple(observation(str(i),0. if i<3 else 100.,source=1 if i%3==2 else 0,
                          frame=str(i//3),index=i%3) for i in range(6))
    rows=(((-1,0.),),((-1,0.),),((-1,-3.),(0,0.),(1,-1.)),
          ((-1,0.),),((-1,0.),),((-1,-3.),(3,0.),(4,-1.)))
    deliveries=[dict(sequence_id='0003',side=side,frame_id=frame,arrival_us=1_000_010,frame_sha256='a'*64)
                for side in ('vehicle-side','infrastructure-side') for frame in ('0','1')]
    pose=dict(sequence_id='0003',frame_id='0',state_us=1_000_000,information_us=1_000_000,
        arrival_us=1_000_010,cache_frame_sha256='a'*64,ego_translation_world=[center,0.,0.],
        novatel_to_world_sha256='b'*64,lidar_to_novatel_sha256='c'*64)
    scope=dict(kind='arrived_vehicle_pose_recovery_allocation_v1',mode=SCOPE_MODE,pose_table_sha256='d'*64,
        radius_m=50.,pose=pose,fallback=None,pose_age_us=100_000,
        motion_policy='hold_last_received_ego_position',raw_point_policy='independent_constant_velocity',
        decision_loss_scope_changed=False,input_detections_filtered=False,gt_model_inputs=False)
    ingestion=dict(kind='persistent_cache_ingestion_v1',configuration_sha256='e'*64,new_deliveries=deliveries,
                   recovery_task_scope=scope)
    result=t.step(raw,rows,frame_id='event',event_id='event',reference_us=1_100_000,decision_us=1_100_000,
                  cache_ingestion=ingestion)
    assert result.audit['decision_indices']==list(range(6)) and t.n==6
    assert result.prediction['predictions'] and all(s['complete_raw_support_retained'] for s in result.audit['components'])
    assert any(p['mean'][0]==100. for p in result.prediction['predictions'])
    assert all(s['weight']==.5 for s in result.audit['components'])
    spent=[s['recovery_steps'] for s in result.audit['components']]
    assert (spent[0]>0 and spent[1]==0) if center==0. else spent==[0,0]
    probability,labels,_=brute(ForestFactors(tuple(o.node for o in raw),rows))
    action=joint_action(t,result)
    def risk(a):return sum(p*sum(labels[a][i]!=labels[h][i] for i in range(6))/6 for h,p in probability.items())
    assert risk(action)-min(map(risk,probability))<=result.audit['model_regret_upper']+1e-12
    t.close()


@pytest.mark.parametrize('kind',['rotation','translation','extra'])
def test_raw_transform_rejects_reflection_nonfinite_or_unknown_fields(kind):
    value=dict(rotation=np.eye(3).tolist(),translation=[[0.],[0.],[0.]])
    if kind=='rotation':value['rotation'][0][0]=-1.
    elif kind=='translation':value['translation'][0][0]=float('nan')
    else:value['GT_ids']=[]
    with pytest.raises(ValueError):transform(value)


def test_task_pose_mode_is_explicit_and_not_accepted_by_other_backends(pose_inputs,tmp_path):
    cache,_,table=pose_inputs
    with pytest.raises(ValueError):diagnostic_configuration('node_beam',task_scoped_recovery=True)
    with pytest.raises(ValueError):BeamRecoveryConfig(recovery_allocation_scope='oracle')
    for backend in ('beam_recovery','beam_recovery_disabled'):
        assert diagnostic_configuration(backend,task_scoped_recovery=True).recovery_allocation_scope==SCOPE_MODE
    tracker=BeamRecoveryTracker(tmp_path/'reject.db',sequence_id='0003')
    with pytest.raises(ValueError,match='exclusively'):
        PersistentForestCacheStream(cache,tracker,GeometryForestScorer(),origin_us=0,ego_poses=table)
    tracker.close()
