from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.event_track_v2x.collect_allocation_training import verified_train_rows
from tools.event_track_v2x.run_train_inference_diagnostic import BACKENDS, diagnostic_configuration
from tools.event_track_v2x.audit_train_inference_comparison import compare, inspect
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV
from test_forest_training_data import prepared_rows

ROOT=Path(__file__).resolve().parents[2]


def pair_file(tmp_path,rows):
    pairs=[dict(vehicle_sequence=r['sequence_id'],infrastructure_sequence=r['sequence_id'],
        vehicle_frame=r['vehicle_frame'],infrastructure_frame=r['infrastructure_frame']) for r in rows]
    path=tmp_path/'cooperative.json'
    path.write_bytes(canonical(pairs))
    return path,pairs


@pytest.mark.parametrize('bad',['full_claim','hash','cross_sequence','duplicate','omitted_sequence'])
def test_shared_train_schedule_rejects_wrong_cohort_or_metadata(prepared_rows,tmp_path,bad):
    _,_,cache,rows=prepared_rows
    path,pairs=pair_file(tmp_path,rows)
    expected=sha_file(path)
    if bad=='cross_sequence': pairs[0]['infrastructure_sequence']='different'
    elif bad=='duplicate': pairs.append(pairs[0])
    elif bad=='omitted_sequence': pairs=pairs[:1]
    elif bad=='hash': expected='f'*64
    path.write_bytes(canonical(pairs))
    if bad not in ('hash','full_claim'): expected=sha_file(path)
    with pytest.raises(ValueError):
        verified_train_rows(cache,path,expected,require_full_train=bad=='full_claim')


def test_real_fresh_process_fixture_comparison_and_teacher_purity(prepared_rows,tmp_path):
    data,_,cache,rows=prepared_rows
    path,_=pair_file(tmp_path,rows)
    assert verified_train_rows(cache,path,sha_file(path),require_full_train=False)==rows
    fitted=fit_dataset(data,sha_file(data/'manifest.json'),tmp_path/'fit',
        config=FitConfig(epochs=1,batch_size=4,hidden=8,heads=2,dropout=0.),require_full_train=False)
    cp=fitted['seeds'][0]
    checkpoint=(tmp_path/'fit'/cp['checkpoint_manifest']).parent
    common=[str(cache.root),cache.manifest_sha256,str(path),sha_file(path),str(checkpoint),cp['checkpoint_sha256'],rows[0]['sequence_id']]
    code='''import sys
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from tools.event_track_v2x.run_train_inference_diagnostic import run
c,h,m,mh,p,ph,s,b,o=sys.argv[1:]
run(VerifiedForestCache(c,h),m,mh,p,ph,o,sequence=s,backend=b,allow_fixture=True)
'''
    runs=[]
    for backend in BACKENDS:
        output=tmp_path/backend
        process=subprocess.run([sys.executable,'-c',code,*common,backend,str(output)],
            cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stdout+process.stderr
        runs.append((output,sha_file(output/'development-inference-receipt.json')))
    config=PersistentCompletionConfig()
    scorer,_=load_identity_checkpoint(checkpoint,cp['checkpoint_sha256'],config=config.state)
    teacher=tmp_path/'teacher'
    replay_rows(cache,rows[:1],teacher,config,learned_scorer=scorer,allocation_teacher=True,
        plan=dict(class_scope=['car'],configuration=asdict(config),identity_checkpoint_sha256=cp['checkpoint_sha256']))
    reference=(teacher,sha_file(teacher/'receipt.json'))
    result=compare(runs,tmp_path/'comparison',teacher=reference,allow_fixture=True)
    assert result['frames']==1 and result['actual_factors_identical']
    assert result['teacher_probe_equivalence']['byte_identical']
    assert result['distinct_reported_process_ids'] and result['same_reported_runtime']
    assert not result['paper_eligible'] and not result['real_tracking_evaluation']
    assert result['identity_counts_are_not_id_switches'] and not result['timing_repeated']
    with pytest.raises(ValueError,match='provenance'):
        compare(runs,tmp_path/'cannot-relabel-fixture')
    assert not (tmp_path/'cannot-relabel-fixture').exists()
    # The new ranking backend is additional, not a silent replacement for the
    # three original predeclared backends or their preserved failure evidence.
    output=tmp_path/'class-bound-joint'
    process=subprocess.run([sys.executable,'-c',code,*common,'class_bound_joint_beam',str(output)],
        cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
    new=inspect(output,sha_file(output/'development-inference-receipt.json'),allow_fixture=True)
    previous=inspect(*runs[2],allow_fixture=True)
    assert new['factor_stream_sha256']==previous['factor_stream_sha256']
    assert new['predictions_sha256']==previous['predictions_sha256']
    assert new['plan']['configuration']['max_batch_frontier']==previous['plan']['configuration']['max_batch_frontier']
    assert not json.loads((output/'receipt.json').read_bytes())['trained_paper_method']
    # Node-wise pruning is an explicitly weaker additional diagnostic, not a
    # completed replacement for the failed joint-current-batch baseline.
    output=tmp_path/'node-beam'
    process=subprocess.run([sys.executable,'-c',code,*common,'node_beam',str(output)],
        cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
    node=inspect(output,sha_file(output/'development-inference-receipt.json'),allow_fixture=True)
    assert node['factor_stream_sha256']==previous['factor_stream_sha256']
    assert node['plan']['backend']=='node_beam' and node['fallback_events']==0
    assert 'node_beam' not in BACKENDS
    assert not json.loads((output/'receipt.json').read_bytes())['complete_batch_beam_ranking']
    for backend in ('beam_recovery','beam_recovery_disabled'):
        output=tmp_path/backend
        process=subprocess.run([sys.executable,'-c',code,*common,backend,str(output)],
            cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stdout+process.stderr
        current=inspect(output,sha_file(output/'development-inference-receipt.json'),allow_fixture=True)
        assert current['factor_stream_sha256']==node['factor_stream_sha256']
        assert current['fallback_events']==0 and backend not in BACKENDS
        receipt=json.loads((output/'receipt.json').read_bytes())
        assert receipt['additional_beam_recovery_enabled'] is (backend=='beam_recovery')
        if backend=='beam_recovery_disabled':
            assert current['predictions_sha256']==node['predictions_sha256']
    from test_recovery_task_scope import make_poses
    pose_path,pose_table=make_poses(cache,tmp_path/'pose-sidecar')
    scoped_code=code.replace('allow_fixture=True',
        'allow_fixture=True,ego_pose_table='+repr(str(pose_path))+',ego_pose_table_sha256='+repr(pose_table.manifest_sha256))
    for backend in ('beam_recovery','beam_recovery_disabled'):
        output=tmp_path/(backend+'-task-scoped')
        process=subprocess.run([sys.executable,'-c',scoped_code,*common,backend,str(output)],
            cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stdout+process.stderr
        scoped=inspect(output,sha_file(output/'development-inference-receipt.json'),allow_fixture=True)
        assert scoped['plan']['ego_pose_table_sha256']==pose_table.manifest_sha256
        assert scoped['factor_stream_sha256']==node['factor_stream_sha256']
        assert scoped['plan']['configuration']['recovery_allocation_scope']=='arrived_vehicle_raw_xy50'
        if backend=='beam_recovery_disabled':assert scoped['predictions_sha256']==node['predictions_sha256']
    # The explicit control changes decoding only, not frozen scoring or inputs.
    control_code=code.replace('allow_fixture=True','allow_fixture=True,max_model_regret=1.')
    for backend,(default_path,default_sha) in zip(BACKENDS[:2],runs[:2]):
        output=tmp_path/(backend+'-no-threshold-fallback')
        process=subprocess.run([sys.executable,'-c',control_code,*common,backend,str(output)],
            cwd=ROOT,env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
        assert process.returncode==0,process.stdout+process.stderr
        control=inspect(output,sha_file(output/'development-inference-receipt.json'),allow_fixture=True)
        default=inspect(default_path,default_sha,allow_fixture=True)
        assert control['plan']['configuration']['state']['max_model_regret']==1.
        assert control['plan']['conditional_bayes_without_threshold_fallback']
        assert control['factor_stream_sha256']==default['factor_stream_sha256']
        assert control['fallback_events']==0
    with (runs[0][0]/'tracking.jsonl').open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError,match='artifact identity'):
        inspect(*runs[0],allow_fixture=True)


@pytest.mark.parametrize('backend',BACKENDS[:2])
def test_decision_control_changes_only_requested_threshold(backend):
    default=asdict(diagnostic_configuration(backend))
    control=asdict(diagnostic_configuration(backend,1.))
    assert default['state']['max_model_regret']==.05
    default['state']['max_model_regret']=1.
    assert control==default


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1.,1.01,'1'])
def test_invalid_decision_control_rejected(value):
    with pytest.raises(ValueError): diagnostic_configuration('component_completion',value)


def test_beam_cannot_accept_ignored_decision_override():
    with pytest.raises(ValueError): diagnostic_configuration('joint_beam',1.)
    with pytest.raises(ValueError): diagnostic_configuration('class_bound_joint_beam',1.)


def test_beam_recovery_controls_only_change_explicit_switch():
    left=asdict(diagnostic_configuration('beam_recovery'))
    right=asdict(diagnostic_configuration('beam_recovery_disabled'))
    assert left.pop('enable_recovery') is True and right.pop('enable_recovery') is False
    assert left==right
    from transvision.models.event_track_v2x.resource_sweep import configuration
    for backend,wrong in (('beam_recovery',False),('beam_recovery_disabled',True)):
        with pytest.raises(ValueError,match='switch differ'):
            configuration(dict(backend=backend,configuration=dict(enable_recovery=wrong)))


def test_comparison_refuses_different_actual_factors_before_output(tmp_path,monkeypatch):
    from tools.event_track_v2x import audit_train_inference_comparison as tool
    plan=dict(cache_sha256='a',cooperative_metadata_sha256='b',identity_checkpoint_sha256='c',
        configuration=dict(state={}),cohort_mode='fixture-only',selected_schedule=[],source_sha256={})
    records=[dict(backend=backend,plan=plan,factor_stream_sha256=str(i)) for i,backend in enumerate(BACKENDS[:2])]
    monkeypatch.setattr(tool,'inspect',lambda p,h,**kw: records[p])
    with pytest.raises(ValueError,match='actual factor stream'):
        tool.compare([(0,'x'),(1,'y')],tmp_path/'wrong',allow_fixture=True)
    assert not (tmp_path/'wrong').exists()


def test_failed_backend_is_retained_and_cannot_be_silently_omitted(tmp_path,monkeypatch):
    from tools.event_track_v2x import audit_train_inference_comparison as tool
    events=[dict(sequence_id='s',frame_id=str(i),factor_rows_sha256=str(i),
        output_payload_sha256='p',output_ids_sha256='i') for i in range(2)]
    common=dict(cache_sha256='a',cooperative_metadata_sha256='b',identity_checkpoint_sha256='c',
        configuration=dict(state={}),cohort_mode='fixture-only',selected_schedule=[],source_sha256={})
    records=[dict(backend=backend,plan=dict(common,backend=backend,runtime=dict(pid=i,host='fixture')),
        factor_stream_sha256='same',events=events) for i,backend in enumerate(BACKENDS[:2])]
    monkeypatch.setattr(tool,'inspect',lambda p,h,**kw: records[p])
    directory=tmp_path/'failed';directory.mkdir()
    (directory/'plan.json').write_bytes(canonical(dict(common,backend='joint_beam')))
    failure=dict(status='failed',partial_outputs_not_final_results=True,completed_frames=1,
        scheduled_frames=2,plan_sha256=sha_file(directory/'plan.json'),error='injected capacity failure')
    (directory/'failure.json').write_bytes(canonical(failure))
    audit=dict(tracking=dict(sequence_id='s',event_id='0',factor_rows_sha256='0'))
    (directory/'tracking.jsonl').write_bytes(canonical(audit)+b'\n')
    for name in ('predictions.jsonl','frame-timings.jsonl'): (directory/name).write_bytes(b'')
    references=[(0,'a'),(1,'b')]
    with pytest.raises(ValueError,match='no silent omissions'):
        tool.compare(references,tmp_path/'omitted',allow_fixture=True)
    result=tool.compare(references,tmp_path/'partial',allow_fixture=True,
        failed_runs=[(directory,sha_file(directory/'failure.json'))])
    assert result['analysis_complete'] and not result['complete_backend_comparison']
    assert result['status']=='partial_due_to_failed_runs'
    assert result['failed_runs'][0]['completed_prefix_actual_factors_match']
    assert result['failed_runs'][0]['metrics_not_computed']
    audit['tracking']['factor_rows_sha256']='different'
    (directory/'tracking.jsonl').write_bytes(canonical(audit)+b'\n')
    with pytest.raises(ValueError,match='different actual factors'):
        tool.compare(references,tmp_path/'changed',allow_fixture=True,
            failed_runs=[(directory,sha_file(directory/'failure.json'))])
