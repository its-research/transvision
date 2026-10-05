"""Publication failure, live lineage and physical reservation software tests.

Synthetic transport never counts as an actual remote experiment or acceptance.
"""
import copy
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

from test_final_refit_teacher_dispatch import FakeTask, dispatch_inputs
from test_seen_val_forest_runtime import gate_fixture


@pytest.fixture
def code(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    pub = importlib.import_module('publish_rbf_seen_val_forest_inputs')
    dispatch = importlib.import_module('submit_rbf_seen_val_bound_forest')
    for label,module in [('publisher',pub),('dispatcher',dispatch)]:
        own = tmp_path/label;own.mkdir();(own/'module.py').write_text('fixture\n')
        (own/'source-freeze.json').write_text('{}')
        monkeypatch.setattr(module,'__file__',str(own/'module.py'))
        monkeypatch.setattr(module,'R',tmp_path)
        monkeypatch.setattr(module,'register',lambda *a:None)
    (tmp_path/'receipts').mkdir();(tmp_path/'artifacts').mkdir()
    monkeypatch.setattr(dispatch,'JOURNAL',tmp_path/'receipts/dispatch.json')
    FakeTask.tasks={};FakeTask.enqueued=[];FakeTask.fail_upload=False;FakeTask.fail_config=False
    return pub,dispatch


def inputs(pub,tmp_path,execute=True):
    records={}
    for role in pub.ROLES:
        path=tmp_path/(role+'.input');path.write_text('software fixture '+role)
        records[role]=dict(path=str(path),key=role,sha256=pub.sha(path),bytes=path.stat().st_size)
    expected=dict(kind=pub.KIND,seed=2027,artifacts={k:{f:v[f] for f in ('key','sha256','bytes')} for k,v in records.items()},
        destination=pub.DESTINATION,project=pub.PROJECT,full_train_main_local_gate={'software_only':True},
        upstream_main={'task_id':'software-main'})
    args=N(seed=2027,execute=execute,output=tmp_path/'artifacts/pub',main_admission=tmp_path/'main.json',main_byte_admission=tmp_path/'bytes.json')
    return args,records,expected


def reader(monkeypatch,pub,corrupt=False):
    def read(task,key,path):
        item=task.artifacts[key];path.write_bytes(item.raw+(b'changed' if corrupt else b''))
        return dict(sha256=pub.sha(path),bytes=path.stat().st_size)
    monkeypatch.setattr(pub,'read_artifact',read)


def test_dry_run_and_missing_main_never_create_or_upload(code,tmp_path,monkeypatch,capsys):
    pub,dispatch=code;args,records,expected=inputs(pub,tmp_path,False)
    pub.run(args,FakeTask,records,expected)
    assert not FakeTask.tasks and not args.output.exists()
    for module,last in [(pub,'output'),(dispatch,'publication')]:
        monkeypatch.setattr(module,'source_gate',lambda:None)
        monkeypatch.setattr(module,'prerequisites',lambda *a:pytest.fail('missing prerequisite must stop before network'))
        monkeypatch.setattr(sys,'argv',['program','--seed','2027','--main-admission',str(tmp_path/'absent'),
            '--main-byte-admission',str(tmp_path/'absent-byte'),'--'+last,str(tmp_path/'not-created'),'--execute'])
        module.main()
    assert not FakeTask.tasks and not (tmp_path/'not-created').exists()
    assert 'waiting_for_' in capsys.readouterr().out


def test_exact_seven_artifacts_read_back_and_reused(code,tmp_path,monkeypatch):
    pub,_=code;args,records,expected=inputs(pub,tmp_path);reader(monkeypatch,pub)
    pub.run(args,FakeTask,records,expected);pub.run(args,FakeTask,records,expected)
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued
    receipt=args.output/'independent-publication.json'
    value=pub.publication(receipt,expected,FakeTask)
    assert set(value['artifacts'])==pub.ROLES
    assert value['independent_cloud_bytes_verified'] is True
    (receipt.parent/'independent-cloud-bytes/cache_archive').write_text('tamper')
    with pytest.raises(AssertionError):pub.publication(receipt,expected,FakeTask)


@pytest.mark.parametrize('failure',['upload','readback','changed-local','missing-role'])
def test_failed_publication_retained_without_retry(code,tmp_path,monkeypatch,failure):
    pub,_=code;args,records,expected=inputs(pub,tmp_path);reader(monkeypatch,pub,failure=='readback')
    if failure=='upload':FakeTask.fail_upload=True
    if failure=='changed-local':Path(records['cache_archive']['path']).write_text('changed')
    if failure=='missing-role':records.pop('events')
    with pytest.raises((AssertionError,RuntimeError)):pub.run(args,FakeTask,records,expected)
    if failure=='missing-role':
        assert not FakeTask.tasks and not args.output.exists();return
    assert len(FakeTask.tasks)==1 and (args.output/'failure.json').is_file()
    assert not (args.output/'independent-publication.json').exists()
    with pytest.raises(AssertionError):pub.run(args,FakeTask,records,expected)
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued


@pytest.mark.parametrize('failure',['running','missing-artifact','extra-artifact','parameters','recipe','script'])
def test_manifest_requires_exact_completed_original_task(code,tmp_path,failure):
    pub,_=code;_,records,_=inputs(pub,tmp_path)
    plan=dict(world_size=4,bootstrap_sha256=hashlib.sha256(b'original').hexdigest())
    job=dict(seed=2027,task_id='main',plan=plan,recipe_sha256=hashlib.sha256(pub.canonical(plan)).hexdigest())
    task=N(id='main',status='completed',artifacts={k:N(hash='a'*64,size=1) for k in
        {'receipt','exclusive-source-manifest'}|{f'replay-rank{i}' for i in range(4)}},data=N(script=N(diff='original')))
    params={'General/plan':json.dumps(plan),'General/recipe_sha256':job['recipe_sha256']}
    task.get_parameters=lambda:params
    gate=dict(seed=2027,main_prerequisite_verified=True,main_acceptance_sha256=records['main_acceptance']['sha256'])
    api=N(get_task=lambda task_id:task)
    assert pub.manifest(2027,records,gate,job,api)['upstream_main']['task_id']=='main'
    if failure=='running':task.status='in_progress'
    if failure=='missing-artifact':task.artifacts.pop('replay-rank0')
    if failure=='extra-artifact':task.artifacts['extra']=N(hash='b'*64,size=1)
    if failure=='parameters':params['General/plan']=json.dumps(dict(plan,world_size=8))
    if failure=='recipe':params['General/recipe_sha256']='changed'
    if failure=='script':task.data.script.diff+='\n'
    with pytest.raises(AssertionError):pub.manifest(2027,records,gate,job,api)


def test_expected_plan_passes_actual_frozen_remote_gate_without_changing_kernel(code,tmp_path):
    pub,_=code;plan,tasks,values,_,_=gate_fixture(tmp_path)
    original=json.loads(tasks['fixture-main'].get_parameters()['General/plan'])
    binding=json.loads(values['binding']);main=dict(seed=plan['seed'],plan=original)
    expected=pub.expected_plan(main,binding,plan['seen_val_input_publication'],8)
    for key in ('configuration','checkpoint','weights_archive','source','source_replacements','exclusive_patches','CPU_capacity_candidate_admission'):
        assert expected[key]==original[key]
    assert expected['bootstrap_sha256']==pub.BOOTSTRAP_SHA and expected['evaluation_scope']==pub.SCOPE
    assert expected['forward_outputs']==binding['forward_artifacts']
    directory=tmp_path/'independent-cloud-bytes';directory.mkdir()
    for key,raw in values.items():(directory/key).write_bytes(raw)
    api=N(get_task=lambda task_id:tasks[task_id])
    pub.remote_gate(expected,tmp_path/'publication.json',api)
    tasks['fixture-main'].status='in_progress'
    with pytest.raises(AssertionError):pub.remote_gate(expected,tmp_path/'publication.json',api)


def dispatch_fixture(code,tmp_path,monkeypatch):
    _,module=code
    result=dispatch_inputs(module,tmp_path,monkeypatch)
    args,base,publication,choice,fleet=result
    base['configuration']['allocation']='bound'
    monkeypatch.setattr(module,'remote_gate',lambda *a:None)
    return result


def test_dispatch_deduplicates_across_GPU_count_preserving_failed_job(code,tmp_path,monkeypatch):
    _,module=code;args,base,path,_,_=dispatch_fixture(code,tmp_path,monkeypatch)
    module.dispatch(args,FakeTask,None,base,path)
    task=next(iter(FakeTask.tasks.values()));task.status='failed'
    module.dispatch(args,FakeTask,None,dict(base,world_size=8),path)
    assert len(FakeTask.tasks)==len(FakeTask.enqueued)==1 and task.status=='failed'
    assert module.semantic(base)!=module.semantic(dict(base,seed=2027))
    assert module.semantic(base)!=module.semantic(dict(base,configuration=dict(allocation='learned')))


@pytest.mark.parametrize('block',['priority','busy','L40','configuration-failure','binding-changed','main-changed'])
def test_dispatch_rechecks_resources_and_upstream_before_enqueue(code,tmp_path,monkeypatch,block):
    _,module=code;args,base,path,choice,fleet=dispatch_fixture(code,tmp_path,monkeypatch)
    if block=='priority':monkeypatch.setattr(module,'core_jobs',lambda:([],[dict(seed=3407)]))
    if block=='busy':monkeypatch.setattr(module,'snapshot',lambda *a:([],fleet))
    if block=='L40':
        choice['eligible_workers']=['L40S:gpu0,1,2,3'];fleet['workers']=[dict(id=choice['eligible_workers'][0])]
        monkeypatch.setattr(module,'snapshot',lambda *a:([choice],fleet))
    if block=='configuration-failure':FakeTask.fail_config=True
    if block=='binding-changed':
        choices=iter([([choice],fleet),([],fleet)]);monkeypatch.setattr(module,'snapshot',lambda *a:next(choices))
    if block=='main-changed':
        calls=[]
        def gate(*a):
            calls.append(1)
            assert len(calls)==1,'upstream main changed after creation'
        monkeypatch.setattr(module,'remote_gate',gate)
    if block in ('configuration-failure','binding-changed','main-changed'):
        with pytest.raises((AssertionError,RuntimeError)):module.dispatch(args,FakeTask,None,base,path)
        assert len(FakeTask.tasks)==1
        assert json.loads(module.JOURNAL.read_bytes())['jobs'][0]['automatic_retry'] is False
    else:
        module.dispatch(args,FakeTask,None,base,path);assert not FakeTask.tasks
    assert not FakeTask.enqueued


@pytest.mark.parametrize('occupied',['worker','queued'])
def test_real_GPU4_GPU8_physical_overlap_blocks_dispatch(code,occupied):
    from submit_rbf_final_identity import available
    now=datetime.datetime.now(datetime.timezone.utc)
    queues=[dict(id='q4',name='GPU4-5090',entries=[]),dict(id='q8',name='GPU8-5090',entries=[])]
    workers=[dict(id='host:gpu0,1,2,3',ip='one-physical-host',queues=[dict(id='q4')],last_activity_time=now.isoformat()),
        dict(id='host:gpu0,1,2,3,4,5,6,7',ip='one-physical-host',queues=[dict(id='q8')],last_activity_time=now.isoformat())]
    if occupied=='worker':workers[1]['task']=dict(id='already-running')
    else:queues[1]['entries']=[dict(task='queued')]
    assert available(workers,queues,now)==[]
