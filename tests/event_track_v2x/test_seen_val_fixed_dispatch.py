"""No-network create-once dispatch and physical reservation checks."""
import copy
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
b=importlib.import_module('submit_seen_val_fixed_baselines')
from test_final_refit_teacher_dispatch import FakeTask
from test_seen_val_fixed_baselines import fixture as remote_fixture


@pytest.fixture
def dispatch_case(tmp_path,monkeypatch):
    (tmp_path/'receipts').mkdir();own=tmp_path/'code';own.mkdir();(own/'source-freeze.json').write_text('{}')
    (own/'dispatch.py').write_text('# fixture')
    producers=tmp_path/'producer'
    for K in (1,4):
        p=producers/f'K{K}';p.mkdir(parents=True);(p/'bootstrap.py').write_text(f'# K{K} fixture\n')
    monkeypatch.setattr(b,'__file__',str(own/'dispatch.py'));monkeypatch.setattr(b,'R',tmp_path)
    monkeypatch.setattr(b,'PRODUCERS',producers);monkeypatch.setattr(b,'JOURNAL',tmp_path/'receipts/dispatch.json')
    monkeypatch.setattr(b,'register',lambda *a:None)
    monkeypatch.setattr(b,'source_gate',lambda:None);monkeypatch.setattr(b,'remote_gate',lambda *a:None)
    monkeypatch.setattr(b,'other_jobs',lambda:([],[]))
    FakeTask.tasks={};FakeTask.enqueued=[];FakeTask.fail_upload=False;FakeTask.fail_config=False
    args=N(K=4,seed=2027,execute=True)
    for key in b.PATH_ARGUMENTS:
        path=tmp_path/(key+'.json');path.write_text('{}');setattr(args,key,path)
    base=dict(seed=2027,baseline_K=4,world_size=4,configuration=dict(allocation='bound'),
        bootstrap_sha256=b.sha(producers/'K4/bootstrap.py'))
    monkeypatch.setattr(b,'qualify',lambda *a:dict(base,world_size=4))
    choice=dict(queue_id='q8',queue_name='GPU8-V100',world_size=8,bindings=[['host',list(range(8))]],
        eligible_workers=['host:gpu0,1,2,3,4,5,6,7'])
    fleet=dict(workers=[dict(id=choice['eligible_workers'][0])])
    monkeypatch.setattr(b,'snapshot',lambda *a:([choice],fleet))
    return args,base,choice,fleet


def test_missing_prerequisites_stop_before_any_network(dispatch_case,tmp_path,monkeypatch,capsys):
    args,_,_,_=dispatch_case
    args.baseline_admission.unlink()
    monkeypatch.setattr(b,'qualify',lambda *a:pytest.fail('missing prerequisite cannot access network'))
    argv=['dispatcher','--K','4','--seed','2027']
    for key in b.PATH_ARGUMENTS:argv+=['--'+key.replace('_','-'),str(getattr(args,key))]
    monkeypatch.setattr(sys,'argv',argv+['--execute']);b.main()
    assert not FakeTask.tasks and not b.JOURNAL.exists()
    assert 'baseline_admission' in capsys.readouterr().out


def test_readonly_default_does_not_write_intent_or_create_task(dispatch_case):
    args,base,_,_=dispatch_case;args.execute=False;b.dispatch(args,FakeTask,None,base)
    assert not FakeTask.tasks and not b.JOURNAL.exists()
    assert not list((b.R/'receipts').glob('*intent*'))


@pytest.mark.parametrize('status',['created','queued','in_progress','completed','failed','stopped'])
def test_same_K_seed_deduplicates_across_world_size_and_terminal_status(dispatch_case,status):
    args,base,_,_=dispatch_case;b.dispatch(args,FakeTask,None,base)
    task=next(iter(FakeTask.tasks.values()));task.status=status
    b.dispatch(args,FakeTask,None,dict(base,world_size=8))
    assert len(FakeTask.tasks)==len(FakeTask.enqueued)==1 and task.status==status
    job=json.loads(b.JOURNAL.read_bytes())['jobs'][0]
    assert job['K']==4 and job['plan']['world_size']==8 and job['input_receipt_sha256']
    assert list((b.R/'receipts').glob('*intent*'))


def test_K_and_seed_remain_distinct_semantic_experiments(dispatch_case):
    _,base,_,_=dispatch_case
    assert b.semantic(base)==b.semantic(dict(base,world_size=8))
    assert b.semantic(base)!=b.semantic(dict(base,seed=1337))
    assert b.semantic(base)!=b.semantic(dict(base,baseline_K=1))


@pytest.mark.parametrize('block',['priority','busy','L40','configuration','GPU-changed','proof-changed','source-changed'])
def test_revalidate_before_enqueue_and_preserve_failed_attempt(dispatch_case,monkeypatch,block):
    args,base,choice,fleet=dispatch_case
    if block=='priority':monkeypatch.setattr(b,'other_jobs',lambda:([],[dict(seed=1337)]))
    elif block=='busy':monkeypatch.setattr(b,'snapshot',lambda *a:([],fleet))
    elif block=='L40':
        choice['eligible_workers']=['L40S:gpu0,1,2,3'];fleet['workers']=[dict(id=choice['eligible_workers'][0])]
    elif block=='configuration':FakeTask.fail_config=True
    elif block=='GPU-changed':
        replies=iter([([choice],fleet),([],fleet)]);monkeypatch.setattr(b,'snapshot',lambda *a:next(replies))
    elif block=='proof-changed':monkeypatch.setattr(b,'qualify',lambda *a:dict(base,configuration=dict(changed=True)))
    elif block=='source-changed':
        def reject():raise AssertionError('source changed')
        monkeypatch.setattr(b,'source_gate',reject)
    if block in ('priority','busy','L40'):
        b.dispatch(args,FakeTask,None,base);assert not FakeTask.tasks
    else:
        with pytest.raises((AssertionError,RuntimeError)):b.dispatch(args,FakeTask,None,base)
        assert len(FakeTask.tasks)==1
        job=json.loads(b.JOURNAL.read_bytes())['jobs'][0]
        assert job['automatic_retry'] is False and job['dispatch_failure_type']
    assert not FakeTask.enqueued


def test_unknown_creation_outcome_preserves_intent_and_blocks_retry(dispatch_case,monkeypatch):
    args,base,_,_=dispatch_case;calls=[]
    def unknown(**kw):calls.append(kw);raise RuntimeError('creation response lost')
    monkeypatch.setattr(FakeTask,'create',unknown)
    with pytest.raises(RuntimeError):b.dispatch(args,FakeTask,None,base)
    assert len(list((b.R/'receipts').glob('*intent*')))==1
    with pytest.raises(AssertionError):b.dispatch(args,FakeTask,None,base)
    assert len(calls)==1 and not FakeTask.enqueued


def test_unjournaled_remote_duplicate_is_not_replaced(dispatch_case):
    args,base,_,_=dispatch_case
    name=f'RBF final-refit complete seen-val fixed K4 seed2027 '+b.semantic(base)[:16]
    FakeTask.create(task_name=name)
    with pytest.raises(AssertionError):b.dispatch(args,FakeTask,None,base)
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued


@pytest.mark.parametrize('occupied',['worker','queued','reservation'])
def test_physical_GPU4_GPU8_overlap_blocks_selection(occupied):
    from submit_rbf_final_identity import available
    now=datetime.datetime.now(datetime.timezone.utc)
    qs=[dict(id='q4',name='GPU4-5090',entries=[]),dict(id='q8',name='GPU8-5090',entries=[])]
    workers=[dict(id='host:gpu0,1,2,3',ip='physical-host',queues=[dict(id='q4')],last_activity_time=now.isoformat()),
        dict(id='host:gpu0,1,2,3,4,5,6,7',ip='physical-host',queues=[dict(id='q8')],last_activity_time=now.isoformat())]
    reserved=[]
    if occupied=='worker':workers[1]['task']=dict(id='other-job')
    elif occupied=='queued':qs[1]['entries']=[dict(task='queued-job')]
    else:reserved=[('physical-host',{0,1,2,3,4,5,6,7})]
    assert available(workers,qs,now,reserved)==[]


@pytest.mark.parametrize('width',[1,4])
def test_execute_actual_frozen_remote_gate_against_verified_local_cloud_fixture(tmp_path,width):
    plan,tasks,_,_,_=remote_fixture(tmp_path,width)
    # Recover the exact bytes consumed by the independent input publication gate.
    # The fixture wrote none yet; its main data envelope points to known real
    # binding/transport/review files and a synthetic completed main receipt.
    from test_seen_val_forest_runtime import gate_fixture
    _,_,values,_,_=gate_fixture(tmp_path)
    directory=tmp_path/'independent-cloud-bytes';directory.mkdir()
    for name,raw in values.items():(directory/name).write_bytes(raw)
    b.remote_gate(plan,tmp_path/'publication.json',N(get_task=lambda task_id:tasks[task_id]))
    tasks['fixture-baseline'].status='in_progress'
    with pytest.raises(AssertionError):b.remote_gate(plan,tmp_path/'publication.json',N(get_task=lambda task_id:tasks[task_id]))
