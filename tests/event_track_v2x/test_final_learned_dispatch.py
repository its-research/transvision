"""No-network controls for the final learned-priority GPU dispatcher."""
import copy
import datetime
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from test_final_refit_teacher_dispatch import FakeTask


@pytest.fixture
def code(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    m=importlib.import_module('submit_final_refit_learned_replay')
    (tmp_path/'receipts').mkdir();(tmp_path/'artifacts').mkdir()
    own=tmp_path/'code';own.mkdir();(own/'module.py').write_text('# software fixture\n');(own/'source-freeze.json').write_text('{}')
    monkeypatch.setattr(m,'R',tmp_path);monkeypatch.setattr(m,'JOURNAL',tmp_path/'receipts/learned.json')
    monkeypatch.setattr(m,'__file__',str(own/'module.py'));monkeypatch.setattr(m,'register',lambda *a:None)
    FakeTask.tasks={};FakeTask.enqueued=[];FakeTask.fail_config=False;FakeTask.fail_upload=False
    return m


def dispatch_inputs(m,tmp_path,monkeypatch,execute=True):
    producer=tmp_path/'producer';producer.mkdir();(producer/'bootstrap.py').write_text('# frozen fixture\n')
    monkeypatch.setattr(m,'PRODUCER',producer)
    base=dict(seed=1337,world_size=4,bootstrap_sha256=m.sha(producer/'bootstrap.py'),model='final-fixture',configuration=dict(method='rbf',allocation='learned'))
    publication=tmp_path/'priority-publication.json';publication.write_text('{}')
    args=N(seed=1337,execute=execute,priority_publication=publication)
    choice=dict(queue_id='q4',queue_name='GPU4-V100',world_size=4,bindings=[['host',[0,1,2,3]]],eligible_workers=['host-V100:gpu0,1,2,3'])
    fleet=dict(workers=[dict(id=choice['eligible_workers'][0])])
    monkeypatch.setattr(m,'reservations',lambda:([],[]))
    monkeypatch.setattr(m,'snapshot',lambda *a:([copy.deepcopy(choice)],copy.deepcopy(fleet)))
    return args,base,choice,fleet


def test_actual_original_plan_changes_only_named_priority_roles(code):
    m=code;r=Path('/Volumes/Data/test/recover-before-fuse')
    parent=r/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004/preparation.json'
    main=json.loads(parent.read_bytes())['seeds'][0]['plan']
    control=json.loads((m.PRODUCER/'source-control.json').read_bytes())
    publication=dict(kind='rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1',
        seed=main['seed'],independent_cloud_bytes_verified=True,learned_replay_accepted=False,
        paper_performance_complete=False,artifacts={k:{} for k in ('audit','checkpoint','weights')},policy_signature='fixture')
    result=m.expected_plan(main,publication,control,'p'*64,'d'*64)
    changed={'world_size','bootstrap_sha256','dispatcher_sha256','exclusive_source_freeze_sha256','exclusive_patches','configuration','scope'}
    assert all(result[k]==v for k,v in main.items() if k not in changed)
    assert result['configuration']==dict(main['configuration'],allocation='learned')
    assert result['source_replacements']==main['source_replacements']
    assert result['priority_publication']==publication
    assert m.semantic(result)==m.semantic(dict(result,world_size=8))
    assert m.semantic(result)!=m.semantic(dict(result,priority_policy_signature='other-policy'))
    bad=copy.deepcopy(publication);bad['independent_cloud_bytes_verified']=False
    with pytest.raises(AssertionError):m.expected_plan(main,bad,control,'p'*64,'d'*64)


def test_default_does_not_create_or_enqueue(code,tmp_path,monkeypatch):
    args,base,_,_=dispatch_inputs(code,tmp_path,monkeypatch,False)
    code.dispatch(args,FakeTask,None,base)
    assert not FakeTask.tasks and not FakeTask.enqueued and not code.JOURNAL.exists()


def test_failed_attempt_reused_across_GPU_cardinality(code,tmp_path,monkeypatch):
    args,base,_,_=dispatch_inputs(code,tmp_path,monkeypatch)
    code.dispatch(args,FakeTask,None,base)
    task=next(iter(FakeTask.tasks.values()));task.status='failed'
    code.dispatch(args,FakeTask,None,dict(base,world_size=8))
    assert len(FakeTask.tasks)==len(FakeTask.enqueued)==1 and task.status=='failed'
    assert len(list((tmp_path/'receipts').glob('*create-intent.json')))==1


@pytest.mark.parametrize('condition',['core-priority','busy','L40','configuration-failure','GPU-changed','unknown-create'])
def test_resource_and_creation_failures_preserved(code,tmp_path,monkeypatch,condition):
    args,base,choice,fleet=dispatch_inputs(code,tmp_path,monkeypatch)
    if condition=='core-priority':monkeypatch.setattr(code,'reservations',lambda:([],[dict(seed=3407)]))
    elif condition=='busy':monkeypatch.setattr(code,'snapshot',lambda *a:([],fleet))
    elif condition=='L40':
        choice['eligible_workers']=['L40S:gpu0,1,2,3'];fleet['workers']=[dict(id=choice['eligible_workers'][0])]
        monkeypatch.setattr(code,'snapshot',lambda *a:([choice],fleet))
    elif condition=='configuration-failure':FakeTask.fail_config=True
    elif condition=='GPU-changed':
        calls=iter([([choice],fleet),([],fleet)])
        monkeypatch.setattr(code,'snapshot',lambda *a:next(calls))
    elif condition=='unknown-create':
        def unknown(**kw):raise TimeoutError('unknown create outcome')
        monkeypatch.setattr(FakeTask,'create',unknown)
    if condition in ('configuration-failure','GPU-changed','unknown-create'):
        with pytest.raises((RuntimeError,AssertionError,TimeoutError)):code.dispatch(args,FakeTask,None,base)
        if condition=='unknown-create':
            with pytest.raises(FileExistsError):code.dispatch(args,FakeTask,None,base)
            assert not code.JOURNAL.exists()
        else:
            job=json.loads(code.JOURNAL.read_bytes())['jobs'][0]
            assert job['automatic_retry'] is False and len(FakeTask.tasks)==1
    else:
        code.dispatch(args,FakeTask,None,base)
        assert not FakeTask.tasks and not code.JOURNAL.exists()
    assert not FakeTask.enqueued


def test_eight_card_worker_physically_blocks_overlapping_four_card_worker(code):
    from submit_rbf_final_identity import available
    now=datetime.datetime.now(datetime.timezone.utc)
    queues=[dict(id='q4',name='GPU4-5090',entries=[]),dict(id='q8',name='GPU8-5090',entries=[])]
    workers=[dict(id='machine:gpu0,1,2,3',ip='host',queues=[dict(id='q4')],last_activity_time=now.isoformat()),
        dict(id='machine:gpu0,1,2,3,4,5,6,7',ip='host',queues=[dict(id='q8')],last_activity_time=now.isoformat(),task=dict(id='running'))]
    assert available(workers,queues,now)==[]
