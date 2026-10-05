"""No-network controls for publication identity and create-once dispatch."""
import copy
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import types

import pytest


@pytest.fixture
def code(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(root))
    pub=importlib.import_module('publish_final_refit_teacher_prerequisite')
    dispatch=importlib.import_module('submit_final_refit_capacity_teacher')
    for name,module in [('publisher',pub),('dispatcher',dispatch)]:
        own=tmp_path/name;own.mkdir();(own/'module.py').write_text('software fixture\n')
        (own/'source-freeze.json').write_text('{}\n')
        monkeypatch.setattr(module,'__file__',str(own/'module.py'))
        monkeypatch.setattr(module,'R',tmp_path)
        monkeypatch.setattr(module,'register',lambda *args:None)
    (tmp_path/'receipts').mkdir();(tmp_path/'artifacts').mkdir()
    monkeypatch.setattr(dispatch,'JOURNAL',tmp_path/'receipts/dispatch.json')
    return pub,dispatch


def proof(module):
    return dict(kind=module.PREREQUISITE_KIND,seed=1337,main_task_id='1'*32,
        main_acceptance_sha256='2'*64,byte_admission_sha256='3'*64,final_model_sha256='4'*64,
        sequence_proof_sha256={f'sequence-{i:02d}.json':'5'*64 for i in range(46)},
        main_prerequisite_verified=True,teacher_runtime_or_targets_admitted=False,
        teacher_task_created=False,learned_Stage2_complete=False,paper_performance_complete=False)


class FakeTask:
    TaskTypes=types.SimpleNamespace(data_processing='data',inference='inference')
    tasks={};enqueued=[];fail_upload=False;fail_config=False

    def __init__(self,name):
        self.id=f'{len(type(self).tasks)+1:032x}';self.name=name;self.status='created'
        self.artifacts={};self.params={};self.data=types.SimpleNamespace(script=types.SimpleNamespace(diff=''))

    @classmethod
    def create(cls,**kw):
        task=cls(kw['task_name']);cls.tasks[task.id]=task;return task

    @classmethod
    def get_tasks(cls,**kw):return [t for t in cls.tasks.values() if re.search(kw['task_name'],t.name)]

    @classmethod
    def get_task(cls,task_id):return cls.tasks[task_id]

    @classmethod
    def enqueue(cls,task,queue_id):cls.enqueued.append((task.id,queue_id));task.status='queued'

    def set_parameters(self,value):self.params={'General/'+k:v for k,v in value.items()}
    def get_parameters(self):return self.params
    def set_script(self,**kw):
        if self.fail_config:raise RuntimeError('injected configuration failure')
        self.data.script.diff=kw['diff']
    def set_base_docker(self,*args,**kw):pass
    def set_packages(self,value):pass
    def add_tags(self,value):pass
    def reload(self):pass
    def mark_completed(self,force):self.status='completed'
    def upload_artifact(self,key,artifact_object,wait_on_upload):
        if self.fail_upload:raise RuntimeError('injected upload failure')
        raw=artifact_object.read_bytes()
        self.artifacts[key]=types.SimpleNamespace(hash=hashlib.sha256(raw).hexdigest(),size=len(raw),raw=raw)
        return True


@pytest.fixture(autouse=True)
def isolated_tasks():
    FakeTask.tasks={};FakeTask.enqueued=[];FakeTask.fail_upload=False;FakeTask.fail_config=False


@pytest.mark.parametrize('mutation',['extra-source','partial','bad-hash','teacher-claimed'])
def test_only_exact_complete_main_proof_can_be_published(code,mutation):
    pub,_=code;value=proof(pub)
    if mutation=='extra-source':value['internal_source']='must not upload'
    elif mutation=='partial':value['sequence_proof_sha256'].pop('sequence-45.json')
    elif mutation=='bad-hash':value['byte_admission_sha256']='not-a-hash'
    elif mutation=='teacher-claimed':value['teacher_runtime_or_targets_admitted']=True
    with pytest.raises(AssertionError):pub.proof_bytes(value)


def publisher_args(tmp_path,execute=True):
    return types.SimpleNamespace(seed=1337,execute=execute,output=tmp_path/'artifacts/pub',
        main_admission=tmp_path/'main.json',main_byte_admission=tmp_path/'main-byte.json')


def mock_readback(monkeypatch,pub):
    def read(task,dest):
        artifact=task.artifacts[pub.KEY];dest.write_bytes(artifact.raw)
        return dict(sha256=artifact.hash,bytes=artifact.size)
    monkeypatch.setattr(pub,'read_artifact',read)


def test_publication_dry_run_never_creates_task_or_output(code,tmp_path):
    pub,_=code;args=publisher_args(tmp_path,False)
    pub.run(args,FakeTask,proof(pub))
    assert not args.output.exists() and not FakeTask.tasks


def test_publication_requires_independent_bytes_and_reuses_exact_completed_task(code,tmp_path,monkeypatch):
    pub,_=code;args=publisher_args(tmp_path);mock_readback(monkeypatch,pub)
    pub.run(args,FakeTask,proof(pub));pub.run(args,FakeTask,proof(pub))
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued
    receipt=args.output/'independent-publication.json'
    spec,local=pub.publication(receipt,proof(pub),FakeTask)
    assert local.read_bytes()==pub.proof_bytes(proof(pub)) and spec['key']==pub.KEY
    local.write_bytes(b'changed')
    with pytest.raises(AssertionError):pub.publication(receipt,proof(pub),FakeTask)


def test_failed_publication_is_not_automatically_recreated(code,tmp_path,monkeypatch):
    pub,_=code;args=publisher_args(tmp_path);mock_readback(monkeypatch,pub);FakeTask.fail_upload=True
    with pytest.raises(RuntimeError):pub.run(args,FakeTask,proof(pub))
    assert (args.output/'task-created-before-upload.json').is_file() and (args.output/'failure.json').is_file()
    with pytest.raises(AssertionError):pub.run(args,FakeTask,proof(pub))
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued


def dispatch_inputs(module,tmp_path,monkeypatch,execute=True):
    producer=tmp_path/'producer';producer.mkdir();(producer/'bootstrap.py').write_text('# frozen test bootstrap\n')
    monkeypatch.setattr(module,'PRODUCER',producer)
    base=dict(seed=1337,world_size=4,bootstrap_sha256=module.sha(producer/'bootstrap.py'),
        final_refit_model_sha256='4'*64,configuration=dict(allocation='teacher',method='rbf'))
    args=types.SimpleNamespace(seed=1337,execute=execute,main_admission=tmp_path/'main.json',main_byte_admission=tmp_path/'main-byte.json')
    publication=tmp_path/'published.json';publication.write_text('{}')
    choice=dict(queue_id='q4',queue_name='GPU4-V100',world_size=4,
        bindings=[['machine',[0,1,2,3]]],eligible_workers=['machine-V100:gpu0,1,2,3'])
    fleet=dict(workers=[dict(id=choice['eligible_workers'][0])])
    monkeypatch.setattr(module,'core_jobs',lambda:([],[]))
    monkeypatch.setattr(module,'snapshot',lambda *args:([copy.deepcopy(choice)],copy.deepcopy(fleet)))
    return args,base,publication,choice,fleet


def test_semantic_identity_excludes_card_count_but_not_model_or_limits(code):
    _,module=code;base=dict(seed=1337,world_size=4,model='a',limits={'work':100})
    assert module.semantic(base)==module.semantic(dict(base,world_size=8))
    assert module.semantic(base)!=module.semantic(dict(base,model='b'))
    assert module.semantic(base)!=module.semantic(dict(base,limits={'work':101}))


def test_dispatch_is_create_once_and_retains_failed_task(code,tmp_path,monkeypatch):
    _,module=code;args,base,publication,_,_=dispatch_inputs(module,tmp_path,monkeypatch)
    module.dispatch(args,FakeTask,None,base,publication)
    task=next(iter(FakeTask.tasks.values()));task.status='failed'
    module.dispatch(args,FakeTask,None,dict(base,world_size=8),publication)
    assert len(FakeTask.tasks)==len(FakeTask.enqueued)==1
    stored=json.loads(module.JOURNAL.read_bytes())['jobs'][0]
    assert stored['task_id']==task.id and stored['status']=='queued'
    assert task.status=='failed'


@pytest.mark.parametrize('block',['priority','busy','L40','configuration-failure','binding-changed'])
def test_dispatch_respects_prerequisites_and_preserves_created_task(code,tmp_path,monkeypatch,block):
    _,module=code;args,base,publication,choice,fleet=dispatch_inputs(module,tmp_path,monkeypatch)
    if block=='priority':monkeypatch.setattr(module,'core_jobs',lambda:([],[{'seed':3407}]))
    elif block=='busy':monkeypatch.setattr(module,'snapshot',lambda *args:([],fleet))
    elif block=='L40':
        choice['eligible_workers']=['L40S:gpu0,1,2,3'];fleet['workers']=[dict(id=choice['eligible_workers'][0])]
        monkeypatch.setattr(module,'snapshot',lambda *args:([choice],fleet))
    elif block=='configuration-failure':FakeTask.fail_config=True
    elif block=='binding-changed':
        calls=iter([([choice],fleet),([],fleet)])
        monkeypatch.setattr(module,'snapshot',lambda *args:next(calls))
    if block in ('configuration-failure','binding-changed'):
        with pytest.raises((RuntimeError,AssertionError)):module.dispatch(args,FakeTask,None,base,publication)
        assert len(FakeTask.tasks)==1
        assert json.loads(module.JOURNAL.read_bytes())['jobs'][0]['automatic_retry'] is False
    else:
        module.dispatch(args,FakeTask,None,base,publication)
        assert not FakeTask.tasks and not module.JOURNAL.exists()
    assert not FakeTask.enqueued


def test_real_physical_intersection_filter_rejects_busy_eight_card_worker(code):
    _,module=code
    from submit_rbf_final_identity import available
    now=datetime.datetime.now(datetime.timezone.utc)
    queues=[dict(id='q4',name='GPU4-5090',entries=[]),dict(id='q8',name='GPU8-5090',entries=[])]
    workers=[dict(id='machine:gpu0,1,2,3',ip='host',queues=[dict(id='q4')],last_activity_time=now.isoformat()),
        dict(id='machine:gpu0,1,2,3,4,5,6,7',ip='host',queues=[dict(id='q8')],last_activity_time=now.isoformat(),task=dict(id='occupied'))]
    assert available(workers,queues,now)==[]
