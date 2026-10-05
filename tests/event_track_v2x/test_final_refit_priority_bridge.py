"""Network-free final teacher provenance controls, not runtime admission."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def code(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    return importlib.import_module('run_final_refit_priority_after_targets')


def task(plan,script,artifacts):
    params={'General/plan':json.dumps(plan),'General/recipe_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    return NS(status='completed',reload=lambda:None,get_parameters=lambda:params,
        data=NS(script=NS(diff=script)),artifacts={k:NS(hash=v['sha256'],size=v['bytes']) for k,v in artifacts.items()})


def example(code,tmp_path,monkeypatch):
    monkeypatch.setattr(code,'TEACHER_BOOTSTRAP',hashlib.sha256(b'teacher').hexdigest())
    monkeypatch.setattr(code,'MAIN_BOOTSTRAP',hashlib.sha256(b'main').hexdigest())
    published=tmp_path/'published.json';published.write_text('{}')
    asset=dict(task='asset',key='data',sha256='a'*64,bytes=3)
    original={k:copy.deepcopy(asset) for k in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest')}
    original.update(forward_outputs=[asset],source_replacements={'runtime':'hash'},world_size=4,
        final_refit_model_sha256='f'*64,original_nested_model_sha256='e'*64,
        configuration=dict(allocation='bound',method='rbf',cap=100),bootstrap_sha256=code.MAIN_BOOTSTRAP)
    ref=dict(task='published',key='proof',sha256=code.sha(published),bytes=published.stat().st_size)
    plan=dict(copy.deepcopy(original),world_size=8,configuration=dict(original['configuration'],allocation='teacher'),
        bootstrap_sha256=code.TEACHER_BOOTSTRAP,final_main_prerequisite_artifact=ref)
    recipe=hashlib.sha256(code.canonical(plan)).hexdigest()
    artifacts={'rank':dict(sha256='c'*64,bytes=5)}
    job=dict(task_id='teacher',plan=plan,recipe_sha256=recipe,allocation_variant='final_refit_exclusive_capacity_raw_witness_teacher_v1')
    main=dict(task_id='main')
    byte=dict(artifacts=artifacts,recipe_sha256=hashlib.sha256(code.canonical(original)).hexdigest())
    target=dict(task_id='teacher',recipe_sha256=recipe,registered_artifacts=artifacts)
    binding=dict(task_id='teacher',original_plan=copy.deepcopy(plan))
    tasks={'main':task(original,'main',artifacts),'teacher':task(plan,'teacher',artifacts),
        'asset':task({},'asset',{'data':asset}),'published':task({},'proof',{'proof':ref})}
    return tasks,job,main,byte,target,binding,published


def test_exact_registered_inputs_accept_different_qualified_world_size(code,tmp_path,monkeypatch):
    tasks,*args=example(code,tmp_path,monkeypatch)
    assert code.live_provenance(tasks.__getitem__,*args) is tasks['teacher']


@pytest.mark.parametrize('mutation',('running','failed','teacher_code','main_code','teacher_plan','artifact_hash',
    'extra_artifact','input_bytes','publication','search_cap','model','legacy_variant'))
def test_changed_runtime_or_upstream_rejected(code,tmp_path,monkeypatch,mutation):
    tasks,job,main,byte,target,binding,published=example(code,tmp_path,monkeypatch)
    if mutation in ('running','failed'):tasks['teacher'].status=mutation
    elif mutation=='teacher_code':tasks['teacher'].data.script.diff='changed'
    elif mutation=='main_code':tasks['main'].data.script.diff='changed'
    elif mutation=='teacher_plan':tasks['teacher'].get_parameters()['General/plan']='{}'
    elif mutation=='artifact_hash':tasks['main'].artifacts['rank'].hash='wrong'
    elif mutation=='extra_artifact':tasks['teacher'].artifacts['extra']=NS(hash='wrong',size=5)
    elif mutation=='input_bytes':tasks['asset'].artifacts['data'].size=4
    elif mutation=='publication':published.write_text('{"changed":true}')
    elif mutation=='legacy_variant':job['allocation_variant']='legacy'
    else:
        plan=job['plan']
        if mutation=='search_cap':plan['configuration']['cap']=101
        else:plan['final_refit_model_sha256']='changed'
        binding['original_plan']=copy.deepcopy(plan)
        job['recipe_sha256']=target['recipe_sha256']=hashlib.sha256(code.canonical(plan)).hexdigest()
        tasks['teacher']=task(plan,'teacher',target['registered_artifacts'])
    with pytest.raises(ValueError):code.live_provenance(tasks.__getitem__,job,main,byte,target,binding,published)


def test_old_runtime_admission_cannot_be_reused(code,tmp_path):
    path=tmp_path/'old.json';path.write_text(json.dumps(dict(kind='rbf_capacity_priority_v7_Linux_CPU_import_runtime_independent_byte_admission_v1')))
    with pytest.raises(ValueError):code.runtime_gate(path,lambda _:pytest.fail('must reject before network'))


def test_complete_frozen_consumer_imports_without_training(code,monkeypatch):
    root=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005')
    if not root.exists():pytest.skip('host frozen source fixture not installed')
    source,proof=code.source_gate(root)
    assert proof['trainer_sha256']==code.TRAINER_SHA
    assert proof['actual_teacher_export_executed'] is proof['priority_fit_started'] is False


def test_observation_error_does_not_stop_or_restart_live_child(code,tmp_path,monkeypatch,capsys):
    epochs=tmp_path/'epochs.jsonl';epochs.write_text('{"epoch":9}\n')
    statuses=iter((None,0));launches=[]
    child=NS(poll=lambda:next(statuses))
    monkeypatch.setattr(code.subprocess,'Popen',lambda *a,**k:launches.append((a,k)) or child)
    monkeypatch.setattr(code.time,'sleep',lambda _:None)
    assert code.monitored(['fixture'],cwd=tmp_path,log=tmp_path/'log',stage='fixture',epoch_path=epochs)>=0
    assert len(launches)==1
    rows=[json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert all(row['ETA_seconds'] is None for row in rows)
