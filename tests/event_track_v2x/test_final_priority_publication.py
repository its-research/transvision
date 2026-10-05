"""No-network publication controls; synthetic fixtures are not fitted models."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from test_final_refit_teacher_dispatch import FakeTask


@pytest.fixture
def code(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    pub=importlib.import_module('publish_final_refit_priority_checkpoint')
    (tmp_path/'receipts').mkdir();(tmp_path/'artifacts').mkdir()
    own=tmp_path/'publisher';own.mkdir();(own/'module.py').write_text('software fixture\n')
    (own/'source-freeze.json').write_text('{}\n')
    monkeypatch.setattr(pub,'R',tmp_path);monkeypatch.setattr(pub,'__file__',str(own/'module.py'))
    monkeypatch.setattr(pub,'register',lambda *args:None)
    FakeTask.tasks={};FakeTask.enqueued=[];FakeTask.fail_upload=False;FakeTask.fail_config=False
    return pub


def fixture_inputs(pub,tmp_path):
    run=tmp_path/'run';(run/'data').mkdir(parents=True);(run/'fit/1337').mkdir(parents=True)
    relatives=('completion.json','data/manifest.json','fit/plan.json','fit/receipt.json',
               'fit/1337/checkpoint.json','fit/1337/weights.npz','fit/1337/epochs.jsonl')
    for name in relatives:(run/name).write_bytes(b'software fixture: '+name.encode())
    audit=dict(kind='rbf_final_priority_local_full_export_and_selected_checkpoint_numeric_audit_v1',
        seed=1337,source_freeze_sha256=pub.AUDITOR_SHA,consumer_freeze_sha256=pub.CONSUMER_SHA,
        export=dict(events=7445,sequences={str(i):{} for i in range(46)},
            exact_export_features_and_targets_verified=True,all_export_groups_independently_scored=True),
        producer_model_or_optimizer_imported=False,learned_replay_accepted=False,
        strict_pipeline_isolated_selection=False,paper_performance_complete=False,
        selection=dict(atol=1e-8,rtol=1e-8),input_hashes={str(run/n):pub.sha(run/n) for n in relatives})
    path=tmp_path/'local-numeric-audit.json';path.write_bytes(pub.canonical(audit))
    return run,path,audit


@pytest.mark.parametrize('corrupt',[None,'partial','driver','consumer','inputs','extra-input','changed-bytes','tolerance','paper-claim'])
def test_only_unchanged_complete_audited_run_can_supply_payload(code,tmp_path,corrupt):
    pub=code;run,path,audit=fixture_inputs(pub,tmp_path)
    if corrupt=='partial':audit['export']['events']=7444
    elif corrupt=='driver':audit['source_freeze_sha256']='0'*64
    elif corrupt=='consumer':audit['consumer_freeze_sha256']='0'*64
    elif corrupt=='inputs':audit['input_hashes'].pop(str(run/'completion.json'))
    elif corrupt=='extra-input':audit['input_hashes'][str(tmp_path/'secret')]='0'*64
    elif corrupt=='changed-bytes':(run/'fit/1337/weights.npz').write_bytes(b'changed')
    elif corrupt=='tolerance':audit['selection']['atol']=1e-5
    elif corrupt=='paper-claim':audit['paper_performance_complete']=True
    if corrupt:
        with pytest.raises(AssertionError):pub.checked_inputs(run,path,audit)
    else:
        paths=pub.checked_inputs(run,path,audit)
        assert set(paths)=={'audit','checkpoint','weights'} and paths['audit']==path


def publication_inputs(pub,tmp_path,execute=True):
    run,path,audit=fixture_inputs(pub,tmp_path)
    paths=pub.checked_inputs(run,path,audit)
    context=dict(seed=1337,policy_signature='fixture-signature',upstream_tasks={
        'main':dict(task_id='1'*32,recipe_sha256='2'*64,bootstrap_sha256='3'*64,artifacts={'fixture':dict(bytes=1,sha256='4'*64)}),
        'teacher':dict(task_id='5'*32,recipe_sha256='6'*64,bootstrap_sha256='7'*64,artifacts={'fixture':dict(bytes=1,sha256='8'*64)})})
    context['qualified_payload']={role:dict(key=pub.KEYS[role],sha256=pub.sha(p),bytes=p.stat().st_size) for role,p in paths.items()}
    return N(seed=1337,execute=execute,output=tmp_path/'artifacts/publication'),paths,context


def mock_readback(pub,monkeypatch,*,fail=False):
    def read(task,key,destination):
        artifact=task.artifacts[key];destination.write_bytes(artifact.raw)
        if fail:raise AssertionError('injected independent byte mismatch')
        return dict(sha256=hashlib.sha256(artifact.raw).hexdigest(),bytes=len(artifact.raw))
    monkeypatch.setattr(pub,'read_artifact',read)


def test_default_never_uploads_or_creates_task(code,tmp_path):
    args,paths,context=publication_inputs(code,tmp_path,False)
    code.run(args,FakeTask,paths,context)
    assert not args.output.exists() and not FakeTask.tasks


def test_payload_rejects_extra_source_or_dataset(code,tmp_path):
    _,paths,context=publication_inputs(code,tmp_path)
    paths['internal-source']=tmp_path/'source.py'
    with pytest.raises(AssertionError):code.payload(paths,context)


@pytest.mark.parametrize('role',['audit','checkpoint','weights'])
def test_qualified_payload_change_refused_before_task_creation(code,tmp_path,role):
    args,paths,context=publication_inputs(code,tmp_path)
    paths[role].write_bytes(b'changed between qualification and upload')
    with pytest.raises(AssertionError):code.run(args,FakeTask,paths,context)
    assert not args.output.exists() and not FakeTask.tasks


def test_completed_three_artifact_publication_reused_exactly(code,tmp_path,monkeypatch):
    args,paths,context=publication_inputs(code,tmp_path);mock_readback(code,monkeypatch)
    code.run(args,FakeTask,paths,context);code.run(args,FakeTask,paths,context)
    assert len(FakeTask.tasks)==1 and not FakeTask.enqueued
    result=code.verify_publication(args.output/'independent-publication.json',paths,context,FakeTask)
    assert result['independent_cloud_bytes_verified'] is True and result['learned_replay_accepted'] is False
    assert set(result['artifacts'])=={'audit','checkpoint','weights'}
    assert all(x['task']==result['task_id'] for x in result['artifacts'].values())


@pytest.mark.parametrize('failure',['upload','readback'])
def test_failure_preserves_created_task_and_prevents_retry(code,tmp_path,monkeypatch,failure):
    args,paths,context=publication_inputs(code,tmp_path)
    mock_readback(code,monkeypatch,fail=failure=='readback');FakeTask.fail_upload=failure=='upload'
    with pytest.raises((RuntimeError,AssertionError)):code.run(args,FakeTask,paths,context)
    assert (args.output/'publication-intent.json').exists() and (args.output/'task-created-before-upload.json').exists()
    assert (args.output/'failure.json').exists()
    with pytest.raises(AssertionError):code.run(args,FakeTask,paths,context)
    assert len(FakeTask.tasks)==1 and next(iter(FakeTask.tasks.values())).status=='created'


@pytest.mark.parametrize('corrupt',['registered-hash','local-bytes','running','extra-artifact','upstream'])
def test_existing_publication_rejects_changed_evidence(code,tmp_path,monkeypatch,corrupt):
    args,paths,context=publication_inputs(code,tmp_path);mock_readback(code,monkeypatch)
    code.run(args,FakeTask,paths,context)
    task=next(iter(FakeTask.tasks.values()));receipt=args.output/'independent-publication.json'
    if corrupt=='registered-hash':task.artifacts[code.KEYS['weights']].hash='0'*64
    elif corrupt=='local-bytes':(args.output/'independent-cloud-bytes'/code.KEYS['weights']).write_bytes(b'changed')
    elif corrupt=='running':task.status='in_progress'
    elif corrupt=='extra-artifact':task.artifacts['extra']=N(hash='0'*64,size=1)
    elif corrupt=='upstream':context=copy.deepcopy(context);context['upstream_tasks']['main']['recipe_sha256']='changed'
    with pytest.raises(AssertionError):code.verify_publication(receipt,paths,context,FakeTask)
