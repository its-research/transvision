"""Finite rejection controls, never full learned replay or paper evidence."""
import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace as N

import pytest


@pytest.fixture
def code(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
    return importlib.import_module('rbf_final_refit_learned_output_binding')


def fixtures():
    policy='fixture-policy';ck='1'*64
    proof=dict(artifacts={'checkpoint':{'sha256':ck},'audit':{'sha256':'2'*64},'weights':{'sha256':'3'*64}},policy_signature=policy,
               upstream_tasks={k:{'task_id':k+'-task'} for k in ('main','teacher')})
    plan=dict(seed=1337,world_size=4,priority_publication=proof,priority_inputs=proof['artifacts'],priority_policy_signature=policy,
              configuration={'allocation':'learned'},cache_manifest={'sha256':'4'*64})
    binding=dict(kind='rbf_final_refit_learned_replay_priority_input_binding_v1',publication=proof,artifacts=proof['artifacts'],
        policy_signature=policy,local_numeric_audit_sha256='2'*64,learned_replay_accepted=False,paper_performance_complete=False,
        main_task_id='main-task',teacher_task_id='teacher-task')
    report=dict(kind='rbf_final_refit_learned_priority_full_train_forest_candidate_v1',
        ranks=[dict(rank=i,seed=1337,priority_policy_signature=policy,priority_checkpoint_sha256=ck) for i in range(4)])
    per=dict(kind='rbf_paper_replay_v1',expected_sequences=['0000'],expected_events=2,fixture=False,
        protocol={'split':'train','dataset':'spd'},configuration=plan['configuration'],cache_sha256='4'*64,
        model_binding=dict(seed=1337,fit_split='train',dataset='spd',priority_policy_signature=policy,priority_checkpoint_sha256=ck))
    receipt=dict(kind='rbf_paper_replay_receipt_v1',completed_sequences=['0000'],completed_events=2,fixture=False,
        files={k:'x' for k in ('plan.json','predictions.jsonl','audit.jsonl','timings.json','resources.json')},databases={'0000':{}})
    return plan,proof,binding,report,per,receipt


@pytest.mark.parametrize('mutation',[None,'rank-policy','rank-checkpoint','rank-seed','rank-duplicate','publication','upstream','claim'])
def test_priority_output_identity_all_ranks(code,mutation):
    plan,proof,binding,report,_,_=copy.deepcopy(fixtures())
    if mutation=='rank-policy':report['ranks'][3]['priority_policy_signature']='other'
    elif mutation=='rank-checkpoint':report['ranks'][1]['priority_checkpoint_sha256']='0'*64
    elif mutation=='rank-seed':report['ranks'][2]['seed']=3407
    elif mutation=='rank-duplicate':report['ranks'][3]['rank']=2
    elif mutation=='publication':binding['publication']={}
    elif mutation=='upstream':binding['teacher_task_id']='other'
    elif mutation=='claim':binding['learned_replay_accepted']=True
    if mutation:
        with pytest.raises(AssertionError):code.verify_priority_report(plan,report,binding,proof)
    else:code.verify_priority_report(plan,report,binding,proof)


@pytest.mark.parametrize('mutation',[None,'policy','checkpoint','sequence','fixture','split','events','cache','resources','database'])
def test_every_sequence_binds_real_fit_and_full_native_files(code,mutation):
    plan,_,_,_,per,receipt=copy.deepcopy(fixtures())
    if mutation=='policy':per['model_binding']['priority_policy_signature']='other'
    elif mutation=='checkpoint':per['model_binding']['priority_checkpoint_sha256']='0'*64
    elif mutation=='sequence':per['expected_sequences']=['0001']
    elif mutation=='fixture':receipt['fixture']=True
    elif mutation=='split':per['protocol']['split']='val'
    elif mutation=='events':receipt['completed_events']=1
    elif mutation=='cache':per['cache_sha256']='0'*64
    elif mutation=='resources':receipt['files'].pop('resources.json')
    elif mutation=='database':receipt['databases']['0001']={}
    if mutation:
        with pytest.raises(AssertionError):code.verify_priority_sequence(plan,per,receipt,'0000',2)
    else:code.verify_priority_sequence(plan,per,receipt,'0000',2)


def test_contained_outputs_reject_escape_and_symlink(code,tmp_path):
    root=tmp_path/'rank';root.mkdir();(root/'ok').write_bytes(b'bytes')
    assert code.contained(root,'ok')==root/'ok'
    (root/'link').symlink_to(root/'ok')
    for name in ('../rank/ok',str(root/'ok'),'link'):
        with pytest.raises(AssertionError):code.contained(root,name)


def generated():
    from prepare_final_refit_learned_readback import build,PARENT
    source,control=build(PARENT.read_text())
    return source,control


def test_generated_reader_keeps_download_extraction_and_factor_math(code):
    source,control=generated()
    assert control['observation_factor_comparison_block_identical']
    assert control['NN_atol']==control['NN_rtol']==1e-4
    actual=Path(code.__file__).with_name('read_final_refit_learned_outputs.py').read_text()
    assert source==actual
    # Only the fixed original factor computation may remain; no producer execution.
    assert 'learned_expansion_order_independently_verified=False' in source
    assert 'full_forest_semantics_or_fresh_state_independently_accepted=False' in source
    assert 'verify_priority_sequence(plan,per_plan,receipt' in source
    assert "contained(directory,database['path'])" in source


@pytest.mark.parametrize('mutation',[None,'raw-hash','row-identity','parents','score','nonfinite','clock','extra-row','missing-row'])
def test_preserved_full_factor_loop_rejects_output_corruption(code,mutation):
    source,_=generated();tree=ast.parse(source)
    norm=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalized_factors')
    env={'math':math};exec(compile(ast.Module(body=[norm],type_ignores=[]),'<independent-normalization>','exec'),env)
    logits=[2.,-3.];values=env['normalized_factors'](logits)
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE observations (i INTEGER,node_id TEXT,raw BLOB,sha TEXT)')
    db.execute('CREATE TABLE potentials (i INTEGER,p INTEGER,w REAL)')
    raw=dict(features=[0.]*203,state_us=100,node={'information_us':90,'arrival_us':100})
    if mutation=='clock':raw['features'][141]=1.
    blob=json.dumps(raw).encode();digest=hashlib.sha256(blob).hexdigest()
    if mutation!='missing-row':db.execute('INSERT INTO observations VALUES (?,?,?,?)',(0,'node',blob,'bad' if mutation=='raw-hash' else digest))
    for parent,value in zip((-1,0),values):
        if mutation=='parents' and parent==0:parent=5
        if mutation=='score' and parent==-1:value+=.01
        if mutation=='nonfinite' and parent==-1:value=float('inf')
        db.execute('INSERT INTO potentials VALUES (?,?,?)',(0,parent,value))
    if mutation=='extra-row':db.execute('INSERT INTO observations VALUES (?,?,?,?)',(1,'extra',blob,digest))
    reference=dict(row=0,node_id='wrong' if mutation=='row-identity' else 'node',context_indices=[0,1],logits=logits)
    env.update(json=json,hashlib=hashlib,db=db,expected={'0000':[reference]},sequence='0000',events={'origin_us_by_sequence':{'0000':100}},
               commits=[None]*2,checked=[],database={'sha256':'fixture'})
    block=source[source.index('                count=0;maximum=0.'):source.index('            finally:')]
    import textwrap
    try:
        if mutation:
            with pytest.raises((AssertionError,IndexError)):exec(textwrap.dedent(block),env)
        else:
            exec(textwrap.dedent(block),env)
            assert env['checked'][0]['nodes']==1 and env['checked'][0]['max_factor_difference_to_admitted_final_refit_forward']==0
    finally:db.close()


@pytest.mark.parametrize('mutation',[None,'status','source','recipe','plan','artifact','extra'])
def test_remote_snapshot_must_still_match_after_download(code,mutation):
    source='fixture-source';plan={'bootstrap_sha256':hashlib.sha256(source.encode()).hexdigest()}
    job={'plan':plan,'recipe_sha256':'recipe'};spec={'sha256':'a'*64,'bytes':5}
    params={'General/plan':json.dumps(plan),'General/recipe_sha256':'recipe'}
    task=N(status='completed',data=N(script=N(diff=source)),artifacts={'receipt':N(hash=spec['sha256'],size=5)},reload=lambda:None,get_parameters=lambda:params)
    if mutation=='status':task.status='in_progress'
    elif mutation=='source':task.data.script.diff='changed'
    elif mutation=='recipe':params['General/recipe_sha256']='other'
    elif mutation=='plan':params['General/plan']='{}'
    elif mutation=='artifact':task.artifacts['receipt'].size=4
    elif mutation=='extra':task.artifacts['extra']=N(hash='x',size=1)
    if mutation:
        with pytest.raises(AssertionError):code.verify_task_unchanged(task,job,{'receipt':spec},'completed')
    else:code.verify_task_unchanged(task,job,{'receipt':spec},'completed')


@pytest.mark.parametrize('mutation',[None,'allocation','new-limits','seed','publication-bytes','task-recipe'])
def test_qualification_rebinds_exact_frozen_main_and_publication(code,tmp_path,monkeypatch,mutation):
    d=code.dispatch
    root=Path('/Volumes/Data/test/recover-before-fuse/source-freezes')
    main=json.loads((root/'rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004/preparation.json').read_bytes())['seeds'][0]['plan']
    control=json.loads((d.PRODUCER/'source-control.json').read_bytes())
    proof=dict(kind='rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1',seed=main['seed'],
        independent_cloud_bytes_verified=True,learned_replay_accepted=False,paper_performance_complete=False,
        artifacts={k:{} for k in ('audit','checkpoint','weights')},policy_signature='fixture')
    path=tmp_path/'publication.json';path.write_bytes(d.canonical(proof))
    main_recipe=hashlib.sha256(d.canonical(main)).hexdigest()
    context={'upstream_tasks':{'main':{'task_id':'main','recipe_sha256':main_recipe}}}
    calls=[]
    def qualify(args,Task):calls.append('fit-qualified');return {},context
    def verify(*a):calls.append('publication-readback');return proof
    monkeypatch.setattr(code.publication,'qualify',qualify)
    monkeypatch.setattr(code.publication,'verify_publication',verify)
    base=d.expected_plan(main,proof,control,code.sha(d.PRODUCER/'bootstrap.py'),code.sha(d.__file__))
    plan=dict(base,world_size=8);identity=d.semantic(base)
    job=dict(seed=main['seed'],task_id='learned',plan=plan,allocation_variant=d.VARIANT,
        semantic_learned_identity=identity,recipe_sha256=hashlib.sha256(d.canonical(plan)).hexdigest(),
        priority_publication=str(path),priority_publication_sha256=code.sha(path))
    params={'General/semantic_learned_identity':identity,'General/recipe_sha256':job['recipe_sha256'],'General/plan':d.canonical(plan).decode()}
    task=N(id='learned',data=N(script=N(diff=(d.PRODUCER/'bootstrap.py').read_text())),get_parameters=lambda:params)
    main_task=N(get_parameters=lambda:{'General/plan':d.canonical(main).decode()})
    Task=N(get_task=lambda task_id:main_task)
    if mutation=='allocation':job['plan']=copy.deepcopy(plan);job['plan']['configuration']['allocation']='bound'
    elif mutation=='new-limits':job['plan']=copy.deepcopy(plan);job['plan']['configuration']['limits']['new_limit']=999
    elif mutation=='seed':job['seed']=1337 if main['seed']!=1337 else 3407
    elif mutation=='publication-bytes':path.write_bytes(b'changed')
    elif mutation=='task-recipe':params['General/recipe_sha256']='other'
    args=N(seed=main['seed'],priority_publication=path)
    if mutation:
        with pytest.raises(AssertionError):code.qualify_job(args,Task,job,task,control)
    else:
        assert code.qualify_job(args,Task,job,task,control)==proof
        assert calls==['fit-qualified','publication-readback']
