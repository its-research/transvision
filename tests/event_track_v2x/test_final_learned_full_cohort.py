"""Full-cohort admission and wiring tests; no real cohort claim."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def code(monkeypatch):
    monkeypatch.syspath_prepend(os.environ.get('RBF_LEARNED_FULL_COHORT_SOURCE_DIRECTORY',
        str(Path(__file__).resolve().parents[2] / 'tools/event_track_v2x')))
    return importlib.import_module('rbf_final_refit_learned_forest_binding')


def fixture(code, world=4):
    plan = dict(seed=1337, method='rbf', world_size=world, scoring_atol=1e-4, scoring_rtol=1e-4,
        checkpoint={'sha256':'c'*64}, final_refit_model_sha256='f'*64, original_nested_model_sha256='o'*64,
        numeric_reference_admission_sha256='n'*64, priority_policy_signature='p'*64,
        priority_inputs={'checkpoint':{'sha256':'a'*64},'weights':{'sha256':'w'*64}},
        configuration={'allocation':'learned','state':{'max_model_regret':1.}})
    recipe=hashlib.sha256(code.outputs.dispatch.canonical(plan)).hexdigest()
    job=dict(seed=1337,task_id='task',plan=plan,recipe_sha256=recipe,priority_publication_sha256='b'*64)
    value=dict(kind=code.outputs.KIND,seed=1337,task_id='task',method='rbf',recipe_sha256=recipe,world_size=world,
        NN_atol=1e-4,NN_rtol=1e-4,source_sha256=code.sha(code.READER/'read_final_refit_learned_outputs.py'),
        final_checkpoint_sha256='c'*64,final_model_sha256='f'*64,input_numeric_index_sha256='n'*64,
        priority_publication_sha256='b'*64,priority_policy_signature='p'*64,priority_checkpoint_sha256='a'*64,
        artifacts={key:{'sha256':'d'*64,'bytes':1} for key in code.outputs.artifact_keys(world)},
        sequences=[dict(sequence_id=f'{i:04d}',events=162 if i<45 else 155,nodes=2,database_sha256='d'*64) for i in range(46)],total_nodes=92)
    for flag in ('all_registered_bytes_verified','all_46_sequences_7445_events_and_final_model_factor_nodes_verified','all_rank_and_sequence_priority_identities_verified'):value[flag]=True
    for flag in ('learned_expansion_order_independently_verified','full_forest_semantics_or_fresh_state_independently_accepted','learned_Stage2_complete','same_resource_performance_accepted','paper_performance_complete'):value[flag]=False
    return value,job


@pytest.mark.parametrize('world',[4,8])
def test_complete_local_metadata_gate(code,world):
    value,job=fixture(code,world)
    assert code.validate_local(value,job,'b'*64)==job['plan']


@pytest.mark.parametrize('mutation',['legacy','partial','duplicate','seed','model','task','recipe','NN_tolerance',
    'unread_bytes','future_claim','priority','priority_checkpoint','publication','missing_rank','missing_binding',
    'reader_source','event_count','node_count','path_escape','database_hash','allocation','regret_contract'])
def test_mixed_partial_or_changed_outputs_rejected(code,mutation):
    value,job=fixture(code)
    if mutation=='legacy':value['kind']='legacy'
    elif mutation=='partial':value['sequences'].pop()
    elif mutation=='duplicate':value['sequences'][1]['sequence_id']=value['sequences'][0]['sequence_id']
    elif mutation=='seed':value['seed']=2027
    elif mutation=='model':value['final_model_sha256']='changed'
    elif mutation=='task':value['task_id']='other'
    elif mutation=='recipe':value['recipe_sha256']='changed'
    elif mutation=='NN_tolerance':value['NN_rtol']=1e-3
    elif mutation=='unread_bytes':value['all_registered_bytes_verified']=False
    elif mutation=='future_claim':value['learned_Stage2_complete']=True
    elif mutation=='priority':value['priority_policy_signature']='changed'
    elif mutation=='priority_checkpoint':value['priority_checkpoint_sha256']='changed'
    elif mutation=='publication':value['priority_publication_sha256']='changed'
    elif mutation=='missing_rank':value['artifacts'].pop('replay-rank0')
    elif mutation=='missing_binding':value['artifacts'].pop('priority-input-binding')
    elif mutation=='reader_source':value['source_sha256']='changed'
    elif mutation=='event_count':value['sequences'][0]['events']-=1
    elif mutation=='node_count':value['total_nodes']+=1
    elif mutation=='path_escape':value['sequences'][0]['sequence_id']='../escape'
    elif mutation=='database_hash':value['sequences'][0]['database_sha256']='z'*64
    else:
        if mutation=='allocation':job['plan']['configuration']['allocation']='bound'
        else:job['plan']['configuration']['state']['max_model_regret']=.5
        value['recipe_sha256']=job['recipe_sha256']=hashlib.sha256(code.outputs.dispatch.canonical(job['plan'])).hexdigest()
    with pytest.raises(AssertionError):code.validate_local(value,job,'b'*64)


def sequence_fixture(code,root):
    value,job=fixture(code)
    job['plan']['cache_manifest']={'sha256':'cache'}
    plan=dict(kind='rbf_paper_replay_v1',expected_sequences=['0000'],expected_events=2,fixture=False,
        protocol={'split':'train','dataset':'spd'},configuration=job['plan']['configuration'],cache_sha256='cache',
        model_binding=dict(seed=1337,fit_split='train',dataset='spd',priority_policy_signature='p'*64,
            priority_checkpoint_sha256='a'*64,checkpoint_sha256='c'*64,model_sha256='f'*64))
    (root/'plan.json').write_text(json.dumps(plan))
    for name in ('predictions.jsonl','audit.jsonl','timings.json','resources.json','forest.sqlite'):
        (root/name).write_text('fixture bytes, not real experimental evidence')
    seq=dict(sequence_id='0000',events=2,database_sha256=code.sha(root/'forest.sqlite'))
    receipt=dict(kind='rbf_paper_replay_receipt_v1',completed_sequences=['0000'],completed_events=2,fixture=False,
        status='software_replay_completed',files={name:code.sha(root/name) for name in ('plan.json','predictions.jsonl','audit.jsonl','timings.json','resources.json')},
        databases={'0000':{'sha256':seq['database_sha256'],'path':'forest.sqlite'}})
    return job,seq,receipt


@pytest.mark.parametrize('mutation',[None,'native_cost_bytes','missing_cost','receipt_status','database_path','priority_identity','db_bytes'])
def test_sequence_files_and_policy_rebound(code,tmp_path,mutation):
    job,seq,receipt=sequence_fixture(code,tmp_path)
    if mutation=='native_cost_bytes':(tmp_path/'resources.json').write_text('changed')
    elif mutation=='missing_cost':receipt['files'].pop('timings.json')
    elif mutation=='receipt_status':receipt['status']='queued'
    elif mutation=='database_path':receipt['databases']['0000']['path']='../escape'
    elif mutation=='priority_identity':job['plan']['priority_policy_signature']='wrong'
    elif mutation=='db_bytes':(tmp_path/'forest.sqlite').write_text('changed')
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    if mutation:
        with pytest.raises(AssertionError):code.check_sequence(tmp_path,seq,job)
    else:assert code.check_sequence(tmp_path,seq,job)==tmp_path/'forest.sqlite'


def cohort_results(job):
    results=[]
    for i in range(46):
        events=162 if i<45 else 155
        trajectory=dict(kind='rbf_independent_exclusive_learned_search_trajectory_checks_v1',database_sha256='d'*64,
            policy_signature=job['plan']['priority_policy_signature'],weights_sha256=job['plan']['priority_inputs']['weights']['sha256'],
            events=events,observations=2,atol=1e-8,rtol=1e-8,all_18_features_and_MLP_scores_independently_recomputed=True,
            all_float64_selections_exactly_replayed_from_independently_checked_features=True,
            ordering_tolerance_used=False,feature_bitwise_parity_claimed=False,candidate_feature_rows=3,
            actual_selected_operations=2,actual_selected_charged_steps=2,
            independent_compensated_arithmetic_order_differences=[dict(event=0)] if i==0 else [])
        results.append(dict(sequence_id=f'{i:04d}',database_sha256='d'*64,structure={'events':events},
            runtime_raw_cache203_context={'rows':2},learned_search_trajectory=trajectory))
    return results


@pytest.mark.parametrize('mutation',[None,'partial','duplicate','model','events','rows','tolerance','ordering','features','source_db','claim'])
def test_full_cohort_aggregation_keeps_scope_and_rounding_evidence(code,mutation):
    _,job=fixture(code);results=cohort_results(job);last=results[-1]['learned_search_trajectory']
    if mutation=='partial':results.pop()
    elif mutation=='duplicate':results[-1]['sequence_id']=results[0]['sequence_id']
    elif mutation=='model':last['weights_sha256']='changed'
    elif mutation=='events':last['events']-=1
    elif mutation=='rows':last['observations']-=1
    elif mutation=='tolerance':last['rtol']=1e-6
    elif mutation=='ordering':last['all_float64_selections_exactly_replayed_from_independently_checked_features']=False
    elif mutation=='features':last['all_18_features_and_MLP_scores_independently_recomputed']=False
    elif mutation=='source_db':last['database_sha256']='changed'
    elif mutation=='claim':last['feature_bitwise_parity_claimed']=True
    if mutation:
        with pytest.raises(AssertionError):code.ordering_summary(results,job)
    else:
        result=code.ordering_summary(results,job)
        assert result['learned_expansion_trajectory_accepted']
        assert result['independent_compensated_arithmetic_order_differences']==[dict(sequence_id='0000',event=0)]
        assert result['ordering_tolerance_used'] is False
        assert 'paper_performance_complete' not in result and 'learned_Stage2_complete' not in result


def test_generated_complete_driver_preserves_existing_oracles(code):
    builder=importlib.import_module('prepare_final_refit_learned_CPU')
    source,control=builder.build(builder.PARENT.read_text())
    assert control['original_five_numeric_oracles_unchanged']
    assert source==Path(code.__file__).with_name('accept_final_refit_learned_cohort.py').read_text()
    assert source.count('validate(v,args,Task)')==2
    assert 'learned_order=ordering(db,s,args,job,progress)' in source
    assert 'learned_search_trajectory=learned_order' in source
    assert 'priority_summary=ordering_summary(results,job)' in source
    assert 'learned_Stage2_complete=False' in source and 'same_resource_performance_accepted=False' in source
    assert 'paper_performance_complete=False' in source
    with pytest.raises(AssertionError):builder.build(builder.PARENT.read_text()+'\n')


@pytest.mark.parametrize('mutation',[None,'running','failed','changed_task_source','changed_registered_artifact',
    'changed_plan','unregistered_receipt','changed_receipt_bytes','wrong_model_rows','changed_weights',
    'mixed_seed','missing_journal','qualifier_rejection'])
def test_live_and_checkpoint_boundary_is_rechecked(code,tmp_path,monkeypatch,mutation):
    import numpy as np
    from types import SimpleNamespace as N
    value,job=fixture(code)
    run=tmp_path/'run';fit=run/'fit/1337';fit.mkdir(parents=True)
    np.savez(fit/'weights.npz',**{f'w{i}':np.zeros(s) for i,s in enumerate(((32,18),(32,),(1,32),(1,)))})
    weights_sha=code.sha(fit/'weights.npz')
    _,signature=code.trajectory.policy_weights(fit/'weights.npz',weights_sha)
    job['plan']['priority_inputs']['weights']['sha256']=weights_sha
    value['priority_policy_signature']=job['plan']['priority_policy_signature']=signature
    source='fixture task bootstrap'
    job['plan']['bootstrap_sha256']=hashlib.sha256(source.encode()).hexdigest()
    publication=tmp_path/'publication.json';publication.write_text('fixture publication')
    value['priority_publication_sha256']=job['priority_publication_sha256']=code.sha(publication)
    value['recipe_sha256']=job['recipe_sha256']=hashlib.sha256(code.outputs.dispatch.canonical(job['plan'])).hexdigest()
    byte=tmp_path/'byte.json';byte.write_text(json.dumps(value))
    receipts=tmp_path/'receipts';receipts.mkdir()
    ledger={'entries':[dict(kind=code.outputs.KIND,receipt=str(byte),receipt_sha256=code.sha(byte))]}
    ledger_path=receipts/'20260928-execution-ledger.json'
    journal=receipts/'journal.json'
    params={'General/plan':json.dumps(job['plan']),'General/recipe_sha256':job['recipe_sha256']}
    task=N(status='completed',reload=lambda:None,data=N(script=N(diff=source)),get_parameters=lambda:params,
        artifacts={k:N(hash=v['sha256'],size=v['bytes']) for k,v in value['artifacts'].items()})
    if mutation in ('running','failed'):task.status='in_progress' if mutation=='running' else 'failed'
    elif mutation=='changed_task_source':task.data.script.diff+='changed'
    elif mutation=='changed_registered_artifact':task.artifacts['replay-rank0'].hash='changed'
    elif mutation=='changed_plan':params['General/plan']='{}'
    elif mutation=='unregistered_receipt':ledger['entries']=[]
    elif mutation=='changed_receipt_bytes':byte.write_text(json.dumps(value)+' ')
    elif mutation=='changed_weights':
        with np.load(fit/'weights.npz') as d:weights={k:d[k] for k in d.files}
        weights['w0']+=.01;np.savez(fit/'weights.npz',**weights)
    elif mutation=='mixed_seed':value['seed']=2027
    ledger_path.write_text(json.dumps(ledger))
    journal.write_text(json.dumps({'jobs':[] if mutation=='missing_journal' else [job]}))
    monkeypatch.setattr(code,'R',tmp_path);monkeypatch.setattr(code,'JOURNAL',journal)
    monkeypatch.setattr(code.outputs,'source_gate',lambda:None)
    def qualify(*a):
        if mutation=='qualifier_rejection':raise AssertionError('actual publication qualification rejected')
    monkeypatch.setattr(code.outputs,'qualify_job',qualify)
    monkeypatch.setattr(code.final_model,'validate_final_model',lambda *a,**k:{'rows':93 if mutation=='wrong_model_rows' else 92})
    args=N(seed=1337,priority_publication=publication,byte_admission=byte,run=run,checkpoint=tmp_path/'checkpoint')
    tasks=N(get_task=lambda task_id:task)
    if mutation:
        with pytest.raises(AssertionError):code.validate(value,args,tasks)
    else:assert code.validate(value,args,tasks)==job
