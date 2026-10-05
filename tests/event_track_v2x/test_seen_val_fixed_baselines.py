"""Separate baseline backends and provenance gates; no real GPU execution."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
b=importlib.import_module('prepare_seen_val_fixed_baselines')
from test_seen_val_forest_runtime import gate_fixture


def cpu_fixture(width,seed,template):
    full=dict(kind='rbf_final_refit_fixed_'+('Top1' if width==1 else 'topK')+'_full_causal_cache203_fresh_state_admission_v1',
        seed=seed,K=width,task_id='fixture-baseline',method='topk',completed_sequences=46,completed_events=7445,
        driver_source_binding=dict(source_freeze_sha256=b.BASELINES[width]['CPU_freeze_sha']),byte_admission_sha256='a'*64,
        atol=1e-8,rtol=1e-8,all_original_schedule_causal_raw_commits_verified=True,
        all_203_features_and_arrived_parent_contexts_independently_verified=True,
        all_stored_branch_states_and_chosen_outputs_numerically_verified=True,
        old_model_full_forest_acceptance_inherited=False,exclusive_partition_or_recovery_accepted=False,
        complete_online_method_accepted=False,same_resource_performance_accepted=False,learned_Stage2_complete=False,
        paper_performance_complete=False,final_model_binding=dict(checkpoint_sha256=template['checkpoint']['sha256'],
        final_model_sha256=template['final_refit_model_sha256']),recipe_sha256=hashlib.sha256(b.canonical(template)).hexdigest())
    payload=dict(kind='rbf_seen_val_fixed_baseline_train_CPU_prerequisite_v1',K=width,seed=seed,task_id='fixture-baseline',
        local_full_train_CPU_receipts_independently_bound=True,full_admission_text=b.canonical(full).decode(),
        full_admission_sha256=hashlib.sha256(b.canonical(full)).hexdigest(),byte_admission_sha256='a'*64,
        sequence_proof_sha256={f'{i:04d}':'f'*64 for i in range(46)},
        registered_artifacts={k:dict(sha256='b'*64,bytes=123) for k in {'receipt'}|{f'replay-rank{i}' for i in range(template['world_size'])}})
    return full,payload


def fixture(tmp_path,width):
    main,tasks,values,main_receipt,_=gate_fixture(tmp_path)
    main['scope']=main['evaluation_scope']
    parent,_,prepared,_=b.source_contract(width)
    original=copy.deepcopy(next(v['plan'] for v in prepared['seeds'] if v['seed']==main['seed']))
    original['world_size']=4
    full,payload=cpu_fixture(width,main['seed'],original)
    source,control=b.derive(width)
    plan=b.make_plan(width,main,payload,hashlib.sha256(source.encode()).hexdigest(),8)
    tasks['fixture-baseline']=N(status='completed',artifacts={k:N(hash=v['sha256'],size=v['bytes']) for k,v in payload['registered_artifacts'].items()},
        data=N(script=N(diff=(parent/'bootstrap.py').read_text())),get_parameters=lambda:{'General/plan':b.canonical(original).decode(),'General/recipe_sha256':full['recipe_sha256']})
    tree=ast.parse(source)
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in ('seen_val_admission','cpu_contract','fixed_seen_val_admission')) or
        (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PARENT_SHA','MAIN_FREEZE_SHA','INPUT_INDEX_SHA','WIDTH','CPU_FREEZE_SHA','BASELINE_SOURCE_SHA','BASELINE_TEMPLATES'))]
    namespace=dict(json=json,hashlib=hashlib,canonical=b.canonical,
        Task=N(get_task=lambda task_id:tasks[task_id]),fetch=lambda spec,path:path.write_bytes(values[spec['key']]),
        write=lambda path,value:path.write_bytes(b.canonical(value)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-frozen-baseline-gates','exec'),namespace)
    return plan,tasks,full,payload,namespace['fixed_seen_val_admission']


@pytest.mark.parametrize('width',[1,4])
def test_real_distinct_baseline_parent_and_main_gate_are_preserved(width):
    source,control=b.derive(width)
    assert control['own_fixed_backend_preserved'] and control['worker_only_change']=='PaperProtocol spd train -> val'
    assert f"width={width}" in source and "from transvision.models.event_track_v2x.paper_runtime import replay,default_configuration" in source
    assert 'exclusive_paper_runtime import' not in source
    assert f"fixed_K{width}_candidate_v1" in source
    assert "'all_21_sequences_3316_events_completed'" in source
    main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    text=ast.get_source_segment(source,main)
    assert text.index('fixed_seen_val_admission(plan,base)')<text.index("for key in ('source'")


@pytest.mark.parametrize('width',[1,4])
@pytest.mark.parametrize('mutation',[None,'main-running','baseline-running','baseline-source','baseline-plan','missing-artifact',
    'width','configuration','input','NN-reference','scope','CPU-partial','CPU-state','CPU-tolerance','CPU-source',
    'CPU-model','CPU-bytes','CPU-missing-sequence','CPU-inherited','CPU-missing','main-plan-digest'])
def test_actual_remote_gates_reject_unqualified_inputs(tmp_path,width,mutation):
    plan,tasks,full,payload,gate=fixture(tmp_path,width)
    if mutation=='main-running':tasks['fixture-main'].status='in_progress'
    elif mutation=='baseline-running':tasks['fixture-baseline'].status='in_progress'
    elif mutation=='baseline-source':tasks['fixture-baseline'].data.script.diff+='\n'
    elif mutation=='baseline-plan':
        params=tasks['fixture-baseline'].get_parameters();d=json.loads(params['General/plan']);d['configuration']['state']['active_limit']=99
        params['General/plan']=b.canonical(d).decode();tasks['fixture-baseline'].get_parameters=lambda:params
    elif mutation=='missing-artifact':tasks['fixture-baseline'].artifacts.pop('replay-rank0')
    elif mutation=='width':plan['baseline_K']=99
    elif mutation=='configuration':plan['configuration']['state']['active_limit']=99
    elif mutation=='input':plan['events']=dict(plan['events'],sha256='changed')
    elif mutation=='NN-reference':plan['forward_outputs']=[]
    elif mutation=='scope':plan['scope']='official independent evaluation'
    elif mutation=='CPU-partial':full['completed_events']=7444
    elif mutation=='CPU-state':full['all_stored_branch_states_and_chosen_outputs_numerically_verified']=False
    elif mutation=='CPU-tolerance':full['atol']=1e-3
    elif mutation=='CPU-source':full['driver_source_binding']['source_freeze_sha256']='changed'
    elif mutation=='CPU-model':full['final_model_binding']['final_model_sha256']='changed'
    elif mutation=='CPU-bytes':full['byte_admission_sha256']='changed'
    elif mutation=='CPU-missing-sequence':payload['sequence_proof_sha256'].pop('0000')
    elif mutation=='CPU-inherited':full['old_model_full_forest_acceptance_inherited']=True
    elif mutation=='CPU-missing':payload['local_full_train_CPU_receipts_independently_bound']=False
    elif mutation=='main-plan-digest':plan['main_seen_val_input_plan_sha256']='changed'
    payload['full_admission_text']=b.canonical(full).decode();payload['full_admission_sha256']=hashlib.sha256(b.canonical(full)).hexdigest()
    if mutation is None:
        gate(plan,tmp_path)
        result=json.loads((tmp_path/'seen-val-baseline-binding.json').read_bytes())
        assert result['K']==width and not result['full_train_baseline_acceptance_inherited']
        assert not result['same_resource_performance_accepted'] and not result['paper_performance_complete']
    else:
        with pytest.raises(AssertionError):gate(plan,tmp_path)


@pytest.mark.parametrize('width',[1,4])
def test_no_real_full_CPU_receipt_is_fabricated(tmp_path,width):
    with pytest.raises((AssertionError,FileNotFoundError)):
        b.local_cpu_prerequisite(width,2027,tmp_path/'absent.json',tmp_path/'absent-byte.json')


def test_K1_cannot_use_K4_CPU_proof():
    _,_,prepared,_=b.source_contract(4);template=dict(prepared['seeds'][0]['plan'],world_size=4)
    _,proof=cpu_fixture(4,template['seed'],template)
    with pytest.raises(AssertionError):b.cpu_contract(proof,1,b.BASELINES[1]['CPU_freeze_sha'])


@pytest.mark.parametrize('mutation',[None,'missing-sequence','duplicate-sequence','GT','state','context',
    'rows','model','database','registration','recipe','symlink'])
def test_local_full_CPU_requires_registered_all_sequence_proofs(tmp_path,monkeypatch,mutation):
    contract=b.source_contract(4);template=dict(contract[2]['seeds'][0]['plan'],world_size=4)
    seed=template['seed'];full,payload=cpu_fixture(4,seed,template)
    monkeypatch.setattr(b,'source_contract',lambda width:contract)
    monkeypatch.setattr(b,'R',tmp_path)
    directory=tmp_path/'full';directory.mkdir();(tmp_path/'receipts').mkdir()
    sequences=[]
    for i in range(46):
        sid=f'{i:04d}';events=7445-45 if i==0 else 1
        sequences.append(dict(sequence_id=sid,events=events,nodes=1,database_sha256='d'*64))
        features=dict(sequence_id=sid,database_sha256='d'*64,events=events,rows=1,
            checkpoint_sha256=template['checkpoint']['sha256'],final_model_sha256=template['final_refit_model_sha256'],
            all_203_features_and_world_state_values_bound_to_raw_cache=True,
            all_contexts_bound_to_exact_first_arrival_history=True,complete_original_sequence=True,
            atol=1e-8,rtol=1e-8,GT_read=False,test_read=False)
        causal=dict(events=events,observations=1,raw_observation_bytes_and_scalar_columns_verified=True,
            all_cumulative_raw_factor_commit_digests_verified=True,all_admitted_information_and_arrivals_before_decision=True,no_old_factor_rescore=True)
        fresh=dict(sequence=sid,database_sha256='d'*64,events=events,states=1,predictions=1,
            stored_states_and_chosen_outputs_checked=True,atol=1e-8,rtol=1e-8)
        value=dict(sequence_id=sid,database_sha256='d'*64,runtime_raw_cache203_context=features,causal=causal,fresh_state=fresh)
        if i==0:
            if mutation=='GT':features['GT_read']=True
            elif mutation=='state':fresh['stored_states_and_chosen_outputs_checked']=False
            elif mutation=='context':features['all_contexts_bound_to_exact_first_arrival_history']=False
            elif mutation=='rows':features['rows']=2
            elif mutation=='model':features['final_model_sha256']='changed'
            elif mutation=='database':fresh['database_sha256']='changed'
            elif mutation=='duplicate-sequence':value['sequence_id']='0001'
            elif mutation=='missing-sequence':continue
        path=directory/f'sequence-{i:02d}.json';path.write_bytes(b.canonical(value))
        if i==0 and mutation=='symlink':
            path.rename(tmp_path/'linked-proof.json');path.symlink_to(tmp_path/'linked-proof.json')
    byte=dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',seed=seed,
        task_id=full['task_id'],recipe_sha256=full['recipe_sha256'],all_registered_bytes_verified=True,
        all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
        full_forest_semantics_or_fresh_state_independently_accepted=False,NN_atol=1e-4,NN_rtol=1e-4,method='topk',
        final_checkpoint_sha256=template['checkpoint']['sha256'],final_model_sha256=template['final_refit_model_sha256'],
        total_nodes=46,sequences=sequences,artifacts=payload['registered_artifacts'])
    byte_path=tmp_path/'byte.json';byte_path.write_bytes(b.canonical(byte))
    full.update(byte_admission_sha256=b.sha(byte_path),observations=46,states=46,predictions=46)
    if mutation=='recipe':full['recipe_sha256']='changed'
    full_path=directory/'acceptance.json';full_path.write_bytes(b.canonical(full))
    entries=[dict(receipt=str(p),receipt_sha256=b.sha(p)) for p in (full_path,byte_path)]
    if mutation=='registration':entries.pop()
    (tmp_path/'receipts/20260928-execution-ledger.json').write_bytes(b.canonical(dict(entries=entries)))
    if mutation is None:
        gate=b.local_cpu_prerequisite(4,seed,full_path,byte_path)
        assert len(gate['sequence_proof_sha256'])==46
        assert gate['full_admission_sha256']==b.sha(full_path)
    else:
        with pytest.raises(AssertionError):b.local_cpu_prerequisite(4,seed,full_path,byte_path)
