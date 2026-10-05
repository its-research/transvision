"""Software-only metadata gates and pinned train-to-val producer changes."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N

import pytest

TOOLS = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
sys.path.insert(0,str(TOOLS))
spec = importlib.util.spec_from_file_location('seen_val_runtime',TOOLS/'prepare_rbf_seen_val_forest_runtime.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def test_pinned_algorithm_only_changes_evaluation_split():
    parent = (b.PARENT/'bootstrap.py').read_text()
    child, control = b.build(parent)
    assert b.literals(parent) == b.literals(child)
    assert control['worker_only_change'] == 'PaperProtocol spd train -> val'
    assert control['scoring_atol'] == control['scoring_rtol'] == 1e-4
    assert not control['dispatch_ready']
    main = next(n for n in ast.parse(child).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    text = ast.get_source_segment(child,main)
    assert text.index('seen_val_admission(plan,base)') < text.index("for key in ('source'")
    assert "len(ev['events'])==3316" in text and "len(ev['origin_us_by_sequence'])==21" in text
    assert "'all_46_sequences_7445_events_completed'" not in child
    with pytest.raises(AssertionError): b.build(parent+'\n')


def gate_fixture(tmp_path):
    plan = copy.deepcopy(json.loads((b.PARENT/'preparation.json').read_bytes())['seeds'][0]['plan'])
    plan['world_size'] = 4
    original = copy.deepcopy(plan)
    seed = plan['seed']
    root = b.R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    transport = b.R/f'artifacts/rbf-seen-val-forest-cache-transport-v1-20261005/seed{seed}/local-transport-readback.json'
    binding = json.loads((root/'input-binding.json').read_bytes())
    t = json.loads(transport.read_bytes())
    values = dict(binding=(root/'input-binding.json').read_bytes(), transport=transport.read_bytes(), review=b.INPUT_INDEX.read_bytes())
    main = dict(kind='rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1',
        seed=seed, task_id='fixture-main', completed_sequences=46, completed_events=7445,
        driver_freeze_sha256=b.MAIN_FREEZE_SHA, checkpoint_sha256=plan['checkpoint']['sha256'],
        final_model_sha256=plan['final_refit_model_sha256'], byte_admission_sha256='a'*64,
        recipe_sha256=hashlib.sha256(canonical(original)).hexdigest(),
        atol=1e-8,rtol=1e-8,old_model_full_forest_acceptance_inherited=False,paper_performance_complete=False)
    for flag in ('strict_declared_support_partition_coverage','padded_float64_mass_arithmetic_verified',
                 'output_class_recovery_provenance_verified','all_raw_factor_and_causal_commits_verified',
                 'all_decisions_search_or_capacity_undecided_semantics_verified',
                 'all_203_feature_recipe_values_independently_verified','all_scorer_contexts_bound_to_original_causal_raw_history'):
        main[flag] = True
    values['main_acceptance'] = canonical(main)
    inputs = {k:dict(task='fixture-publication',key=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in values.items()}
    for role,digest,size in [('cache_archive',t['archive']['sha256'],t['archive']['bytes']),
                             ('cache_manifest',t['cache_manifest_sha256'],123),
                             ('events',t['events_sha256'],(root/'events.json').stat().st_size)]:
        inputs[role]=dict(task='fixture-publication',key=role,sha256=digest,bytes=size)
        plan[role]=inputs[role]
    plan['forward_outputs']=binding['forward_artifacts']
    artifacts={key:N(hash='b'*64,size=123) for key in {'receipt','exclusive-source-manifest'}|{f'replay-rank{i}' for i in range(4)}}
    upstream=dict(task_id='fixture-main',artifacts={k:dict(sha256=v.hash,bytes=v.size) for k,v in artifacts.items()})
    main_gate=dict(kind='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1',seed=seed,
        main_prerequisite_verified=True,main_task_id='fixture-main',main_acceptance_sha256=inputs['main_acceptance']['sha256'],
        sequence_proof_sha256={str(i):'x'*64 for i in range(46)},byte_admission_sha256='a'*64,
        final_model_sha256=plan['final_refit_model_sha256'])
    plan['seen_val_input_publication']=dict(kind='rbf_seen_val_complete_forest_inputs_cloud_bytes_v1',seed=seed,
        independent_cloud_bytes_verified=True,artifacts=inputs,full_train_main_local_gate=main_gate,upstream_main=upstream)
    plan['evaluation_scope']='SPD seen-val exploratory scheduled snapshots; bound allocator baseline'
    params={'General/plan':canonical(original).decode(),'General/recipe_sha256':main['recipe_sha256']}
    tasks={'fixture-publication':N(status='completed',artifacts={k:N(hash=v['sha256'],size=v['bytes']) for k,v in inputs.items()}),
           'fixture-main':N(status='completed',artifacts=artifacts,data=N(script=N(diff=(b.PARENT/'bootstrap.py').read_text())),get_parameters=lambda:params)}
    # Stub only ClearML transport; all gate code is the actual generated gate.
    source = inspect_gate()
    namespace=dict(json=json,hashlib=hashlib,canonical=canonical,INPUT_INDEX_SHA=b.INPUT_INDEX_SHA,
        PARENT_SHA=b.PARENT_SHA,MAIN_FREEZE_SHA=b.MAIN_FREEZE_SHA,Task=N(get_task=lambda task_id:tasks[task_id]),
        fetch=lambda spec,path:path.write_bytes(values[spec['key']]),
        write=lambda path,value:path.write_bytes(canonical(value)))
    exec(compile(source,'software-only-seen-val-gate','exec'),namespace)
    return plan,tasks,values,main,namespace['seen_val_admission']


def inspect_gate():
    import inspect
    return inspect.getsource(b.seen_val_admission)


@pytest.mark.parametrize('corrupt', [None,'publication_unread','main_running','main_partial','wrong_driver',
    'wrong_main_model','missing_context','relaxed_tolerance','missing_main_rank','changed_source',
    'changed_limits','changed_checkpoint','changed_forward','wrong_split_scope','missing_local_gate','wrong_review',
    'changed_archive','extra_publication_artifact'])
def test_gate_rejects_unqualified_remote_inputs(tmp_path,corrupt):
    plan,tasks,values,main,gate=gate_fixture(tmp_path)
    pub=plan['seen_val_input_publication']
    if corrupt=='publication_unread':pub['independent_cloud_bytes_verified']=False
    elif corrupt=='main_running':tasks['fixture-main'].status='in_progress'
    elif corrupt=='main_partial':main['completed_events']=7444
    elif corrupt=='wrong_driver':main['driver_freeze_sha256']='changed'
    elif corrupt=='wrong_main_model':main['final_model_sha256']='changed'
    elif corrupt=='missing_context':main['all_scorer_contexts_bound_to_original_causal_raw_history']=False
    elif corrupt=='relaxed_tolerance':main['atol']=1e-3
    elif corrupt=='missing_main_rank':tasks['fixture-main'].artifacts.pop('replay-rank0')
    elif corrupt=='changed_source':tasks['fixture-main'].data.script.diff+='\n'
    elif corrupt=='changed_limits':plan['configuration']['state']['max_frontier']+=1
    elif corrupt=='changed_checkpoint':plan['checkpoint']['sha256']='changed'
    elif corrupt=='changed_forward':plan['forward_outputs']=[]
    elif corrupt=='wrong_split_scope':plan['evaluation_scope']='formal unseen validation'
    elif corrupt=='missing_local_gate':pub['full_train_main_local_gate']['main_prerequisite_verified']=False
    elif corrupt=='wrong_review':pub['artifacts']['review']['sha256']='changed'
    elif corrupt=='changed_archive':plan['cache_archive']=dict(plan['cache_archive'],sha256='changed')
    elif corrupt=='extra_publication_artifact':tasks['fixture-publication'].artifacts['extra']=N(hash='x',size=1)
    values['main_acceptance']=canonical(main)
    if corrupt is None:
        gate(plan,tmp_path)
        assert json.loads((tmp_path/'seen-val-input-binding.json').read_bytes())['paper_performance_complete'] is False
    else:
        with pytest.raises(AssertionError):gate(plan,tmp_path)
