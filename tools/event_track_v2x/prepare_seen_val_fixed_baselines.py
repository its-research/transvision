"""Prepare distinct fixed K1/K4 seen-val producers and exact train CPU gates.

No task is created. Each backend is inherited from its own frozen producer;
publication and complete train CPU admission remain execution prerequisites.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME = 'rbf-seen-val-fixed-K1-K4-GPU-producers-v1-20261005'
BOUND = R/'source-freezes/rbf-seen-val-bound-forest-GPU-producer-v1-20261005'
BOUND_SHA = 'd9cacf90832ca7e8ba6f25a3ce15d89bac970db857acccea8c4eb0345a60d98f'
BASELINES = {
    1: dict(name='Top1', source_sha='c01e5a017a40c8f6398805df99975d58cf3f4ee2ac44657c96d75eaad3c8d37e',
        preparation_sha='f565cc985c3fb67cf606286c02144be9ef88d243535b04b8707a783d8f2e5078',
        CPU_freeze_sha='6ebb95230c82d5564000e911f3c4c45939783954f5937b27e4d553d03a171e35'),
    4: dict(name='topK', source_sha='c1106f2bbc9628dd622f3c691459c6f71dff55362f1a5ffa9b5c57196dc75bd2',
        preparation_sha='fdda5760036bf847d91b85fcfca792fa2870b10749c2c6f56cbde27b4fd7078a',
        CPU_freeze_sha='7dbc44e80ac6b13c232a59004fb3baaaa67ebde9b6e95c4890ce57a74f2f47f4')}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def cpu_contract(payload, width, freeze_sha):
    """Validate the exact receipt envelope; independent algorithms are not rerun."""
    assert payload['kind'] == 'rbf_seen_val_fixed_baseline_train_CPU_prerequisite_v1'
    assert type(width) is int and width in (1,4) and payload['K'] == width
    assert payload['local_full_train_CPU_receipts_independently_bound'] is True
    raw = payload['full_admission_text'].encode()
    assert hashlib.sha256(raw).hexdigest() == payload['full_admission_sha256']
    full = json.loads(raw)
    name = 'Top1' if width == 1 else 'topK'
    assert full['kind'] == 'rbf_final_refit_fixed_'+name+'_full_causal_cache203_fresh_state_admission_v1'
    assert full['seed'] == payload['seed'] and full['task_id'] == payload['task_id']
    assert full['K'] == width and full['method'] == 'topk'
    assert full['completed_sequences'] == 46 and full['completed_events'] == 7445
    assert full['driver_source_binding']['source_freeze_sha256'] == freeze_sha
    assert full['byte_admission_sha256'] == payload['byte_admission_sha256']
    assert full['atol'] == full['rtol'] == 1e-8
    for key in ('all_original_schedule_causal_raw_commits_verified',
                'all_203_features_and_arrived_parent_contexts_independently_verified',
                'all_stored_branch_states_and_chosen_outputs_numerically_verified'):
        assert full[key] is True
    for key in ('old_model_full_forest_acceptance_inherited','exclusive_partition_or_recovery_accepted',
                'complete_online_method_accepted','same_resource_performance_accepted','learned_Stage2_complete','paper_performance_complete'):
        assert full[key] is False
    assert len(payload['sequence_proof_sha256']) == 46
    for sid,digest in payload['sequence_proof_sha256'].items():
        assert sid and '/' not in sid and '\\' not in sid and sid not in ('.','..')
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    return full


def fixed_seen_val_admission(plan, base):
    """Injected remote gate; seven existing published inputs remain unchanged."""
    assert plan['baseline_K'] == WIDTH
    seed = plan['seed']; assert seed in (1337,2027,3407)
    main_plan = plan['main_seen_val_input_plan']
    assert main_plan['seed'] == seed
    assert hashlib.sha256(canonical(main_plan)).hexdigest() == plan['main_seen_val_input_plan_sha256']
    seen_val_admission(main_plan,base)
    for key in ('cache_archive','cache_manifest','events','checkpoint','weights_archive','source',
                'forward_outputs','final_refit_model_sha256','artifact_files_host','scoring_atol','scoring_rtol'):
        assert plan[key] == main_plan[key], ('changed common seen-val input',key)
    template = BASELINE_TEMPLATES[str(seed)]
    for key in ('configuration','checkpoint','weights_archive','source','parent_bootstrap_sha256',
                'final_refit_model_sha256','original_nested_model_sha256','method','scoring_atol','scoring_rtol'):
        assert plan[key] == template[key], ('changed fixed baseline contract',key)
    assert plan['method'] == 'topk'
    assert plan['configuration']['state']['active_limit'] == WIDTH
    assert plan['configuration']['state']['decision_mode'] == 'retained'
    assert plan['scope'] == plan['evaluation_scope'] == f'SPD seen-val exploratory scheduled snapshots; fixed K{WIDTH} baseline'
    assert plan['baseline_variant'] == f'final_refit_seen_val_fixed_K{WIDTH}_v1'
    payload = plan['baseline_train_CPU_prerequisite']
    full = cpu_contract(payload,WIDTH,CPU_FREEZE_SHA)
    assert full['seed'] == seed
    assert full['final_model_binding']['checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert full['final_model_binding']['final_model_sha256'] == plan['final_refit_model_sha256']
    task = Task.get_task(task_id=full['task_id']); assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == BASELINE_SOURCE_SHA
    params = task.get_parameters(); original = json.loads(params['General/plan'])
    assert original['world_size'] in (4,8)
    assert original == dict(template,world_size=original['world_size'])
    assert hashlib.sha256(canonical(original)).hexdigest() == params['General/recipe_sha256'] == full['recipe_sha256']
    artifacts = payload['registered_artifacts']
    assert set(task.artifacts) == set(artifacts) == {'receipt'}|{f'replay-rank{i}' for i in range(original['world_size'])}
    for key,spec in artifacts.items():
        assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
    write(base/'seen-val-baseline-binding.json',dict(seed=seed,K=WIDTH,
        baseline_train_CPU_prerequisite=payload,main_input_plan_sha256=plan['main_seen_val_input_plan_sha256'],
        baseline_configuration_unchanged=True,full_train_baseline_acceptance_inherited=False,
        measured_network_arrival_history_verified=False,same_resource_performance_accepted=False,paper_performance_complete=False))


def paths(width):
    item=BASELINES[width]
    parent=R/f"source-freezes/rbf-final-refit-full-train-fixed-{item['name']}-GPU-v1-20261004"
    cpu=R/f"source-freezes/rbf-final-refit-fixed-{item['name']}-full-independent-CPU-v2-receipt-rows-20261004"
    return parent,cpu


def source_contract(width):
    item=BASELINES[width];parent,cpu=paths(width)
    assert sha(parent/'bootstrap.py')==item['source_sha'] and sha(parent/'preparation.json')==item['preparation_sha']
    prepared=json.loads((parent/'preparation.json').read_bytes())
    for name,spec in prepared['execution_sources'].items():assert sha(parent/name)==spec['sha256']
    assert sha(cpu/'source-freeze.json')==item['CPU_freeze_sha']
    frozen=json.loads((cpu/'source-freeze.json').read_bytes())
    for name,spec in frozen['sources'].items():assert sha(cpu/name)==spec['sha256']
    for spec in frozen['unchanged_references']:assert sha(spec['path'])==spec['sha256']
    return parent,cpu,prepared,frozen


def local_cpu_prerequisite(width, seed, full_path, byte_path):
    """Require registered complete receipts, all 46 sequence proofs and exact source."""
    parent,cpu,prepared,frozen=source_contract(width)
    ledger=json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']
    for path in (full_path,byte_path):
        assert path.resolve().is_relative_to(R) and not any(p.is_symlink() for p in (path,*path.parents))
        assert any(e.get('receipt')==str(path) and e.get('receipt_sha256')==sha(path) for e in ledger)
    full=json.loads(full_path.read_bytes());byte=json.loads(byte_path.read_bytes())
    template=next(v['plan'] for v in prepared['seeds'] if v['seed']==seed)
    assert full['seed']==byte['seed']==seed and full['task_id']==byte['task_id']
    assert full['byte_admission_sha256']==sha(byte_path) and full['recipe_sha256']==byte['recipe_sha256']
    assert byte['all_registered_bytes_verified'] is True
    assert byte['all_46_sequences_7445_events_and_final_model_factor_nodes_verified'] is True
    assert byte['full_forest_semantics_or_fresh_state_independently_accepted'] is False
    assert byte['NN_atol']==byte['NN_rtol']==1e-4 and byte['method']=='topk'
    expected_kind=('rbf_final_refit_full_train_fixed_Top1_independent_bytes_events_factor_admission_v1' if width==1
        else 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1')
    assert byte['kind']==expected_kind
    assert full['final_model_binding']['checkpoint_sha256']==byte['final_checkpoint_sha256']==template['checkpoint']['sha256']
    assert full['final_model_binding']['final_model_sha256']==byte['final_model_sha256']==template['final_refit_model_sha256']
    assert full['observations']==byte['total_nodes']
    seq={v['sequence_id']:v for v in byte['sequences']};assert len(seq)==len(byte['sequences'])==46
    assert sum(s['events'] for s in seq.values())==7445
    assert sum(s['nodes'] for s in seq.values())==byte['total_nodes']
    proofs={};states=predictions=0
    files=sorted(full_path.parent.glob('sequence-*.json'));assert len(files)==46
    for path in files:
        assert path.is_file() and not path.is_symlink()
        value=json.loads(path.read_bytes());sid=value['sequence_id'];assert sid in seq and sid not in proofs
        assert value['database_sha256']==seq[sid]['database_sha256']
        features,causal,fresh=(value[k] for k in ('runtime_raw_cache203_context','causal','fresh_state'))
        assert features['sequence_id']==fresh['sequence']==sid
        assert features['database_sha256']==fresh['database_sha256']==seq[sid]['database_sha256']
        assert features['events']==causal['events']==fresh['events']==seq[sid]['events']
        assert features['rows']==causal['observations']==seq[sid]['nodes']
        assert features['checkpoint_sha256']==template['checkpoint']['sha256']
        assert features['final_model_sha256']==template['final_refit_model_sha256']
        for key in ('all_203_features_and_world_state_values_bound_to_raw_cache','all_contexts_bound_to_exact_first_arrival_history','complete_original_sequence'):
            assert features[key] is True
        for key in ('raw_observation_bytes_and_scalar_columns_verified','all_cumulative_raw_factor_commit_digests_verified','all_admitted_information_and_arrivals_before_decision','no_old_factor_rescore'):
            assert causal[key] is True
        assert fresh['stored_states_and_chosen_outputs_checked'] is True
        assert fresh['atol']==fresh['rtol']==features['atol']==features['rtol']==1e-8
        assert features['GT_read'] is features['test_read'] is False
        states+=fresh['states'];predictions+=fresh['predictions'];proofs[sid]=sha(path)
    assert full['states']==states and full['predictions']==predictions
    payload=dict(kind='rbf_seen_val_fixed_baseline_train_CPU_prerequisite_v1',K=width,seed=seed,task_id=full['task_id'],
        local_full_train_CPU_receipts_independently_bound=True,full_admission_text=full_path.read_text(),
        full_admission_sha256=sha(full_path),byte_admission_sha256=sha(byte_path),sequence_proof_sha256=proofs,
        registered_artifacts=byte['artifacts'])
    cpu_contract(payload,width,BASELINES[width]['CPU_freeze_sha'])
    return payload


def derive(width):
    parent,cpu,prepared,_=source_contract(width)
    original=(parent/'bootstrap.py').read_text();source=original
    assert sha(BOUND/'bootstrap.py')==BOUND_SHA
    bound_source=(BOUND/'bootstrap.py').read_text();bound_tree=ast.parse(bound_source)
    gate=next(n for n in bound_tree.body if isinstance(n,ast.FunctionDef) and n.name=='seen_val_admission')
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in bound_tree.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PARENT_SHA','MAIN_FREEZE_SHA','INPUT_INDEX_SHA')}
    name=BASELINES[width]['name'];label='Top-1' if width==1 else 'Top-K'
    old_scope=('full paired train, distinct frozen final-refit checkpoint, unchanged fixed Top-1 backend and limits; independent replay and same-resource comparison pending' if width==1 else
        'full paired train, distinct frozen final-refit checkpoint, unchanged fixed Top-K K4 backend and limits; independent replay and same-resource comparison pending')
    edits=[("PaperProtocol('spd','train')","PaperProtocol('spd','val')"),
        (f"task_name='final-refit fixed {label} full-train control'",f"task_name='final-refit fixed K{width} SPD seen-val control'"),
        ("base=Path('rbf-original-joint-coupled-replay')",f"base=Path('rbf-seen-val-fixed-K{width}-replay')"),
        (" try:\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):",
         " try:\n  fixed_seen_val_admission(plan,base)\n  for key in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest'):"),
        ("assert len(ev['events'])==7445 and len(ev['origin_us_by_sequence'])==46",
         "assert ev['kind']=='rbf_matching_seen_val_complete_forest_events_v1' and ev['split']=='val' and ev['seed']==plan['seed'];assert len(ev['events'])==3316 and len(ev['origin_us_by_sequence'])==21;assert ev['arrival_policy']=='scheduled_pair_snapshot_at_reference_plus_100ms' and ev['measured_network_arrival_history_verified'] is False"),
        (f"'kind':'rbf_final_refit_full_train_fixed_{name}_candidate_v1'",f"'kind':'rbf_final_refit_seen_val_fixed_K{width}_candidate_v1'"),
        ("'all_46_sequences_7445_events_completed'","'all_21_sequences_3316_events_completed'"),
        (old_scope,f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline; full val CPU, resource and metric admission pending'),
        (" task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)",
         " for key in ('seen-val-input-binding','seen-val-baseline-binding'):\n  if (base/(key+'.json')).exists():task.upload_artifact(key,artifact_object=base/(key+'.json'),wait_on_upload=True)\n task.upload_artifact('receipt',artifact_object=base/'receipt.json',wait_on_upload=True)")]
    for before,after in edits:
        assert source.count(before)==1,(width,before)
        source=source.replace(before,after,1)
    constants.update(WIDTH=width,BASELINE_SOURCE_SHA=BASELINES[width]['source_sha'],CPU_FREEZE_SHA=BASELINES[width]['CPU_freeze_sha'],
        BASELINE_TEMPLATES={str(v['seed']):v['plan'] for v in prepared['seeds']})
    injected='\n'+'\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n\n'
    injected+=ast.get_source_segment(bound_source,gate)+'\n\n'+inspect.getsource(cpu_contract)+'\n'+inspect.getsource(fixed_seen_val_admission)+'\n'
    source=source.replace('\ndef main():',injected+'\ndef main():',1)
    compile(source,'bootstrap.py','exec')
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    left,right=functions(original),functions(source)
    for key in left:
        if key not in ('work','main'):assert left[key]==right[key]
    work=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='work')
    assert functions(ast.get_source_segment(original,work).replace("PaperProtocol('spd','train')","PaperProtocol('spd','val')"))['work']==right['work']
    assert functions(ast.get_source_segment(bound_source,gate))['seen_val_admission']==right['seen_val_admission']
    return source,dict(K=width,parent_source_sha256=BASELINES[width]['source_sha'],edits=edits,
        own_fixed_backend_preserved=True,worker_only_change='PaperProtocol spd train -> val',
        original_bound_input_admission_function_unchanged=True,train_CPU_receipt_required=True,
        actual_GPU_execution=False,full_seen_val_baseline_accepted=False,same_resource_performance_accepted=False)


def make_plan(width, main_plan, payload, bootstrap_sha, world_size):
    _,_,prepared,_=source_contract(width)
    seed=main_plan['seed'];assert world_size in (4,8)
    full=cpu_contract(payload,width,BASELINES[width]['CPU_freeze_sha']);assert full['seed']==seed
    base=next(v['plan'] for v in prepared['seeds'] if v['seed']==seed)
    assert base['checkpoint']==main_plan['checkpoint'] and base['final_refit_model_sha256']==main_plan['final_refit_model_sha256']
    inherited={k:v for k,v in base.items() if k.endswith('admission_sha256') or k.endswith('completion_sha256')}
    plan=copy.deepcopy({k:v for k,v in base.items() if k not in inherited and k!='dispatcher_sha256'})
    plan.update(bootstrap_sha256=bootstrap_sha,world_size=world_size,baseline_K=width,
        baseline_variant=f'final_refit_seen_val_fixed_K{width}_v1',inherited_train_admissions=inherited,
        main_seen_val_input_plan=main_plan,main_seen_val_input_plan_sha256=hashlib.sha256(canonical(main_plan)).hexdigest(),
        baseline_train_CPU_prerequisite=payload,scope=f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline',
        evaluation_scope=f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline',
        full_forest_independently_accepted=False,paper_performance_complete=False)
    for key in ('cache_archive','cache_manifest','events','forward_outputs'):plan[key]=main_plan[key]
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--software-log',type=Path,required=True);args=parser.parse_args()
    out=R/'source-freezes'/NAME;assert not out.exists()
    log=args.software_log.read_text();assert '59 passed' in log and 'FAILED' not in log and 'ERROR' not in log
    produced={k:derive(k) for k in (1,4)}
    out.mkdir();refs={str(BOUND/'bootstrap.py'):BOUND_SHA}
    for width,(source,control) in produced.items():
        directory=out/f'K{width}';directory.mkdir();(directory/'bootstrap.py').write_text(source)
        new(directory/'source-control.json',control)
        parent,cpu,prepared,frozen=source_contract(width)
        refs[str(parent/'preparation.json')]=BASELINES[width]['preparation_sha']
        refs[str(cpu/'source-freeze.json')]=BASELINES[width]['CPU_freeze_sha']
        refs.update({str(parent/name):v['sha256'] for name,v in prepared['execution_sources'].items()})
        refs.update({str(cpu/name):v['sha256'] for name,v in frozen['sources'].items()})
        refs.update({v['path']:v['sha256'] for v in frozen['unchanged_references']})
    shutil.copyfile(__file__,out/Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name('rbf_nested_seen_val_v2_common.py'),out/'rbf_nested_seen_val_v2_common.py')
    test=Path(__file__).resolve().parents[2]/'tests/event_track_v2x/test_seen_val_fixed_baselines.py';shutil.copyfile(test,out/test.name)
    shutil.copyfile(args.software_log,out/'software-tests.log')
    new(out/'source-freeze.json',dict(kind=NAME,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()},
        references=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],
        software_tests=59,software_tests_only=True,missing_execution_dependencies=['complete train baseline CPU proof','main full train acceptance and seen-val input publication','separate dispatch and val output adapters'],
        actual_GPU_execution=False,train_or_val_baseline_completed=False,same_resource_performance_accepted=False,paper_performance_complete=False))
    register(out/'source-freeze.json',NAME)
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),GPU_tasks_created=0)))


if __name__=='__main__':main()
