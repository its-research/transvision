"""Freeze a separate final-model-bound driver around unchanged CPU oracles."""
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha
from rbf_final_refit_forest_binding import INDEX, JOURNAL, OLD_JOURNAL, original_checkpoint, validate_final_model, validate_local_output

D = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004'
OLD = R/'source-freezes/rbf-independent-action-capacity-undecided-v2-20261002'
CACHE = R/'source-freezes/rbf-independent-runtime-cache203-context-v1-20261002/cache203.py'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def main():
    assert not D.exists(), 'immutable preparation already exists; inspect it without overwriting'
    old_driver = (OLD/'accept_real_cohort.py').read_text()
    assert sha(OLD/'accept_real_cohort.py') == '5cc190231525031ce665821576b6a1e82d55917affdf25103e71c2e9fbaf2629'
    assert sha(CACHE) == '90eed7a5e907eafe6d9c4ab06111c3c1dd3df5bc1b5b07953b4128566d83cc35'
    driver_tree = ast.parse(old_driver)
    validate_node = next(v for v in driver_tree.body if isinstance(v, ast.FunctionDef) and v.name == 'validate')
    lines = old_driver.splitlines(keepends=True)
    driver = ''.join(lines[:validate_node.lineno-1]) + 'from rbf_final_refit_forest_binding import validate, source_gate\n' + ''.join(lines[validate_node.end_lineno:])
    edits = [
        ('D=Path(__file__).parent', 'D=Path('+repr(str(OLD))+')'),
        ('from cache203 import CacheAdmission,verify_database as feature_database', 'from final_cache203 import CacheAdmission,verify_database as feature_database'),
        ("freeze['future_cohort_driver']==Path(__file__).name", "freeze['future_cohort_driver']=='accept_real_cohort.py'"),
        ("v=json.loads(args.byte_admission.read_text());job=validate(v)", "final_driver_binding=source_gate();v=json.loads(args.byte_admission.read_text());job=validate(v)"),
        ("(args.output/'binding.json').write_text", "binding.update(final_driver_binding);binding.update(feature_admission.final_model_binding);binding['old_model_full_forest_acceptance_inherited']=False\n (args.output/'binding.json').write_text"),
        ("kind='rbf_full_exclusive_capacity_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'", "kind='rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'"),
        ("(args.output/'acceptance.json').write_text(json.dumps(final,indent=2)+'\\n');print(json.dumps(final),flush=True)", "(args.output/'acceptance.json').write_text(json.dumps(final,indent=2)+'\\n');register(args.output/'acceptance.json',final['kind']);print(json.dumps(final),flush=True)"),
    ]
    for before, after in edits:
        driver = replace_once(driver, before, after)
    driver = driver.replace('from clearml import Task\n', 'from clearml import Task\nfrom rbf_nested_seen_val_v2_common import register\n', 1)
    original_cache = CACHE.read_text()
    cache_edits = [
        ("assert sha(checkpoint) == forward_receipt['plan']['checkpoint']['sha256']", "original_path = original_checkpoint(seed)\n        assert sha(original_path) == forward_receipt['plan']['checkpoint']['sha256']\n        self.final_model_binding = validate_final_model(seed, checkpoint, original=original_path)"),
        ("original_frozen_runtime_source_sha256=source_sha)", "original_frozen_runtime_source_sha256=source_sha)\n        self.binding.update(self.final_model_binding)\n        self.binding['independently_admitted_forward_sha256'] = self.final_model_binding['full_NN_numeric_completion_sha256']\n        self.binding['upstream_cache_original_forward_admission_retained'] = True"),
    ]
    cache = original_cache
    for before, after in cache_edits:
        cache = replace_once(cache, before, after)
    cache = cache.replace('import numpy as np\n', 'import numpy as np\nfrom rbf_final_refit_forest_binding import original_checkpoint, validate_final_model\n', 1)
    compile(driver, 'accept_final_refit_cohort.py', 'exec')
    compile(cache, 'final_cache203.py', 'exec')
    def function_map(source):
        tree = ast.parse(source)
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert function_map(original_cache) == function_map(cache), 'independent numeric/context functions must be unchanged'
    def method_map(source):
        cls = next(v for v in ast.parse(source).body if isinstance(v, ast.ClassDef) and v.name == 'CacheAdmission')
        return {v.name: ast.dump(v, include_attributes=False) for v in cls.body if isinstance(v, ast.FunctionDef) and v.name != '__init__'}
    assert method_map(original_cache) == method_map(cache), 'raw-frame reconstruction must be unchanged'
    old_main = next(v for v in ast.parse(old_driver).body if isinstance(v, ast.FunctionDef) and v.name == 'main')
    new_main = next(v for v in ast.parse(driver).body if isinstance(v, ast.FunctionDef) and v.name == 'main')
    def sequence_loop(function):
        attempt = next(v for v in function.body if isinstance(v, ast.Try))
        return next(v for v in attempt.body if isinstance(v, ast.For))
    assert ast.dump(sequence_loop(old_main), include_attributes=False) == ast.dump(sequence_loop(new_main), include_attributes=False)
    jobs = json.loads(JOURNAL.read_bytes())['jobs']
    old_jobs = json.loads(OLD_JOURNAL.read_bytes())['jobs']
    assert len(jobs) == 3
    entries = json.loads(INDEX.read_bytes())['seeds']
    bindings = []
    for job in jobs:
        entry = next(v for v in entries if v['seed'] == job['seed'])
        bindings.append(validate_final_model(job['seed'], Path(entry['training_byte_proof']).parent/'checkpoint', plan=job['plan']))
        assert sha(original_checkpoint(job['seed'])) == json.loads((R/f'artifacts/rbf-joint-identity-all-row-GPU-admission-v1-20261001/seed{job["seed"]}/receipt.json').read_bytes())['plan']['checkpoint']['sha256']
    unchanged = [OLD/'accept_real_cohort.py', CACHE,
        OLD/'decoder_capacity.py', OLD/'source-freeze.json', OLD/'source-control.json',
        R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py',
        R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002/oracle.py',
        R/'source-freezes/rbf-independent-legal-action-search-v1-20261002/decoder.py',
        R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002/causal.py']
    unchanged_sources = [dict(path=str(p),sha256=sha(p)) for p in unchanged]
    D.mkdir()
    (D/'accept_final_refit_cohort.py').write_text(driver)
    (D/'final_cache203.py').write_text(cache)
    for name in ('rbf_final_refit_forest_binding.py', 'rbf_nested_seen_val_v2_common.py', 'prepare_rbf_final_refit_forest_CPU.py'):
        shutil.copyfile(Path(__file__).with_name(name), D/name)
    shutil.copyfile(OLD/'decoder_capacity.py', D/'decoder_capacity.py')
    source_control = dict(kind='rbf_final_refit_independent_forest_CPU_source_control_v1',
        original_driver_sha256=sha(OLD/'accept_real_cohort.py'), original_cache_oracle_sha256=sha(CACHE),
        old_validator_replaced_by_separate_final_model_recipe_and_output_gate=True,
        driver_literal_edits=[dict(before=a, after=b) for a,b in edits],
        cache_constructor_literal_edits=[dict(before=a, after=b) for a,b in cache_edits],
        all_independent_feature_context_numerical_function_AST_identical=True,
        raw_cache_frame_reconstruction_AST_identical=True,
        full_46_sequence_loop_and_all_five_oracle_calls_AST_identical=True,
        software_gate_sources_unchanged=True, numerical_tolerances_unchanged=True,
        old_model_full_forest_acceptance_inherited=False, full_forest_accepted=False)
    new(D/'source-control.json', source_control)
    freeze = dict(kind='rbf_final_refit_full_forest_independent_CPU_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:sha(p) for p in D.iterdir()},
        unchanged_independent_sources=unchanged_sources,
        index_sha256=sha(INDEX), dispatch_journal_sha256=sha(JOURNAL),
        final_model_bindings=bindings, full_forest_accepted=False,
        learned_Stage2_complete=False, paper_performance_complete=False)
    new(D/'source-freeze.json', freeze)
    sys.path.insert(0, str(D))
    # Use the generated constructor on accepted metadata, without running a
    # single database/numeric check a second time.
    spec = importlib.util.spec_from_file_location('final_model_cache_metadata_control', D/'final_cache203.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    metadata_controls = []
    for job in jobs:
        entry = next(v for v in entries if v['seed'] == job['seed'])
        admission = module.CacheAdmission(job['seed'], Path(entry['training_byte_proof']).parent/'checkpoint')
        assert admission.final_model_binding['final_model_sha256'] == job['plan']['final_refit_model_sha256']
        assert len(admission.events) == 46 and sum(map(len, admission.events.values())) == 7445
        metadata_controls.append(dict(seed=job['seed'], events=7445, sequences=46, final_binding=admission.final_model_binding))
    job = jobs[0]
    old_job = next(v for v in old_jobs if v['seed'] == job['seed'])
    forward = json.loads(Path(next(v for v in entries if v['seed'] == job['seed'])['prediction_byte_proof']).read_bytes())
    event_document = json.loads((R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{job["seed"]}/events.json').read_bytes())
    counts = {s: sum(v['sequence_id'] == s for v in event_document['events']) for s in event_document['origin_us_by_sequence']}
    fixture = dict(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1',
        all_registered_bytes_verified=True, all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
        full_forest_semantics_or_fresh_state_independently_accepted=False,
        learned_Stage2_complete=False, same_resource_performance_accepted=False, paper_performance_complete=False,
        seed=job['seed'], task_id=job['task_id'], method='rbf', recipe_sha256=job['recipe_sha256'],
        world_size=job['plan']['world_size'], NN_atol=1e-4, NN_rtol=1e-4,
        final_checkpoint_sha256=job['plan']['checkpoint']['sha256'], final_model_sha256=job['plan']['final_refit_model_sha256'],
        input_numeric_index_sha256=job['plan']['numeric_reference_admission_sha256'],
        artifacts={k:{} for k in ('receipt','exclusive-source-manifest', *(f'replay-rank{i}' for i in range(job['plan']['world_size'])))},
        sequences=[dict(sequence_id=s, events=counts[s], nodes=forward['sequence_counts'][s], database_sha256='a'*64) for s in sorted(counts)],
        total_nodes=forward['rows'])
    validate_local_output(fixture, job, old_job)
    rejected = []
    mutations = {
        'old_nested_admission_kind': lambda v,j: v.update(kind='rbf_exclusive_capacity_undecided_train_replay_output_byte_coverage_factor_admission_v1'),
        'topK_cannot_use_main_gate': lambda v,j: v.update(method='topk'),
        'missing_factor_gate': lambda v,j: v.update(all_46_sequences_7445_events_and_final_model_factor_nodes_verified=False),
        'old_model_claim': lambda v,j: v.update(final_model_sha256=j['plan']['original_nested_model_sha256']),
        'wrong_checkpoint': lambda v,j: v.update(final_checkpoint_sha256='a'*64),
        'wrong_task': lambda v,j: v.update(task_id='0'*32),
        'wrong_seed': lambda v,j: v.update(seed=2027 if j['seed'] != 2027 else 1337),
        'partial_cohort': lambda v,j: v['sequences'].pop(),
        'wrong_total_nodes': lambda v,j: v.update(total_nodes=v['total_nodes']+1),
        'missing_rank_artifact': lambda v,j: v['artifacts'].pop('replay-rank0'),
        'widened_NN_tolerance': lambda v,j: v.update(NN_atol=1e-3),
        'premature_full_forest_acceptance': lambda v,j: v.update(full_forest_semantics_or_fresh_state_independently_accepted=True),
        'premature_paper_performance': lambda v,j: v.update(paper_performance_complete=True),
        'wrong_numeric_index': lambda v,j: v.update(input_numeric_index_sha256='b'*64),
        'unsafe_sequence_path': lambda v,j: v['sequences'][0].update(sequence_id='../outside'),
        'changed_work_cap': lambda v,j: j['plan']['configuration']['limits'].update(max_decision_nodes=8192),
        'changed_frozen_source': lambda v,j: j['plan']['source'].update(sha256='c'*64),
    }
    for name, mutation in mutations.items():
        value, bad_job = copy.deepcopy(fixture), copy.deepcopy(job)
        mutation(value, bad_job)
        if bad_job['plan'] != job['plan']:
            bad_job['recipe_sha256'] = value['recipe_sha256'] = hashlib.sha256(canonical(bad_job['plan'])).hexdigest()
        try:
            validate_local_output(value, bad_job, old_job)
        except (AssertionError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('mutation admitted: '+name)
    gate = dict(kind='rbf_final_refit_full_forest_CPU_metadata_source_gate_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_freeze_sha256=sha(D/'source-freeze.json'), real_three_seed_cache_checkpoint_metadata_controls=metadata_controls,
        rejected_mutations=rejected, fixture_does_not_assert_actual_output_bytes_or_semantics=True,
        full_forests_replayed=False, GPU_tasks_repeated=False, old_running_oracles_preserved=True,
        full_forest_accepted=False, learned_Stage2_complete=False, paper_performance_complete=False,
        next_dependency='completed producer, independent byte/event/factor admission, then this full forest CPU driver')
    receipt = R/'receipts/rbf-final-refit-full-forest-independent-CPU-v3-preparation-20261004.json'
    new(receipt, gate)
    register(receipt, gate['kind'])
    print(json.dumps(dict(preparation=str(receipt), sources=str(D), rejected_mutations=len(rejected),
        model_bound_metadata_seeds=[v['seed'] for v in metadata_controls], full_forest_accepted=False)))


if __name__ == '__main__':
    main()
