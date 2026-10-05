"""Read-only final-model forest prerequisite for a future capacity teacher.

This does not admit teacher targets or launch experiments. In particular an
old-model forest receipt, K baseline, partial cohort, or capacity-undecided
decision labelled optimal cannot open this gate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
DRIVER = ROOT/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
DRIVER_FREEZE_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
KIND = 'rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'
TRUE_FLAGS = (
    'strict_declared_support_partition_coverage', 'padded_float64_mass_arithmetic_verified',
    'output_class_recovery_provenance_verified', 'all_raw_factor_and_causal_commits_verified',
    'all_decisions_search_or_capacity_undecided_semantics_verified',
    'all_203_feature_recipe_values_independently_verified',
    'all_scorer_contexts_bound_to_original_causal_raw_history')
FALSE_FLAGS = (
    'old_model_full_forest_acceptance_inherited', 'complete_online_method_accepted',
    'learned_Stage2_complete', 'same_resource_performance_accepted', 'paper_performance_complete',
    'full_model_posterior_optimum_regret_independently_computed')
STAGES = ('structure', 'causal', 'fresh_state', 'conditional_action_search', 'runtime_raw_cache203_context')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary_gate(value, seed, model):
    require(type(seed) is int and seed in (1337, 2027, 3407), 'unsupported seed')
    require(value['kind'] == KIND and value['seed'] == seed, 'final-refit main receipt required')
    require(value['completed_sequences'] == 46 and value['completed_events'] == 7445, 'full train cohort required')
    for key in TRUE_FLAGS:
        require(value[key] is True, 'missing independent scope: '+key)
    for key in FALSE_FLAGS:
        require(value[key] is False, 'unsupported inherited or performance claim: '+key)
    count = value['capacity_undecided_component_decisions']
    require(type(count) is int and count >= 0, 'invalid capacity count')
    require(value['conditional_full_legal_action_search_and_declared_gap_verified'] is (count == 0),
            'capacity undecided cannot be counted as full optimal search')
    require(value['atol'] == value['rtol'] == 1e-8, 'fixed forest tolerance required')
    for key, expected in model.items():
        require(value[key] == expected, 'final model provenance mismatch: '+key)
    require(value['final_model_sha256'] != value['original_model_sha256'], 'old model cannot replace final refit')


def cohort_gate(value, byte, parts):
    for key in ('seed', 'task_id', 'recipe_sha256'):
        require(value[key] == byte[key], 'byte/main identity mismatch: '+key)
    require(byte['method'] == 'rbf', 'main forest required, not a fixed-K baseline')
    expected = {s['sequence_id']: s for s in byte['sequences']}
    actual = {s['sequence_id']: s for s in parts}
    require(len(expected) == len(byte['sequences']) == len(actual) == len(parts) == 46,
            '46 unique sequence proofs required')
    require(set(actual) == set(expected), 'sequence identities differ')
    require(sum(s['events'] for s in expected.values()) == 7445, 'full event coverage required')
    for sid, part in actual.items():
        seq = expected[sid]
        require(part['database_sha256'] == seq['database_sha256'], 'sequence database differs')
        for stage in STAGES:
            require(part[stage]['events'] == seq['events'], 'partial stage: '+stage)
            if stage != 'causal':
                require(part[stage]['database_sha256'] == seq['database_sha256'], 'stage database differs')
        require(part['causal']['observations'] == part['runtime_raw_cache203_context']['rows'] == seq['nodes'],
                'raw row coverage differs')
        for key in ('strict_explicit_residual_partition_pass', 'declared_fixed_support_coverage_pass',
                    'all_padded_mass_arithmetic_independently_verified'):
            require(part['structure'][key] is True, 'structure scope missing: '+key)
        for key in ('raw_observation_bytes_and_scalar_columns_verified',
                    'all_cumulative_raw_factor_commit_digests_verified',
                    'all_admitted_information_and_arrivals_before_decision', 'no_old_factor_rescore'):
            require(part['causal'][key] is True, 'causal scope missing: '+key)
        state, action, context = (part[k] for k in STAGES[2:])
        require(state['stored_states_and_chosen_outputs_checked'] is True, 'all fresh states required')
        require(action['capacity_undecided_counted_as_optimal'] is False, 'capacity semantics changed')
        for key in ('all_selected_actions_raw_support_legal', 'bounded_search_gap_and_shared_budget_verified',
                    'action_domain_not_restricted_to_retained_posterior', 'own_selected_state_export_verified'):
            require(action[key] is True, 'action scope missing: '+key)
        for key in ('all_203_features_and_world_state_values_bound_to_raw_cache',
                    'all_contexts_bound_to_exact_first_arrival_history', 'complete_original_sequence'):
            require(context[key] is True, 'raw context scope missing: '+key)
        require(context['GT_read'] is False and context['test_read'] is False, 'input boundary changed')
        for key in ('checkpoint_sha256', 'final_model_sha256', 'full_NN_numeric_completion_sha256'):
            require(context[key] == value[key], 'sequence final model differs: '+key)
        require(context['model_total_rows'] == byte['total_nodes'], 'model-total row namespace differs')
        for stage in ('structure', 'fresh_state', 'conditional_action_search', 'runtime_raw_cache203_context'):
            require(part[stage]['atol'] == part[stage]['rtol'] == 1e-8, 'stage tolerance changed')
    for summary, stage, key in (
        ('capacity_undecided_component_decisions', 'conditional_action_search', 'capacity_undecided_component_decisions'),
        ('fresh_conditional_states', 'fresh_state', 'states'), ('fresh_predictions', 'fresh_state', 'predictions'),
        ('model_output_class_recoveries', 'structure', 'output_class_recoveries_verified'),
        ('actions_outside_retained_classes', 'conditional_action_search', 'actions_outside_retained_classes'),
        ('incomplete_component_action_searches', 'conditional_action_search', 'incomplete_component_searches')):
        counts = [p[stage][key] for p in parts]
        require(all(type(n) is int and n >= 0 for n in counts), 'invalid stage counts')
        require(sum(counts) == value[summary], 'summary total differs: '+summary)


def gate(acceptance, byte_admission, seed):
    acceptance, byte_admission = Path(acceptance), Path(byte_admission)
    value, byte = (json.loads(p.read_bytes()) for p in (acceptance, byte_admission))
    require(sha(DRIVER/'source-freeze.json') == DRIVER_FREEZE_SHA, 'independent driver freeze changed')
    freeze = json.loads((DRIVER/'source-freeze.json').read_bytes())
    for name, digest in freeze['sources'].items():
        require(sha(DRIVER/name) == digest, 'frozen validator source changed: '+name)
    model = next(v for v in freeze['final_model_bindings'] if v['seed'] == seed)
    summary_gate(value, seed, model)
    require(value['byte_admission_sha256'] == sha(byte_admission), 'byte receipt changed')
    ledger = json.loads((ROOT/'receipts/20260928-execution-ledger.json').read_bytes())
    registered = [e for e in ledger['entries'] if Path(e.get('receipt', '')) == acceptance]
    require(len(registered) == 1 and registered[0]['receipt_sha256'] == sha(acceptance),
            'registered completed main acceptance required')
    # Load only the source-pinned metadata gate, never the expensive full oracle.
    dependency = 'rbf_nested_seen_val_v2_common'
    loaded = sys.modules.get(dependency)
    if loaded is not None:
        require(sha(loaded.__file__) == freeze['sources'][dependency+'.py'], 'ambient dependency differs')
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(DRIVER))
        spec = importlib.util.spec_from_file_location('final_teacher_frozen_main_binding', DRIVER/'rbf_final_refit_forest_binding.py')
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
        for key, digest in binding.source_gate().items():
            require(value[key] == digest, 'main driver binding differs: '+key)
        job = binding.validate(byte)  # checks live completed Task/artifacts and final-model lineage
    finally:
        sys.path[:] = original_path
    parts_paths = sorted(acceptance.parent.glob('sequence-*.json'))
    cohort_gate(value, byte, [json.loads(p.read_bytes()) for p in parts_paths])
    from clearml import Task
    params = Task.get_task(task_id=job['task_id']).get_parameters()
    require(params['General/recipe_sha256'] == value['recipe_sha256'], 'registered recipe differs')
    require(json.loads(params['General/plan']) == job['plan'], 'registered plan differs')
    return dict(kind='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1',
                seed=seed, main_task_id=job['task_id'], main_acceptance_sha256=sha(acceptance),
                byte_admission_sha256=sha(byte_admission), final_model_sha256=value['final_model_sha256'],
                sequence_proof_sha256={p.name: sha(p) for p in parts_paths},
                main_prerequisite_verified=True, teacher_runtime_or_targets_admitted=False,
                teacher_task_created=False, learned_Stage2_complete=False, paper_performance_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--byte-admission', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(1337, 2027, 3407), required=True)
    args = parser.parse_args()
    print(json.dumps(gate(args.acceptance, args.byte_admission, args.seed), allow_nan=False))
