"""Corrupt receipt controls; fixture admission is never experiment evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('teacher_prereq', Path(__file__).resolve().parents[2]/
    'tools/event_track_v2x/rbf_final_refit_teacher_prerequisites.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def fixture():
    model = dict(checkpoint_sha256='f'*64, final_model_sha256='a'*64,
                 original_model_sha256='b'*64, full_NN_numeric_completion_sha256='c'*64)
    value = dict(kind=gate.KIND, seed=2027, task_id='final-task', recipe_sha256='d'*64,
                 completed_sequences=46, completed_events=7445, atol=1e-8, rtol=1e-8,
                 capacity_undecided_component_decisions=46,
                 conditional_full_legal_action_search_and_declared_gap_verified=False,
                 fresh_conditional_states=92, fresh_predictions=138, model_output_class_recoveries=46,
                 actions_outside_retained_classes=46, incomplete_component_action_searches=46, **model)
    value.update({k: True for k in gate.TRUE_FLAGS})
    value.update({k: False for k in gate.FALSE_FLAGS})
    parts, sequences = [], []
    for i in range(46):
        sid, count, digest = f'{i:04d}', (200 if i == 45 else 161), f'{i:064x}'
        seq = dict(sequence_id=sid, database_sha256=digest, events=count, nodes=3)
        sequences.append(seq)
        stages = {k: dict(events=count, database_sha256=digest, atol=1e-8, rtol=1e-8) for k in gate.STAGES}
        stages['structure'].update(output_class_recoveries_verified=1,
            strict_explicit_residual_partition_pass=True, declared_fixed_support_coverage_pass=True,
            all_padded_mass_arithmetic_independently_verified=True)
        stages['causal'].update(observations=3, raw_observation_bytes_and_scalar_columns_verified=True,
            all_cumulative_raw_factor_commit_digests_verified=True,
            all_admitted_information_and_arrivals_before_decision=True, no_old_factor_rescore=True)
        stages['fresh_state'].update(states=2, predictions=3, stored_states_and_chosen_outputs_checked=True)
        stages['conditional_action_search'].update(capacity_undecided_counted_as_optimal=False,
            all_selected_actions_raw_support_legal=True, bounded_search_gap_and_shared_budget_verified=True,
            action_domain_not_restricted_to_retained_posterior=True, own_selected_state_export_verified=True,
            capacity_undecided_component_decisions=1, actions_outside_retained_classes=1, incomplete_component_searches=1)
        stages['runtime_raw_cache203_context'].update(rows=3, model_total_rows=138,
            all_203_features_and_world_state_values_bound_to_raw_cache=True,
            all_contexts_bound_to_exact_first_arrival_history=True, complete_original_sequence=True,
            GT_read=False, test_read=False, **model)
        parts.append(dict(sequence_id=sid, database_sha256=digest, **stages))
    byte = dict(seed=2027, task_id='final-task', recipe_sha256='d'*64, method='rbf', total_nodes=138, sequences=sequences)
    return value, model, byte, parts


def test_complete_metadata_and_capacity_semantics():
    value, model, byte, parts = fixture()
    gate.summary_gate(value, 2027, model)
    gate.cohort_gate(value, byte, parts)


@pytest.mark.parametrize('key,bad', [
    ('kind', 'rbf_full_exclusive_capacity_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'),
    ('seed', 1337), ('completed_events', 7444), ('completed_sequences', 45),
    ('old_model_full_forest_acceptance_inherited', True), ('atol', 1e-4),
    ('capacity_undecided_component_decisions', True),
    ('conditional_full_legal_action_search_and_declared_gap_verified', True),
    ('checkpoint_sha256', '0'*64), ('learned_Stage2_complete', True)])
def test_summary_rejects_old_partial_or_overclaimed_receipts(key, bad):
    value, model, _, _ = fixture()
    value[key] = bad
    with pytest.raises(ValueError):
        gate.summary_gate(value, 2027, model)


@pytest.mark.parametrize('mutation', ['duplicate', 'baseline', 'database', 'model', 'row_namespace',
                                     'partial_stage', 'optimal', 'state', 'leak', 'sum', 'causal', 'support'])
def test_cohort_rejects_mixed_model_partial_and_corrupted_evidence(mutation):
    value, _, byte, parts = fixture()
    first = parts[0]
    if mutation == 'duplicate': parts[-1] = copy.deepcopy(first)
    elif mutation == 'baseline': byte['method'] = 'topk'
    elif mutation == 'database': first['fresh_state']['database_sha256'] = '0'*63+'1'
    elif mutation == 'model': first['runtime_raw_cache203_context']['final_model_sha256'] = 'b'*64
    elif mutation == 'row_namespace': first['runtime_raw_cache203_context']['model_total_rows'] = 3
    elif mutation == 'partial_stage': first['causal']['events'] -= 1
    elif mutation == 'optimal': first['conditional_action_search']['capacity_undecided_counted_as_optimal'] = True
    elif mutation == 'state': first['fresh_state']['stored_states_and_chosen_outputs_checked'] = False
    elif mutation == 'leak': first['runtime_raw_cache203_context']['GT_read'] = True
    elif mutation == 'sum': value['fresh_predictions'] += 1
    elif mutation == 'causal': first['causal']['no_old_factor_rescore'] = False
    elif mutation == 'support': first['structure']['declared_fixed_support_coverage_pass'] = False
    with pytest.raises(ValueError):
        gate.cohort_gate(value, byte, parts)
