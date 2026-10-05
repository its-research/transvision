"""Full learned cohort provenance and checkpoint bindings, without dispatch."""
import hashlib
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha
import rbf_final_refit_forest_binding as final_model
import rbf_final_refit_learned_output_binding as outputs
import rbf_independent_learned_trajectory as trajectory

MAIN = R / 'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
MAIN_SHA = 'f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
READER = R / 'source-freezes/rbf-final-refit-learned-priority-independent-output-reader-v1-20261005'
READER_SHA = '886d70f0ac5dc3264dd75419a06f81cbb85a3b40e89bd22ea2367f5e8b49593e'
TRAJECTORY = R / 'source-freezes/rbf-independent-learned-search-trajectory-v1-20261005'
TRAJECTORY_SHA = '7822cd9ab9269e5b28f4585f5c58b330daf95d2ed4c76a1799d0cd0b3feef00e'
JOURNAL = R / 'receipts/rbf-final-refit-learned-priority-full-train-GPU-dispatch-20261005.json'
KIND = 'rbf_final_refit_learned_full_cohort_independent_forest_and_priority_scope_acceptance_v1'


def arguments(parser):
    outputs.arguments(parser)
    for name in ('byte-admission', 'output', 'checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)


def source_gate():
    own = Path(__file__).resolve().parent
    freeze = json.loads((own / 'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_final_refit_learned_full_independent_CPU_source_v1'
    for name, item in freeze['sources'].items():
        assert sha(own / name) == item['sha256'] and (own / name).stat().st_size == item['bytes']
    for reference in freeze['references']:
        assert sha(reference['path']) == reference['sha256']
    for directory, expected in ((MAIN, MAIN_SHA), (READER, READER_SHA), (TRAJECTORY, TRAJECTORY_SHA)):
        assert sha(directory / 'source-freeze.json') == expected
    for module, origin in ((final_model, MAIN), (outputs, READER), (trajectory, TRAJECTORY)):
        path = Path(module.__file__).resolve()
        assert path.parent == own and sha(path) == sha(origin / path.name), 'ambient or modified acceptance module'
    for name in ('final_cache203.py', 'decoder_capacity.py'):
        assert sha(own / name) == sha(MAIN / name)
    outputs.source_gate()
    gate = R / 'receipts/rbf-independent-learned-search-trajectory-software-gate-20261005.json'
    proof = json.loads(gate.read_bytes())
    assert proof['source_freeze_sha256'] == TRAJECTORY_SHA and proof['frozen_tests_passed'] == 26
    assert proof['actual_producer_transitive_bytes_checked'] is True
    assert sha(proof['entrypoint_proof']) == proof['entrypoint_proof_sha256']
    assert proof['numeric_atol'] == proof['numeric_rtol'] == 1e-8
    assert proof['ordering_tolerance_used'] is proof['feature_bitwise_parity_claimed'] is False
    assert proof['actual_dataset_and_full_causal_trajectory_accepted'] is False
    return dict(driver_freeze_sha256=sha(own / 'source-freeze.json'),
        driver_source_sha256=sha(own / 'accept_final_refit_learned_cohort.py'),
        final_binding_source_sha256=sha(__file__), final_cache_source_sha256=sha(own / 'final_cache203.py'),
        source_control_sha256=sha(own / 'source-control.json'),
        learned_trajectory_source_freeze_sha256=TRAJECTORY_SHA,
        learned_trajectory_software_gate_sha256=sha(gate))


def validate_local(value, job, publication_sha):
    """Reject partial, mixed-seed or legacy outputs before any network query."""
    plan = job['plan']
    assert value['kind'] == outputs.KIND
    for key in ('all_registered_bytes_verified', 'all_46_sequences_7445_events_and_final_model_factor_nodes_verified',
                'all_rank_and_sequence_priority_identities_verified'):
        assert value[key] is True
    for key in ('learned_expansion_order_independently_verified', 'full_forest_semantics_or_fresh_state_independently_accepted',
                'learned_Stage2_complete', 'same_resource_performance_accepted', 'paper_performance_complete'):
        assert value[key] is False
    assert type(value['seed']) is int and value['seed'] == job['seed'] == plan['seed'] in (1337, 2027, 3407)
    assert value['task_id'] == job['task_id'] and value['method'] == plan['method'] == 'rbf'
    assert value['recipe_sha256'] == job['recipe_sha256'] == hashlib.sha256(outputs.dispatch.canonical(plan)).hexdigest()
    assert value['world_size'] == plan['world_size'] in (4, 8)
    assert value['NN_atol'] == value['NN_rtol'] == plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert value['source_sha256'] == sha(READER / 'read_final_refit_learned_outputs.py')
    assert value['final_checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert value['final_model_sha256'] == plan['final_refit_model_sha256'] != plan['original_nested_model_sha256']
    assert value['input_numeric_index_sha256'] == plan['numeric_reference_admission_sha256']
    assert value['priority_publication_sha256'] == publication_sha == job['priority_publication_sha256']
    assert value['priority_policy_signature'] == plan['priority_policy_signature']
    assert value['priority_checkpoint_sha256'] == plan['priority_inputs']['checkpoint']['sha256']
    assert plan['configuration']['allocation'] == 'learned'
    assert plan['configuration']['state']['max_model_regret'] == 1.
    assert set(value['artifacts']) == outputs.artifact_keys(plan['world_size'])
    sequences = value['sequences']
    assert len(sequences) == len({s['sequence_id'] for s in sequences}) == 46
    assert sum(s['events'] for s in sequences) == 7445
    assert sum(s['nodes'] for s in sequences) == value['total_nodes']
    for sequence in sequences:
        sid, digest = sequence['sequence_id'], sequence['database_sha256']
        assert isinstance(sid, str) and sid and '/' not in sid and '\\' not in sid and sid not in ('.', '..')
        assert type(sequence['events']) is int and sequence['events'] > 0
        assert type(sequence['nodes']) is int and sequence['nodes'] >= 0
        assert isinstance(digest, str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef')
    return plan


def validate(value, args, Task):
    control = outputs.source_gate()
    jobs = [j for j in json.loads(JOURNAL.read_bytes())['jobs'] if j['task_id'] == value['task_id']]
    assert len(jobs) == 1
    job = jobs[0]
    assert args.seed == value['seed']
    plan = validate_local(value, job, sha(args.priority_publication))
    ledger = json.loads((R / 'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(e.get('kind') == outputs.KIND and e.get('receipt') == str(args.byte_admission.absolute())
        and e.get('receipt_sha256') == sha(args.byte_admission) for e in ledger['entries']), 'unregistered or changed byte admission'
    assert not any(p.is_symlink() for p in (args.byte_admission, *args.byte_admission.parents))
    task = Task.get_task(task_id=job['task_id'])
    assert str(task.status) == 'completed'
    outputs.qualify_job(args, Task, job, task, control)
    outputs.verify_task_unchanged(task, job, value['artifacts'], 'completed')
    model = final_model.validate_final_model(args.seed, args.checkpoint, plan=plan)
    assert value['total_nodes'] == model['rows']
    priority_weights = args.run.absolute() / 'fit' / str(args.seed) / 'weights.npz'
    assert sha(priority_weights) == plan['priority_inputs']['weights']['sha256']
    _, signature = trajectory.policy_weights(priority_weights, plan['priority_inputs']['weights']['sha256'])
    assert signature == plan['priority_policy_signature']
    return job


def check_sequence(directory, sequence, job):
    """Rebind the exact files used by each oracle, including native cost logs."""
    plan = job['plan']
    receipt = json.loads(outputs.contained(directory, 'receipt.json').read_bytes())
    per_plan = json.loads(outputs.contained(directory, 'plan.json').read_bytes())
    sid = sequence['sequence_id']
    outputs.verify_priority_sequence(plan, per_plan, receipt, sid, sequence['events'])
    assert per_plan['model_binding']['checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert per_plan['model_binding']['model_sha256'] == plan['final_refit_model_sha256']
    assert receipt['status'] == 'software_replay_completed'
    for name, digest in receipt['files'].items():
        assert sha(outputs.contained(directory, name)) == digest
    info = receipt['databases'][sid]
    assert info['sha256'] == sequence['database_sha256']
    database = outputs.contained(directory, info['path'])
    assert sha(database) == info['sha256']
    return database


def ordering(database, sequence, args, job, progress):
    plan = job['plan']
    return trajectory.verify_database(database, sequence['database_sha256'],
        args.run.absolute() / 'fit' / str(args.seed) / 'weights.npz',
        plan['priority_inputs']['weights']['sha256'], plan['priority_policy_signature'], progress)


def ordering_summary(results, job):
    assert len(results) == 46 and len({r['sequence_id'] for r in results}) == 46
    total = rows = operations = charged = 0
    differences = []
    for result in results:
        value = result['learned_search_trajectory']
        assert value['kind'] == 'rbf_independent_exclusive_learned_search_trajectory_checks_v1'
        assert value['database_sha256'] == result['database_sha256']
        assert value['policy_signature'] == job['plan']['priority_policy_signature']
        assert value['weights_sha256'] == job['plan']['priority_inputs']['weights']['sha256']
        assert value['events'] == result['structure']['events']
        assert value['observations'] == result['runtime_raw_cache203_context']['rows']
        assert value['atol'] == value['rtol'] == 1e-8
        assert value['all_18_features_and_MLP_scores_independently_recomputed'] is True
        assert value['all_float64_selections_exactly_replayed_from_independently_checked_features'] is True
        assert value['ordering_tolerance_used'] is value['feature_bitwise_parity_claimed'] is False
        total += value['events'];rows += value['candidate_feature_rows']
        operations += value['actual_selected_operations'];charged += value['actual_selected_charged_steps']
        differences.extend(dict(sequence_id=result['sequence_id'], **item)
                           for item in value['independent_compensated_arithmetic_order_differences'])
    assert total == 7445
    return dict(learned_expansion_trajectory_accepted=True,
        all_float64_selections_exactly_replayed_from_independently_checked_features=True,
        priority_policy_signature=job['plan']['priority_policy_signature'],
        priority_weights_sha256=job['plan']['priority_inputs']['weights']['sha256'],
        priority_checkpoint_sha256=job['plan']['priority_inputs']['checkpoint']['sha256'],
        candidate_feature_rows=rows, selected_search_operations=operations, selected_charged_steps=charged,
        independent_compensated_arithmetic_order_differences=differences,
        ordering_tolerance_used=False, feature_bitwise_parity_claimed=False,
        priority_numeric_atol=1e-8, priority_numeric_rtol=1e-8)
