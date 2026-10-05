"""Fail-closed final-refit bindings for the separate full-forest CPU oracle.

These metadata gates reuse accepted bytes and NN results. They do not repeat
numeric admission or turn source/software checks into forest acceptance.
"""
import hashlib
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha

INDEX = R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'
JOURNAL = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
OLD_JOURNAL = R/'receipts/rbf-exclusive-action-capacity-undecided-GPU4-v2-dispatch-20261002.json'
OLD_DRIVER = R/'source-freezes/rbf-independent-action-capacity-undecided-v2-20261002'
PRODUCER = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def original_checkpoint(seed):
    return R/f'artifacts/rbf-joint-identity-full-independent-numpy-v1-20261001/seed{seed}/weights_archive-unpack/training/seed-{seed}/checkpoint.json'


def validate_final_model(seed, checkpoint, original=None, plan=None):
    assert type(seed) is int and seed in (1337, 2027, 3407)
    index = json.loads(INDEX.read_bytes())
    assert index['kind'] == 'rbf_new_nested_selected_all_class_final_refit_three_seed_independent_full_numeric_index_v1'
    for flag in ('all_registered_bytes_accepted', 'all_46_original_paired_train_sequences_and_query_rows_accepted',
                 'full_independent_joint_attention_cross_source_temporal_motion_numerical_pass'):
        assert index[flag] is True
    assert index['rows'] == 677744 and index['numeric_failed_rows'] == 0
    assert index['atol'] == index['rtol'] == 1e-4 and index['tolerance_changed'] is False
    assert index['paper_performance_complete'] is False
    entry = next(v for v in index['seeds'] if v['seed'] == seed)
    for name in ('training_byte_proof', 'prediction_byte_proof', 'numeric_completion'):
        assert sha(entry[name]) == entry[name+'_sha256']
    train = json.loads(Path(entry['training_byte_proof']).read_bytes())
    forward = json.loads(Path(entry['prediction_byte_proof']).read_bytes())
    numeric = json.loads(Path(entry['numeric_completion']).read_bytes())
    assert train['kind'] == 'rbf_new_final_refit_six_artifacts_independent_byte_metadata_admission_v1'
    assert train['seed'] == forward['seed'] == numeric['seed'] == seed
    assert train['task_id'] == entry['training_task_id'] and forward['task_id'] == entry['forward_task_id']
    assert train['all_six_registered_artifact_bytes_independently_read'] is True
    assert train['frozen_nested_epoch_and_full_train_recipe_verified'] is True
    assert forward['kind'] == 'rbf_new_final_refit_all_row_independent_byte_coverage_admission_v1'
    assert forward['all_registered_output_bytes_verified'] is True
    assert forward['all_46_train_sequences_row_coverage_verified'] is True
    assert forward['train_byte_admission_sha256'] == entry['training_byte_proof_sha256']
    assert numeric['kind'] == 'rbf_new_final_refit_full_independent_float64_joint_reference_v1'
    assert numeric['full_independent_numeric_pass'] is True and numeric['numeric_failed_rows'] == 0
    assert numeric['prediction_proof_sha256'] == entry['prediction_byte_proof_sha256']
    assert numeric['atol'] == numeric['rtol'] == 1e-4
    assert numeric['rows'] == forward['rows'] == entry['rows']
    assert len(forward['sequence_counts']) == len(numeric['sequences']) == 46
    assert sum(forward['sequence_counts'].values()) == entry['rows']
    assert {v['sequence_id']: v['rows'] for v in numeric['sequences']} == forward['sequence_counts']
    assert sha(checkpoint) == train['artifacts']['checkpoint']['sha256']
    ck = json.loads(Path(checkpoint).read_bytes())
    old = json.loads(Path(original or original_checkpoint(seed)).read_bytes())
    assert ck['seed'] == old['seed'] == seed and ck['labels_in_model_inputs'] is False
    assert ck['selection'] == 'frozen_nested_selected_epoch_full_train_refit'
    assert ck['selected_epoch'] == entry['epochs'] and ck['validation_or_test_selection'] is False
    assert ck['weights']['sha256'] == train['weights_sha256'] == forward['weights_sha256'] == numeric['weights_sha256']
    assert ck['model_sha256'] == train['model_sha256'] and ck['model_sha256'] != old['model_sha256']
    assert ck['row_protocol'] == old['row_protocol'] and ck['frozen_cache_identity'] == old['frozen_cache_identity']
    protocol = ck['row_protocol']
    assert protocol['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert protocol['class_scope'] == ['car', 'bicycle', 'pedestrian'] and protocol['evaluation_class'] == 'car'
    assert protocol['minimum_raw_score'] == .05 and protocol['maximum_detections'] == 64
    assert protocol['old_row_rescoring'] is False
    if plan is not None:
        assert plan['seed'] == seed and plan['checkpoint'] == train['artifacts']['checkpoint']
        assert plan['weights_archive'] == train['artifacts']['identity-training']
        assert plan['final_refit_model_sha256'] == ck['model_sha256']
        assert plan['original_nested_model_sha256'] == old['model_sha256']
        assert plan['numeric_reference_admission_sha256'] == sha(INDEX)
        assert plan['final_refit_training_byte_admission_sha256'] == entry['training_byte_proof_sha256']
        assert plan['forward_full_byte_and_coverage_admission_sha256'] == entry['prediction_byte_proof_sha256']
        assert plan['final_refit_NN_numeric_completion_sha256'] == entry['numeric_completion_sha256']
        assert plan['forward_outputs'] == [v for k, v in sorted(forward['artifacts'].items()) if k.startswith('predictions-rank')]
    return dict(seed=seed, checkpoint_sha256=sha(checkpoint), final_model_sha256=ck['model_sha256'],
        original_model_sha256=old['model_sha256'], index_sha256=sha(INDEX),
        training_byte_proof_sha256=entry['training_byte_proof_sha256'],
        forward_byte_proof_sha256=entry['prediction_byte_proof_sha256'],
        full_NN_numeric_completion_sha256=entry['numeric_completion_sha256'],
        rows=entry['rows'], original_cache_and_row_protocol_unchanged=True,
        old_model_full_forest_acceptance_inherited=False, full_forest_accepted=False)


def validate_local_output(value, job, old_job):
    """Pure output/recipe gate; failures happen before any remote API calls."""
    assert value['kind'] == 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'
    assert value['all_registered_bytes_verified'] is True
    assert value['all_46_sequences_7445_events_and_final_model_factor_nodes_verified'] is True
    assert value['full_forest_semantics_or_fresh_state_independently_accepted'] is False
    assert value['learned_Stage2_complete'] is value['same_resource_performance_accepted'] is value['paper_performance_complete'] is False
    plan = job['plan']
    assert value['seed'] == job['seed'] == plan['seed'] in (1337, 2027, 3407)
    assert value['task_id'] == job['task_id'] and value['method'] == plan['method'] == 'rbf'
    assert value['recipe_sha256'] == job['recipe_sha256'] == hashlib.sha256(canonical(plan)).hexdigest()
    assert value['world_size'] == plan['world_size'] in (4, 8)
    assert value['NN_atol'] == value['NN_rtol'] == plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert value['final_checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert value['final_model_sha256'] == plan['final_refit_model_sha256'] != plan['original_nested_model_sha256']
    assert value['input_numeric_index_sha256'] == plan['numeric_reference_admission_sha256']
    assert old_job['seed'] == job['seed']
    for name in ('configuration', 'exclusive_patches', 'source_replacements', 'CPU_capacity_candidate_admission',
                 'exclusive_source_freeze_sha256', 'source', 'events', 'cache_archive', 'cache_manifest'):
        assert plan[name] == old_job['plan'][name], ('changed frozen core/input', name)
    assert plan['configuration']['backend'] == 'exclusive_root_partition_regions_v1'
    assert plan['configuration']['limits']['residual_partition_version'] == 1
    expected = {'receipt', 'exclusive-source-manifest', *(f'replay-rank{i}' for i in range(plan['world_size']))}
    assert set(value['artifacts']) == expected
    sequences = value['sequences']
    assert len(sequences) == len({v['sequence_id'] for v in sequences}) == 46
    assert sum(v['events'] for v in sequences) == 7445
    assert sum(v['nodes'] for v in sequences) == value['total_nodes']
    for seq in sequences:
        sid = seq['sequence_id']
        assert isinstance(sid, str) and sid and '/' not in sid and '\\' not in sid and sid not in ('.', '..')
        assert type(seq['events']) is int and seq['events'] > 0
        assert type(seq['nodes']) is int and seq['nodes'] >= 0
        digest = seq['database_sha256']
        assert isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    return plan


def source_gate():
    directory = Path(__file__).resolve().parent
    freeze = json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_final_refit_full_forest_independent_CPU_source_v1'
    for name, digest in freeze['sources'].items():
        assert sha(directory/name) == digest
    for item in freeze['unchanged_independent_sources']:
        assert sha(item['path']) == item['sha256']
    return dict(driver_freeze_sha256=sha(directory/'source-freeze.json'),
        final_binding_source_sha256=sha(__file__), final_cache_source_sha256=sha(directory/'final_cache203.py'),
        driver_source_sha256=sha(directory/'accept_final_refit_cohort.py'),
        source_control_sha256=sha(directory/'source-control.json'))


def validate(value):
    source_gate()
    assert value['kind'] == 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'
    job = next(v for v in json.loads(JOURNAL.read_bytes())['jobs'] if v['task_id'] == value['task_id'])
    old_job = next(v for v in json.loads(OLD_JOURNAL.read_bytes())['jobs'] if v['seed'] == value['seed'])
    plan = validate_local_output(value, job, old_job)
    entry = next(v for v in json.loads(INDEX.read_bytes())['seeds'] if v['seed'] == value['seed'])
    checkpoint = Path(entry['training_byte_proof']).parent/'checkpoint'
    binding = validate_final_model(value['seed'], checkpoint, plan=plan)
    assert value['total_nodes'] == binding['rows']
    producer = json.loads((PRODUCER/'preparation.json').read_bytes())
    template = next(v['plan'] for v in producer['seeds'] if v['seed'] == value['seed'])
    assert {k:v for k,v in plan.items() if k != 'world_size'} == {k:v for k,v in template.items() if k != 'world_size'}
    cf_path = R/'source-freezes/rbf-exclusive-action-capacity-undecided-GPU4-v2-20261002/source-freeze.json'
    assert sha(cf_path) == plan['exclusive_source_freeze_sha256']
    cf = json.loads(cf_path.read_bytes())
    assert cf['source_replacements'] == plan['source_replacements']
    assert cf['configuration_sha256'] == hashlib.sha256(canonical(plan['configuration'])).hexdigest()
    cpu = R/'artifacts/rbf-exclusive-main-action-capacity-undecided-CPU-v1-20261002/independent-readback/diagnostic-byte-binding-receipt.json'
    proof = json.loads(cpu.read_bytes())
    assert sha(cpu) == plan['CPU_capacity_candidate_admission']['independent_receipt_sha256']
    assert proof['task_id'] == plan['CPU_capacity_candidate_admission']['task_id']
    assert proof['CPU_committed_events'] == 86 and proof['CPU_error_binding'] is None
    assert proof['capacity_undecided_event_committed'] is True and proof['first85_original_prediction_bytes_identical'] is True
    assert proof['source_core_repair_tests']['passed'] is True
    from clearml import Task
    task = Task.get_task(task_id=value['task_id'])
    assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    assert set(task.artifacts) == set(value['artifacts'])
    for key, spec in value['artifacts'].items():
        assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
    return job
