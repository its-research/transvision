"""Final-refit fixed TopK input/output bindings; no same-resource claim."""
import hashlib
import json
from pathlib import Path
import sys

R = Path('/Volumes/Data/test/recover-before-fuse')
MAIN_CPU = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004'
sys.path.insert(0, str(MAIN_CPU))
from rbf_final_refit_forest_binding import INDEX, validate_final_model
from rbf_nested_seen_val_v2_common import sha

PRODUCER = R/'source-freezes/rbf-final-refit-full-train-fixed-topK-GPU-v1-20261004'
JOURNAL = R/'receipts/rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json'
OLD_JOURNAL = R/'receipts/rbf-original-joint-coupled-topk-GPU4-v5-native34-dispatch-20261001.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_local_output(value, job, template, old_job):
    assert value['kind'] == 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'
    assert value['all_registered_bytes_verified'] is True
    assert value['all_46_sequences_7445_events_and_final_model_factor_nodes_verified'] is True
    assert value['full_forest_semantics_or_fresh_state_independently_accepted'] is False
    assert value['learned_Stage2_complete'] is value['same_resource_performance_accepted'] is value['paper_performance_complete'] is False
    plan = job['plan']
    assert value['method'] == plan['method'] == 'topk'
    assert type(value['seed']) is int and value['seed'] == job['seed'] == plan['seed'] in (1337, 2027, 3407)
    assert value['task_id'] == job['task_id']
    assert value['recipe_sha256'] == job['recipe_sha256'] == hashlib.sha256(canonical(plan)).hexdigest()
    assert value['world_size'] == plan['world_size'] in (4, 8)
    assert {k:v for k,v in plan.items() if k != 'world_size'} == {k:v for k,v in template.items() if k != 'world_size'}
    assert value['NN_atol'] == value['NN_rtol'] == plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert value['final_checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert value['final_model_sha256'] == plan['final_refit_model_sha256'] != plan['original_nested_model_sha256']
    assert value['input_numeric_index_sha256'] == plan['numeric_reference_admission_sha256']
    assert old_job['seed'] == job['seed']
    for key in ('configuration', 'source', 'events', 'cache_archive', 'cache_manifest'):
        assert plan[key] == old_job['plan'][key]
    config = plan['configuration']
    assert config['method'] == 'topk' and config['history_features'] is True
    assert config['state']['active_limit'] == 4 and config['state']['decision_mode'] == 'retained'
    assert config['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert set(value['artifacts']) == {'receipt', *(f'replay-rank{i}' for i in range(plan['world_size']))}
    sequences = value['sequences']
    assert len(sequences) == len({x['sequence_id'] for x in sequences}) == 46
    assert sum(x['events'] for x in sequences) == 7445
    assert sum(x['nodes'] for x in sequences) == value['total_nodes']
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
    assert freeze['kind'] == 'rbf_final_refit_fixed_topK_independent_CPU_source_v1'
    for name, spec in freeze['sources'].items():
        assert sha(directory/name) == spec['sha256'] and (directory/name).stat().st_size == spec['bytes']
    for item in freeze['unchanged_references']:
        assert sha(item['path']) == item['sha256']
    main_freeze = json.loads((MAIN_CPU/'source-freeze.json').read_bytes())
    for name, digest in main_freeze['sources'].items():
        assert sha(MAIN_CPU/name) == digest
    assert freeze['reference_final_cache_sha256'] == sha(MAIN_CPU/'final_cache203.py')
    return dict(source_freeze_sha256=sha(directory/'source-freeze.json'),
        driver_sha256=sha(directory/'accept_final_refit_topk_cohort.py'),
        binding_sha256=sha(__file__), final_model_cache_constructor_sha256=sha(MAIN_CPU/'final_cache203.py'))


def validate_output(value):
    source_binding = source_gate()
    assert value['kind'] == 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'
    assert value['method'] == 'topk'
    job = next(v for v in json.loads(JOURNAL.read_bytes())['jobs'] if v['task_id'] == value['task_id'])
    template = next(v['plan'] for v in json.loads((PRODUCER/'preparation.json').read_bytes())['seeds'] if v['seed'] == value['seed'])
    old_job = next(v for v in json.loads(OLD_JOURNAL.read_bytes())['jobs'] if v['seed'] == value['seed'])
    plan = validate_local_output(value, job, template, old_job)
    entry = next(v for v in json.loads(INDEX.read_bytes())['seeds'] if v['seed'] == value['seed'])
    checkpoint = Path(entry['training_byte_proof']).parent/'checkpoint'
    model_binding = validate_final_model(value['seed'], checkpoint, plan=plan)
    assert value['total_nodes'] == model_binding['rows']
    from clearml import Task
    task = Task.get_task(task_id=value['task_id'])
    assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    assert set(task.artifacts) == set(value['artifacts'])
    for key, spec in value['artifacts'].items():
        assert task.artifacts[key].hash == spec['sha256'] and task.artifacts[key].size == spec['bytes']
    return job, checkpoint, model_binding, source_binding
