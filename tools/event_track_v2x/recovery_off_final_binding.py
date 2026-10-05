"""Separate, fail-closed binding for recovery-off full independent verification.

Only final model/cache admissions transfer. Main-forest structural, action and
state acceptance never transfers. This module contacts ClearML only in validate.
"""
import contextlib
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha

S = R/'source-freezes'
PRODUCER = S/'rbf-final-refit-recovery-off-bound-GPU-v1-20261005'
PARENT = S/'rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
JOURNAL = R/'receipts/rbf-final-refit-recovery-off-bound-GPU-dispatch-20261005.json'
MODEL_D = S/'rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
STRUCTURE_D = S/'rbf-independent-exclusive-forest-structure-recovery-v1-20261002'
FRESH = S/'rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py'
BYTE_KIND = 'rbf_final_refit_recovery_off_bound_independent_bytes_events_factor_admission_v1'
READER_SHA = 'ff3e4fc4c4023ac894289e355c63c07d384c9eaa9b86b520947d319d749f2444'
INPUT_SHA = '760e966dbeb9b4652c69d6641985ae5be806c179cc6d3dbf6a08261102a1481f'
CANDIDATE_SHA = 'b778aee162f177066da207b70ecc807578be687373583a6b9748fa63b3aa9b70'
UNCHANGED = {
    MODEL_D/'final_cache203.py': 'f1c2c1d5eb2b575b0029740deefb2afb37573da5302c9f1fe5ac67c792375801',
    MODEL_D/'rbf_final_refit_forest_binding.py': 'd51fdf5ff8de7d7d7cc48d38ffedaa58f6f9fb3db7563013947a14112e438e92',
    MODEL_D/'rbf_nested_seen_val_v2_common.py': '58e616f116155b9bd2a9f0ed52b6b0ebc47895c6e9957271a6d88db51b6502ec',
    STRUCTURE_D/'causal.py': '0579f22b617b17ba9139c2a8ce67cbff8c6e1c1e1f154422a311ebfb2cdcad32',
    STRUCTURE_D/'oracle.py': '9db8a185698163081b18ac0beb7bf40ec408dfde8b5ac95c1d8786170797e939',
    S/'rbf-independent-identity-forest-audit-v1-20261001/oracle.py': '24be6bf660cbecadc3102320e18e5b12ff06063b050e4dfc44307d4defeea8a8',
    FRESH: '46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58',
}
REQUIRED_SOURCES = {
    'recovery_off_final_binding.py', 'accept_recovery_off_final_cohort.py',
    'recovery_off_structure_oracle.py', 'recovery_off_action_oracle.py',
    'rbf_nested_seen_val_v2_common.py',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and nested imports need a stable, explicitly chosen identity.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextlib.contextmanager
def aliases(values):
    previous = {name: sys.modules.get(name) for name in values}
    sys.modules.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_unchanged_oracles():
    for path, expected in UNCHANGED.items():
        assert sha(path) == expected, ('changed independent source', str(path))
    common = load_module('_rbf_recovery_original_common', MODEL_D/'rbf_nested_seen_val_v2_common.py')
    with aliases({'rbf_nested_seen_val_v2_common': common}):
        model = load_module('_rbf_recovery_original_final_model_binding', MODEL_D/'rbf_final_refit_forest_binding.py')
    with aliases({'rbf_final_refit_forest_binding': model}):
        cache = load_module('_rbf_recovery_original_final_cache203', MODEL_D/'final_cache203.py')
    # causal only imports canonical/digest/sha from this old structure module;
    # do not call its original, incompatible structural verification function.
    structure_helpers = load_module('_rbf_recovery_original_structure_helpers', STRUCTURE_D/'oracle.py')
    with aliases({'oracle': structure_helpers}):
        causal = load_module('_rbf_recovery_original_causal', STRUCTURE_D/'causal.py')
    fresh = load_module('_rbf_recovery_original_fresh', FRESH)
    return model, cache, causal, fresh


def source_gate(directory=None):
    """Require a real integrated software qualification, not just source files."""
    directory = Path(directory or Path(__file__).resolve().parent)
    freeze = json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_recovery_off_full_independent_CPU_source_v1'
    assert REQUIRED_SOURCES <= set(freeze['sources'])
    for name, expected in freeze['sources'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(directory/relative) == expected
    assert freeze['unchanged_independent_sources'] == {str(p): h for p, h in UNCHANGED.items()}
    for path, expected in UNCHANGED.items():
        assert sha(path) == expected
    gate = json.loads((directory/'qualification.json').read_bytes())
    assert gate['kind'] == 'rbf_recovery_off_full_independent_CPU_software_gate_v1'
    assert gate['source_freeze_sha256'] == sha(directory/'source-freeze.json')
    assert gate['sources'] == freeze['sources']
    for flag in ('integrated_pass', 'binding_negative_controls_passed',
                 'restricted_structure_negative_controls_passed', 'restricted_action_negative_controls_passed',
                 'unchanged_cache_causal_fresh_schema_integration_passed'):
        assert gate[flag] is True, flag
    assert gate['original_experiment_acceptance_inherited'] is False
    assert gate['real_cohort_accepted'] is gate['paper_performance_complete'] is False
    assert gate['evidence']
    for item in gate['evidence']:
        assert sha(item['path']) == item['sha256']
    return dict(driver_source_freeze_sha256=sha(directory/'source-freeze.json'),
                software_qualification_sha256=sha(directory/'qualification.json'),
                sources=freeze['sources'], unchanged_independent_sources=freeze['unchanged_independent_sources'])


def validate_local_output(value, job, template, parent_plan):
    """Pure rejection gates before network or large cache reads."""
    assert value['kind'] == BYTE_KIND
    for flag in ('all_registered_bytes_verified', 'all_46_sequences_7445_events_and_final_model_factor_nodes_verified'):
        assert value[flag] is True
    for flag in ('full_forest_semantics_or_fresh_state_independently_accepted', 'learned_Stage2_complete',
                 'same_resource_performance_accepted', 'paper_performance_complete'):
        assert value[flag] is False
    plan = job['plan']
    assert value['seed'] == job['seed'] == plan['seed'] in (1337, 2027, 3407)
    assert value['method'] == plan['method'] == 'rbf'
    assert value['task_id'] == job['task_id']
    assert value['recipe_sha256'] == job['recipe_sha256'] == hashlib.sha256(canonical(plan)).hexdigest()
    assert {k: v for k, v in plan.items() if k != 'world_size'} == {k: v for k, v in template.items() if k != 'world_size'}
    assert type(plan['world_size']) is int and value['world_size'] == plan['world_size'] in (4, 8)
    assert value['NN_atol'] == value['NN_rtol'] == plan['scoring_atol'] == plan['scoring_rtol'] == 1e-4
    assert value['source_sha256'] == READER_SHA
    assert value['final_checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert value['final_model_sha256'] == plan['final_refit_model_sha256'] != plan['original_nested_model_sha256']
    assert value['input_numeric_index_sha256'] == plan['numeric_reference_admission_sha256']
    assert plan['recovery_off_source_freeze_sha256'] == CANDIDATE_SHA
    assert plan['recovery_off_real_checkpoint_input_contract_sha256'] == INPUT_SHA
    for flag in ('original_exclusive_acceptance_inherited', 'recovery_off_independent_semantics_accepted',
                 'full_forest_independently_accepted', 'learned_Stage2_complete', 'paper_performance_complete'):
        assert plan[flag] is False
    configuration = copy.deepcopy(parent_plan['configuration'])
    configuration['backend'] = 'exclusive_event_boundary_recovery_off_v1'
    configuration['limits']['recovery_off_version'] = 1
    assert plan['configuration'] == configuration
    assert plan['configuration']['allocation'] == 'bound'
    for name in ('seed', 'source', 'events', 'cache_archive', 'cache_manifest', 'checkpoint', 'weights_archive',
                 'final_refit_model_sha256', 'original_nested_model_sha256', 'numeric_reference_admission_sha256',
                 'forward_outputs', 'final_refit_training_byte_admission_sha256',
                 'forward_full_byte_and_coverage_admission_sha256', 'final_refit_NN_numeric_completion_sha256'):
        assert plan[name] == parent_plan[name], ('unexpected ablation input change', name)
    expected = {'receipt', 'exclusive-source-manifest', *(f'replay-rank{i}' for i in range(plan['world_size']))}
    assert set(value['artifacts']) == expected
    rows = value['sequences']
    assert len(rows) == len({v['sequence_id'] for v in rows}) == 46
    assert sum(v['events'] for v in rows) == 7445
    assert sum(v['nodes'] for v in rows) == value['total_nodes']
    for row in rows:
        sid = row['sequence_id']
        assert isinstance(sid, str) and sid and '/' not in sid and '\\' not in sid and sid not in ('.', '..')
        assert type(row['events']) is int and row['events'] > 0
        assert type(row['nodes']) is int and row['nodes'] >= 0
        digest = row['database_sha256']
        assert isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    return plan


def validate(value, checkpoint):
    source_gate()
    prepared = json.loads((PRODUCER/'preparation.json').read_bytes())
    parent = json.loads((PARENT/'preparation.json').read_bytes())
    jobs = [job for job in json.loads(JOURNAL.read_bytes())['jobs'] if job['task_id'] == value['task_id']]
    assert len(jobs) == 1
    job = jobs[0]
    template = next(v['plan'] for v in prepared['seeds'] if v['seed'] == value['seed'])
    parent_plan = next(v['plan'] for v in parent['seeds'] if v['seed'] == value['seed'])
    plan = validate_local_output(value, job, template, parent_plan)
    gate_path = PRODUCER/'recovery_off_dispatch_gate.py'
    assert sha(gate_path) == prepared['execution_sources'][gate_path.name]['sha256']
    producer_gate = load_module('_rbf_recovery_original_dispatch_gate', gate_path)
    producer_proof = producer_gate.require_qualified_source(PRODUCER, prepared)
    for item in prepared['prerequisite_receipts']:
        assert sha(item['path']) == item['sha256']
    model, _, _, _ = load_unchanged_oracles()
    model_binding = model.validate_final_model(value['seed'], checkpoint, plan=plan)
    assert value['total_nodes'] == model_binding['rows']
    from clearml import Task
    task = Task.get_task(task_id=value['task_id'])
    assert str(task.status) == 'completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    parameters = task.get_parameters()
    assert json.loads(parameters['General/plan']) == plan
    assert parameters['General/recipe_sha256'] == job['recipe_sha256']
    assert set(task.artifacts) == set(value['artifacts'])
    for key, record in value['artifacts'].items():
        assert task.artifacts[key].hash == record['sha256'] and task.artifacts[key].size == record['bytes']
    return job, dict(model_binding, producer_qualification_sha256=sha(PRODUCER/'source-qualification.json'),
                     producer_preparation_sha256=sha(PRODUCER/'preparation.json'),
                     producer_source_qualified=producer_proof['actual_deployed_sources_match_qualified_candidate'],
                     original_experiment_acceptance_inherited=False)
