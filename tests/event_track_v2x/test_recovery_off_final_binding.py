"""Reject mismatched experiment identities before any remote or large reads."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
import recovery_off_final_binding as binding
from accept_recovery_off_final_cohort import assert_database_binding


def fixture(world=4):
    parent = dict(seed=1337, source={'sha256': 's'}, events={'sha256': 'e'},
        cache_archive={'sha256': 'a'}, cache_manifest={'sha256': 'm'},
        checkpoint={'sha256': 'c'}, weights_archive={'sha256': 'w'},
        final_refit_model_sha256='final', original_nested_model_sha256='old',
        numeric_reference_admission_sha256='numeric', forward_outputs=['forward'],
        final_refit_training_byte_admission_sha256='train',
        forward_full_byte_and_coverage_admission_sha256='coverage', final_refit_NN_numeric_completion_sha256='completion',
        configuration=dict(method='rbf', backend='exclusive_root_partition_regions_v1', allocation='bound',
                           state={'candidate_protocol': 'rbf-all-class-top64-v1'}, limits={'residual_partition_version': 1}))
    plan = copy.deepcopy(parent)
    plan.update(method='rbf', world_size=world, scoring_atol=1e-4, scoring_rtol=1e-4,
                original_exclusive_acceptance_inherited=False, recovery_off_independent_semantics_accepted=False,
                full_forest_independently_accepted=False, learned_Stage2_complete=False, paper_performance_complete=False,
                recovery_off_source_freeze_sha256=binding.CANDIDATE_SHA,
                recovery_off_real_checkpoint_input_contract_sha256=binding.INPUT_SHA)
    plan['configuration']['backend'] = 'exclusive_event_boundary_recovery_off_v1'
    plan['configuration']['limits']['recovery_off_version'] = 1
    job = dict(seed=1337, task_id='new-recovery-off-task', plan=plan,
               recipe_sha256=hashlib.sha256(binding.canonical(plan)).hexdigest())
    template = copy.deepcopy(plan); template['world_size'] = None
    sequences = [dict(sequence_id=f'seq{i}', events=1 if i else 7400, nodes=1, database_sha256='a'*64) for i in range(46)]
    value = dict(kind=binding.BYTE_KIND, all_registered_bytes_verified=True,
                 all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
                 full_forest_semantics_or_fresh_state_independently_accepted=False,
                 learned_Stage2_complete=False, same_resource_performance_accepted=False, paper_performance_complete=False,
                 seed=1337, method='rbf', task_id=job['task_id'], recipe_sha256=job['recipe_sha256'], world_size=world,
                 NN_atol=1e-4, NN_rtol=1e-4, source_sha256=binding.READER_SHA,
                 final_checkpoint_sha256='c', final_model_sha256='final', input_numeric_index_sha256='numeric',
                 artifacts={k: {} for k in ['receipt', 'exclusive-source-manifest', *[f'replay-rank{i}' for i in range(world)]]},
                 sequences=sequences, total_nodes=46)
    return value, job, template, parent


@pytest.mark.parametrize('world', [4, 8])
def test_exact_independent_identity(world):
    args = fixture(world)
    assert binding.validate_local_output(*args)['world_size'] == world


@pytest.mark.parametrize('field,value', [
    ('kind', 'rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'),
    ('source_sha256', 'main-reader'), ('seed', 2027), ('method', 'topk'),
    ('task_id', 'main-task'), ('recipe_sha256', 'different'), ('world_size', 8),
    ('NN_atol', 1e-3), ('NN_rtol', 1e-3), ('final_checkpoint_sha256', 'old-checkpoint'),
    ('final_model_sha256', 'old'), ('input_numeric_index_sha256', 'old-index'),
    ('all_registered_bytes_verified', False),
    ('all_46_sequences_7445_events_and_final_model_factor_nodes_verified', False),
    ('full_forest_semantics_or_fresh_state_independently_accepted', True),
    ('learned_Stage2_complete', True), ('same_resource_performance_accepted', True),
    ('paper_performance_complete', True), ('total_nodes', 45),
])
def test_wrong_local_receipt_rejected(field, value):
    args = fixture(); args[0][field] = value
    with pytest.raises(AssertionError): binding.validate_local_output(*args)


@pytest.mark.parametrize('mutation', ['missing_rank', 'extra_rank', 'duplicate_sequence', 'missing_sequence',
                                     'event_count', 'unsafe_path', 'invalid_hash', 'negative_rows'])
def test_full_coverage_not_a_subset(mutation):
    args = fixture(); value = args[0]
    if mutation == 'missing_rank': value['artifacts'].pop('replay-rank0')
    if mutation == 'extra_rank': value['artifacts']['replay-rank4'] = {}
    if mutation == 'duplicate_sequence': value['sequences'][1]['sequence_id'] = 'seq0'
    if mutation == 'missing_sequence': value['sequences'].pop()
    if mutation == 'event_count': value['sequences'][0]['events'] -= 1
    if mutation == 'unsafe_path': value['sequences'][0]['sequence_id'] = '../seq0'
    if mutation == 'invalid_hash': value['sequences'][0]['database_sha256'] = 'z'*64
    if mutation == 'negative_rows': value['sequences'][0]['nodes'] = -1; value['total_nodes'] -= 2
    with pytest.raises(AssertionError): binding.validate_local_output(*args)


@pytest.mark.parametrize('mutation', ['learned', 'old_backend', 'extra_config', 'cache', 'checkpoint',
                                     'larger_cap', 'inherited', 'wrong_candidate', 'wrong_input', 'template'])
def test_rehashed_plan_cannot_change_ablation_contract(mutation):
    value, job, template, parent = fixture(); plan = job['plan']
    if mutation == 'learned': plan['configuration']['allocation'] = 'learned'
    if mutation == 'old_backend': plan['configuration']['backend'] = parent['configuration']['backend']
    if mutation == 'extra_config': plan['configuration']['new_experiment'] = True
    if mutation == 'cache': plan['cache_archive']['sha256'] = 'other-cache'
    if mutation == 'checkpoint': plan['checkpoint']['sha256'] = 'other-checkpoint'; value['final_checkpoint_sha256'] = 'other-checkpoint'
    if mutation == 'larger_cap': plan['configuration']['limits']['max_events'] = 999999
    if mutation == 'inherited': plan['original_exclusive_acceptance_inherited'] = True
    if mutation == 'wrong_candidate': plan['recovery_off_source_freeze_sha256'] = 'new'
    if mutation == 'wrong_input': plan['recovery_off_real_checkpoint_input_contract_sha256'] = 'new'
    if mutation != 'template': template = copy.deepcopy(plan); template['world_size'] = None
    else: plan['unexpected'] = True
    value['recipe_sha256'] = job['recipe_sha256'] = hashlib.sha256(binding.canonical(plan)).hexdigest()
    with pytest.raises(AssertionError): binding.validate_local_output(value, job, template, parent)


def database_fixture(path):
    config = dict(state={'candidate_protocol': 'rbf-all-class-top64-v1'}, recovery_off_version=1)
    audit = dict(kind='persistent_exclusive_event_boundary_recovery_off_v1', recovery_enabled=False, original_model_risk_certified=False)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB)')
        db.execute('CREATE TABLE events(ordinal INTEGER,audit BLOB)')
        db.executemany('INSERT INTO meta VALUES(?,?)', [('sequence_id', json.dumps('seq0')), ('config', json.dumps(config))])
        db.execute('INSERT INTO events VALUES(0,?)', (json.dumps(audit),))
    return dict(state=config['state'], limits={'recovery_off_version': 1}), audit


def test_database_configuration_schema_binding(tmp_path):
    path = tmp_path/'fixture.sqlite'; config, _ = database_fixture(path)
    assert_database_binding(path, binding.sha(path), 'seq0', config)
    config['limits']['recovery_off_version'] = 2
    with pytest.raises(AssertionError): assert_database_binding(path, binding.sha(path), 'seq0', config)


@pytest.mark.parametrize('field,value', [('kind', 'main'), ('recovery_enabled', True), ('original_model_risk_certified', True)])
def test_database_cannot_relabel_main_as_ablation(tmp_path, field, value):
    path = tmp_path/'fixture.sqlite'; config, audit = database_fixture(path)
    audit[field] = value
    with sqlite3.connect(path) as db: db.execute('UPDATE events SET audit=?', (json.dumps(audit),))
    with pytest.raises(AssertionError): assert_database_binding(path, binding.sha(path), 'seq0', config)


def test_missing_full_integration_gate_is_closed(tmp_path):
    with pytest.raises(FileNotFoundError): binding.source_gate(tmp_path)


def test_unchanged_oracles_loaded_from_frozen_paths_without_module_pollution():
    # Reads and imports only the small frozen independent source files, no task,
    # cache, checkpoint, full numerical run, or source freeze is constructed.
    old = {name: sys.modules.get(name) for name in ('oracle', 'rbf_final_refit_forest_binding', 'rbf_nested_seen_val_v2_common')}
    model, cache, causal, fresh = binding.load_unchanged_oracles()
    assert Path(model.__file__) == binding.MODEL_D/'rbf_final_refit_forest_binding.py'
    assert Path(cache.__file__) == binding.MODEL_D/'final_cache203.py'
    assert Path(causal.__file__) == binding.STRUCTURE_D/'causal.py'
    assert Path(fresh.__file__) == binding.FRESH
    assert cache.validate_final_model is model.validate_final_model
    assert fresh.ATOL == fresh.RTOL == cache.ATOL == cache.RTOL == 1e-8
    assert old == {name: sys.modules.get(name) for name in old}


def test_original_causal_and_fresh_run_on_frozen_recovery_off_schema(tmp_path):
    from test_recovery_off_structure_oracle import fixture as producer_fixture
    database = tmp_path/'recovery-off.sqlite'
    expected = producer_fixture(database)
    _, _, causal, fresh = binding.load_unchanged_oracles()
    raw = causal.verify_database(database, expected)
    states = fresh.verify_database(database, expected)
    assert raw['events'] == states['events'] == 4 and raw['observations'] == 4
    assert states['states'] > 0 and states['predictions'] > 0
    assert states['stored_states_and_chosen_outputs_checked'] is True
    assert states['identity_search_and_partition_bound_acceptance'] is False
    assert binding.sha(database) == expected


def test_rehashed_branch_state_mutation_still_rejected_by_original_fresh(tmp_path):
    from test_recovery_off_structure_oracle import fixture as producer_fixture
    database = tmp_path/'recovery-off.sqlite'
    producer_fixture(database)
    _, _, _, fresh = binding.load_unchanged_oracles()
    with sqlite3.connect(database) as db:
        table = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'pc*_states' ORDER BY name LIMIT 1").fetchone()[0]
        handle, raw = db.execute(f'SELECT h,payload FROM {table} LIMIT 1').fetchone()
        value = json.loads(raw); value['mean'][0] += .1
        db.execute(f'UPDATE {table} SET payload=? WHERE h=?', (json.dumps(value), handle))
    with pytest.raises(AssertionError): fresh.verify_database(database, binding.sha(database))
