"""Software fixtures for irreversible learned allocation, not experiment admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import types

import numpy as np
import pytest

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
ORACLE = Path(os.environ.get('RBF_RECOVERY_OFF_LEARNED_TRAJECTORY_ORACLE',
    str(Path(__file__).resolve().parents[2] / 'tools/event_track_v2x/rbf_independent_recovery_off_learned_trajectory.py')))


@pytest.fixture(scope='module')
def oracle():
    spec = importlib.util.spec_from_file_location('new_learned_reference', ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def producer():
    from transvision.models.event_track_v2x.recovery_off_allocation import RecoveryOffLearned
    from transvision.models.event_track_v2x.recovery_off_tracking import RecoveryOffConfig
    from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
    from transvision.models.event_track_v2x.identity_forest import IdentityNode
    from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy
    return types.SimpleNamespace(**locals())


def fixture(producer, path, budget=4, *, zero_weights=False, decision_cap=32, frontier_cap=128):
    path.mkdir()
    rng = np.random.default_rng(1337)
    arrays = [rng.normal(0, .05, shape) for shape in ((32, 18), (32,), (1, 32), (1,))]
    if zero_weights:
        arrays = [np.zeros_like(a) for a in arrays]
    policy = producer.FrozenPriorityPolicy(arrays)
    weight_path = path / 'weights.npz'
    np.savez(weight_path, **{f'w{i}': a for i, a in enumerate(arrays)})
    config = producer.RecoveryOffConfig(
        state=producer.PaperForestTrackingConfig(decision_mode='all-legal-hamming', expansion_budget=budget,
            active_limit=2, max_frontier=frontier_cap, window_us=100 if decision_cap == 2 else 250000,
            paper_action_budget=20, max_model_regret=1.),
        max_decision_nodes=decision_cap, max_total_frontier=frontier_cap * 2)
    database = path / 'forest.sqlite'
    tracker = producer.RecoveryOffLearned(database, sequence_id='0003', config=config, allocation_policy=policy)

    def raw(i, state, frame=None):
        features = np.zeros(203)
        features[0], features[138], features[200], features[201] = i / 100, 1., .8, .8
        return producer.RawIdentityDetection('0003',
            producer.IdentityNode(f'd{i}', i % 2, state, state, str(state) if frame is None else frame),
            i, state, [float(i), 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9) * .2, .8, features, 'a' * 64)

    batches = [
        (100000, [], []),
        (200000, [raw(0, 200000), raw(1, 200000)], [[(-1, -.3)], [(-1, -.2)]]),
        (300000, [raw(2, 300000), raw(3, 300000)], [[(-1, -1.), (0, -.1)], [(-1, -1.2), (1, -.1)]]),
        (400000, [raw(4, 400000), raw(5, 400000)], [[(-1, -1.5), (0, -.2), (2, -.05)], [(-1, -1.), (1, -.2), (3, -.1)]]),
        (500000, [raw(6, 500000)], [[(-1, -2.), (2, -.2), (3, -.3)]]),
        (600000, [], []),
        (900000, [], []),
    ]
    try:
        for event, (clock, raw_rows, factors) in enumerate(batches):
            tracker.step(raw_rows, factors, frame_id=str(event), reference_us=clock, decision_us=clock, event_id=str(event))
    finally:
        tracker.close()
    return database, weight_path, policy.signature


def verify(oracle, inputs):
    db, weights, signature = inputs
    return oracle.verify_database(db, oracle.numeric.sha(db), weights, oracle.numeric.sha(weights), signature)


@pytest.mark.parametrize('budget,zero_weights,decision_cap,frontier_cap', [
    (0, False, 32, 128), (1, False, 32, 128), (4, False, 32, 128),
    (100, False, 32, 128), (4, True, 32, 128), (4, False, 2, 128),
    (100, False, 32, 2),
])
def test_restricted_producer_trajectory(oracle, producer, tmp_path, budget, zero_weights, decision_cap, frontier_cap):
    result = verify(oracle, fixture(producer, tmp_path / 'fixture', budget, zero_weights=zero_weights,
                                   decision_cap=decision_cap, frontier_cap=frontier_cap))
    assert result['events'] == 7 and result['observations'] == 7 and result['merge_events'] == 1
    assert result['maximum_archived_components'] == 2 and result['zero_loss_scope_component_events'] > 0
    assert result['all_float64_selections_exactly_replayed_from_independently_checked_features']
    assert result['ordering_tolerance_used'] is False and result['feature_bitwise_parity_claimed'] is False
    assert not result['actual_dataset_and_full_causal_trajectory_accepted']
    assert not result['full_forest_semantics_or_fresh_state_independently_accepted']
    assert result['atol'] == result['rtol'] == 1e-8
    assert result['historical_support_rebuilt_from_prior_commit_active_union_output']
    assert result['original_model_risk_certified'] is False
    assert result['restricted_component_events'] > 0
    if budget == 100 and frontier_cap == 128:
        assert result['excluded_archived_prefix_occurrences'] > 0


@pytest.fixture(scope='module')
def valid(oracle, producer, tmp_path_factory):
    inputs = fixture(producer, tmp_path_factory.mktemp('learned_trajectory') / 'source')
    verify(oracle, inputs)
    return inputs


@pytest.mark.parametrize('corrupt', ['candidate_omission', 'candidate_order', 'feature', 'score', 'chosen',
    'charged_step', 'work_kind', 'stopped_early', 'future_window', 'prefix_count', 'prefix_bytes',
    'policy_signature', 'budget', 'feature_count', 'factors', 'weights', 'bad_weight_shape', 'history', 'unrestricted_risk', 'priority_scope', 'schema'])
def test_tampering_rejected(oracle, valid, tmp_path, corrupt):
    db, weights, signature = valid
    db_copy, weights_copy = tmp_path / 'forest.sqlite', tmp_path / 'weights.npz'
    shutil.copyfile(db, db_copy)
    shutil.copyfile(weights, weights_copy)
    connection = sqlite3.connect(db_copy)
    try:
        if corrupt in ('weights', 'bad_weight_shape'):
            with np.load(weights_copy) as data:
                arrays = {k: data[k] for k in data.files}
            arrays['w0'] = arrays['w0'] + .01 if corrupt == 'weights' else arrays['w0'][:1]
            np.savez(weights_copy, **arrays)
        elif corrupt == 'history':
            rows=connection.execute('SELECT ordinal,audit FROM events ORDER BY ordinal').fetchall()
            ordinal,audit=next((o,json.loads(a)) for o,a in rows if any(c['historical_support']['clauses'] for c in json.loads(a)['components']))
            target=next(c for c in audit['components'] if c['historical_support']['clauses'])
            target['historical_support']['clauses']=[]
            forged=hashlib.sha256(oracle.numeric.canonical(target['historical_support'])).hexdigest()
            target['historical_support_sha256']=forged
            target['decision']['support_sha256']=forged
            connection.execute('UPDATE events SET audit=? WHERE ordinal=?',(json.dumps(audit),ordinal))
        elif corrupt == 'schema':
            connection.execute("UPDATE meta SET v=? WHERE k='schema'", (json.dumps('persistent_exclusive_completion_learned_allocation_v1'),))
        elif corrupt in ('policy_signature', 'budget'):
            key = 'allocation_policy_signature' if corrupt == 'policy_signature' else 'config'
            value = json.loads(connection.execute('SELECT v FROM meta WHERE k=?', (key,)).fetchone()[0])
            if corrupt == 'policy_signature':value = 'changed'
            else:value['state']['expansion_budget'] += 1
            connection.execute('UPDATE meta SET v=? WHERE k=?', (json.dumps(value), key))
        elif corrupt == 'prefix_bytes':
            connection.execute("UPDATE pc1_prefixes SET sha=? WHERE h=1", ('f' * 64,))
        else:
            rows = connection.execute('SELECT ordinal,audit FROM events ORDER BY ordinal').fetchall()
            ordinal, audit = next((o, json.loads(a)) for o, a in rows if any(
                len(s['priority_selection']['candidates']) > 1 for s in json.loads(a)['allocation_trace']))
            selected = next(s for s in audit['allocation_trace'] if len(s['priority_selection']['candidates']) > 1)
            records = selected['priority_selection']['candidates']
            if corrupt == 'candidate_omission':records.pop()
            elif corrupt == 'candidate_order':records.reverse()
            elif corrupt == 'feature':records[0]['features'][1] += .01
            elif corrupt == 'score':records[0]['priority_score'] += .01
            elif corrupt == 'chosen':selected['component'] = next(r['component'] for r in records if r['component'] != selected['component'])
            elif corrupt == 'charged_step':selected['charged_search_steps'] += 1
            elif corrupt == 'work_kind':selected['work_kind'] = 'changed'
            elif corrupt == 'stopped_early':audit['allocation_trace'] = []
            elif corrupt == 'future_window':audit['decision_indices'].pop()
            elif corrupt == 'prefix_count':audit['components'][0]['prefix_nodes'] -= 1
            elif corrupt == 'feature_count':audit['priority_feature_rows'] += 1
            elif corrupt == 'factors':audit['appended_rows'][0][0][1] += .1
            elif corrupt == 'unrestricted_risk':audit['original_model_risk_certified']=True
            elif corrupt == 'priority_scope':audit['priority_decision_support_scope']='original_model'
            connection.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(audit), ordinal))
        reseal(connection, oracle)
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AssertionError):
        verify(oracle, (db_copy, weights_copy, signature))


def test_reference_imports_exclude_producer(oracle):
    source = ORACLE.read_text()
    assert 'from transvision' not in source and 'import transvision' not in source
    assert hashlib.sha256(oracle.REFERENCE.read_bytes()).hexdigest() == oracle.REFERENCE_SHA


def test_zero_weight_tie_requires_smallest_component(oracle, producer, tmp_path):
    inputs = fixture(producer, tmp_path / 'tie', 4, zero_weights=True)
    verify(oracle, inputs)
    connection = sqlite3.connect(inputs[0])
    rows = connection.execute('SELECT ordinal,audit FROM events ORDER BY ordinal').fetchall()
    ordinal, audit = next((o, json.loads(a)) for o, a in rows if any(
        len(s['priority_selection']['candidates']) > 1 for s in json.loads(a)['allocation_trace']))
    selected = next(s for s in audit['allocation_trace'] if len(s['priority_selection']['candidates']) > 1)
    assert all(r['priority_score'] == 0 for r in selected['priority_selection']['candidates'])
    selected['component'] = max(r['component'] for r in selected['priority_selection']['candidates'])
    connection.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(audit), ordinal))
    reseal(connection, oracle)
    connection.commit();connection.close()
    with pytest.raises(AssertionError, match='exact tie break'):
        verify(oracle, inputs)


def reseal(connection, oracle):
    """Rehash mutations so semantic negative tests cannot pass on chain checks alone."""
    ph=ah='0'*64
    config=json.loads(connection.execute("SELECT v FROM meta WHERE k='config'").fetchone()[0])
    config_sha=hashlib.sha256(oracle.numeric.canonical(config)).hexdigest()
    for ordinal,pb,ab in connection.execute('SELECT ordinal,prediction,audit FROM events ORDER BY ordinal').fetchall():
        p,a=json.loads(pb),json.loads(ab)
        p['previous_commit_sha256']=ph
        ph=hashlib.sha256(oracle.numeric.canonical({k:v for k,v in p.items() if k!='commit_sha256'})).hexdigest()
        p['commit_sha256']=ph
        a['previous_audit_sha256']=ah
        a['prediction_sha256']=ph
        a['configuration_sha256']=config_sha
        ah=hashlib.sha256(oracle.numeric.canonical(a)).hexdigest()
        connection.execute('UPDATE events SET prediction=?,audit=? WHERE ordinal=?',
            (oracle.numeric.canonical(p),oracle.numeric.canonical(a),ordinal))
    state=json.loads(connection.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
    state.update(prediction_sha256=ph,audit_sha256=ah)
    connection.execute("UPDATE meta SET v=? WHERE k='state'",(json.dumps(state),))


def test_restrictions_preserve_correlated_whole_histories(oracle):
    clauses=(((0,1,2),((0,0,2),(0,1,0))),)
    assert oracle.allows(clauses,(0,0,2)) and oracle.allows(clauses,(0,1,0))
    assert oracle.allows(clauses,(0,0)) and oracle.allows(clauses,(0,1))
    assert not oracle.allows(clauses,(0,0,0))
    assert not oracle.allows(clauses,(0,1,2))


def test_cli_source_gate_requires_exact_checker_and_pinned_independent_reference(oracle, tmp_path, monkeypatch):
    import sys
    cli_path=ORACLE.with_name('check_recovery_off_learned_search_trajectory.py')
    monkeypatch.setitem(sys.modules,'rbf_independent_recovery_off_learned_trajectory',oracle)
    spec=importlib.util.spec_from_file_location('restricted_trajectory_cli',cli_path)
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for source in (ORACLE,cli_path):shutil.copyfile(source,tmp_path/source.name)
    freeze=dict(kind='rbf_independent_recovery_off_learned_search_trajectory_source_v1',
        sources={p.name:dict(sha256=oracle.numeric.sha(p),bytes=p.stat().st_size) for p in tmp_path.iterdir()},
        references=[dict(path=str(oracle.REFERENCE),sha256=oracle.REFERENCE_SHA)])
    manifest=tmp_path/'source-freeze.json'
    manifest.write_text(json.dumps(freeze))
    monkeypatch.setattr(oracle,'__file__',str(tmp_path/ORACLE.name))
    assert cli.source_gate(tmp_path)==oracle.numeric.sha(manifest)
    freeze['sources']={}
    manifest.write_text(json.dumps(freeze))
    with pytest.raises(AssertionError):cli.source_gate(tmp_path)


def test_cli_source_changes_fail_admission(oracle, tmp_path, monkeypatch):
    import sys
    cli_path=ORACLE.with_name('check_recovery_off_learned_search_trajectory.py')
    monkeypatch.setitem(sys.modules,'rbf_independent_recovery_off_learned_trajectory',oracle)
    spec=importlib.util.spec_from_file_location('restricted_trajectory_cli_change',cli_path)
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for source in (ORACLE,cli_path):shutil.copyfile(source,tmp_path/source.name)
    freeze=dict(kind='rbf_independent_recovery_off_learned_search_trajectory_source_v1',
        sources={p.name:dict(sha256=oracle.numeric.sha(p),bytes=p.stat().st_size) for p in tmp_path.iterdir()},
        references=[dict(path=str(oracle.REFERENCE),sha256=oracle.REFERENCE_SHA)])
    (tmp_path/'source-freeze.json').write_text(json.dumps(freeze))
    monkeypatch.setattr(oracle,'__file__',str(tmp_path/ORACLE.name))
    path=tmp_path/ORACLE.name
    path.write_text(path.read_text()+'\n# changed after freeze\n')
    with pytest.raises(AssertionError):cli.source_gate(tmp_path)
