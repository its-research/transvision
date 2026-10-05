"""Real frozen producer fixtures, then independent reconstruction and tampering."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import types
import tarfile

import numpy as np
import pytest

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
SOURCE = ROOT / 'source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005/source'
ORACLE = Path(os.environ.get('RBF_LEARNED_TRAJECTORY_ORACLE',
    str(Path(__file__).resolve().parents[2] / 'tools/event_track_v2x/rbf_independent_learned_trajectory.py')))


@pytest.fixture(scope='module')
def oracle():
    spec = importlib.util.spec_from_file_location('new_learned_reference', ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def producer():
    with pytest.MonkeyPatch.context() as patch:
        for name in list(sys.modules):
            if name.startswith('transvision.') or name == 'transvision':
                patch.delitem(sys.modules, name)
        for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
            module = types.ModuleType(name)
            module.__path__ = [str(SOURCE.joinpath(*name.split('.')))]
            patch.setitem(sys.modules, name, module)
        from transvision.models.event_track_v2x.exclusive_completion_tracking import ExclusiveCompletionLearned, PersistentExclusiveCompletionConfig
        from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, PaperForestTrackingConfig
        from transvision.models.event_track_v2x.identity_forest import IdentityNode
        from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy
        # Prove that fixtures execute the producer's exact transitive bytes,
        # including its three frozen core replacements, not ambient modules.
        archive = ROOT / 'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == '038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
        with tarfile.open(archive) as stream:
            manifest = json.load(stream.extractfile('source-manifest.json'))
        expected = {row['path']: row['sha256'] for row in manifest['files']}
        bootstrap = ROOT / 'source-freezes/rbf-final-refit-learned-priority-full-train-GPU-v1-20261005/bootstrap.py'
        assert hashlib.sha256(bootstrap.read_bytes()).hexdigest() == '65c4af1f1a811fa0982daf912e7de8da0b857c776b38e7ce7023b60afaa0c43f'
        for node in ast.parse(bootstrap.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('PATCHES', 'REPLACEMENTS'):
                expected.update({name: value['sha256'] for name, value in ast.literal_eval(node.value).items()})
        for name, module in list(sys.modules.items()):
            if name.startswith('transvision.') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                relative = str(path.relative_to(SOURCE))
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[relative]
        yield types.SimpleNamespace(**locals())


def fixture(producer, path, budget=4, *, zero_weights=False, decision_cap=32, frontier_cap=128):
    path.mkdir()
    rng = np.random.default_rng(1337)
    arrays = [rng.normal(0, .05, shape) for shape in ((32, 18), (32,), (1, 32), (1,))]
    if zero_weights:
        arrays = [np.zeros_like(a) for a in arrays]
    policy = producer.FrozenPriorityPolicy(arrays)
    weight_path = path / 'weights.npz'
    np.savez(weight_path, **{f'w{i}': a for i, a in enumerate(arrays)})
    config = producer.PersistentExclusiveCompletionConfig(
        state=producer.PaperForestTrackingConfig(decision_mode='all-legal-hamming', expansion_budget=budget,
            active_limit=2, max_frontier=frontier_cap, window_us=100 if decision_cap == 2 else 250000,
            paper_action_budget=20, max_model_regret=1.),
        max_decision_nodes=decision_cap, max_total_frontier=frontier_cap * 2)
    database = path / 'forest.sqlite'
    tracker = producer.ExclusiveCompletionLearned(database, sequence_id='0003', config=config, allocation_policy=policy)

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
def test_frozen_producer_trajectory(oracle, producer, tmp_path, budget, zero_weights, decision_cap, frontier_cap):
    result = verify(oracle, fixture(producer, tmp_path / 'fixture', budget, zero_weights=zero_weights,
                                   decision_cap=decision_cap, frontier_cap=frontier_cap))
    assert result['events'] == 7 and result['observations'] == 7 and result['merge_events'] == 1
    assert result['maximum_archived_components'] == 2 and result['zero_loss_scope_component_events'] > 0
    assert result['all_float64_selections_exactly_replayed_from_independently_checked_features']
    assert result['ordering_tolerance_used'] is False and result['feature_bitwise_parity_claimed'] is False
    assert not result['actual_dataset_and_full_causal_trajectory_accepted']
    assert not result['full_forest_semantics_or_fresh_state_independently_accepted']
    assert result['atol'] == result['rtol'] == 1e-8


@pytest.fixture(scope='module')
def valid(oracle, producer, tmp_path_factory):
    inputs = fixture(producer, tmp_path_factory.mktemp('learned_trajectory') / 'source')
    verify(oracle, inputs)
    return inputs


@pytest.mark.parametrize('corrupt', ['candidate_omission', 'candidate_order', 'feature', 'score', 'chosen',
    'charged_step', 'work_kind', 'stopped_early', 'future_window', 'prefix_count', 'prefix_bytes',
    'policy_signature', 'budget', 'feature_count', 'factors', 'weights', 'bad_weight_shape'])
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
            connection.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(audit), ordinal))
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
    connection.commit();connection.close()
    with pytest.raises(AssertionError, match='exact tie break'):
        verify(oracle, inputs)
