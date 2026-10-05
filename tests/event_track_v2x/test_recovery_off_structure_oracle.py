"""Read-only oracle tests against the frozen recovery-off producer, not itself."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

FROZEN = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005')
sys.path.insert(0, str(FROZEN))
from transvision.models.event_track_v2x import recovery_off_tracking as producer
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig, RawIdentityDetection
from transvision.models.event_track_v2x.identity_forest import IdentityNode

ORACLE = Path(__file__).resolve().parents[2]/'tools/event_track_v2x/recovery_off_structure_oracle.py'
_spec = importlib.util.spec_from_file_location('tested_recovery_off_structure_oracle', ORACLE)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def observation(name, time, source=0, frame=None, arrival=None):
    features = np.zeros(203)
    features[138], features[200], features[201] = 1, .8, .8
    return RawIdentityDetection('0003', IdentityNode(name, source, time, arrival or time+10, frame or name),
        0, time, [0., 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9)*.2, .8, features, 'a'*64)


def fixture(path, *, budget=100, width=2, max_nodes=4096):
    assert Path(producer.__file__).resolve().is_relative_to(FROZEN), 'test must use the frozen producer'
    config = producer.RecoveryOffConfig(state=PaperForestTrackingConfig(
        active_limit=width, expansion_budget=budget, decision_mode='all-legal-hamming', max_model_regret=1.),
        max_decision_nodes=max_nodes)
    t = producer.RecoveryOffTracker(path, sequence_id='0003', config=config)
    t.step([observation('a', 1000000), observation('b', 1000001, 1), observation('c', 1000002)],
           [((-1, 0.),), ((-1, -.2), (0, -.3)), ((-1, 0.),)],
           frame_id='first', event_id='first', reference_us=1100000, decision_us=1100000,
           decision_indices=tuple(range(min(3, max_nodes))))
    t.step([observation('bridge', 900000, 1, arrival=1200000)],
           [((-1, -.5), (1, -.2), (2, -.1))],
           frame_id='bridge', event_id='bridge', reference_us=1300000, decision_us=1300000,
           decision_indices=tuple(range(min(4, max_nodes))))
    t.step([], [], frame_id='empty', event_id='empty', reference_us=1400000, decision_us=1400000,
           decision_indices=tuple(range(min(4, max_nodes))))
    t.step([], [], frame_id='expired', event_id='expired', reference_us=10000000, decision_us=10000000,
           decision_indices=())
    return t.close()


def rehash_events(db, change):
    """Re-sign complete audit chains so semantic mutations are not hash tests."""
    previous = '0'*64
    for ordinal, raw in list(db.execute('SELECT ordinal,audit FROM events ORDER BY ordinal')):
        a = json.loads(raw)
        change(ordinal, a)
        a['previous_audit_sha256'] = previous
        previous = oracle.digest(a)
        db.execute('UPDATE events SET audit=? WHERE ordinal=?', (oracle.canonical(a), ordinal))
    state = json.loads(db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
    state['audit_sha256'] = previous
    db.execute("UPDATE meta SET v=? WHERE k='state'", (oracle.canonical(state),))


@pytest.mark.parametrize('budget,width,max_nodes', [(0, 1, 4096), (1, 1, 4096), (100, 1, 4096), (100, 2, 4096), (100, 2, 2)])
def test_frozen_full_database_scope(tmp_path, budget, width, max_nodes):
    path = tmp_path/'candidate.db'
    expected = fixture(path, budget=budget, width=width, max_nodes=max_nodes)
    result = oracle.verify_database(path, expected)
    assert result['events'] == 4 and result['predecessor_clauses'] >= 4
    assert result['restricted_support_cut_coverage_verified']
    assert result['restricted_residual_mass_arithmetic_verified']
    assert not result['conditional_action_search_verified']
    assert not result['fresh_branch_states_verified'] and not result['original_support_coverage_claimed']
    assert oracle.sha(path) == expected


def test_clauses_preserve_whole_vector_correlation():
    context = oracle.ComponentContext({}, None, range(3), [((0, 1, 2), ((0, 0, 2), (0, 1, 1)))])
    assert context.allowed((0, 0, 2)) and context.allowed((0, 1, 1))
    assert not context.allowed((0, 0, 1)) and not context.allowed((0, 1, 2))


@pytest.mark.parametrize('mutation', ['wrong_predecessor', 'missing_clause', 'marginal_cartesian_class',
    'original_risk', 'false_raw_support', 'wrong_mass', 'drop_residual', 'duplicate_residual',
    'false_region_scope', 'wrong_branch', 'claim_recovery'])
def test_rehashed_semantic_mutations_rejected(tmp_path, mutation):
    path = tmp_path/'mutated.db'
    fixture(path)
    db = sqlite3.connect(path)
    def mutate(ordinal, a):
        if ordinal != 1:
            return
        s = a['components'][0]
        if mutation == 'wrong_predecessor': s['predecessors'] = []
        elif mutation == 'missing_clause': s['historical_support']['clauses'].pop()
        elif mutation == 'marginal_cartesian_class':
            s['historical_support']['clauses'][0][1].append([99]*len(s['historical_support']['clauses'][0][0]))
        elif mutation == 'original_risk': a['model_regret_upper'] = 0.
        elif mutation == 'false_raw_support': s['complete_raw_support_retained'] = True
        elif mutation == 'wrong_mass': s['log_partition_upper'] += .1
        elif mutation == 'drop_residual':
            assert s['frontier']
            s['frontier'].pop()
        elif mutation == 'duplicate_residual':
            assert s['frontier']
            s['frontier'].append(copy.deepcopy(s['frontier'][0]))
        elif mutation == 'false_region_scope':
            assert s['frontier']
            s['frontier'][0]['gross_upper_may_include_pruned_histories'] = False
        elif mutation == 'wrong_branch': s['branches'].pop()
        elif mutation == 'claim_recovery': s['recovery_events'] = [{'previous_component': 1}]
    rehash_events(db, mutate)
    db.commit()
    db.close()
    with pytest.raises(AssertionError): oracle.verify_database(path, oracle.sha(path))


def test_final_mutable_metadata_is_checked_but_not_used_as_history(tmp_path):
    path = tmp_path/'meta.db'
    fixture(path)
    db = sqlite3.connect(path)
    c = db.execute('SELECT component FROM component_catalog WHERE live=1').fetchone()[0]
    state = json.loads(db.execute(f"SELECT v FROM pc{c}_meta WHERE k='state'").fetchone()[0])
    state['active'] = []
    db.execute(f"UPDATE pc{c}_meta SET v=? WHERE k='state'", (oracle.canonical(state),))
    db.commit()
    db.close()
    with pytest.raises(AssertionError): oracle.verify_database(path, oracle.sha(path))


def test_raw_factor_view_and_input_byte_binding(tmp_path):
    path = tmp_path/'raw.db'
    expected = fixture(path)
    with pytest.raises(AssertionError): oracle.verify_database(path, '0'*64)
    db = sqlite3.connect(path)
    db.execute('UPDATE potentials SET w=w+.1 WHERE i=1 AND p=-1')
    db.commit()
    db.close()
    with pytest.raises(AssertionError): oracle.verify_database(path, oracle.sha(path))
    assert oracle.sha(path) != expected


def test_rehashed_archived_pruned_output_is_rejected_even_when_raw_legal(tmp_path):
    path = tmp_path/'forbidden-output.db'
    t = producer.RecoveryOffTracker(path, sequence_id='0003', config=producer.RecoveryOffConfig(
        state=PaperForestTrackingConfig(active_limit=1, expansion_budget=100,
                                       decision_mode='all-legal-hamming', max_model_regret=1.)))
    first = t.step([observation('a', 1000000), observation('b', 1000001, 1)],
                  [((-1, 0.),), ((-1, 0.), (0, -8.))],
                  frame_id='first', event_id='first', reference_us=1100000, decision_us=1100000)
    t.step([], [], frame_id='next', event_id='next', reference_us=1200000, decision_us=1200000)
    t.close()
    db = sqlite3.connect(path)
    s = first.audit['components'][0]
    admitted = {row['handle'] for row in s['active']} | {s['output_handle']}
    discarded = [(h, sha) for h, sha in db.execute(f"SELECT h,sha FROM pc{s['component']}_prefixes WHERE depth=2") if h not in admitted]
    assert discarded
    h, sha = discarded[0]
    def mutate(ordinal, a):
        if ordinal == 1:
            a['components'][0]['output_handle'] = h
            a['components'][0]['output_sha256'] = sha
    rehash_events(db, mutate)
    db.commit()
    db.close()
    with pytest.raises(AssertionError, match='recovered pruned'):
        oracle.verify_database(path, oracle.sha(path))


def test_malformed_archive_view_cannot_fabricate_edge_components(tmp_path):
    path = tmp_path/'view.db'
    fixture(path)
    db = sqlite3.connect(path)
    c = db.execute('SELECT component FROM component_catalog WHERE live=1').fetchone()[0]
    db.execute(f'DROP VIEW pc{c}_potentials')
    db.execute(f'CREATE VIEW pc{c}_potentials AS SELECT i,p,w+.1 AS w FROM potentials')
    db.commit()
    db.close()
    with pytest.raises(AssertionError, match='immutable raw factor'):
        oracle.verify_database(path, oracle.sha(path))
