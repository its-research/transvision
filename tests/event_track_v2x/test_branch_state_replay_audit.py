"""Real raw-state replay and fault injection; no GT or network."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tools.event_track_v2x import audit_branch_state_replay as tool
from tools.event_track_v2x.run_train_inference_diagnostic import run as run_inference
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig, PersistentBeamTracker
from transvision.models.event_track_v2x.persistent_reachable_slot_bound_beam import (
    PersistentReachableSlotBoundBeamConfig, PersistentReachableSlotBoundBeamTracker)
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig, BeamRecoveryTracker
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV
from test_forest_tracking import observation
from test_forest_training_data import prepared_rows
from test_train_inference_diagnostic import pair_file
from test_persistent_component_tracking import step


@pytest.fixture(params=['node', 'joint', 'recover'])
def scene(tmp_path, request):
    cls, config = {
        'node': (PersistentBeamTracker, PersistentBeamConfig),
        'joint': (PersistentReachableSlotBoundBeamTracker, PersistentReachableSlotBoundBeamConfig),
        'recover': (BeamRecoveryTracker, BeamRecoveryConfig),
    }[request.param]
    path = tmp_path / 'scene.sqlite'
    t = cls(path, sequence_id='0003', config=config(state=ForestTrackingConfig(active_limit=2)))
    configuration = asdict(t.config)
    commits = [step(t, time=900_000, event='empty')]
    raw = (observation('a', -2, index=0), observation('b', 2, index=1))
    commits.append(step(t, raw, (((-1, 0.),), ((-1, 0.),)), event='birth'))
    raw = (observation('c', 1, source=1, frame='merge', state_us=1_150_000, arrival_us=1_200_000),)
    commits.append(step(t, raw, (((-1, -4.), (0, -.3), (1, 0.)),), time=1_250_000, event='merge'))
    # Reopening must not prevent historical membership/ancestry reconstruction.
    digest = t.close()
    t = cls.open(path, expected_database_sha256=digest,
                 expected_prediction_sha256=commits[-1].prediction['commit_sha256'])
    raw = (observation('d', .5, frame='later', state_us=1_280_000, arrival_us=1_300_000),)
    commits.append(step(t, raw, (((-1, -5.), (2, 0.)),), time=1_350_000, event='extend'))
    commits.append(step(t, time=1_400_000, event='rescore',
        rescored_rows=[(2, ((-1, -4.), (0, 3.), (1, -2.)))]))
    raw = (observation('e', -.5, source=1, frame='late', state_us=1_160_000, arrival_us=1_450_000),)
    commits.append(step(t, raw, (((-1, -5.), (3, 0.)),), time=1_500_000, event='late'))
    commits.append(step(t, time=4_000_000, event='expired'))
    t.close()
    return path, configuration, [(c.audit, c.prediction) for c in commits]


def checker_for(path, configuration, **kwargs):
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.set_authorizer(tool.read_only_tables)
    return tool.FreshStateAudit(db, '0003', configuration, **kwargs)


def test_every_event_rebuilt_after_merging_rescoring_reopen_late_evidence_and_expiry(scene):
    path, config, commits = scene
    original = tool.sha_file(path)
    checker = checker_for(path, config)
    with closing(checker.db):
        for audit, prediction in commits:
            result = checker.check(audit, prediction)
            assert result['entire_frame_exact']
        assert checker.visits == sum(a['observation_count'] for a, _ in commits)
        assert checker.events[0]['components'] == 0
        assert checker.events[1]['components'] == 2
        assert checker.events[2]['components'] == 1
        assert checker.events[-1]['output_boxes'] == 0
        assert checker.events[-1]['expired_components'] == 1
        assert len(checker.observations) == 5
    assert original == tool.sha_file(path)


@pytest.mark.parametrize('table', ['states', 'potentials', 'weights', 'pc1_states', 'pc1_weights', 'pc1_potentials', 'meta'])
def test_sql_authorizer_forbids_state_caches_final_factors_and_mutation(scene, table):
    path, config, _ = scene
    checker = checker_for(path, config)
    with closing(checker.db):
        with pytest.raises(sqlite3.DatabaseError):
            checker.db.execute('SELECT * FROM ' + table).fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            checker.db.execute('DELETE FROM observations')


def reseal(audit, prediction):
    prediction['commit_sha256'] = tool.digest({k: v for k, v in prediction.items() if k != 'commit_sha256'})
    audit['prediction_sha256'] = prediction['commit_sha256']


@pytest.mark.parametrize('mode', ['mean', 'covariance', 'score', 'missing_id', 'extra_id', 'branch_digest',
    'audit_chain', 'prediction_chain', 'factor_digest', 'append_coverage', 'component_missing',
    'component_duplicate', 'depth', 'prefix', 'expiry', 'future_reference', 'configuration'])
def test_faults_cannot_be_hidden_by_valid_prediction_commit_hash(scene, mode):
    path, config, commits = scene
    checker = checker_for(path, config)
    with closing(checker.db):
        checker.check(*commits[0])
        audit, prediction = deepcopy(commits[1])
        box = prediction['predictions'][0]
        if mode == 'mean': box['mean'][0] += .01
        elif mode == 'covariance': box['covariance'][0][0] += .01
        elif mode == 'score': box['score'] -= .01
        elif mode == 'missing_id': prediction['predictions'].pop()
        elif mode == 'extra_id': prediction['predictions'].append(dict(box, track_id='0003:extra'))
        elif mode == 'branch_digest': audit['components'][0]['branches'][0]['state_sha256'] = 'f' * 64
        elif mode == 'audit_chain': audit['previous_audit_sha256'] = 'f' * 64
        elif mode == 'prediction_chain': prediction['previous_commit_sha256'] = 'f' * 64
        elif mode == 'factor_digest': audit['factor_rows_sha256'] = 'f' * 64
        elif mode == 'append_coverage': audit['new_observations'] += 1
        elif mode == 'component_missing': audit['components'].pop()
        elif mode == 'component_duplicate': audit['components'].append(audit['components'][0])
        elif mode == 'depth': audit['components'][0]['nodes'] += 1
        elif mode == 'prefix': audit['components'][0]['output_sha256'] = 'f' * 64
        elif mode == 'expiry': audit['components'][0]['expired_output_only'] = True
        elif mode == 'future_reference': prediction['box_reference_timestamp_us'] = prediction['decision_timestamp_us'] + 1
        elif mode == 'configuration': audit['configuration_sha256'] = 'f' * 64
        reseal(audit, prediction)
        with pytest.raises(ValueError): checker.check(audit, prediction)
        assert len(checker.events) == 1


@pytest.mark.parametrize('mode', ['raw_hash', 'future_raw', 'future_component', 'overlap', 'prefix_root'])
def test_corrupted_immutable_inputs_rejected_without_using_final_live_flags(scene, mode):
    path, config, commits = scene
    with sqlite3.connect(path) as db:
        if mode in ('raw_hash', 'future_raw'):
            raw = json.loads(db.execute('SELECT raw FROM observations WHERE i=0').fetchone()[0])
            if mode == 'future_raw': raw['node']['arrival_us'] = 99_000_000
            else: raw['mean'][0] += 10
            encoded = tool.canonical(raw)
            db.execute('UPDATE observations SET raw=?,sha=? WHERE i=0',
                (encoded, hashlib.sha256(encoded).hexdigest() if mode == 'future_raw' else 'f'*64))
        elif mode == 'future_component': db.execute('UPDATE component_catalog SET created_us=999000000 WHERE component=1')
        elif mode == 'overlap': db.execute('UPDATE component_members SET global_i=0 WHERE component=2')
        elif mode == 'prefix_root': db.execute('UPDATE pc1_prefixes SET root=999 WHERE depth=1')
    checker = checker_for(path, config)
    with closing(checker.db):
        checker.check(*commits[0])
        with pytest.raises(ValueError): checker.check(*commits[1])


def test_cap_fails_before_reading_new_raw_inputs_and_repeated_event_rejected(scene):
    path, config, commits = scene
    checker = checker_for(path, config, max_node_visits=1)
    with closing(checker.db):
        checker.check(*commits[0])
        with pytest.raises(ValueError, match='cap'): checker.check(*commits[1])
        assert not checker.observations and checker.visits == 0
        with pytest.raises(ValueError, match='duplicate'): checker.check(*commits[0])


def test_rescore_must_keep_historical_support(scene):
    path, config, commits = scene
    checker = checker_for(path, config)
    with closing(checker.db):
        for pair in commits[:4]: checker.check(*pair)
        audit, prediction = deepcopy(commits[4])
        audit['rescored_rows'][0][1].append([3, 0.])
        with pytest.raises(ValueError, match='support'): checker.check(audit, prediction)


def test_fresh_process_sealed_fixture_full_wrapper_and_failed_cap_receipt(prepared_rows, tmp_path, monkeypatch):
    data, _, cache, rows = prepared_rows
    path, _ = pair_file(tmp_path, rows)
    fit = fit_dataset(data, tool.sha_file(data/'manifest.json'), tmp_path/'fit',
        config=FitConfig(epochs=1, batch_size=4, hidden=8, heads=2, dropout=0.), require_full_train=False)
    cp = fit['seeds'][0]
    checkpoint = (tmp_path/'fit'/cp['checkpoint_manifest']).parent
    for k, v in THREAD_ENV.items(): monkeypatch.setenv(k, v)
    replay = tmp_path/'replay'
    run_inference(cache, path, tool.sha_file(path), checkpoint, cp['checkpoint_sha256'], replay,
        sequence=rows[0]['sequence_id'], backend='reachable_slot_bound_joint_beam', allow_fixture=True)
    receipt_sha = tool.sha_file(replay/'development-inference-receipt.json')
    with pytest.raises(ValueError, match='provenance'):
        tool.run(replay, receipt_sha, tmp_path/'not-real')
    assert not (tmp_path/'not-real').exists()
    command = 'from tools.event_track_v2x.audit_branch_state_replay import run; import sys; run(*sys.argv[1:],allow_fixture=True)'
    process = subprocess.run([sys.executable, '-c', command, str(replay), receipt_sha, str(tmp_path/'audit')],
        cwd=tool.ROOT, env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path/'audit'/'audit.json').read_bytes())
    assert result['frames'] == 1 and result['entire_sequence_exact']
    assert not result['GT_read'] and not result['cached_states_read'] and not result['paper_eligible']
    assert not result['identity_correctness_verified'] and not result['unchosen_branches_audited']
    with pytest.raises(ValueError, match='cap'):
        tool.run(replay, receipt_sha, tmp_path/'failed-cap', max_node_visits=1, allow_fixture=True)
    failure = json.loads((tmp_path/'failed-cap'/'failure.json').read_bytes())
    assert failure['status'] == 'failed' and failure['partial_events_not_acceptance']
    assert not (tmp_path/'failed-cap'/'audit.json').exists()
    with pytest.raises(ValueError, match='new nonsymlink'):
        tool.run(replay, receipt_sha, tmp_path/'audit', allow_fixture=True)
