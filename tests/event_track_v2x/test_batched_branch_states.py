from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import numpy as np
import pytest

from transvision.models.event_track_v2x.batched_branch_states import StateBatch, BatchedStateExclusiveTracker
from transvision.models.event_track_v2x.exclusive_completion_tracking import ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.tracking_v2 import propagate, ci
from test_forest_tracking import observation
from test_persistent_component_tracking import step, joint_action


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_batch_against_serial_full_covariance_yaw_and_time(device):
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            pytest.skip('real CUDA device unavailable; not a GPU acceptance')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(720)
    n = 37
    means, other = rng.normal(size=(2, n, 9))
    means[:, 6] += np.pi
    other[:, 6] -= np.pi
    a, b = rng.normal(size=(2, n, 9, 9))
    cov, ocov = a @ a.swapaxes(-1, -2)+np.eye(9)*.1, b @ b.swapaxes(-1, -2)+np.eye(9)*.1
    dt = rng.uniform(-.2, .8, n)
    got = StateBatch(device, max_batch=64).update(means, cov, other, ocov, dt, density=.1, weight=.3)
    for i in range(n):
        m, c = propagate(means[i], cov[i], dt[i], .1)
        m, c = ci(m, c, other[i], ocov[i], .3)
        np.testing.assert_allclose(got[0][i], m, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(got[1][i], c, atol=1e-8, rtol=1e-8)
    assert not np.shares_memory(got[0], means)


@pytest.mark.parametrize('batch_size', [1, 4])
def test_exclusive_replay_preserves_actions_work_and_late_arrival(tmp_path, batch_size):
    config = PersistentExclusiveCompletionConfig(state=ForestTrackingConfig(
        active_limit=3, expansion_budget=8, max_model_regret=1.))
    serial = ExclusiveCompletionTracker(tmp_path/'serial.sqlite', sequence_id='0003', config=config)
    batched = BatchedStateExclusiveTracker(tmp_path/'batch.sqlite', sequence_id='0003', config=config, max_batch=batch_size)
    original = None
    try:
        for frame in range(5):
            time = 1_000_000+frame*100_000
            state_time = time if frame != 3 else 1_050_000  # real late ordering path
            raw = tuple(observation(str(frame)+'-'+str(i), x=i*3+.1*frame,
                state_us=state_time, arrival_us=time, frame=str(frame), index=i) for i in range(4))
            rows = []
            for i in range(4):
                if frame:
                    rows.append(((-1, -12.), ((frame-1)*4+i, 0.), ((frame-1)*4+(i+1)%4, -15.)))
                else:
                    rows.append(((-1, 0.),))
            a = step(serial, raw, rows, time=time, event=str(frame))
            b = step(batched, raw, rows, time=time, event=str(frame))
            assert joint_action(serial, a) == joint_action(batched, b)
            for key in ('state_updates', 'search_steps', 'log_partition_upper', 'model_regret_upper'):
                assert a.audit[key] == b.audit[key]
            for x, y in zip(a.audit['components'], b.audit['components'], strict=True):
                for key in ('active', 'frontier', 'output_handle'):
                    assert x[key] == y[key]
            assert len(a.prediction['predictions']) == len(b.prediction['predictions'])
            for x, y in zip(a.prediction['predictions'], b.prediction['predictions'], strict=True):
                for key in ('track_id', 'class_label', 'score'):
                    assert x[key] == y[key]
                for key in ('mean', 'covariance'):
                    np.testing.assert_allclose(x[key], y[key], atol=1e-8, rtol=1e-8)
            if frame == 0:
                original = b.prediction_json
        assert batched.state_batch.rows > 0
        assert batched.state_batch.maximum_batch == batch_size
        assert batched.db.execute('SELECT prediction FROM events WHERE ordinal=0').fetchone()[0] == original
        batched.state_batch.max_batch = 2
        with pytest.raises(ValueError, match='settings changed'):
            step(batched, time=2_000_000, event='changed')
    finally:
        serial.close()
        batched.close()


def test_caps_invalid_covariance_and_resume_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        StateBatch(max_batch=0)
    batch = StateBatch(max_batch=1)
    with pytest.raises(ValueError):
        batch.update(np.ones((2, 9)), np.ones((2, 9, 9)), np.ones((2, 9)), np.ones((2, 9, 9)), [0, 0], density=.1, weight=.5)
    with pytest.raises(np.linalg.LinAlgError):
        batch.update(np.ones((1, 9)), -np.eye(9)[None], np.ones((1, 9)), np.eye(9)[None], [0], density=.1, weight=.5)
    with pytest.raises(ValueError, match='resume'):
        BatchedStateExclusiveTracker.open(tmp_path/'existing.sqlite')


def test_work_cap_rolls_back_without_stale_observation_cache(tmp_path):
    config = PersistentExclusiveCompletionConfig(state=ForestTrackingConfig(
        active_limit=2, expansion_budget=4, max_replay_operations=1))
    tracker = BatchedStateExclusiveTracker(tmp_path/'cap.sqlite', sequence_id='0003', config=config)
    try:
        raw = (observation('a'), observation('b', source=1))
        with pytest.raises(ValueError, match='work cap'):
            step(tracker, raw, (((-1, 0.),), ((-1, 0.),)))
        assert not tracker.raw_state_cache
        assert tracker.db.execute('SELECT count(*) FROM observations').fetchone()[0] == 0
        assert tracker.db.execute('SELECT count(*) FROM events').fetchone()[0] == 0
        got = step(tracker, (observation('replacement'),), (((-1, 0.),),))
        assert got.audit['state_updates'] == 1
    finally:
        tracker.close()


def test_sql_cache_is_bounded_namespace_safe_and_keeps_transaction_guard():
    from transvision.models.event_track_v2x.batched_branch_states import _CachedNamespace
    from transvision.models.event_track_v2x.persistent_component_store import _Namespace
    shared = OrderedDict()
    db = sqlite3.connect(':memory:')
    try:
        a, b = [_CachedNamespace(_Namespace(db, i), shared) for i in (1, 2)]
        for _ in range(2):
            assert a._sql('SELECT raw FROM observations WHERE i=?') == 'SELECT raw FROM pc1_observations WHERE i=?'
            assert b._sql('SELECT raw FROM observations WHERE i=?') == 'SELECT raw FROM pc2_observations WHERE i=?'
            with pytest.raises(ValueError, match='transaction'):
                a._sql('BEGIN IMMEDIATE')
        for i in range(400):
            a._sql('SELECT '+str(i))
        assert len(shared) == 256
    finally:
        db.close()


def test_nonprofile_qualifier_checks_real_persisted_states(tmp_path):
    replay = tmp_path/'input'
    replay.mkdir()
    serial = ExclusiveCompletionTracker(replay/'sequence.sqlite', sequence_id='0003',
        config=PersistentExclusiveCompletionConfig(state=ForestTrackingConfig(expansion_budget=4)))
    step(serial, (observation('a'),), (((-1, 0.),),))
    step(serial, (observation('b', state_us=1_200_000),), (((-1, -12.), (0, 0.)),),
         time=1_300_000, event='second')
    checksum = serial.close()
    receipt = replay/'receipt.json'
    receipt.write_text(json.dumps(dict(status='software_replay_completed',
        databases={'0003': dict(path='sequence.sqlite', sha256=checksum)})))
    output = tmp_path/'qualified'
    tool = Path(__file__).resolve().parents[2]/'tools/event_track_v2x/qualify_batched_branch_states.py'
    subprocess.run([sys.executable, str(tool), '--replay', str(replay),
        '--receipt-sha256', hashlib.sha256(receipt.read_bytes()).hexdigest(),
        '--sequence', '0003', '--events', '2', '--no-profile', '--output', str(output)],
        check=True, capture_output=True, text=True, timeout=60)
    result = json.loads((output/'candidate-check.json').read_bytes())
    assert result['materialized_branch_states_checked'] > 0
    assert result['discrete_actions_factors_work_identical']
    assert result['profiling_enabled'] is False
    assert result['production_promotion_allowed'] is False
