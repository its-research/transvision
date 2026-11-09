import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_causal_detector_motion import estimate, stream
from tools.event_track_v2x.v2v4real_causal_detector_motion_oracle import scalar_estimate


def frame(ordinal, centres, classes=None, side='vehicle-side', pose=None):
    s = np.array([list(p) + [2, 4, 1, 0] for p in centres], dtype=np.float32).reshape(-1, 7)
    return dict(sequence_id='s', side=side, frame_ordinal=ordinal,
                source_to_world=np.eye(4) if pose is None else pose,
                states_source_legacy7=s,
                class_indices=np.array([0]*len(s) if classes is None else classes, dtype=np.int64),
                row_sha256=f'{ordinal:064x}')


def compare(now, old, speed=100, gap=3):
    kwargs = dict(max_speed_mps=speed, max_gap_ordinals=gap)
    actual = estimate(now, old, **kwargs)
    oracle = scalar_estimate(now, old, **kwargs)
    for k in ('velocity_valid', 'predecessor_candidates'):
        np.testing.assert_array_equal(actual[k], oracle[k])
    np.testing.assert_allclose(actual['velocity_source'], np.asarray(oracle['velocity_source']).reshape(-1, 2),
                               atol=1e-4, rtol=1e-4, equal_nan=True)
    assert actual['reasons'] == oracle['reasons']
    assert actual['dependencies'] == oracle['dependencies']
    return actual


def test_world_static_moving_and_rotating_source():
    old = frame(0, [[10, 0, 0]])
    pose = np.eye(4); pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; pose[0, 3] = 1
    now = frame(1, [[0, -9, 0]], pose=pose)
    a = compare(now, old)
    np.testing.assert_array_equal(a['velocity_source'], [[0, 0]])
    assert a['velocity_valid'][0]


def test_all_classes_compete_without_gt_and_mutual_one_to_one():
    old = frame(0, [[0, 0, 0], [4, 0, 0], [0, 0, 0]], classes=[0, 0, 1])
    now = frame(1, [[.2, 0, 0], [.3, 0, 0], [4.2, 0, 0], [.1, 0, 0]], classes=[0, 0, 0, 1])
    a = compare(now, old)
    assert a['velocity_valid'].tolist() == [True, False, True, True]
    assert a['predecessor_candidates'].tolist() == [0, -1, 1, 2]
    assert a['reasons'][1] == 'not_mutual_nearest'
    bad = dict(now, track_ids=[1, 2, 3, 4])
    with pytest.raises(ValueError, match='GT-free'): estimate(bad, old, max_speed_mps=100, max_gap_ordinals=3)


def test_exact_ties_gap_gate_and_initial_unavailable():
    old = frame(0, [[-1, 0, 0], [1, 0, 0]])
    a = compare(frame(1, [[0, 0, 0]]), old)
    assert a['reasons'] == ['ambiguous_nearest'] and np.isnan(a['velocity_source']).all()
    a = compare(frame(4, [[0, 0, 0]]), old)
    assert a['reasons'] == ['gap_exceeded']
    a = compare(frame(1, [[10, 0, 0]]), old, speed=1)
    assert a['reasons'] == ['no_allowed_predecessor']
    a = compare(frame(0, [[1, 0, 0]]), None)
    assert not a['velocity_valid'].any() and np.isnan(a['velocity_source']).all()


def test_empty_breaks_history_and_sources_isolated():
    frames = [frame(0, [[0, 0, 0]]), frame(0, [[0, 0, 0]], side='infrastructure-side'),
              frame(1, []), frame(2, [[1, 0, 0]])]
    rows = list(stream(frames, max_speed_mps=100, max_gap_ordinals=3))
    assert all(not row['velocity_valid'].any() for row in rows)
    assert rows[-1]['dependencies'] == [frames[-1]['row_sha256'], frames[-2]['row_sha256']]


def test_noncausal_and_invalid_pose_rejected():
    now = frame(1, [[0, 0, 0]])
    for old in (frame(1, []), frame(2, []), frame(0, [], side='infrastructure-side')):
        with pytest.raises(ValueError): estimate(now, old, max_speed_mps=100, max_gap_ordinals=3)
    p = np.eye(4); p[0, 0] = 2
    with pytest.raises(ValueError): estimate(frame(2, [], pose=p), now, max_speed_mps=100, max_gap_ordinals=3)


def test_seeded_scalar_oracle_differential_and_no_input_mutation():
    rng = np.random.default_rng(1337)
    for _ in range(40):
        old = frame(0, rng.normal(size=(12, 3))*10, classes=rng.integers(0, 3, 12))
        now = frame(1, old['states_source_legacy7'][:, :3] + rng.normal(size=(12, 3))*.1,
                    classes=old['class_indices'].copy())
        original = now['states_source_legacy7'].tobytes()
        compare(now, old)
        assert now['states_source_legacy7'].tobytes() == original
