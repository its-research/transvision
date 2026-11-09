import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_motion_train_targets import targets

TRAIN = dict(split='train', official_test_read=False)

def frame(ordinal, x=0., translation=0., ids=None, side='vehicle-side'):
    ids = [1] if ids is None else ids
    p = np.eye(4); p[0, 3] = translation
    return dict(sequence_id='a', side=side, frame_ordinal=ordinal, track_ids=ids,
                states_source=np.array([[x, 0, 0, 2, 4, 1, 0]]*len(ids)).reshape(-1, 7), source_to_world=p)

def test_test_rejected_before_iterating():
    def forbidden():
        raise AssertionError('test labels opened')
        yield
    with pytest.raises(ValueError): targets(dict(split='official_test', official_test_read=True), forbidden())

def test_moving_sensor_static_world_and_initial_validity():
    a, b = targets(TRAIN, [frame(0, 10), frame(1, 9, 1)])
    assert not a['velocity_valid'][0] and np.isnan(a['velocity_source']).all()
    np.testing.assert_array_equal(b['velocity_source'], [[0, 0]])
    assert b['predecessor_ordinals'] == [0]

def test_source_isolation_and_empty_history():
    a, b, c, d = targets(TRAIN, [frame(0), frame(0, side='infrastructure-side'), frame(1, ids=[]), frame(2, 2)])
    assert not b['velocity_valid'].any() and c['velocity_source'].shape == (0, 2)
    np.testing.assert_array_equal(d['velocity_source'], [[10, 0]])

def test_duplicate_identity_breaks_history_and_duplicate_time_rejected():
    a, b, c = targets(TRAIN, [frame(0), frame(1, ids=[1, 1]), frame(2)])
    assert not b['velocity_valid'].any() and not c['velocity_valid'].any()
    with pytest.raises(ValueError): targets(TRAIN, [frame(1), frame(0)])
    with pytest.raises(ValueError): targets(TRAIN, [frame(0), frame(0)])

def test_rotated_source_velocity():
    a = frame(0); b = frame(1, 0)
    b['states_source'][0, 1] = -1
    b['source_to_world'][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(targets(TRAIN, [a, b])[1]['velocity_source'], [[0, -10]], atol=0, rtol=0)
