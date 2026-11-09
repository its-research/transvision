import numpy as np
import pytest

from tools.event_track_v2x.v2v4real_motion_independent_oracle import scalar_target_rows


def frame(ordinal, ids, centers, x=0, side='vehicle-side'):
    pose = np.eye(4); pose[0, 3] = x
    return dict(sequence_id='scene', side=side, frame_ordinal=ordinal,
                track_ids=ids, states_source=np.asarray([[c, 0, 0, 1, 1, 1, 0] for c in centers]).reshape(-1, 7), source_to_world=pose)


def test_scalar_static_world_and_empty_gap():
    out = list(scalar_target_rows([frame(0, [1], [4]), frame(1, [], []), frame(2, [1], [2], x=2)]))
    assert not out[0]['velocity_valid'][0]
    assert out[1]['velocity_source'].shape == (0, 2)
    np.testing.assert_array_equal(out[2]['velocity_source'], [[0, 0]])
    assert out[2]['predecessor_ordinals'] == [0]


def test_scalar_ambiguity_source_isolation_and_order():
    out = list(scalar_target_rows([frame(0, [1], [0]), frame(1, [1, 1], [1, 2]), frame(2, [1], [3]), frame(3, [1], [3], side='infrastructure-side')]))
    assert all(not x['velocity_valid'].any() for x in out)
    with pytest.raises(ValueError, match='noncausal'):
        list(scalar_target_rows([frame(0, [], []), frame(0, [], [])]))
