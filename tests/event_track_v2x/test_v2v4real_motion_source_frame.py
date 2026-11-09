import hashlib

import numpy as np
import pytest
import yaml

from tools.event_track_v2x.v2v4real_motion_source_frame import source_train_frame


def fixture(objects):
    pose = np.eye(4); pose[0, 3] = 20
    raw = yaml.safe_dump(dict(lidar_pose=pose.tolist(), vehicles=objects)).encode()
    binding = dict(label_sha256=hashlib.sha256(raw).hexdigest(), cav_id='1',
                   side='infrastructure-side', frame_ordinal=2,
                   source_to_world=pose.tolist(), sequence_id='scene', frame_key='000002')
    return raw, binding


def vehicle(associated=-1, x=3):
    return dict(obj_type='Car', location=[x, 0, 0], center=[0, 0, 0],
                extent=[2, 1, 1], angle=[0, 0, 0], ass_id=associated)


def test_geometry_stays_source_local_and_duplicate_ids_preserved():
    raw, b = fixture({1: vehicle(7), 2: vehicle(7)})
    out = source_train_frame(dict(split='train', official_test_read=False), raw, b)
    assert out['track_ids'] == [7, 7]
    np.testing.assert_allclose(out['states_source'][:, 0], 3)
    assert out['raw_classes'] == ['Car', 'Car']


def test_pedestrian_excluded_before_geometry_and_empty_roi():
    raw, b = fixture({1: dict(obj_type='Pedestrian'), 2: vehicle(x=1000)})
    out = source_train_frame(dict(split='train', official_test_read=False), raw, b)
    assert out['states_source'].shape == (0, 7)
    assert out['track_ids'] == []


def test_split_byte_and_pose_rejections():
    raw, b = fixture({1: vehicle()})
    with pytest.raises(ValueError, match='before label parsing'):
        source_train_frame(dict(split='official_test', official_test_read=True), b'bad', {})
    with pytest.raises(ValueError, match='SHA'):
        source_train_frame(dict(split='train', official_test_read=False), raw + b' ', b)
    b['source_to_world'][0][3] += 1
    with pytest.raises(ValueError, match='pose differ'):
        source_train_frame(dict(split='train', official_test_read=False), raw, b)
