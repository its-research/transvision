"""Closed numerical YAML grammar, independent of Python object unpickling."""
import base64
import json
import struct

import numpy as np
import pytest
import yaml

from transvision.models.event_track_v2x.v2v4real_inputs import (
    V2V4RealInputError, load_raw_yaml, pose_projection, pose_to_world,
    source_to_target, load_prepared_frames,
)
from tests.event_track_v2x.test_v2v4real_inputs import _fixture
from tools.event_track_v2x.prepare_v2v4real_inputs import prepare_native_inputs


def native_doc(matrix=None):
    matrix = np.eye(4) if matrix is None else matrix
    payload = base64.b64encode(struct.pack('<16d', *matrix.ravel())).decode()
    return f'''gps:
- !!python/object/apply:numpy.core.multiarray.scalar
  - &dtype !!python/object/apply:numpy.dtype
    args: [f8, false, true]
    state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]
  - !!binary AAAAAAAACEA=
lidar_pose: &pose !!python/object/apply:numpy.core.multiarray._reconstruct
  args: [!!python/name:numpy.ndarray '', !!python/tuple [0], !!binary Yg==]
  state: !!python/tuple [1, !!python/tuple [4, 4], *dtype, false, !!binary {payload}]
true_ego_pos: *pose
vehicles: {{}}
'''.encode()


def test_native_matrix_and_scalar_are_data_not_object_construction():
    m = pose_to_world([10., 2., .5, 10., 20., -5.])
    data = native_doc(m)
    result = load_raw_yaml(data)
    assert result['gps'] == [3.]
    np.testing.assert_array_equal(result['lidar_pose'], m)
    assert result['true_ego_pos'] is result['lidar_pose']
    assert set(pose_projection(result)) == {'lidar_pose', 'source_to_world'}
    np.testing.assert_array_equal(pose_projection(result)['source_to_world'], m)
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(data)  # No global registration into PyYAML's loader.


@pytest.mark.parametrize('old,new', [
    (b'numpy.core.multiarray._reconstruct', b'os.system'),
    (b'numpy.ndarray', b'builtins.eval'), (b'[f8, false, true]', b'[O8, false, true]'),
    (b'[f8, false, true]', b'[f8, true, true]'),
    (b'[4, 4]', b'[4000000000, 4]'), (b'[4, 4]', b'[true, 4]'),
    (b'*dtype, false', b'*dtype, true'), (b'!!binary Yg==', b'!!binary YQ=='),
    (b'!!python/tuple [0]', b'!!python/tuple [false]'),
    (b'AAAAAAAACEA=', b'AA=='), (b"[3, '<'", b"[true, '<'"),
    (b'null, -1, -1, 0', b'null, -1, -1, false'),
    (b'!!python/name:numpy.ndarray \'\'', b'!!python/name:numpy.ndarray execute'),
])
def test_unrecognized_or_malformed_numeric_records_rejected(old, new):
    with pytest.raises(V2V4RealInputError):
        load_raw_yaml(native_doc().replace(old, new))


def test_nonfinite_native_array_and_recursive_alias_rejected():
    m = np.eye(4); m[0, 0] = np.nan
    with pytest.raises(V2V4RealInputError, match='non-finite'):
        load_raw_yaml(native_doc(m))
    with pytest.raises(V2V4RealInputError):
        load_raw_yaml(native_doc().replace(b'!!binary Yg==', b'*pose'))


@pytest.mark.parametrize('change', ['scale', 'reflection', 'singular', 'bottom', 'nan', 'bool'])
def test_invalid_native_transforms_rejected(change):
    m = np.eye(4)
    if change == 'scale': m[0, 0] = 1.01
    if change == 'reflection': m[0, 0] = -1
    if change == 'singular': m[0, 0] = 0
    if change == 'bottom': m[3, 0] = .1
    if change == 'nan': m[0, 0] = np.nan
    if change == 'bool': m = m.astype(bool)
    with pytest.raises(V2V4RealInputError): pose_to_world(m)


def test_small_serialization_residual_is_preserved_not_projected_to_so3():
    m = np.eye(4); m[0, 0] += 6e-7
    np.testing.assert_array_equal(pose_to_world(m), m)
    np.testing.assert_allclose(source_to_target(np.eye(4), m), np.linalg.inv(m), atol=1e-15)


def test_matrix_preparation_and_hash_bound_reload(tmp_path):
    source, ego, evidence = _fixture(tmp_path)
    m = pose_to_world([3., 4., 5., 2., 3., 4.])
    for p in source.glob('*/*/*.yaml'): p.write_bytes(native_doc(m))
    output = tmp_path / 'projection'
    receipt = prepare_native_inputs(source, output, dataset_split='train', ego_agents=ego, source_evidence=evidence)
    manifest, frames = load_prepared_frames(output / 'inputs', expected_manifest_sha256=receipt['input_manifest_sha256'])
    assert manifest['kind'] == 'v2v4real_pose_lidar_projection_v2'
    for row in frames: np.testing.assert_array_equal(row['source_to_world'], m)
    # A v1 manifest must not be allowed to disguise native-matrix input as Euler.
    manifest['kind'] = 'v2v4real_pose_lidar_projection_v1'
    manifest['pose_convention'] = 'xyz-roll-yaw-pitch-degrees-RzRyNegRxNeg-v1'
    (output / 'inputs' / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(V2V4RealInputError, match='legacy'):
        load_prepared_frames(output / 'inputs')
