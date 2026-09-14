import hashlib
import struct

import numpy as np
import pytest

from transvision.models.event_track_v2x.v2v4real_inputs import V2V4RealInputError
from transvision.models.event_track_v2x.v2v4real_pcd import decode_pcd, read_native_pcd


def pcd(red_values=(0, 128, 255)):
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\n'
              'COUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n').format(n=len(red_values))
    rows = []
    for i, red in enumerate(red_values):
        word = red << 16 | 0x1234
        encoded = struct.unpack('<f', struct.pack('<I', word))[0]
        rows.append(f'{i+.25} {-i-.5} {i+.75} {encoded:.10e}\n')
    return (header + ''.join(rows)).encode()


def test_rgb_bits_are_not_used_as_raw_intensity_and_order_is_preserved():
    raw = pcd()
    cloud = decode_pcd(raw)
    assert cloud.xyzi.dtype == np.float32 and cloud.xyzi.shape == (3, 4)
    np.testing.assert_array_equal(cloud.xyzi[:, 3], np.asarray([0, 128/255, 1], dtype=np.float32))
    np.testing.assert_array_equal(cloud.xyzi[:, 0], [.25, 1.25, 2.25])
    np.testing.assert_array_equal(cloud.rgb_bits, [0x1234, 0x801234, 0xff1234])
    assert cloud.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert not cloud.xyzi.flags.writeable and not cloud.rgb_bits.flags.writeable


@pytest.mark.parametrize('old,new', [
    (b'x y z rgb', b'x y z intensity'), (b'x y z rgb', b'x y z x'),
    (b'4 4 4 4', b'4 4 4 8'), (b'F F F F', b'F F F U'),
    (b'COUNT 1 1 1 1', b'COUNT 1 1 1 3'), (b'DATA ascii', b'DATA binary'),
    (b'DATA ascii', b'DATA binary_compressed'), (b'WIDTH 3', b'WIDTH 2'),
    (b'HEIGHT 1', b'HEIGHT 0'), (b'POINTS 3', b'POINTS 3.0'),
    (b'POINTS 3', b'POINTS 2000000000'), (b'VERSION 0.7', b'VERSION 0.7\nVERSION 0.7'),
    (b'COUNT 1 1 1 1', b'COUNT 1 1 1 1\nBOGUS 1'),
    (b'0.25', b'nan'), (b'0.25', b'inf'), (b'0.25', b'1e100'),
    (b'0.25', b'not-a-number'), (b'0.25', b'0.25 1'),
    (b'0.25', b'0.25#'), (b'VIEWPOINT 0 0 0 1', b'VIEWPOINT 0 0 0 0'),
])
def test_reject_unknown_schema_wrong_counts_nonfinite_and_partial_tokens(old, new):
    with pytest.raises(V2V4RealInputError): decode_pcd(pcd().replace(old, new))


@pytest.mark.parametrize('mutation', ['missing-row', 'extra-row', 'extra-token', 'no-data', 'unicode', 'long-header'])
def test_no_partial_or_silent_read(mutation):
    raw = pcd()
    if mutation == 'missing-row': raw = b'\n'.join(raw.splitlines()[:-1]) + b'\n'
    if mutation == 'extra-row': raw += b'1 2 3 0\n'
    if mutation == 'extra-token': raw += b'garbage\n'
    if mutation == 'no-data': raw = raw.split(b'DATA')[0]
    if mutation == 'unicode': raw = raw.replace(b'FIELDS', b'\xffIELDS')
    if mutation == 'long-header': raw = b'#' * 2048 + b'\n' + raw
    with pytest.raises(V2V4RealInputError): decode_pcd(raw)


def test_file_hash_binding_and_symlink_rejection(tmp_path):
    path = tmp_path / 'frame.pcd'; raw = pcd(); path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert read_native_pcd(path, expected_sha256=digest).source_sha256 == digest
    with pytest.raises(V2V4RealInputError, match='SHA-256 differs'):
        read_native_pcd(path, expected_sha256='0' * 64)
    linked = tmp_path / 'linked.pcd'; linked.symlink_to(path)
    with pytest.raises(V2V4RealInputError, match='symlink'):
        read_native_pcd(linked, expected_sha256=digest)


def test_all_256_red_levels_match_open3d_when_installed(tmp_path):
    o3d = pytest.importorskip('open3d')
    path = tmp_path / 'levels.pcd'; raw = pcd(tuple(range(256))); path.write_bytes(raw)
    cloud = o3d.io.read_point_cloud(str(path))
    reference = np.hstack((np.asarray(cloud.points), np.asarray(cloud.colors)[:, :1])).astype(np.float32)
    np.testing.assert_array_equal(decode_pcd(raw).xyzi, reference)


def test_complete_projection_audit_reads_no_native_yaml_or_gt(tmp_path, monkeypatch):
    from tests.event_track_v2x.test_v2v4real_inputs import _fixture
    from tools.event_track_v2x.prepare_v2v4real_inputs import prepare_native_inputs
    from tools.event_track_v2x.audit_v2v4real_pcd import audit
    from transvision.models.event_track_v2x import v2v4real_inputs
    source, ego, evidence = _fixture(tmp_path)
    for path in source.glob('*/*/*.pcd'): path.write_bytes(pcd())
    output = tmp_path / 'prepared'
    receipt = prepare_native_inputs(source, output, dataset_split='train', ego_agents=ego, source_evidence=evidence)
    def reject(*a, **k):
        raise AssertionError('PCD consumer must not read native YAML/GT')
    monkeypatch.setattr(v2v4real_inputs, 'load_raw_yaml', reject)
    result = audit(output / 'inputs', receipt['input_manifest_sha256'])
    assert result['source_frames'] == 4 and result['total_points'] == 12
    assert result['no_points_dropped'] and result['point_order_preserved']
    assert not result['source_yaml_or_GT_read']
    assert not result['all_frames_exact_oracle_parity']
    assert not result['paper_eligible'] and not result['detector_executed']


def test_oracle_mismatch_cannot_be_reported_as_parity(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from tests.event_track_v2x.test_v2v4real_inputs import _fixture
    from tools.event_track_v2x.prepare_v2v4real_inputs import prepare_native_inputs
    from tools.event_track_v2x.audit_v2v4real_pcd import audit
    source, ego, evidence = _fixture(tmp_path)
    for path in source.glob('*/*/*.pcd'): path.write_bytes(pcd())
    output = tmp_path / 'prepared'
    receipt = prepare_native_inputs(source, output, dataset_split='train', ego_agents=ego, source_evidence=evidence)
    fake = SimpleNamespace(__version__='0.19.0', io=SimpleNamespace(read_point_cloud=lambda *a, **k:
        SimpleNamespace(points=np.zeros((3, 3)), colors=np.zeros((3, 3)))))
    monkeypatch.setitem(sys.modules, 'open3d', fake)
    with pytest.raises(ValueError, match='Open3D XYZI mismatch'):
        audit(output / 'inputs', receipt['input_manifest_sha256'], open3d_oracle=True)
