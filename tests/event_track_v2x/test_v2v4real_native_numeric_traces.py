"""Trace transport/binding boundaries on synthetic data, not GPU acceptance."""
import json

import numpy as np
import pytest

from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import CORE_SHA, digest, verify


def fixture(tmp_path):
    anchors = np.zeros((1, 1, 1, 7), dtype=np.float32)
    anchors[..., 3:6] = 1
    np.savez(tmp_path / 'trace.npz', anchors=anchors,
             psm=np.full((1, 1, 1, 1), -10, dtype=np.float32),
             regression=np.zeros((1, 7, 1, 1), dtype=np.float32),
             bev=np.ones((1, 128, 1, 1), dtype=np.float32),
             observed_anchor_scores=np.array([1 / (1 + np.exp(10))], dtype=np.float32))
    np.savez(tmp_path / 'raw.npz', states_source_legacy7=np.empty((0, 7)),
             raw_scores=np.empty(0), appearance=np.empty((0, 128)),
             appearance_valid=np.empty(0, dtype=bool), class_indices=np.empty(0, dtype=int))
    row = dict(sequence_id='test01', frame_ordinal=0, side='vehicle-side', frame_id='native-0',
               pcd_sha256='c' * 64, payload='raw.npz', payload_sha256=digest(tmp_path / 'raw.npz'), candidates=0)
    raw = tmp_path / 'raw.jsonl'; raw.write_text(json.dumps(row) + '\n')
    trace = dict(row, payload='trace.npz', payload_sha256=digest(tmp_path / 'trace.npz'),
                 checkpoint_sha256='a' * 64, source_archive_sha256='b' * 64,
                 inference_core_sha256=CORE_SHA, raw_rows_sha256=digest(raw), TF32_enabled=False)
    index = tmp_path / 'trace.jsonl'; index.write_text(json.dumps(trace) + '\n')
    args = dict(trace_index=index, trace_sha=digest(index), raw_index=raw, raw_sha=digest(raw),
                trace_root=tmp_path, raw_root=tmp_path, checkpoint_sha='a' * 64,
                source_archive_sha='b' * 64, lidar_range=[-1, -1, -1, 1, 1, 1])
    return args, trace


def test_empty_raw_row_is_checked_and_scope_is_limited(tmp_path):
    args, _ = fixture(tmp_path)
    result = verify(**args)
    assert result['rows'] == 1 and result['candidates'] == 0
    assert result['independent_post_head_math_verified']
    assert not result['network_forward_verified'] and not result['paper_metric_complete']


@pytest.mark.parametrize('change', ['checkpoint', 'frame', 'duplicate', 'payload', 'escape'])
def test_binding_and_transport_corruption_fail_closed(tmp_path, change):
    args, trace = fixture(tmp_path)
    if change == 'checkpoint': trace['checkpoint_sha256'] = 'd' * 64
    if change == 'frame': trace['frame_id'] = 'different'
    if change == 'escape': trace['payload'] = '../trace.npz'
    if change == 'payload': (tmp_path / 'trace.npz').write_bytes(b'corrupt')
    text = json.dumps(trace) + '\n'
    if change == 'duplicate': text *= 2
    args['trace_index'].write_text(text)
    args['trace_sha'] = digest(args['trace_index'])
    with pytest.raises(ValueError): verify(**args)
