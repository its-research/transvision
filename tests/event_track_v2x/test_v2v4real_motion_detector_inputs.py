import hashlib
import io
import json
import tarfile
import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_motion_detector_inputs import selected_payloads


def fixture():
    b = io.BytesIO()
    np.savez(b, states_source_legacy7=np.array([[0, 0, 0, 2, 4, 1, 0]], dtype=np.float32),
             raw_scores=np.array([.1]), class_indices=np.array([2]),
             appearance=np.zeros((1, 128)), appearance_valid=np.array([False]))
    payload = b.getvalue()
    m = dict(kind='v2v4real_train_raw_native_features_v1', split='train', official_test_read=False,
             gt_read=False, candidate_protocol='rbf-all-class-top64-v1',
             matching_candidate_rule='raw_score_ge_0.05_stable_top64_before_nms', rows=1)
    row = dict(sequence_id='a', frame_id='0', side='vehicle-side', payload='payloads/a.npz',
               payload_sha256=hashlib.sha256(payload).hexdigest(), candidates=1)
    m['rows_sha256']=hashlib.sha256((json.dumps(row)+'\n').encode()).hexdigest()
    m['anchor_file_sha256']=hashlib.sha256(b'a').hexdigest()
    return m, row, payload


def archive(m, payload, extra=False):
    _, row, _ = fixture()
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode='w') as tar:
        entries = [('manifest.json', json.dumps(m).encode()), ('anchors.npy', b'a'), ('payloads/a.npz', payload)]
        entries.append(('predictions.jsonl', (json.dumps(row)+'\n').encode()))
        if extra: entries.append(('extra', b'x'))
        for name, data in entries:
            member = tarfile.TarInfo(name);member.size=len(data);tar.addfile(member, io.BytesIO(data))
    b.seek(0);return b


def test_test_rejected_before_archive_open():
    m, row, payload = fixture();m['split']='official_test'
    def forbidden(): raise AssertionError('opened test archive')
    with pytest.raises(ValueError, match='train-only'): selected_payloads(m, [row], {('a','0','vehicle-side')}, forbidden)


def test_raw_payload_kept_byte_identical_all_class():
    m, row, payload = fixture();key=('a','0','vehicle-side')
    data=selected_payloads(m,[row],{key},archive(m,payload))[key]
    assert data['payload_bytes']==payload and data['arrays']['class_indices'].tolist()==[2]


def test_corruption_inventory_and_selection_rejected():
    m,row,payload=fixture();key=('a','0','vehicle-side')
    with pytest.raises(ValueError, match='SHA'): selected_payloads(m,[row],{key},archive(m,payload+b'x'))
    with pytest.raises(ValueError, match='unexpected'): selected_payloads(m,[row],{key},archive(m,payload,True))
    with pytest.raises(ValueError, match='covered'): selected_payloads(m,[row],{('b','0','vehicle-side')},archive(m,payload))


def test_archive_index_pin_not_just_member_allowlist():
    m,row,payload=fixture();key=('a','0','vehicle-side')
    m['rows_sha256']='0'*64
    with pytest.raises(ValueError, match='index bytes'):selected_payloads(m,[row],{key},archive(m,payload))
