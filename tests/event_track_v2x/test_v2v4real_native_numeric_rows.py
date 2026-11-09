"""Synthetic driver boundary tests, not original CUDA runtime acceptance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.event_track_v2x import capture_v2v4real_native_numeric_rows as m


def setup(tmp_path, monkeypatch):
    class Detector:
        device = 'cuda:0'
        checkpoint_sha256 = m.CHECKPOINTS[1337]
        checkpoint = tmp_path / 'checkpoint'
    monkeypatch.setattr(m, 'asset', lambda path, sha: path)
    core = SimpleNamespace(FrozenPointPillar=Detector, __file__=tmp_path/'core.py')
    pcd = tmp_path/'pcd'; pcd.write_bytes(b'synthetic')
    raw = dict(states_source_legacy7=np.empty((0, 7)), raw_scores=np.empty(0),
        class_indices=np.empty(0, dtype=int), appearance=np.empty((0, 128)),
        appearance_valid=np.empty(0, dtype=bool))
    np.savez(tmp_path/'raw.npz', **raw)
    r = dict(sequence_id='test', frame_ordinal=0, frame_id='0', side='vehicle-side',
        payload='raw.npz', payload_sha256=m.digest(tmp_path/'raw.npz'),
        pcd_sha256=m.digest(pcd), candidates=0)
    index=tmp_path/'raw.jsonl'; index.write_text(json.dumps(r)+'\n')
    original=dict(sequence_id='test', frame_ordinal=0, frame_key='0', is_ego=True,
        pcd_path='pcd', pcd_sha256=m.digest(pcd))
    calls=[]
    def capture(*args, **kwargs):
        calls.append(1)
        return dict(raw, states=np.empty((0, 9))), {'bev': np.ones((1,128,1,1), dtype='float32')}
    monkeypatch.setattr(m,'capture_predict',capture)
    args=dict(records=[original], inputs_root=tmp_path, detector=Detector(), core=core,
        lidar_range=[-1,-1,-1,1,1,1], read_pcd=lambda *a, **k: SimpleNamespace(xyzi=np.zeros((1,4))),
        mask_ego=lambda x:x, mask_range=lambda x,r:x, seed=1337, source_archive=tmp_path/'source',
        accepted_raw_index=index, accepted_raw_sha=m.digest(index), accepted_raw_root=tmp_path,
        output=tmp_path/'output', max_trace_bytes=1024)
    return args,calls


def test_driver_keeps_empty_frame_and_marks_producer_only(tmp_path, monkeypatch):
    args,calls=setup(tmp_path,monkeypatch); progress=[]
    result=m.capture_rows(**args,progress=progress.append)
    assert calls==[1] and result['rows']==1
    assert not result['independent_numeric_acceptance']
    row=json.loads((args['output']/'traces.jsonl').read_text())
    assert row['raw_rows_sha256']==args['accepted_raw_sha']
    assert progress[0]['completed_rows']==1 and progress[0]['eta_seconds']==0


@pytest.mark.parametrize('case',['duplicate','pcd','cpu','budget'])
def test_driver_rejects_scope_asset_device_and_truncation(tmp_path,monkeypatch,case):
    args,calls=setup(tmp_path,monkeypatch)
    if case=='duplicate': args['records']*=2
    if case=='pcd': (tmp_path/'pcd').write_bytes(b'changed')
    if case=='cpu': args['detector'].device='cpu'
    if case=='budget': args['max_trace_bytes']=1
    with pytest.raises(ValueError):m.capture_rows(**args)
    assert not (args['output']/'traces.jsonl').exists()
    assert calls == ([1] if case=='budget' else [])
    if case=='budget':
        capacity=json.loads((args['output']/'capacity-preflight.json').read_text())
        assert capacity['within_budget'] is False
        assert capacity['projected_cohort_uncompressed_upper_bound_bytes']>args['max_trace_bytes']
