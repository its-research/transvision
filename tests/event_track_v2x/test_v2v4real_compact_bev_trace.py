"""Lossless sampling-cell checks; synthetic inputs, not actual GPU acceptance."""
import numpy as np
import pytest

from tools.event_track_v2x.v2v4real_compact_bev_trace import compact_trace,compact_features
from tools.event_track_v2x.v2v4real_native_numeric_reference import (
    bev_features,decode_heads,stable_observed_top64,legacy_boxes,verify_raw_row)


def trace(channels=128,empty=False):
    rng=np.random.default_rng(734)
    a=np.zeros((3,5,2,7),dtype=np.float32);a[...,3:6]=1
    a[...,0]=rng.uniform(-1.5,1.5,size=a.shape[:-1]);a[...,1]=rng.uniform(-1.5,1.5,size=a.shape[:-1])
    a.reshape(-1,7)[:4,:2]=[[-1,-1],[1,1],[-1,1],[0,0]]
    p=np.full((1,2,3,5),-10 if empty else 0,dtype=np.float32)
    r=np.zeros((1,14,3,5),dtype=np.float32)
    scores=np.full(30,1/(1+np.exp(10)) if empty else .5,dtype=np.float32)
    return dict(psm=p,regression=r,anchors=a,observed_anchor_scores=scores,
        bev=rng.normal(size=(1,channels,9,11)).astype(np.float32))


@pytest.mark.parametrize('channels',[3,64,128,384])
def test_cells_preserve_full_reference_including_boundaries_outside_and_ties(channels):
    t=trace(channels);bounds=[-1,-1,-1,1,1,1];c=compact_trace(t,bounds)
    boxes,scores=decode_heads(t['psm'],t['regression'],t['anchors'])
    selected=stable_observed_top64(t['observed_anchor_scores'],scores)
    expected,valid=bev_features(t['bev'],boxes[selected],bounds)
    actual,v=compact_features(boxes=boxes[selected],lidar_range=bounds,
        **{k:c[k] for k in ('bev_shape','bev_cell_indices','bev_cell_values')})
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(v,valid)
    values=t['bev'][0,:,c['bev_cell_indices'][:,0],c['bev_cell_indices'][:,1]]
    np.testing.assert_array_equal(c['bev_cell_values'],values)
    result=verify_raw_row(**c,lidar_range=bounds,states_legacy7=legacy_boxes(boxes[selected]),
        raw_scores=t['observed_anchor_scores'][selected],appearance=actual,appearance_valid=v,
        class_indices=np.zeros(len(selected),dtype=int))
    assert not result['network_forward_verified']
    assert 'bev' not in c and c['bev_cell_values'].nbytes<t['bev'].nbytes


def test_empty_prediction_retains_heads_scores_and_explicit_empty_cells():
    t=trace(empty=True);c=compact_trace(t,[-1,-1,-1,1,1,1])
    assert c['bev_cell_indices'].shape==(0,2) and c['bev_cell_values'].shape==(0,128)
    assert c['psm'] is t['psm'] and c['observed_anchor_scores'] is t['observed_anchor_scores']


@pytest.mark.parametrize('case',['missing','duplicate','quantized'])
def test_incomplete_or_quantized_sampling_proof_fails(case):
    t=trace();bounds=[-1,-1,-1,1,1,1];c=compact_trace(t,bounds)
    boxes,_=decode_heads(t['psm'],t['regression'],t['anchors'])
    if case=='missing':c['bev_cell_indices']=c['bev_cell_indices'][1:];c['bev_cell_values']=c['bev_cell_values'][1:]
    if case=='duplicate':c['bev_cell_indices'][0]=c['bev_cell_indices'][1]
    if case=='quantized':c['bev_cell_values']=c['bev_cell_values'].astype(np.float16)
    with pytest.raises(ValueError):compact_features(boxes=boxes,lidar_range=bounds,
        **{k:c[k] for k in ('bev_shape','bev_cell_indices','bev_cell_values')})
