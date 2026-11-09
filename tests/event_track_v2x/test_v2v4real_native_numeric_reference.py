"""Compare independent math with the unchanged, hash-pinned original Torch code."""
import hashlib
from pathlib import Path
import tarfile
import types

import numpy as np
import pytest
import torch

from tools.event_track_v2x import v2v4real_native_numeric_reference as ref


@pytest.fixture(scope='module')
def original():
    archive=Path('/Volumes/Data/test/recover-before-fuse/historical/live-audit-20260926/fulltrain-native-export-source.tar.gz')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='f47f391db33e02813b05e10f598ecf9240877463a8e32b61bc5841a8769d38ea'
    with tarfile.open(archive) as t:
        raw=t.extractfile('project/transvision/models/event_track_v2x/paper_pointpillar.py').read()
    assert hashlib.sha256(raw).hexdigest()=='5138ba83aa381b722828265b4dfb4934bd53cc533685fd004db862b3158cd48b'
    module=types.ModuleType('unchanged_original_head_reference')
    exec(compile(raw,'hash-pinned-original-paper_pointpillar.py','exec'),module.__dict__)
    return module


@pytest.mark.parametrize('channels',[64,128,384])
def test_original_decode_and_bilinear_pooling(original,channels):
    rng=np.random.default_rng(1337+channels)
    psm=rng.normal(size=(1,2,7,9)).astype('float32')
    regression=rng.normal(0,.1,size=(1,14,7,9)).astype('float32')
    anchors=rng.normal(size=(7,9,2,7)).astype('float32');anchors[...,3:6]=[1.5,2,4]
    boxes,scores=ref.decode_heads(psm,regression,anchors)
    actual_boxes,actual_scores=original.decode_heads(torch.from_numpy(psm),torch.from_numpy(regression),anchors)
    np.testing.assert_allclose(actual_boxes.numpy(),boxes,atol=ref.ATOL,rtol=ref.RTOL)
    np.testing.assert_allclose(actual_scores.numpy(),scores,atol=ref.ATOL,rtol=ref.RTOL)
    selected=ref.stable_observed_top64(actual_scores.numpy(),scores)
    expected=sorted(np.flatnonzero(actual_scores.numpy()>=.05),key=lambda i:(-float(actual_scores[i]),int(i)))[:64]
    np.testing.assert_array_equal(selected,expected)
    bev=rng.normal(size=(1,channels,5,8)).astype('float32')
    bounds=[-5,-4,-2,5,4,2]
    # Exact edges, just outside, and an interior interpolation point.
    centers=np.array([[-5,-4],[5,4],[5.001,0],[0,-4.001],[1.1,-.7]])
    sample_boxes=np.zeros((len(centers),7));sample_boxes[:,:2]=centers
    features,valid=ref.bev_features(bev,sample_boxes,bounds)
    observed,observed_valid=original.bev_features(torch.from_numpy(bev),sample_boxes,bounds)
    np.testing.assert_array_equal(valid,observed_valid)
    np.testing.assert_allclose(features,observed,atol=ref.ATOL,rtol=ref.RTOL)
    np.testing.assert_array_equal(features[~valid],0)


def test_zero_empty_and_stable_ties():
    assert ref.bev_features(np.zeros((1,64,2,2)),np.zeros((0,7)),[-1,-1,-1,1,1,1])[0].shape==(0,128)
    values,valid=ref.bev_features(np.zeros((1,64,2,2)),np.zeros((1,7)),[-1,-1,-1,1,1,1])
    assert not valid.any() and not values.any()
    scores=np.ones(100,dtype='float32')*.5
    np.testing.assert_array_equal(ref.stable_observed_top64(scores,scores.astype('float64')),np.arange(64))


def test_score_tolerance_not_relaxed_and_explicit_dtype_required():
    with pytest.raises(AssertionError):ref.stable_observed_top64(np.array([.8],dtype='float32'),np.array([.7]))
    with pytest.raises(ValueError):ref.stable_observed_top64(np.array([.8],dtype='float64'),np.array([.8]))
    with pytest.raises(ValueError):ref.decode_heads(np.zeros((1,2,3,4)),np.zeros((1,14,3,4)),np.zeros((3,4,2,7)))


def test_seven_state_layout_has_no_fabricated_velocity():
    raw=np.array([[1,2,3,4,5,6,0]],dtype=float)
    actual=ref.legacy_boxes(raw)
    assert actual.shape==(1,7)
    np.testing.assert_allclose(actual,[[1,2,3,5,6,4,-np.pi/2]])


def test_full_raw_row_requires_all_anchor_scores_and_rejects_corruption(original):
    psm=np.zeros((1,2,2,2),dtype='float32')
    regression=np.zeros((1,14,2,2),dtype='float32')
    anchors=np.zeros((2,2,2,7),dtype='float32');anchors[...,3:6]=[1,2,3]
    bev=np.ones((1,64,4,4),dtype='float32');bounds=[-2,-2,-2,2,2,2]
    boxes,scores=original.decode_heads(torch.from_numpy(psm),torch.from_numpy(regression),anchors)
    boxes=boxes.numpy().astype('float64');scores=scores.numpy()
    features,valid=original.bev_features(torch.from_numpy(bev),boxes,bounds)
    states=boxes[:,[0,1,2,4,5,3,6]].copy();states[:,6]=-np.pi/2
    args=dict(psm=psm,regression=regression,anchors=anchors,bev=bev,lidar_range=bounds,
              observed_anchor_scores=scores,states_legacy7=states,raw_scores=scores.astype('float64'),
              appearance=features,appearance_valid=valid,class_indices=np.zeros(8,dtype='int64'))
    result=ref.verify_raw_row(**args)
    assert result['candidates']==8 and result['network_forward_verified'] is False
    assert result['paper_metric_complete'] is False
    corrupted=states.copy();corrupted[0,0]=.1
    with pytest.raises(AssertionError):ref.verify_raw_row(**dict(args,states_legacy7=corrupted))
    with pytest.raises(ValueError):ref.verify_raw_row(**dict(args,class_indices=np.ones(8,dtype='int64')))
    with pytest.raises(ValueError):ref.verify_raw_row(**dict(args,observed_anchor_scores=scores[:2]))
