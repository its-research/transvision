import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest

from tools.event_track_v2x.build_v2v4real_full_causal_motion import apply, execute
from tools.event_track_v2x.v2v4real_causal_detector_motion import estimate
from tests.event_track_v2x.test_v2v4real_causal_detector_motion import frame


def model():
    a=np.eye(9);a[0,7]=a[7,0]=.2
    return dict(sides={s:dict(initial_prior=dict(velocity_source=[3.,-2.]),
        statuses=dict(matched=dict(covariance=a.tolist()),unavailable=dict(covariance=(a*2).tolist())))
        for s in ('vehicle-side','infrastructure-side')})


def test_first_prior_is_finite_and_preserves_full_cross_terms():
    f=frame(0,[[10,0,0]]);p=estimate(f,None,max_speed_mps=40.,max_gap_ordinals=1)
    v,c=apply(f,p,model())
    assert v.dtype==f['states_source_legacy7'].dtype
    np.testing.assert_array_equal(v,[[3,-2]])
    assert not p['velocity_valid'].any() and c[0,0,7]==.4


def test_empty_and_measured_zero_distinct():
    old=frame(0,[[10,0,0]]);f=frame(1,[[10,0,0]])
    p=estimate(f,old,max_speed_mps=40.,max_gap_ordinals=1);v,c=apply(f,p,model())
    assert p['velocity_valid'].all()
    np.testing.assert_array_equal(v,[[0,0]])
    assert c[0,0,7]==.2
    f=frame(2,[]);p=estimate(f,old,max_speed_mps=40.,max_gap_ordinals=1);v,c=apply(f,p,model())
    assert v.shape==(0,2) and c.shape==(0,9,9)


def test_invalid_covariance_rejected():
    f=frame(0,[[0,0,0]]);p=estimate(f,None,max_speed_mps=40.,max_gap_ordinals=1);m=model()
    m['sides']['vehicle-side']['statuses']['unavailable']['covariance'][0][0]=-1
    with pytest.raises(np.linalg.LinAlgError):apply(f,p,m)


def test_unaccepted_fit_rejected_before_raw_inputs(tmp_path):
    p=tmp_path/'fit.json';p.write_text(json.dumps(dict(kind='wrong',official_test_used=False)))
    args=SimpleNamespace(output=tmp_path/'output',fit_acceptance=p,
                         fit_acceptance_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='complete independent'):execute(args)
