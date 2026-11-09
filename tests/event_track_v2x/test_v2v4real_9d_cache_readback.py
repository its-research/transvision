import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tools.event_track_v2x.verify_v2v4real_9d_cache import probability,verify


def test_scalar_probability_closed_forms_and_clipping():
    p=probability([0,.2,.5,.8,1],2.,0.)
    np.testing.assert_allclose(p[1:4],[1/17,.5,16/17],atol=1e-15)
    assert np.isfinite(p).all() and 0<p[0]<p[-1]<1
    assert np.all(np.diff(p)>0)


def test_scalar_probability_extreme_logits_and_empty():
    np.testing.assert_array_equal(probability([.2,.8],1.,1000.),[1,1])
    np.testing.assert_array_equal(probability([.2,.8],1.,-1000.),[0,0])
    assert probability([],1.,0.).shape==(0,)


@pytest.mark.parametrize('flag',['fixture','gt_payload_opened'])
def test_nonproduction_or_GT_bridge_rejected_before_arrays(tmp_path,flag):
    p=tmp_path/'cache-raw-bridge-receipt.json';d=dict(kind='v2v4real_raw_top64_explicit_motion_contract_bridge_v1',
        fixture=False,gt_payload_opened=False,motion_independent_admission_imported=True);d[flag]=True
    p.write_text(json.dumps(d));digest=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='actual GT-free'):
        verify(SimpleNamespace(scope=tmp_path,output=tmp_path/'independent',bridge_sha256=digest))
