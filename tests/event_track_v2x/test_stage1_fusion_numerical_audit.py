import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[2]/'tools/event_track_v2x'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('audit_stage1_fusion_numerics',ROOT/'audit_stage1_fusion_numerics.py')
audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)


def auditor():
    x=object.__new__(audit.Audit);x.counts={'covariance_preimages':0}
    x.maximum_mean=x.maximum_covariance=x.maximum_scalar=0.
    return x


def record(mean,covariance):
    return dict(mean=mean.tolist(),covariance_sha256=hashlib.sha256(audit.canonical(covariance.tolist())).hexdigest(),covariance_trace=float(np.trace(covariance)),position_covariance_upper=covariance[:3,:3][np.triu_indices(3)].tolist())


def test_covariance_reference_needs_exact_preimage_even_inside_float_tolerance():
    p=np.eye(9);m=np.zeros(9);changed=p.copy();changed[7,8]=changed[8,7]=1e-12
    with pytest.raises(ValueError,match='SHA preimage'):
        auditor().record(record(m,p),(m,p),changed)


def test_independent_cross_terms_cannot_inherit_matching_original_SHA():
    p=np.eye(9);m=np.zeros(9);bad=p.copy();bad[4,8]=bad[8,4]=.001
    with pytest.raises(ValueError,match='complete covariance differs'):
        auditor().record(record(m,p),(m,bad),p)


def test_temporal_transport_retains_offdiagonal_velocity_and_absolute_noise():
    m=np.zeros(9);m[7]=3;p=np.eye(9);p[1,8]=p[8,1]=.2
    out,c=audit.transport(m,p,-.5)
    assert out[0]==-1.5 and c[0,7]==-.5 and c[1,8]==pytest.approx(-.3)
    assert c[0,0]==pytest.approx(1.3) and c[1,1]==pytest.approx(1.1)


def test_original_input_symlinks_are_rejected(tmp_path):
    p=tmp_path/'actual';p.write_text('x');link=tmp_path/'alias';link.symlink_to(p)
    with pytest.raises(ValueError,match='nonlinked'):audit.ordinary(link)
