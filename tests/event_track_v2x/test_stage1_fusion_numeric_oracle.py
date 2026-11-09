import importlib.util,math
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('stage1_fusion_numeric_oracle',Path(__file__).parents[2]/'tools/event_track_v2x/stage1_fusion_numeric_oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def meta(rotation=None):
    return dict(lidar_to_world_row_rotation=np.eye(3).tolist() if rotation is None else rotation,
                lidar_to_world_translation=[10.,20.,30.],box_reference_timestamp_us=1000000)


def state():return np.array([1.,2.,3.,2.,4.,1.,-.5*math.pi,5.,6.])


def test_full_legacy_jacobian_keeps_dimension_and_yaw_cross_terms():
    p=np.eye(9);p[3,6]=p[6,3]=.2;p[4,7]=p[7,4]=.1
    m,c=oracle.physical_source(state(),p,meta(),1000000)
    np.testing.assert_allclose(m,[11.,22.,33.,4.,2.,1.,0.,5.,6.],atol=1e-14)
    assert c[4,6]==c[6,4]==-.2 and c[3,7]==c[7,3]==.1


def test_rotation_and_transport_propagate_velocity_position_covariance():
    r=[[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]
    m,p=oracle.physical_source(state(),np.eye(9),meta(r),2000000)
    np.testing.assert_allclose(m,[2.,26.,33.,4.,2.,1.,math.pi/2,-6.,5.],atol=1e-14)
    assert p[0,0]==pytest.approx(2.1) and p[0,7]==pytest.approx(1.)
    assert p[1,1]==pytest.approx(2.1) and p[1,8]==pytest.approx(1.)


def test_infrastructure_after_reference_but_before_deadline_transport_has_negative_cross_terms():
    _,p=oracle.physical_source(state(),np.eye(9),meta(),900000)
    assert p[0,7]==pytest.approx(-.1) and p[0,0]==pytest.approx(1.02)


def test_pose_with_undefined_upright_projection_is_rejected():
    r=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    with pytest.raises(ValueError,match='heading undefined'):
        oracle.physical_source(state(),np.eye(9),meta(r),1000000)


def test_CI_diagonal_information_solution_and_circular_yaw():
    a,b=state(),state()+2.;a[6]=math.pi-.1;b[6]=-math.pi+.1
    m,p=oracle.ci(a,np.eye(9),b,np.eye(9)*3,.5)
    np.testing.assert_allclose(m[[0,1,2,3,4,5,7,8]],a[[0,1,2,3,4,5,7,8]]+.5)
    assert abs(oracle.angle(m[6]-(math.pi-.05)))<1e-14
    np.testing.assert_allclose(p,np.eye(9)*1.5)


def test_mixture_preserves_between_hypothesis_variance_and_circular_angles():
    a,b=state(),state();b[0]+=4.;a[6]=math.pi-.1;b[6]=-math.pi+.1
    m,p=oracle.mixture([a,b],[np.eye(9),np.eye(9)],[.5,.5])
    assert m[0]==3. and p[0,0]==5.
    assert abs(oracle.angle(m[6]-math.pi))<1e-14 and p[6,6]==pytest.approx(1.01)
    assert p[0,6]==pytest.approx(.2)


def test_bad_mass_is_rejected_and_full_covariance_comparison_checks_off_diagonals():
    with pytest.raises(ValueError,match='without renormalization'):
        oracle.mixture([state()],[np.eye(9)],[.9])
    bad=np.eye(9);bad[7,8]=bad[8,7]=.01
    with pytest.raises(ValueError,match='complete covariance differs'):
        oracle.compare_covariance(bad,np.eye(9))
