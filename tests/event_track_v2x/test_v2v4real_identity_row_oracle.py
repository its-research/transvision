import math
import numpy as np
from tools.event_track_v2x.verify_v2v4real_identity_rows import observation,parents


def test_world_feature_geometry_and_full_covariance_cross_term():
    m=dict(lidar_to_world_row_rotation=np.eye(3).tolist(),lidar_to_world_translation=[1,2,3],
        box_reference_timestamp_us=100000,side='vehicle-side')
    state=[4,5,6,2,4,1,-math.pi/2,3,4];cov=np.eye(9);cov[0,7]=cov[7,0]=.2
    s,c,f=observation(state,cov,m,np.zeros(128),.8,.9,False)
    np.testing.assert_allclose(s,[5,7,9,4,2,1,0,3,4],atol=1e-15)
    assert c[0,7]==.2 and f.shape==(203,) and f[141]==.001 and f[143]==.1
    np.testing.assert_allclose(f[200:],[.8,.9,0])


def test_all_causal_parents_gate_and_same_frame_exclusion():
    a=dict(state_us=np.array([0,0,0,100000]),source=np.array([0,1,1,0]),frame_id=np.array(['0','0','0','1']),
        arrival_us=np.array([100000]*3+[200000]),decision_us=np.array([100000]*3+[200000]),mean=np.zeros((4,9)))
    a['mean'][0,0]=8;a['mean'][1,0]=8;a['mean'][2,0]=8.01
    assert parents(a,3)==[0,1]
    assert parents(a,2)==[0]


def test_parent_limit_uses_stable_index_before_sorted_context_order():
    a=dict(state_us=np.zeros(11,dtype=int),source=np.r_[np.ones(10,dtype=int),0],frame_id=np.array(['a']*10+['b']),
        arrival_us=np.zeros(11,dtype=int),decision_us=np.zeros(11,dtype=int),mean=np.zeros((11,9)))
    assert parents(a,10)==list(range(8))
