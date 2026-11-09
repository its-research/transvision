import numpy as np
from tools.event_track_v2x.v2v4real_motion_fit_oracle import match, initial_prior, errors, moments
from tools.event_track_v2x.v2v4real_motion_fit import offline_matches, prior, residuals


def test_independent_score_order_and_collision_matching():
    rng = np.random.default_rng(2027)
    for _ in range(30):
        boxes = rng.normal(size=(20, 7)); gt = rng.normal(size=(9, 7)); scores = rng.random(20)
        np.testing.assert_array_equal(match(boxes,scores,gt), offline_matches(boxes,scores,gt))


def test_scalar_full9_residual_axes_wrap_initial_and_second_moment():
    target = np.array([[1,2,3,4,5,6,np.pi/2]])
    row = dict(side='vehicle-side', states_source_legacy7=np.array([[2,2,3,6,4,6,np.pi-.1]]),
               target_states_source=target, target_velocity_source=np.array([[3.,4.]]),
               target_velocity_valid=np.ones(1,dtype=bool), gt_matches=np.array([0]))
    pred = dict(velocity_valid=np.zeros(1,dtype=bool), velocity_source=np.array([[np.nan,np.nan]]))
    p = dict(**{'vehicle-side':{'velocity_source':[1.,2.]}})
    oracle = errors(row,pred,[0],p)['unavailable']
    _,valid,produced = residuals(row,pred,p)
    assert not valid.any()
    np.testing.assert_allclose(oracle,produced,atol=1e-12)
    m = moments(oracle*10)
    assert abs(m['covariance'][0][7] + 1.9) < 1e-12
    assert np.linalg.eigvalsh(m['covariance']).min() > 0


def test_scalar_prior_holdout_exclusion():
    targets = [dict(group=g,side=s,velocity_source=np.tile([v,v+1],(10,1)),velocity_valid=np.ones(10,dtype=bool))
               for g,v in [('fit',2.),('held',100.)] for s in ('vehicle-side','infrastructure-side')]
    assert initial_prior(targets,{'fit'}) == prior(targets,{'fit'})
