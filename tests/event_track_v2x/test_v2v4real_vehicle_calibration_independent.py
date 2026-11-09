import ast
import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from tools.event_track_v2x import verify_v2v4real_vehicle_calibration as checker


def test_matching_is_stable_strict_boundary_same_class_and_one_to_one():
    states=[[0,0],[.1,0],[2,0],[10,0],[10,0]]
    labels=checker.match_labels(states,[.8,.8,.9,.7,.6],[0,0,0,1,0],[[0,0],[10,0]],[0,0])
    assert labels.tolist()==[1,0,0,0,1]
    # Exact-distance tie chooses the first GT in the ordered source input.
    assert checker.match_labels([[0,0],[-1,0]],[.9,.8],[0,0],[[-1,0],[1,0]],[0,0]).tolist()==[1,0]
    assert checker.match_labels([],[],[],[],[]).shape==(0,)


def test_objective_gradient_matches_central_difference_without_regularization():
    scores=np.array([.05,.2,.3,.6,.7,.99]);x=checker.logit(scores);y=np.array([0,0,1,0,1,1])
    theta=np.array([1.8,-.2]);loss,gradient=checker.objective(theta,x,y)
    numeric=[]
    for i in range(2):
        delta=np.zeros(2);delta[i]=1e-5
        numeric.append((checker.objective(theta+delta,x,y)[0]-checker.objective(theta-delta,x,y)[0])/2e-5)
    np.testing.assert_allclose(gradient,numeric,rtol=1e-8,atol=1e-10)
    expected=np.mean(np.logaddexp(0,theta[0]*x+theta[1])-y*(theta[0]*x+theta[1]))
    assert loss==pytest.approx(expected,abs=1e-15)


def test_projected_gradient_handles_all_constrained_boundaries():
    np.testing.assert_array_equal(checker.projected_gradient([0,100],[1,-1]),[0,0])
    np.testing.assert_array_equal(checker.projected_gradient([100,-100],[-1,1]),[0,0])
    np.testing.assert_array_equal(checker.projected_gradient([0,0],[-1,.2]),[-1,.2])


def examples():
    scores=np.repeat([.05,.1,.2,.4,.7,.9],10)
    y=np.concatenate([np.array([1]*n+[0]*(10-n)) for n in (1,2,3,4,6,8)])
    x=checker.logit(scores)
    optimum=minimize(lambda t:checker.objective(t,x,y),[1.,0.],jac=True,
        bounds=[(0.,100.),(-100.,100.)],method='L-BFGS-B')
    return scores,y,dict(slope=optimum.x[0],intercept=optimum.x[1])


def test_complete_probability_and_diagnostics_crosscheck():
    scores,y,model=examples();probabilities,evidence=checker.verify_parameters(scores,y,model)
    assert evidence['projected_gradient_inf']<=1e-5 and evidence['L2']==0
    checker.compare_diagnostics(checker.diagnostics(probabilities,y),checker.diagnostics(probabilities,y))
    known=checker.diagnostics([0.,.5,1.],[0,1,1])
    assert known['brier']==pytest.approx(1/12)
    assert known['ece']==pytest.approx(1/6)
    assert known['bins'][0]['count']==known['bins'][5]['count']==known['bins'][9]['count']==1
    assert known['bins'][1]['confidence'] is None


@pytest.mark.parametrize('bad',['slope','intercept','negative_slope','too_large','old_labels'])
def test_changed_calibration_or_old_car_targets_are_rejected(bad):
    scores,y,model=examples()
    if bad=='slope':model['slope']+=.01
    elif bad=='intercept':model['intercept']+=.01
    elif bad=='negative_slope':model['slope']=-.01
    elif bad=='too_large':model['slope']=100.01
    else:y=y.copy();y[0]=1-y[0]
    with pytest.raises(ValueError):checker.verify_parameters(scores,y,model)


@pytest.mark.parametrize('bad',['nll','brier','ece','count','bin_count','empty_bin'])
def test_diagnostic_corruption_is_rejected(bad):
    expected=checker.diagnostics([.1,.8],[0,1]);actual=copy.deepcopy(expected)
    if bad in ('nll','brier','ece'):actual[bad]+=.001
    elif bad=='count':actual['count']+=1
    elif bad=='bin_count':actual['bins'][1]['count']+=1
    else:actual['bins'][0]['confidence']=0.
    with pytest.raises(ValueError):checker.compare_diagnostics(actual,expected)


def test_checker_has_no_producer_function_imports():
    tree=ast.parse(Path(checker.__file__).read_bytes())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x.startswith(('transvision','tools.event_track_v2x')) for x in imports)
    assert checker.RTOL==1e-10 and checker.ATOL==1e-12 and checker.PROJECTED_GRADIENT_TOL==1e-5
