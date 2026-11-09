"""Independent nine-state source, CI and circular-mixture numerical equations.

No predictor, cache adapter or production math helper is imported. These operate
only on original admitted detector states or sealed association alternatives.
"""
import math
import numpy as np

MEAN_ATOL = 1e-8
COVARIANCE_ATOL = 1e-8
RTOL = 1e-10


def angle(x): return (float(x) + math.pi) % (2 * math.pi) - math.pi


def gaussian(mean, covariance):
    m, p = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    if m.shape != (9,) or p.shape != (9,9) or not np.isfinite(m).all() or not np.isfinite(p).all():
        raise ValueError('finite complete original nine-state Gaussian required')
    if not np.allclose(p, p.T, atol=1e-12, rtol=1e-10): raise ValueError('symmetric covariance required')
    np.linalg.cholesky(p)
    return m.copy(), p.copy()


def physical_source(mean, covariance, metadata, target_us):
    """Compose legacy-axis, pose and causal transport Jacobians independently."""
    m, p = gaussian(mean, covariance)
    rotation = np.asarray(metadata['lidar_to_world_row_rotation'], dtype=float)
    translation = np.asarray(metadata['lidar_to_world_translation'], dtype=float)
    if rotation.shape != (3,3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('complete finite original pose required')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(rotation),1,atol=2e-5):
        raise ValueError('proper original row-vector rotation required')
    if type(target_us) is not int or type(metadata['box_reference_timestamp_us']) is not int:
        raise ValueError('native integer reference timestamps required')
    delta = (target_us - metadata['box_reference_timestamp_us'])/1e6
    converted_yaw = angle(-m[6] - math.pi/2)
    c,s = math.cos(converted_yaw),math.sin(converted_yaw)
    heading = np.array([c,s,0.])@rotation
    tangent = np.array([-s,c,0.])@rotation
    norm = float(heading[0]**2+heading[1]**2)
    if norm < 1e-8: raise ValueError('upright projected heading undefined')
    out = m.copy();out[3],out[4] = m[4],m[3]
    out[:3] = m[:3]@rotation+translation
    out[6] = math.atan2(float(heading[1]),float(heading[0]))
    out[7:9] = m[7:9]@rotation[:2,:2]
    out[0] += delta*out[7];out[1] += delta*out[8]
    legacy = np.eye(9);legacy[3:5,3:5] = [[0.,1.],[1.,0.]];legacy[6,6] = -1.
    pose = np.eye(9);pose[:3,:3] = rotation.T;pose[7:9,7:9] = rotation[:2,:2].T
    pose[6,6] = (heading[0]*tangent[1]-heading[1]*tangent[0])/norm
    motion = np.eye(9);motion[0,7] = motion[1,8] = delta
    transform = motion@pose@legacy
    cov = transform@p@transform.T+np.eye(9)*.1*abs(delta)
    return out,(cov+cov.T)/2


def ci(first, first_covariance, second, second_covariance, weight=.5):
    left,p = gaussian(first,first_covariance);right,q = gaussian(second,second_covariance)
    if not math.isfinite(weight) or not 0 <= weight <= 1: raise ValueError('CI weight outside original interval')
    right[6] = left[6]+angle(right[6]-left[6])
    # Solve the information equations, rather than importing the original
    # inverse/covariance multiplication implementation.
    identity = np.eye(9)
    information = weight*np.linalg.solve(p,identity)+(1-weight)*np.linalg.solve(q,identity)
    vector = weight*np.linalg.solve(p,left)+(1-weight)*np.linalg.solve(q,right)
    mean = np.linalg.solve(information,vector);cov = np.linalg.solve(information,identity)
    mean[6] = angle(mean[6])
    return mean,(cov+cov.T)/2


def mixture(means,covariances,weights):
    if not len(weights) or len(means)!=len(weights) or len(covariances)!=len(weights):
        raise ValueError('complete mixture alternatives required')
    w=np.asarray(weights,dtype=float)
    if not np.isfinite(w).all() or np.any(w<0) or abs(math.fsum(w)-1)>1e-12:
        raise ValueError('original posterior mass must sum to one without renormalization')
    checked=[gaussian(m,p) for m,p in zip(means,covariances)]
    anchor=checked[0][0][6]
    for m,_ in checked:m[6]=anchor+angle(m[6]-anchor)
    mean=np.array([math.fsum(float(m[j])*float(weight) for (m,_),weight in zip(checked,w)) for j in range(9)])
    cov=np.zeros((9,9))
    for (m,p),weight in zip(checked,w):
        deviation=m-mean;cov+=weight*(p+np.outer(deviation,deviation))
    mean[6]=angle(mean[6]);cov=(cov+cov.T)/2
    np.linalg.cholesky(cov)
    return mean,cov


def compare_mean(actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    difference=actual-expected
    if actual.shape!=(9,) or expected.shape!=(9,):raise ValueError('complete nine means required')
    difference[6]=angle(difference[6])
    limits=MEAN_ATOL+RTOL*abs(expected)
    if not np.isfinite(difference).all() or np.any(abs(difference)>limits):
        raise ValueError('independent complete physical mean differs')
    return float(np.max(abs(difference)))


def compare_covariance(actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=(9,9) or expected.shape!=(9,9):raise ValueError('complete covariance cross terms required')
    if not np.isfinite(actual).all() or not np.allclose(actual,expected,atol=COVARIANCE_ATOL,rtol=RTOL):
        raise ValueError('independent complete covariance differs')
    np.linalg.cholesky(actual)
    return float(np.max(abs(actual-expected)))
