#!/usr/bin/env python3
"""Independently recompute fitted parameters from pinned example arrays.

This is parameter readback only. It does not independently reconstruct examples
from raw predictions/GT and cannot by itself grant D2 or paper acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CLASSES = ('car', 'bicycle', 'pedestrian')
SIDES = ('vehicle-side', 'infrastructure-side')
CONFIG = {'score': {'logit_clip': 1e-6, 'l2': .0001},
          'covariance': {'minimum_class_samples': 32, 'shrinkage': .1,
                         'floor_std': [.05]*6 + [.01, .1, .1]}, 'match_distance_m': 2.}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, name):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a, b, rtol=1e-10, atol=1e-12), 'recomputed parameter differs: '+name)


def score_oracle(scores, targets):
    x = np.log(np.clip(scores, 1e-6, 1-1e-6)/(1-np.clip(scores, 1e-6, 1-1e-6)))
    y = targets.astype(float)
    def objective(theta):
        z = theta[0]*x + theta[1]
        residual = expit(z)-y
        return (float(np.mean(np.logaddexp(0, z)-y*z)+.0001*theta[0]**2/2),
                np.array([np.mean(residual*x)+.0001*theta[0], np.mean(residual)]))
    prior = (y.sum()+1)/(len(y)+2)
    optimum = minimize(objective, [0., np.log(prior/(1-prior))], jac=True,
                       method='L-BFGS-B', bounds=[(0., 20.), (-30., 30.)],
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
    need(optimum.success and np.isfinite(optimum.x).all(), 'independent score fit did not converge')
    calibrated = expit(optimum.x[0]*x+optimum.x[1])
    return {'slope': float(optimum.x[0]), 'intercept': float(optimum.x[1]),
            'examples': len(y), 'positives': int(y.sum()), 'logit_clip': 1e-6,
            'in_sample_brier_raw': float(np.mean((scores-y)**2)),
            'in_sample_brier_calibrated': float(np.mean((calibrated-y)**2)),
            'optimizer_converged': True}


def covariance_oracle(residuals):
    values = residuals.copy()
    values[:, 6] = (values[:, 6]+np.pi) % (2*np.pi)-np.pi
    second = values.T@values/len(values)
    matrix = .9*second+.1*np.diag(np.diag(second))+np.diag(np.square([.05]*6+[.01, .1, .1]))
    np.linalg.cholesky(matrix)
    return matrix, values.mean(0), float(np.linalg.eigvalsh(matrix).min())


def verify(path, expected_sha256):
    need(not path.is_symlink() and sha(path) == expected_sha256, 'calibration identity differs')
    c = json.loads(path.read_bytes())
    need(c.get('kind') == 'eventtrack_train_calibration_v1' and c.get('config') == CONFIG
         and c.get('candidate_policy') == 'raw-score>=0.05/all-class-top64',
         'calibration recipe differs from historical frozen numerical settings')
    e = c.get('evidence', {})
    need(e.get('in_sample_detector_predictions') is True
         and all(e.get(k) is False for k in ('held_out_gt_used_for_fitting',
             'official_validation_used_for_selection', 'test_payloads_read', 'paper_eligible')),
         'calibration scope differs')
    need(set(c['example_records']) == {side+'/'+name for side in SIDES for name in CLASSES}
         and set(c['sides']) == set(SIDES), 'example/model group coverage differs')
    checked = []
    for side in SIDES:
        need(set(c['sides'][side]) == set(CLASSES), 'coarse class coverage differs')
        examples = {}
        for name in CLASSES:
            record = c['example_records'][side+'/'+name]
            relative = Path(record['path'])
            need(relative.name == record['path'], 'unsafe example path')
            source = path.parent/relative
            need(not source.is_symlink() and source.stat().st_size == record['bytes']
                 and sha(source) == record['sha256'], 'example artifact bytes differ')
            with np.load(source, allow_pickle=False) as z:
                need(set(z.files) == {'scores', 'targets', 'residuals'}, 'example array fields differ')
                scores, targets, residuals = [z[k] for k in ('scores', 'targets', 'residuals')]
            need(scores.ndim == 1 and len(scores) > 0 and targets.shape == scores.shape
                 and set(np.unique(targets)) == {0, 1} and np.isfinite(scores).all()
                 and np.all((scores >= 0)&(scores <= 1)) and residuals.ndim == 2
                 and residuals.shape[1] == 9 and np.isfinite(residuals).all(), 'invalid examples')
            examples[name] = (scores, targets, residuals)
        pooled = np.concatenate([examples[name][2] for name in CLASSES])
        need(len(pooled) >= 32, 'insufficient pooled covariance support')
        for name, (scores, targets, residuals) in examples.items():
            model = c['sides'][side][name]
            score = score_oracle(scores, targets)
            need(set(model['score']) == set(score), 'score parameter fields differ')
            for key, value in score.items():
                if type(value) in (int, bool):
                    need(type(model['score'][key]) is type(value) and model['score'][key] == value,
                         'score support/convergence differs')
                else:
                    close(model['score'][key], value, key)
            fallback = len(residuals) < 32
            chosen = pooled if fallback else residuals
            matrix, mean, minimum = covariance_oracle(chosen)
            covariance = model['covariance']
            close(covariance['matrix'], matrix, 'covariance matrix')
            close(covariance['mean_residual_not_corrected'], mean, 'residual mean')
            close(covariance['minimum_eigenvalue'], minimum, 'minimum eigenvalue')
            need(covariance['samples'] == len(chosen) and model['class_covariance_support'] == len(residuals)
                 and model['covariance_source'] == ('side-pooled-fallback' if fallback else 'side-class')
                 and covariance['interpretation'] == 'conditional_on_2m_true_positive; includes_bias_second_moment; not_false_positive_uncertainty',
                 'covariance sample/fallback/interpretation differs')
            checked.append({'side': side, 'class': name, 'examples': len(scores),
                            'positives': int(targets.sum()), 'residuals': len(residuals),
                            'pooled_fallback': fallback})
    return {'kind': 'canonical_calibration_parameters_independent_example_readback',
            'calibration_sha256': expected_sha256, 'groups': checked,
            'independent_parameters_recomputed': True,
            'raw_GT_examples_independently_reconstructed': False,
            'numeric_tolerance': {'relative': 1e-10, 'absolute': 1e-12},
            'formal_v2_ready': False, 'paper_eligible': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.calibration, a.expected_sha256)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    with a.receipt.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CALIBRATION_PARAMETERS_RECOMPUTED '+sha(a.receipt), flush=True)


if __name__ == '__main__':
    main()
