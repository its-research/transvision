"""Train-only binary score calibration and explicitly scoped diagnostics."""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _binary(probabilities, labels):
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    if (p.ndim != 1 or not len(p) or p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all()):
        raise ValueError('nonempty aligned binary observations required')
    return p, y


@dataclass(frozen=True)
class ExistenceCalibration:
    slope: float
    intercept: float
    fit_split: str
    fit_groups: tuple

    def __post_init__(self):
        if (self.fit_split != 'train' or not self.fit_groups or self.slope < 0 or not all(math.isfinite(x) for x in (self.slope, self.intercept))):
            raise ValueError('finite monotone train-only calibration required')

    def apply(self, probabilities):
        p = np.asarray(probabilities, float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('invalid raw probability')
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        return expit(self.slope * (np.log(clipped) - np.log1p(-clipped)) + self.intercept)


def fit_existence(probabilities, labels, *, protocol, groups, fit_groups):
    protocol.require_train()
    p, y = _binary(probabilities, labels)
    if len(groups) != len(p) or not set(groups) <= set(fit_groups) or len(set(y)) != 2:
        raise ValueError('calibration requires both labels in declared training groups')
    p = np.clip(p, 1e-7, 1 - 1e-7)
    x = np.log(p) - np.log1p(-p)

    def objective(theta):
        logits = theta[0] * x + theta[1]
        loss = np.mean(np.logaddexp(0., logits) - y * logits)
        residual = expit(logits) - y
        return loss, np.array([np.mean(residual * x), np.mean(residual)])

    result = minimize(objective, [1., 0.], jac=True, bounds=[(0., 100.), (-100., 100.)], method='L-BFGS-B')
    if not result.success:
        raise ValueError('existence calibration failed: ' + result.message)
    return ExistenceCalibration(float(result.x[0]), float(result.x[1]), 'train', tuple(sorted(set(groups))))


def binary_diagnostics(probabilities, labels, *, bins=10):
    p, y = _binary(probabilities, labels)
    if type(bins) is not int or bins < 1:
        raise ValueError('positive bin count required')
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    rows, ece = [], 0.
    for i in range(bins):
        selected = bucket == i
        n = int(selected.sum())
        confidence, frequency = (float(p[selected].mean()), float(y[selected].mean())) if n else (None, None)
        if n:
            ece += n / len(p) * abs(confidence - frequency)
        rows.append(dict(bin=i, count=n, confidence=confidence, frequency=frequency))
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(
        nll=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        brier=float(np.mean((p - y)**2)),
        ece=ece,
        count=len(p),
        bins=rows,
        scope='binary_events_not_joint_posterior_TV',
        joint_probability_certificate=False)


def separate_identity_losses(logits, contexts, targets):
    """Named cross-source and temporal set-valued row surrogates, not joint
    NLL."""
    import torch
    if not len(logits) == len(contexts) == len(targets) or not logits:
        raise ValueError('aligned nonempty training rows required')
    terms = {'cross_source': [], 'temporal': []}
    for row, context, target in zip(logits, contexts, targets):
        if len(row) != len(target.known) or len(row) != len(context.indices) or not torch.isfinite(row).all():
            raise ValueError('invalid identity training row')
        source = context.observations[-1].node.source_id
        cross = [False] + [o.node.source_id != source for o in context.observations[:-1]]
        for name, mask in [('cross_source', [True] + cross[1:]), ('temporal', [True] + [not x for x in cross[1:]])]:
            known = torch.tensor([k and m for k, m in zip(target.known, mask)], device=row.device)
            positive = torch.tensor([p and m for p, m in zip(target.positives, mask)], device=row.device)
            if positive.any() and known.sum() > positive.sum():
                terms[name].append(torch.logsumexp(row[known], 0) - torch.logsumexp(row[positive], 0))
    zero = sum(row.sum() * 0 for row in logits)
    result = {name: torch.stack(values).mean() if values else zero for name, values in terms.items()}
    return dict(result, loss=result['cross_source'] + result['temporal'], counts={name: len(values) for name, values in terms.items()}, objective='set_valued_row_surrogates')
