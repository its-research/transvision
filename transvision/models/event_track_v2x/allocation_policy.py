"""Causal search-state features and immutable learned expansion priorities.

Priorities estimate signed one-operation MODEL decision-bound reduction per
charged search step. They are not probabilities, upper bounds, real ID losses,
or a claim of optimal sequential value of computation. No GT/identity tokens
or future evidence are inputs. Deterministic inference keeps every legal region.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .detection_cache_v2 import canonical


FEATURES = ('loss_weight', 'eta_upper', 'conditional_risk', 'optimization_gap',
    'model_regret_upper', 'active_entropy', 'top_two_log_gap', 'log_nodes',
    'log_loss_nodes', 'log_active', 'log_frontier', 'frontier_min_depth_fraction',
    'frontier_max_depth_fraction', 'remaining_budget_fraction', 'proposal_operation',
    'log_operation_steps', 'log_prefix_nodes', 'weighted_eta')
RECIPE = 'causal-component-search-state-priority-v1'
TARGET = 'signed_weighted_model_decision_bound_reduction_per_charged_step_v1'


def decision_bound(kernel, mass, scope, fallback):
    active, weights, retained, _, eta = mass
    _, decision = kernel._decode(active, weights, retained, eta, scope, fallback, fallback_on_risk=False)
    return decision


def priority_features(kernel, mass, scope, fallback, weight, operation, remaining, budget):
    """Only current solver statistics; no raw identities, absolute time or GT."""
    decision = decision_bound(kernel, mass, scope, fallback)
    active, logs, retained, _, eta = mass
    probabilities = [math.exp(w-retained) for w in logs]
    entropy = -math.fsum(p*math.log(p) for p in probabilities if p > 0)/math.log(max(2, len(active)))
    depth = [kernel._prefix(h).depth/kernel.n for h in kernel.frontier] if kernel.n else []
    values = (weight, eta, decision['conditional_risk'] or 0., decision['optimization_gap'] or 0.,
        decision['risk_bound'], entropy, min(50., logs[0]-logs[1])/50. if len(logs) > 1 else 1.,
        math.log1p(kernel.n), math.log1p(len(scope)), math.log1p(len(active)), math.log1p(len(depth)),
        min(depth, default=1.), max(depth, default=1.), remaining/max(1, budget),
        float(operation['base'] is not None), math.log1p(operation['requested_steps']),
        math.log1p(kernel.prefix_count), weight*eta)
    if len(values) != len(FEATURES) or not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite causal priority features')
    return tuple(values), decision


@dataclass(frozen=True, init=False)
class FrozenPriorityPolicy:
    """Small tanh MLP. Array buffers are immutable bytes, not writable views."""
    _buffers: tuple
    _shapes: tuple
    signature: str

    def __init__(self, weights):
        arrays = tuple(np.asarray(v, dtype='<f8') for v in weights)
        if (len(arrays) != 4 or arrays[0].ndim != 2 or arrays[0].shape[1] != len(FEATURES)
                or not 1 <= arrays[0].shape[0] <= 256 or arrays[1].shape != (arrays[0].shape[0],)
                or arrays[2].shape != (1, arrays[0].shape[0]) or arrays[3].shape != (1,)
                or any(not np.isfinite(a).all() for a in arrays)):
            raise ValueError('finite supported priority MLP weights required')
        buffers, shapes = tuple(a.tobytes() for a in arrays), tuple(a.shape for a in arrays)
        object.__setattr__(self, '_buffers', buffers)
        object.__setattr__(self, '_shapes', shapes)
        h = hashlib.sha256(canonical([RECIPE, FEATURES, TARGET, shapes]))
        for buffer in buffers:
            h.update(buffer)
        object.__setattr__(self, 'signature', h.hexdigest())

    @property
    def weights(self):
        # ndarray dtype/shape metadata remains mutable even when its buffer is
        # readonly. Never expose a persistent view used by later inference.
        return tuple(np.frombuffer(b, dtype='<f8').reshape(s) for b, s in zip(self._buffers, self._shapes))

    def scores(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError('finite priority feature rows required')
        a, b, c, d = self.weights
        with np.errstate(over='raise', invalid='raise'):
            result = np.tanh(np.tanh(x @ a.T+b) @ c.T+d)[:, 0]
        if not np.isfinite(result).all():
            raise ValueError('nonfinite learned priority; no silent confidence fallback')
        return result
