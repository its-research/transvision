"""Frozen 203-feature checkpoints and retained one-to-one association hypotheses.

The network architecture and encoder input exactly match the full-train v1
checkpoint; this is not the incompatible 208-feature GT-canary model.
"""
from __future__ import annotations

import heapq
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
import torch
from torch import nn

from .detection_cache_v2 import DetectionCacheV2, sha_file
from .prediction_features import encode_features, choose_candidates

SCHEMA_SHA = 'e449a9c4356ee641e7d65829103f85be873f8045a4b317da80ea5b4ee8405ef5'
PLAN_SHA = 'ddc6d32fc8e9d55b71c0139d67f67e85f027b01367a5cd70edcaefbb61a3b809'
DATA_SHA = '4219dd8ab9f0350553356e63d36496631097ebf581d77e9bf08a6502eff6f65a'
CALIBRATION_SHA = '418bcb07757656ddf1ef8e02694b82797ecc5d1ea7804a8ba6be55d47476735f'
CHECKPOINTS = {
    1337: 'de2363f4269f807e50200f5742ab55a6ed113110cf4764b5623f3000fb7696b0',
    2027: 'a8a785b41d539390e284c9fc191993eeb0592c6a0f21cd5239f32526b24f04f8',
    3407: '55ff4f9ca449e91fd155e7b7403a8103556ccb1c5e69fd4c43de65f0b7fd28c7',
}


class PredictedAssociation(nn.Module):
    def __init__(self, hidden=128, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(203, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.left_dustbin = nn.Linear(hidden, 1)
        self.right_dustbin = nn.Linear(hidden, 1)

    def forward(self, left, right):
        l, r = self.encoder(left), self.encoder(right)
        a, b = l[:, :, None, :].expand(-1, -1, r.shape[1], -1), r[:, None, :, :].expand(-1, l.shape[1], -1, -1)
        pair = self.pair(torch.cat([a, b, torch.abs(a - b), a * b], -1)).squeeze(-1)
        return pair, self.left_dustbin(l).squeeze(-1), self.right_dustbin(r).squeeze(-1)


def load_frozen_model(path, seed, device='cpu'):
    if seed not in CHECKPOINTS or sha_file(path) != CHECKPOINTS[seed]:
        raise ValueError('unrecognized or changed final checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'kind': 'eventtrack_predicted_association_checkpoint_v1', 'seed': seed, 'epoch': 24,
                'feature_schema_sha256': SCHEMA_SHA, 'plan_sha256': PLAN_SHA,
                'data_manifest_sha256': DATA_SHA}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError('checkpoint training/feature contract differs: ' + key)
    model = PredictedAssociation()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


def frame_features(frame, reference_metadata, origin_us, pair_delta_s, calibration):
    if not isinstance(frame, DetectionCacheV2):
        raise TypeError('the learned path requires formal DetectionCacheV2')
    if frame.metadata['calibration_sha256'] != CALIBRATION_SHA:
        raise ValueError('checkpoint calibration identity differs')
    selected = choose_candidates(frame.raw_scores, {'minimum_raw_score': .05, 'maximum_per_side': 64})
    arrays = {'scores': frame.raw_scores, 'class_indices': frame.class_indices,
              'appearance_128': frame.appearance, 'appearance_valid': frame.appearance_valid}
    features = encode_features(frame.states, frame.covariances, arrays, selected,
                               frame.metadata, reference_metadata, pair_delta_s, origin_us, calibration)
    return selected, features


def k_best_assignments(cost, count=3):
    """Murty partitioning on rectangular row-complete assignments, deterministic ties."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1] or np.isnan(cost).any() or count < 1:
        raise ValueError('invalid assignment cost/count')
    n = cost.shape[0]
    if n == 0:
        return [(0., ())]
    serial = itertools.count()
    heap = []
    def push(fixed, forbidden):
        constrained = cost.copy()
        for i, j in forbidden:
            constrained[i, j] = np.inf
        for i, j in fixed:
            value = constrained[i, j]
            constrained[i, :] = np.inf
            constrained[:, j] = np.inf
            constrained[i, j] = value
        try:
            rows, columns = linear_sum_assignment(constrained)
        except ValueError:
            return
        score = float(cost[rows, columns].sum())
        if len(rows) == n and np.isfinite(constrained[rows, columns]).all():
            heapq.heappush(heap, (score, tuple(columns.tolist()), next(serial), fixed, forbidden))
    push((), frozenset())
    result, seen = [], set()
    while heap and len(result) < count:
        score, columns, _, fixed, forbidden = heapq.heappop(heap)
        if columns not in seen:
            result.append((score, columns)); seen.add(columns)
        fixed_rows = {i for i, _ in fixed}
        prefix = list(fixed)
        for i in range(n):
            if i in fixed_rows:
                continue
            push(tuple(prefix), forbidden | {(i, columns[i])})
            prefix.append((i, columns[i]))
    return result


def association_hypotheses(pair, left_dustbin, right_dustbin, left_classes, right_classes, top_h=3):
    """Retain Top-H joint assignments plus explicit all-unmatched alternative.

Weights are a normalized truncated energy distribution, not calibrated identity
probabilities. No assignment is silently forced; right dustbin costs are included.
"""
    pair = np.asarray(pair, float)
    n, m = pair.shape
    ld, rd = np.asarray(left_dustbin, float), np.asarray(right_dustbin, float)
    if ld.shape != (n,) or rd.shape != (m,) or not all(np.isfinite(x).all() for x in [pair, ld, rd]):
        raise ValueError('nonfinite/mismatched association logits')
    if not n or not m:
        return [{'pairs': [], 'unmatched_left': list(range(n)), 'unmatched_right': list(range(m)), 'weight': 1., 'energy': 0.}]
    pair = pair.copy()
    pair[np.asarray(left_classes)[:, None] != np.asarray(right_classes)[None, :]] = -np.inf
    row = np.c_[pair, ld]; row -= logsumexp(row, axis=1)[:, None]
    col = np.c_[pair.T, rd]; col -= logsumexp(col, axis=1)[:, None]
    costs = np.full((n, m+n), np.inf)
    costs[:, :m] = -row[:, :m] - col[:, :n].T + col[:, -1][None, :]
    costs[np.arange(n), m+np.arange(n)] = -row[:, -1]
    hypotheses = k_best_assignments(costs, top_h)
    null = tuple(m+i for i in range(n))
    if null not in {x[1] for x in hypotheses}:
        hypotheses.append((float(costs[np.arange(n), null].sum()), null))
    energies = np.array([x[0] for x in hypotheses]) - col[:, -1].sum()
    weights = np.exp(-energies - logsumexp(-energies))
    result = []
    for (cost, columns), energy, weight in zip(hypotheses, energies, weights):
        pairs = [(i, j) for i, j in enumerate(columns) if j < m]
        result.append({'pairs': pairs, 'unmatched_left': [i for i, j in enumerate(columns) if j >= m],
                       'unmatched_right': sorted(set(range(m)) - {j for _, j in pairs}),
                       'energy': float(energy), 'weight': float(weight)})
    return result
