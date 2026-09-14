"""Prediction-only neural/geometry potentials for the SAME forest support.

Forest rows jointly use cross-source and temporal heads, rather than displaying identity probabilities that never affect the next decision. Row normalization is not global
posterior normalization and is never used as a mass bound.
"""
from __future__ import annotations
import math

import numpy as np
import torch
from torch.nn import functional as F

from .forest_tracking import FEATURE_RECIPE, RawIdentityDetection
from .identity_forest import ForestFactors, digest
from .learned_identity import CausalIdentityInput, RecoverableIdentityModel
from .recoverable_identity import model_digest
from .tracking_v2 import propagate


def geometry_parent_logit(current, previous, *, process_noise=.1):
    mean, cov = propagate(np.asarray(previous.mean), np.asarray(previous.covariance), (current.state_us - previous.state_us) / 1e6, process_noise)
    innovation = np.asarray(current.mean[:2]) - mean[:2]
    uncertainty = cov[:2, :2] + np.asarray(current.covariance)[:2, :2]
    sign, logdet = np.linalg.slogdet(uncertainty)
    if sign <= 0:
        raise ValueError('nonpositive motion innovation covariance')
    return float(-.5 * (innovation @ np.linalg.solve(uncertainty, innovation) + logdet) - math.log(2 * math.pi))


def _check(observations, support, decision_us, max_nodes, max_pairs):
    if (type(max_nodes) is not int or max_nodes < 1 or type(max_pairs) is not int or max_pairs < 1 or type(decision_us) is not int or decision_us < 0
            or len(observations) > max_nodes or len(observations)**2 > max_pairs):
        raise ValueError('neural token/pair budget or decision time invalid')
    if any(type(o) is not RawIdentityDetection or o.node.arrival_us > decision_us for o in observations):
        raise ValueError('future or non-prediction input')
    # Validate identity support BEFORE any network allocation.
    return ForestFactors(tuple(o.node for o in observations), tuple(tuple((p, 0.) for p in row) for row in support))


def neural_parent_logits(model, observations, support, decision_us, *, max_nodes=128, max_pairs=16384, geometry_weight=1., process_noise=.1, history_enabled=True):
    """Differentiable support-aligned logits; no identity targets enter
    forward.

    Every arrived observation is a query and history token. Re-scoring old queries with subsequently arrived evidence is legal at the current decision time. It does not
    retroactively change any previously committed prediction.
    """
    if (not isinstance(model, RecoverableIdentityModel) or not math.isfinite(geometry_weight) or geometry_weight < 0 or not math.isfinite(process_noise) or process_noise < 0):
        raise ValueError('invalid model or motion configuration')
    _check(observations, support, decision_us, max_nodes, max_pairs)
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    indices = [[i for i, obs in enumerate(observations) if obs.node.source_id == side] for side in (0, 1)]
    order = indices[0] + indices[1]
    position = {index: rank for rank, index in enumerate(order)}
    local = [{index: rank for rank, index in enumerate(group)} for group in indices]

    def features(ids):
        return torch.tensor([observations[i].features for i in ids], dtype=dtype, device=device).reshape(-1, 203)

    def times(ids, key):
        return torch.tensor([getattr(observations[i].node, key) for i in ids], dtype=torch.long, device=device)

    all_ids = list(range(len(observations)))
    batch = CausalIdentityInput(
        features(indices[0]), features(indices[1]), features(all_ids), times(indices[0], 'information_us'), times(indices[1], 'information_us'), times(all_ids, 'information_us'),
        times(indices[0], 'arrival_us'), times(indices[1], 'arrival_us'), times(all_ids, 'arrival_us'),
        torch.tensor([o.node.source_id for o in observations], dtype=torch.long, device=device), decision_us)
    output = model(batch) if history_enabled else model(batch, history_enabled=False)
    rows = []
    for i, parents in enumerate(support):
        source = observations[i].node.source_id
        source_unmatched = (output.left_unmatched if source == 0 else output.right_unmatched)[local[source][i]]
        row = []
        for parent in parents:
            if parent == -1:
                value = output.temporal_unmatched[position[i]] + source_unmatched
            else:
                value = output.temporal[position[i], parent]
                if observations[parent].node.source_id != source:
                    left, right = (i, parent) if source == 0 else (parent, i)
                    value = value + output.pair[local[0][left], local[1][right]]
                else:
                    value = value + source_unmatched
                value = value + geometry_weight * geometry_parent_logit(observations[i], observations[parent], process_noise=process_noise)
            row.append(value)
        rows.append(torch.stack(row))
    return tuple(rows)


def parent_supervision_loss(logits, support, targets):
    """Offline parent-pointer CE, not a globally normalized forest likelihood.

    None targets explicitly mark rows whose positive parent was outside candidate support. Such rows are excluded, not mislabeled as true births. The data runner must report this
    count and candidate coverage separately.
    """
    if len(logits) != len(support) or len(targets) != len(logits):
        raise ValueError('one aligned target/support per row required')
    terms, omitted = [], 0
    for row, parents, target in zip(logits, support, targets):
        if row.ndim != 1 or len(row) != len(parents) or not torch.isfinite(row).all():
            raise ValueError('finite support-aligned logits required')
        if target is None:
            omitted += 1
            continue
        if type(target) is not int or target not in parents:
            raise ValueError('supervision parent outside support')
        label = torch.tensor([parents.index(target)], device=row.device)
        terms.append(F.cross_entropy(row[None], label))
    if not terms:
        raise ValueError('no supervised rows; cannot report a zero training loss')
    return {'loss': torch.stack(terms).mean(), 'supervised_rows': len(terms), 'positive_outside_support_rows': omitted}


class LearnedForestScorer:
    feature_recipe = FEATURE_RECIPE

    def __init__(self, model, *, max_nodes=128, max_pairs=16384, geometry_weight=1., process_noise=.1, history_enabled=True):
        if (not isinstance(model, RecoverableIdentityModel) or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters())):
            raise ValueError('frozen eval-mode identity model required')
        self.model = model
        self.options = dict(max_nodes=max_nodes, max_pairs=max_pairs, geometry_weight=geometry_weight, process_noise=process_noise)
        if type(history_enabled) is not bool:
            raise TypeError('explicit history ablation flag required')
        if not history_enabled:
            self.options['history_enabled'] = False
        self.model_sha256 = model_digest(model)

    @property
    def signature(self):
        return digest(['learned_forest', self.feature_recipe, self.model_sha256, self.options])

    def __call__(self, observations, support, decision_us):
        if (any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()) or model_digest(self.model) != self.model_sha256):
            raise ValueError('frozen model changed')
        with torch.inference_mode():
            logits = neural_parent_logits(self.model, observations, support, decision_us, **self.options)
            rows = tuple(
                tuple((p, float(value)) for p, value in zip(parents,
                                                            F.log_softmax(row, dim=0).to(device='cpu', dtype=torch.float64).tolist())) for parents, row in zip(support, logits))
        return ForestFactors(tuple(o.node for o in observations), rows)


class GeometryForestScorer:
    """Declared geometry baseline, never a substitute for the learned
    method."""
    feature_recipe = FEATURE_RECIPE

    def __init__(self, *, birth_logit=-4., process_noise=.1):
        if not math.isfinite(birth_logit) or not math.isfinite(process_noise) or process_noise < 0:
            raise ValueError('invalid geometry baseline configuration')
        self.birth_logit, self.process_noise = birth_logit, process_noise

    @property
    def signature(self):
        return digest(['geometry_forest', self.feature_recipe, self.birth_logit, self.process_noise])

    def __call__(self, observations, support, decision_us):
        _check(observations, support, decision_us, 4096, 4096**2)
        rows = []
        for i, parents in enumerate(support):
            values = [self.birth_logit if p < 0 else geometry_parent_logit(observations[i], observations[p], process_noise=self.process_noise) for p in parents]
            maximum = max(values)
            normalizer = maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))
            rows.append(tuple((p, v - normalizer) for p, v in zip(parents, values)))
        return ForestFactors(tuple(o.node for o in observations), tuple(rows))
