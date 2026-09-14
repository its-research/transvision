"""Causal, source-conditioned visual identity potentials for Recover Before Fuse.

This is a trainable module, not a trained or calibrated tracking result.  It
leaves DetectionCacheV2 and the sealed 203-feature checkpoint unchanged.  A
temporal attention path actually affects current association; probabilities
and deterministic partition bounds remain separate responsibilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FEATURE_DIM = 203


@dataclass(frozen=True)
class CausalIdentityInput:
    """Prediction-only features; history can include either physical source.

    Timestamps are integer microseconds in the receiver clock.  Node features
    must already have been generated without labels/future data.  This type
    validates availability; it cannot attest the producer's provenance.
    """

    left: Tensor
    right: Tensor
    history: Tensor
    left_information_us: Tensor
    right_information_us: Tensor
    history_information_us: Tensor
    left_arrival_us: Tensor
    right_arrival_us: Tensor
    history_arrival_us: Tensor
    history_source: Tensor
    decision_us: int


@dataclass(frozen=True)
class IdentityPotentials:
    pair: Tensor
    left_unmatched: Tensor
    right_unmatched: Tensor
    temporal: Tensor
    temporal_unmatched: Tensor
    context_left: Tensor
    context_right: Tensor


def _integer_vector(value, size, name, device):
    if (not isinstance(value, Tensor) or value.shape != (size,)
            or value.dtype not in (torch.int32, torch.int64) or value.device != device):
        raise ValueError(f'{name} must be an integer vector on the feature device')


class RecoverableIdentityModel(nn.Module):
    """Shared visual encoder + causal temporal attention + structured potentials.

    No ground-truth identity is accepted by forward.  Hidden size/head count are
    implementation defaults, not parameters selected on official validation.
    """

    def __init__(self, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        if (type(hidden) is not int or type(heads) is not int or hidden < 1
                or heads < 1 or hidden % heads or not math.isfinite(dropout)
                or not 0 <= dropout < 1):
            raise ValueError('invalid hidden/head/dropout configuration')
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.source_embedding = nn.Embedding(2, hidden)
        self.time_embedding = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.temporal_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden)
        self.pair_scorer = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.temporal_scorer = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.left_unmatched = nn.Linear(hidden, 1)
        self.right_unmatched = nn.Linear(hidden, 1)
        self.temporal_unmatched = nn.Linear(hidden, 1)

    def _validate(self, batch):
        if not isinstance(batch, CausalIdentityInput) or type(batch.decision_us) is not int or batch.decision_us < 0:
            raise ValueError('expected causal identity input with nonnegative integer decision_us')
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for name in ('left', 'right', 'history'):
            x = getattr(batch, name)
            if (not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] != FEATURE_DIM
                    or x.device != device or x.dtype != dtype or not torch.isfinite(x).all()):
                raise ValueError(f'{name} must contain finite (N,203) features matching the model')
            info, arrival = getattr(batch, name + '_information_us'), getattr(batch, name + '_arrival_us')
            _integer_vector(info, len(x), name + '_information_us', device)
            _integer_vector(arrival, len(x), name + '_arrival_us', device)
            if torch.any(info < 0) or torch.any(arrival < info) or torch.any(arrival > batch.decision_us):
                raise ValueError('future, negative or causally inconsistent feature timestamps')
        _integer_vector(batch.history_source, len(batch.history), 'history_source', device)
        if torch.any((batch.history_source < 0) | (batch.history_source > 1)):
            raise ValueError('source must be 0 or 1')

    def _embed(self, features, source, information, arrival, decision):
        # Subtract integer timestamps before float conversion to preserve precision.
        age = torch.stack(((decision - information), (decision - arrival)), -1).to(features.dtype) / 1e6
        # Bounded coordinate encoding; no truncation or alteration of causal checks.
        age = torch.log1p(age)
        return self.encoder(features) + self.source_embedding(source) + self.time_embedding(age)

    @staticmethod
    def _pairs(a, b, scorer):
        aa = a[:, None].expand(-1, len(b), -1)
        bb = b[None].expand(len(a), -1, -1)
        return scorer(torch.cat((aa, bb, torch.abs(aa - bb), aa * bb), -1)).squeeze(-1)

    def forward(self, batch: CausalIdentityInput) -> IdentityPotentials:
        self._validate(batch)
        n = len(batch.left)
        current = torch.cat((batch.left, batch.right), 0)
        source = torch.cat((torch.zeros(n, dtype=torch.long, device=current.device),
                            torch.ones(len(batch.right), dtype=torch.long, device=current.device)))
        embedded = self._embed(current, source,
            torch.cat((batch.left_information_us, batch.right_information_us)),
            torch.cat((batch.left_arrival_us, batch.right_arrival_us)), batch.decision_us)
        history = self._embed(batch.history, batch.history_source,
            batch.history_information_us, batch.history_arrival_us, batch.decision_us)
        if len(history) and len(current):
            attended, _ = self.temporal_attention(embedded[None], history[None], history[None], need_weights=False)
            context = self.context_norm(embedded + attended[0])
        else:
            context = self.context_norm(embedded)
        left, right = context[:n], context[n:]
        result = IdentityPotentials(
            self._pairs(left, right, self.pair_scorer), self.left_unmatched(left).squeeze(-1),
            self.right_unmatched(right).squeeze(-1), self._pairs(context, history, self.temporal_scorer),
            self.temporal_unmatched(context).squeeze(-1), left, right)
        if any(not torch.isfinite(value).all() for value in vars(result).values()):
            raise ValueError('identity network produced nonfinite potentials or context')
        return result


def _targets(value, shape, device, name):
    if (not isinstance(value, Tensor) or value.shape != shape or value.device != device
            or not torch.isfinite(value).all() or torch.any((value != 0) & (value != 1))
            or torch.any(value.sum(1) > 1)):
        raise ValueError(f'{name} must be a finite row-exclusive binary target matrix')


def identity_training_loss(output: IdentityPotentials, cross_target: Tensor,
                           temporal_target: Tensor, temporal_weight=1.0):
    """Offline targets only; cross-target one-to-one, temporal targets row-exclusive.

    Multiple current-source observations may refer to the same historical token.
    Targets never enter the forward inference object. This loss does not claim
    exact global-assignment likelihood or probability calibration.
    """
    if not math.isfinite(temporal_weight) or temporal_weight < 0:
        raise ValueError('temporal_weight must be finite and nonnegative')
    _targets(cross_target, output.pair.shape, output.pair.device, 'cross_target')
    if torch.any(cross_target.sum(0) > 1):
        raise ValueError('cross targets must be one-to-one')
    _targets(temporal_target, output.temporal.shape, output.temporal.device, 'temporal_target')

    def assignment(logits, unmatched, target):
        if logits.shape[0] == 0:
            return logits.sum() * 0 + unmatched.sum() * 0
        if logits.shape[1] == 0:
            labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        else:
            labels = torch.where(target.sum(1) > 0, target.to(torch.long).argmax(1),
                                 torch.full((len(logits),), logits.shape[1], device=logits.device, dtype=torch.long))
        return F.cross_entropy(torch.cat((logits, unmatched[:, None]), -1), labels)

    cross = (assignment(output.pair, output.left_unmatched, cross_target)
             + assignment(output.pair.T, output.right_unmatched, cross_target.T)) / 2
    temporal = assignment(output.temporal, output.temporal_unmatched, temporal_target)
    return {'total': cross + temporal_weight * temporal, 'cross_assignment': cross,
            'temporal_assignment': temporal}


def detached_log_potentials(output: IdentityPotentials):
    """Return finite log-potentials, never a posterior-mass certificate.

    Each matched edge uses its row/column normalized score. All-unmatched has
    positive weight. Full normalization and residual bounds belong to the bank.
    """
    row = F.log_softmax(torch.cat((output.pair, output.left_unmatched[:, None]), -1), -1)
    col = F.log_softmax(torch.cat((output.pair.T, output.right_unmatched[:, None]), -1), -1)
    return tuple(x.detach().to(device='cpu', dtype=torch.float64).numpy().copy()
                 for x in (row[:, :-1] + col[:, :-1].T, row[:, -1], col[:, -1]))
