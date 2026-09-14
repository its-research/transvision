"""Connect causal learned potentials to a fixed-support recoverable local bank.

This adapter is NOT a complete 3D lifecycle tracker or a calibrated posterior.
Node identity/ordering and support remain fixed for one bounded ambiguity
window. Birth/death, overlapping windows and state trajectories need a tracker
adapter before real-data method evaluation. Absolute neural re-scoring is not
misrepresented as an independent measurement likelihood.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

from .hypothesis_bank import EvidenceUpdate, HypothesisBank, LogAssociationFactors
from .learned_identity import CausalIdentityInput, RecoverableIdentityModel, detached_log_potentials


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64):
        raise ValueError('unsupported reproducible tensor dtype')
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()).hexdigest()


def model_digest(model):
    return hashlib.sha256(json.dumps({name: _tensor_digest(value) for name, value in sorted(model.state_dict().items())},
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _input_digest(batch):
    return hashlib.sha256(json.dumps({name: _tensor_digest(value) if isinstance(value, torch.Tensor) else value
        for name, value in vars(batch).items()}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class LearnedBankSnapshot:
    bank: object
    left_node_ids: tuple[str, ...]
    right_node_ids: tuple[str, ...]
    input_sha256: str
    model_sha256: str
    evidence_semantics: str = 'absolute_learned_potential_rescore_not_independent_Bayes_likelihood'
    calibrated_real_posterior: bool = False
    complete_3d_tracker: bool = False


class LearnedHypothesisSession:
    """Bounded local integration; caller must supply a frozen eval-mode model."""

    def __init__(self, model, batch, *, left_node_ids, right_node_ids, allowed=None,
                 active_limit=4, window_us=2_000_000, max_updates=64, max_history_tokens=512,
                 max_current_tokens=128, max_cross_pairs=4096, max_temporal_pairs=65536,
                 bank_limits=None):
        if (not isinstance(model, RecoverableIdentityModel)
                or any(module.training for module in model.modules())):
            raise ValueError('an eval-mode RecoverableIdentityModel is required')
        if not isinstance(batch, CausalIdentityInput):
            raise TypeError('expected prediction-only causal identity input')
        for value in (window_us, max_updates, max_history_tokens, max_current_tokens,
                      max_cross_pairs, max_temporal_pairs):
            if type(value) is not int or value < 1:
                raise ValueError('window/update/history limits must be positive integers')
        self.model = model
        self.left_node_ids = self._ids(left_node_ids, len(batch.left))
        self.right_node_ids = self._ids(right_node_ids, len(batch.right))
        self.window_us, self.max_updates, self.max_history_tokens = window_us, max_updates, max_history_tokens
        self.max_current_tokens, self.max_cross_pairs = max_current_tokens, max_cross_pairs
        self.max_temporal_pairs = max_temporal_pairs
        self.start_us = batch.decision_us
        self._check_input(batch)
        self.model_sha256 = model_digest(model)
        with torch.inference_mode():
            output = model(batch)
        self._factors = LogAssociationFactors(*detached_log_potentials(output), allowed=allowed)
        self.bank = HypothesisBank(self._factors, active_limit=active_limit, information_us=batch.decision_us,
                                   **(bank_limits or {}))
        self._initial_input = _input_digest(batch)
        self._updates = {}
        self._initial_snapshot = None

    @staticmethod
    def _ids(values, expected):
        result = tuple(values)
        if (len(result) != expected or len(set(result)) != len(result)
                or any(not isinstance(x, str) or not x for x in result)):
            raise ValueError('node IDs must be unique nonempty strings matching feature rows')
        return result

    def _check_input(self, batch):
        if not isinstance(batch, CausalIdentityInput):
            raise TypeError('expected prediction-only causal identity input')
        if len(batch.history) > self.max_history_tokens:
            raise ValueError('history budget exceeded; no silent token selection')
        current = len(batch.left) + len(batch.right)
        if (current > self.max_current_tokens or len(batch.left) * len(batch.right) > self.max_cross_pairs
                or current * len(batch.history) > self.max_temporal_pairs):
            raise ValueError('neural pair/token budget exceeded; no silent candidate selection')
        if not self.start_us <= batch.decision_us <= self.start_us + self.window_us:
            raise ValueError('outside the immutable component window; explicit new window required')
        self.model._validate(batch)
        for information in (batch.left_information_us, batch.right_information_us, batch.history_information_us):
            if torch.any(information < batch.decision_us - self.window_us):
                raise ValueError('feature history exceeds fixed lag; no silent expiry')

    def _check_model(self):
        if (any(module.training for module in self.model.modules())
                or model_digest(self.model) != self.model_sha256):
            raise ValueError('model changed during local inference session')

    def start(self, *, expansion_budget):
        self._check_model()
        if self._initial_snapshot is None:
            bank = self.bank.advance(decision_us=self.start_us, expansion_budget=expansion_budget)
            self._initial_snapshot = LearnedBankSnapshot(bank, self.left_node_ids, self.right_node_ids,
                                                        self._initial_input, self.model_sha256)
        return self._initial_snapshot

    def rescore(self, evidence_id, batch, *, left_node_ids, right_node_ids, expansion_budget):
        self._check_model()
        if self._initial_snapshot is None:
            raise ValueError('start the bank before adding evidence')
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError('evidence_id must be nonempty')
        if tuple(left_node_ids) != self.left_node_ids or tuple(right_node_ids) != self.right_node_ids:
            raise ValueError('node universe/order changed; this requires a different local component')
        self._check_input(batch)
        if len(batch.left) != len(self.left_node_ids) or len(batch.right) != len(self.right_node_ids):
            raise ValueError('node feature count changed')
        fingerprint = _input_digest(batch)
        if evidence_id in self._updates:
            previous, snapshot = self._updates[evidence_id]
            if previous != fingerprint:
                raise ValueError('conflicting duplicate evidence ID')
            return snapshot
        if len(self._updates) >= self.max_updates:
            raise ValueError('update budget exceeded; no hidden history removal')
        with torch.inference_mode():
            output = self.model(batch)
        new = LogAssociationFactors(*detached_log_potentials(output), allowed=self._factors.allowed)
        # Re-score the same model distribution, instead of double-counting a
        # full history-dependent network output as a fresh independent factor.
        delta = LogAssociationFactors(
            np.asarray(new.log_pair) - np.asarray(self._factors.log_pair),
            np.asarray(new.log_left_unmatched) - np.asarray(self._factors.log_left_unmatched),
            np.asarray(new.log_right_unmatched) - np.asarray(self._factors.log_right_unmatched),
            self._factors.allowed)
        bank = self.bank.advance(decision_us=batch.decision_us, expansion_budget=expansion_budget,
            evidence=EvidenceUpdate(evidence_id, batch.decision_us, delta))
        snapshot = LearnedBankSnapshot(bank, self.left_node_ids, self.right_node_ids, fingerprint, self.model_sha256)
        self._factors = new
        self._updates[evidence_id] = (fingerprint, snapshot)
        return snapshot

    @staticmethod
    def audit(snapshot):
        return asdict(snapshot)
