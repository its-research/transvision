"""Candidate batching of already-arrived row contexts; not enabled by default.

The numerical primitive is the existing, independently checked
``batched_row_logits``. A batch never spans a sequence or a decision time.
This adapter changes execution grouping, not candidates, model parameters,
forest limits, historical factors or the order of commits.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .forest_potentials import LearnedForestScorer
from .forest_row_context import ForestRowContext
from .forest_training import batched_row_logits
from .identity_forest import ForestFactors
from .recoverable_identity import model_digest


class BatchedLearnedRowScorer:
    execution_recipe = 'rbf-same-arrival-batched-row-context-candidate-v1'

    def __init__(self, original, *, max_batch=64):
        if type(original) is not LearnedForestScorer:
            raise TypeError('the original frozen learned scorer is required')
        if type(max_batch) is not int or not 1 <= max_batch <= 64:
            raise ValueError('batch size must stay within the admitted 1..64 range')
        if original.options.get('history_enabled', True) is not True:
            raise ValueError('the admitted batch primitive requires actual temporal history')
        self.original = original
        self.max_batch = max_batch
        self._signature = original.signature
        self._options = dict(original.options)
        self._model_sha256 = original.model_sha256

    @property
    def signature(self):
        # This is the original semantic model/options identity. The distinct
        # execution recipe and code hash must also be recorded by a runner.
        if self.original.signature != self._signature or self.original.options != self._options:
            raise ValueError('frozen scorer options changed')
        return self._signature

    def _check_model(self):
        model = self.original.model
        if (any(module.training for module in model.modules())
                or any(parameter.requires_grad for parameter in model.parameters())
                or model_digest(model) != self._model_sha256):
            raise ValueError('frozen model changed')
        return model

    def __call__(self, observations, support, decision_us):
        # Ordinary calls keep the unchanged deployment path.
        return self.original(observations, support, decision_us)

    def score_contexts(self, contexts):
        contexts = tuple(contexts)
        self.signature
        if not contexts:
            return ()
        if (any(type(context) is not ForestRowContext for context in contexts)
                or len({context.decision_us for context in contexts}) != 1
                or len({context.observations[-1].sequence_id for context in contexts}) != 1):
            raise ValueError('only contexts from one actual sequence and decision may be batched')
        for context in contexts:
            n = len(context.indices)
            if n > min(9, self._options['max_nodes']) or n*n > self._options['max_pairs']:
                raise ValueError('context exceeds the unchanged candidate/token/pair limits')
        results = []
        with torch.inference_mode():
            for start in range(0, len(contexts), self.max_batch):
                chunk = contexts[start:start+self.max_batch]
                model = self._check_model()
                logits = batched_row_logits(model, chunk, max_nodes=9, max_batch=self.max_batch,
                    geometry_weight=self._options['geometry_weight'], process_noise=self._options['process_noise'])
                if len(logits) != len(chunk):
                    raise ValueError('batch primitive changed query coverage')
                self._check_model()
                for context, row in zip(chunk, logits, strict=True):
                    if row.ndim != 1 or len(row) != len(context.indices) or not torch.isfinite(row).all():
                        raise ValueError('batch primitive changed finite parent support')
                    weights = F.log_softmax(row, dim=0).to(device='cpu', dtype=torch.float64).tolist()
                    # Every preceding context row has the singleton birth
                    # support; its log-softmax is exactly zero. Only the last
                    # row enters the persistent adapter, just as before.
                    rows = tuple(((-1, 0.),) for _ in context.indices[:-1])
                    rows += (tuple(zip(context.support[-1], weights, strict=True)),)
                    results.append(ForestFactors(tuple(obs.node for obs in context.observations), rows))
        return tuple(results)
