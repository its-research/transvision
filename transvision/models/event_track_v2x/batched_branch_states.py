"""Opt-in float64 branch-state execution; identity search remains unchanged.

Batch members are independent root histories. Each history keeps its original
update order and late arrivals use the original chronological replay. No state
is shared between roots or averaged across identity hypotheses.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict

import numpy as np

from .detection_cache_v2 import canonical
from .exclusive_completion_tracking import (
    ExclusiveCompletionTracker, _ExclusiveStore, _ExclusivePartitionKernel,
)
from .recoverable_states import _state
from .persistent_component_store import _Namespace
from .residual_frontier import RECIPE as REGION_RECIPE, residual_log_upper
from .identity_forest import digest

RECIPE = 'independent-root-wave-float64-state-candidate-v1'


class _CachedNamespace(_Namespace):
    """Bounded SQL-template translation cache; values stay SQL parameters."""

    def __init__(self, namespace, shared):
        self.connection, self.prefix = namespace.connection, namespace.prefix
        self.shared = shared

    def _sql(self, sql):
        key = (self.prefix, sql)
        if key not in self.shared:
            self.shared[key] = super()._sql(sql)
            while len(self.shared) > 256:
                self.shared.popitem(last=False)
        self.shared.move_to_end(key)
        return self.shared[key]


class StateBatch:
    """Bounded numerical batches. CUDA is explicit, never silently emulated."""

    def __init__(self, device='cpu', max_batch=64):
        if type(max_batch) is not int or not 1 <= max_batch <= 4096:
            raise ValueError('state batch must be in 1..4096')
        self.device, self.max_batch = device, max_batch
        self.calls = self.rows = self.maximum_batch = 0
        self.torch = None
        if device != 'cpu':
            import torch
            d = torch.device(device)
            if d.type != 'cuda' or d.index is None or not torch.cuda.is_available():
                raise ValueError('explicit available cuda:N device required')
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise ValueError('TF32 must be disabled for candidate validation')
            self.torch = torch

    def update(self, means, covariances, other, other_cov, deltas, *, density, weight):
        values = [np.asarray(v, dtype=np.float64) for v in
                  (means, covariances, other, other_cov, deltas)]
        n = len(values[0])
        if (not 1 <= n <= self.max_batch or
                [v.shape for v in values] != [(n, 9), (n, 9, 9), (n, 9), (n, 9, 9), (n,)] or
                any(not np.isfinite(v).all() for v in values) or
                not math.isfinite(density) or density < 0 or
                not math.isfinite(weight) or not 0 < weight < 1):
            raise ValueError('finite bounded float64 state batch required')
        for cov in (values[1], values[3]):
            if not np.allclose(cov, cov.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
                raise ValueError('symmetric covariance required')
            np.linalg.cholesky(cov)
        if self.torch is None:
            xp = np
            mean, cov, other, other_cov, dt = [v.copy() for v in values]
            eye = np.eye(9)
            f = np.broadcast_to(eye, (n, 9, 9)).copy()
        else:
            xp = self.torch
            mean, cov, other, other_cov, dt = [xp.tensor(v, dtype=xp.float64, device=self.device) for v in values]
            eye = xp.eye(9, dtype=xp.float64, device=self.device)
            f = eye.expand(n, 9, 9).clone()
        f[:, 0, 7], f[:, 1, 8] = dt, dt
        mean = (f @ mean[..., None])[..., 0]
        cov = f @ cov @ f.swapaxes(-1, -2) + eye * (density * abs(dt))[:, None, None]
        other[:, 6] = mean[:, 6] + (other[:, 6] - mean[:, 6] + math.pi) % (2 * math.pi) - math.pi
        a, b = xp.linalg.inv(cov), xp.linalg.inv(other_cov)
        result_cov = xp.linalg.inv(weight*a + (1-weight)*b)
        result_mean = (result_cov @ (weight*a @ mean[..., None] + (1-weight)*b @ other[..., None]))[..., 0]
        result_mean[:, 6] = (result_mean[:, 6] + math.pi) % (2 * math.pi) - math.pi
        result_cov = .5 * (result_cov + result_cov.swapaxes(-1, -2))
        if self.torch is not None:
            result_mean, result_cov = result_mean.cpu().numpy(), result_cov.cpu().numpy()
        if not np.isfinite(result_mean).all() or not np.isfinite(result_cov).all():
            raise ValueError('nonfinite batched state output')
        self.calls += 1
        self.rows += n
        self.maximum_batch = max(self.maximum_batch, n)
        return result_mean, result_cov


class _BatchedStateKernel(_ExclusivePartitionKernel):
    def residual_regions(self, active, weights):
        # Many frontier handles have the same depth. A retained leaf has only
        # one ancestor at that depth; never repeat the binary-lifting traversal
        # for every frontier handle. Cache is call-local and hard-capped.
        ancestors = OrderedDict()
        excluded_count = {h: 0 for h in active}
        regions = []
        for h in sorted(self.frontier):
            prefix = self._prefix(h)
            excluded = []
            for leaf, weight in zip(active, weights):
                key = (leaf, prefix.depth)
                if key not in ancestors:
                    ancestors[key] = self.ancestor(leaf, prefix.depth)
                    while len(ancestors) > self.config.prefix_cache_entries:
                        ancestors.popitem(last=False)
                ancestors.move_to_end(key)
                if ancestors[key] == h:
                    if self._prefix(leaf).depth != self.n:
                        raise ValueError('residual exclusion must reference a complete current class')
                    excluded_count[leaf] += 1
                    excluded.append(dict(handle=leaf, sha256=self._prefix(leaf).sha256, log_weight=weight))
            gross = self._upper(h)
            value = residual_log_upper(gross, [r['log_weight'] for r in excluded], nodes=self.n)
            body = dict(handle=h, sha256=prefix.sha256, depth=prefix.depth,
                        nodes=self.n, factor_revision=self.revision, region_recipe=REGION_RECIPE,
                        excluded_leaves=excluded, log_gross_upper=gross, **value)
            body['region_sha256'] = digest(body)
            regions.append(body)
        if any(v > 1 for v in excluded_count.values()):
            raise ValueError('retained class is inside overlapping gross prefix regions')
        if sum(excluded_count.values()) > self.config.state.active_limit:
            raise ValueError('residual exclusion storage exceeds retained-class capacity')
        return regions

    def _observation(self, index):
        # One shared, hard-capped cache per event. Namespace keys isolate
        # component-local indices; rollback never retains entries.
        cache = self.raw_state_cache
        key = (self.db.prefix, index)
        if key not in cache:
            cache[key] = super()._observation(index)
            while len(cache) > self.config.prefix_cache_entries:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def _prefill_states(self, handles):
        chains = []
        count = 0
        for handle in handles:
            pending, cursor = [], handle
            while cursor is not None:
                if self.db.execute('SELECT 1 FROM states WHERE h=?', (cursor,)).fetchone():
                    break
                pending.append(cursor)
                count += 1
                if count > self.config.prefix_cache_entries:
                    # Preserve the admitted serial algorithm for a dependency
                    # graph larger than the candidate's bounded scheduler.
                    return
                cursor = self._prefix(cursor).previous_root
            if pending:
                chains.append(pending)
        while chains:
            updates = []
            for chain in chains:
                h = chain.pop()
                prefix = self._prefix(h)
                obs = self._observation(prefix.depth - 1)
                row = self.db.execute('SELECT payload FROM states WHERE h=?', (prefix.previous_root,)).fetchone()
                state = json.loads(row[0]) if row else None
                order = [obs.state_us, obs.node.source_id, obs.node.node_id]
                if state is None or order <= state['last_order']:
                    # Birth validation and late/out-of-order replay are exactly
                    # the original implementation, including all work charges.
                    super()._ensure_state(h)
                else:
                    updates.append((h, obs, state, order))
            for offset in range(0, len(updates), self.state_batch.max_batch):
                chunk = updates[offset:offset+self.state_batch.max_batch]
                for _ in chunk:
                    self._charge_state()
                means, covs = self.state_batch.update(
                    [s['mean'] for _, _, s, _ in chunk],
                    [s['covariance'] for _, _, s, _ in chunk],
                    [o.mean for _, o, _, _ in chunk],
                    [o.covariance for _, o, _, _ in chunk],
                    [(o.state_us-s['last_us'])/1e6 for _, o, s, _ in chunk],
                    density=self.config.state.process_noise, weight=self.config.state.ci_weight)
                for (h, obs, state, order), mean, cov in zip(chunk, means, covs, strict=True):
                    mean, cov = _state(mean, cov, obs.score)
                    value = dict(mean=mean, covariance=cov, first_us=state['first_us'],
                                 last_us=obs.state_us, last_order=order,
                                 max_score=max(state['max_score'], obs.score))
                    self.db.execute('INSERT INTO states VALUES(?,?)', (h, canonical(value)))
            chains = [chain for chain in chains if chain]

    def _predict(self, handle, reference):
        latest, cursor = {}, handle
        while cursor:
            prefix = self._prefix(cursor)
            latest.setdefault(prefix.root, cursor)
            cursor = prefix.parent_handle
        self._prefill_states([h for _, h in sorted(latest.items())])
        return super()._predict(handle, reference)


class _BatchedStateStore(_ExclusiveStore):
    def open_kernel(self, component, config, *, caches=None):
        kernel = super().open_kernel(component, config, caches=caches)
        kernel.__class__ = _BatchedStateKernel
        kernel.db = _CachedNamespace(kernel.db, self.owner.sql_template_cache)
        kernel.state_batch = self.owner.state_batch
        kernel.raw_state_cache = self.owner.raw_state_cache
        return kernel


class BatchedStateExclusiveTracker(ExclusiveCompletionTracker):
    """Distinct candidate store. Existing/frozen trackers are never patched."""
    SCHEMA = 'experimental_exclusive_batched_state_v1'

    def __init__(self, path, *, sequence_id, config=None, device='cpu', max_batch=64):
        self.state_batch = StateBatch(device, max_batch)
        self.raw_state_cache = OrderedDict()
        self.sql_template_cache = OrderedDict()
        self.execution = dict(recipe=RECIPE, device=device, max_batch=max_batch)
        super().__init__(path, sequence_id=sequence_id, config=config)
        self._set('state_execution', self.execution)

    def _setup_store(self):
        super()._setup_store()
        self.store = _BatchedStateStore(self, max_components=self.config.max_components,
                                      max_member_rows=self.config.max_member_rows)

    @classmethod
    def open(cls, *args, **kwargs):
        raise ValueError('candidate resume requires separate admission; use a new store')

    def step(self, *args, **kwargs):
        if self.execution != dict(recipe=RECIPE, device=self.state_batch.device, max_batch=self.state_batch.max_batch):
            raise ValueError('state execution settings changed')
        self.raw_state_cache.clear()
        try:
            return super().step(*args, **kwargs)
        finally:
            self.raw_state_cache.clear()

    def _audit_properties(self):
        return dict(super()._audit_properties(), state_execution=self.execution,
                    execution_extra_cache_caps=dict(raw_observations=self.config.prefix_cache_entries,
                                                   ancestor_pairs_per_call=self.config.prefix_cache_entries,
                                                   SQL_templates=256),
                    state_execution_independently_accepted=False)
