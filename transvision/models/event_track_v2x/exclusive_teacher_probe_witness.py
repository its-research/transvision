"""Read-only raw witnesses for a separately named exclusive offline teacher.

Witnesses survive each probe rollback, inside the outer event transaction.
They do not enter the deployment bank, its inputs, priority order, or budget.
Their database bytes and encoding cost are reported as offline teacher costs.
No existing teacher task or source package is changed by this new class.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib

from .allocation_policy import decision_bound
from .detection_cache_v2 import canonical
from .exclusive_completion_tracking import ExclusiveCompletionTeacher

RECIPE = 'exclusive_raw_search_before_after_probe_witness_v1'
INITIAL_RECIPE = 'exclusive_all_live_initial_search_and_catalog_v1'


class ExclusiveWitnessTeacher(ExclusiveCompletionTeacher):
    SCHEMA = 'persistent_exclusive_completion_teacher_raw_probe_witness_v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db.execute('CREATE TABLE IF NOT EXISTS priority_probe_witnesses('
                        'sha TEXT PRIMARY KEY, raw_bytes INTEGER NOT NULL, payload BLOB NOT NULL)')
        self._witness_encoding_seconds = 0.
        self._witness_peak_raw_encoding_bytes = 0

    @classmethod
    def open(cls, *args, **kwargs):
        instance = super().open(*args, **kwargs)
        # The inherited immutable reopen bypasses __init__. Historical witness
        # counts stay in SQL; timing and transient encoding peaks describe this
        # process instance, rather than inventing cumulative historical costs.
        instance._witness_encoding_seconds = 0.
        instance._witness_peak_raw_encoding_bytes = 0
        return instance

    def _snapshot(self, kernel):
        # SQL reads only: do not populate solver caches, compute new branches,
        # promote leaves or alter weight/state bookkeeping to collect evidence.
        prefixes = []
        for row in kernel.db.execute('SELECT h,parent,depth,choice,root,previous_root,jumps,sha '
                                     'FROM prefixes ORDER BY h'):
            prefixes.append([*row[:6], bytes(row[6]).decode(), row[7]])
        slots = list(kernel.db.execute('SELECT source,frame FROM observations ORDER BY i'))
        factors = [[] for _ in slots]
        for i, parent, weight in kernel.db.execute('SELECT i,p,w FROM potentials ORDER BY i,p'):
            factors[i].append([parent, weight])
        if len(prefixes) != kernel.prefix_count or len(slots) != kernel.n:
            raise ValueError('raw probe witness differs from actual search storage')
        return dict(kind='exclusive_raw_kernel_search_snapshot_v1',
                    sequence_id=kernel.sequence_id, nodes=kernel.n,
                    revision=kernel.revision, prefix_count=kernel.prefix_count,
                    active=sorted(kernel.active), frontier=sorted(kernel.frontier),
                    prefixes=prefixes, slots=slots, factors=factors)

    def _store_witness(self, value):
        started = time.perf_counter()
        raw = canonical(value)
        digest = hashlib.sha256(raw).hexdigest()
        payload = zlib.compress(raw, level=1)
        prior = self.db.execute('SELECT raw_bytes,payload FROM priority_probe_witnesses '
                                'WHERE sha=?', (digest,)).fetchone()
        if prior is None:
            self.db.execute('INSERT INTO priority_probe_witnesses VALUES(?,?,?)',
                            (digest, len(raw), sqlite3.Binary(payload)))
        elif prior[0] != len(raw) or zlib.decompress(bytes(prior[1])) != raw:
            raise ValueError('immutable probe witness collision')
        self._witness_encoding_seconds += time.perf_counter() - started
        self._witness_peak_raw_encoding_bytes = max(self._witness_peak_raw_encoding_bytes, len(raw))
        return digest

    def _probe_uncached(self, record, state):
        # The original probe executor, bound calculation, budget and rollback
        # remain unchanged. Only raw SQL snapshots bracket the operation.
        component, operation = record['component'], record['operation']
        kernel = self.kernels[component]
        self.probe_executions += 1
        saved = kernel.prefix_count, set(kernel.active), set(kernel.frontier), kernel.state_updates
        cache = tuple(self.shared_cache.items())
        before = self._snapshot(kernel)
        self.db.execute('SAVEPOINT allocation_probe')
        try:
            done, limited = self._execute_allocation_work(kernel, operation, state['decision_us'],
                                                         state['prefix_count'], state['frontier_count'])
            after_bound = decision_bound(kernel, kernel._mass(), state['scopes'][component],
                                         state['contexts'][component]['fallback'])['risk_bound']
            target = state['weights'][component] * (record['model_bound_before'] - after_bound) / max(1, done)
            after = self._snapshot(kernel)
            result = dict(target=target, model_bound_after=after_bound,
                          charged_steps=done, resource_limited=limited)
        finally:
            self.db.execute('ROLLBACK TO allocation_probe')
            self.db.execute('RELEASE allocation_probe')
            kernel.prefix_count, kernel.active, kernel.frontier, kernel.state_updates = saved
            self.shared_cache.clear()
            for key, value in cache:
                self.shared_cache[key] = value
        # These rows join the current event transaction, rather than the
        # rolled-back probe. A failed event therefore cannot leave accepted
        # orphan labels. Cached probes keep the same immutable witness refs.
        return dict(result, probe_witness=dict(
            recipe=RECIPE, component=component,
            before_sha256=self._store_witness(before),
            after_sha256=self._store_witness(after),
            operation=dict(operation), scope=list(state['scopes'][component]),
            loss_weight=state['weights'][component],
            effective_prefix_cap=min(kernel.config.max_prefix_nodes,
                kernel.prefix_count + self.config.max_total_prefix_nodes - state['prefix_count']),
            effective_frontier_cap=min(kernel.config.state.max_frontier,
                len(kernel.frontier) + self.config.max_total_frontier - state['frontier_count'])))

    def _select_allocation(self, candidates, **state):
        component, payload = super()._select_allocation(candidates, **state)
        payload['allocation_training']['raw_probe_witness_recipe'] = RECIPE
        payload['allocation_training']['causal_search_context'] = dict(
            decision_us=state['decision_us'], remaining=state['remaining'],
            expansion_budget=self.config.state.expansion_budget,
            global_prefix_count=state['prefix_count'], global_frontier_count=state['frontier_count'],
            scope_by_component={str(key): list(value) for key, value in state['scopes'].items()},
            components={str(key): dict(
                nodes=kernel.n, prefix_count=kernel.prefix_count,
                frontier_count=len(kernel.frontier),
                members=list(self.store.members(key)),
                loss_weight=state['weights'][key],
                proposal_handles=list(state['proposals'][key]),
                completion_bases=sorted(self._completion_bases[key]))
                for key, kernel in sorted(self.kernels.items())})
        return component, payload

    def _prepare_allocation_proposals(self, proposals, **state):
        super()._prepare_allocation_proposals(proposals, **state)
        # Record every live component, including zero-loss/fully enumerated
        # ones that will never produce a label. Archived prefix rows still
        # consume the shared cap. These are current SQL reads, not new search.
        catalog = []
        for component, live, created in self.db.execute(
                'SELECT component,live,created_us FROM component_catalog ORDER BY component'):
            count = self.db.execute(f'SELECT count(*) FROM pc{component}_prefixes').fetchone()[0]
            catalog.append(dict(component=component, live=bool(live),
                                created_us=created, prefix_count=count))
        self._initial_search_witnesses = dict(
            recipe=INITIAL_RECIPE, catalog=catalog,
            components={str(c): self._store_witness(self._snapshot(kernel))
                        for c, kernel in sorted(self.kernels.items())},
            proposals={str(c): list(values) for c, values in sorted(proposals.items())},
            completion_bases={str(c): sorted(values) for c, values in sorted(self._completion_bases.items())})

    def _allocation_summary(self):
        count, raw, compressed = self.db.execute(
            'SELECT count(*),coalesce(sum(raw_bytes),0),coalesce(sum(length(payload)),0) '
            'FROM priority_probe_witnesses').fetchone()
        return dict(super()._allocation_summary(), raw_probe_witness_recipe=RECIPE,
                    raw_probe_witnesses=count, raw_probe_witness_uncompressed_bytes=raw,
                    raw_probe_witness_compressed_bytes=compressed,
                    raw_probe_witness_encoding_seconds=self._witness_encoding_seconds,
                    raw_probe_witness_peak_single_encoding_bytes=self._witness_peak_raw_encoding_bytes,
                    raw_probe_witness_timing_scope='current_teacher_process_instance',
                    initial_search_witnesses=self._initial_search_witnesses,
                    raw_probe_witness_cost_is_offline_teacher_cost=True,
                    teacher_peak_process_memory_must_include_witness_encoding=True)

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA,
                    independent_probe_target_verification_pending=True)
