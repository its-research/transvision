"""Single-transaction, shared-raw persistent component identity tracking.

This coordinator uses exact candidate-support components, not a confidence cut.
It shares one expansion budget and one cache budget across live components.
Graph bridges retain predecessor prefixes and restart from complete raw support.
The default is deterministic; a separate learned scheduler reuses the same
search operations and certified-in-exact-arithmetic bounds. Real-data resource
and tracking validation remain incomplete for both schedulers.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import sqlite3

import numpy as np

from .detection_cache_v2 import canonical
from .forest_tracking import ForestTrackingConfig, RawIdentityDetection
from .identity_forest import digest
from .persistent_component_store import PersistentComponentStore
from .persistent_forest import (
    PersistentForestConfig, PersistentForestTracker, PersistentForestCommit, _path, _file_hash,
)


@dataclass(frozen=True)
class PersistentComponentConfig(PersistentForestConfig):
    max_components: int = 4096
    max_member_rows: int = 1_000_000
    max_total_prefix_nodes: int = 1_000_000
    max_total_frontier: int = 65536
    max_total_cache_entries: int = 8192

    def __post_init__(self):
        super().__post_init__()
        if any(type(v) is not int or v < 1 for k, v in asdict(self).items() if k != 'state'):
            raise ValueError('positive shared component limits required')


class _GlobalLRU(OrderedDict):
    """One capped value store plus bounded per-namespace recency indexes.

Indexes contain references to existing keys, not a second copy of values.
Namespace eviction must not scan unrelated components' cached prefixes.
"""
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.namespace_order = {}

    def __setitem__(self, key, value):
        if key not in self:
            self.counts[key[0]] = self.counts.get(key[0], 0)+1
            self.namespace_order.setdefault(key[0], OrderedDict())[key] = None
        super().__setitem__(key, value)

    def _removed(self, key):
        del self.namespace_order[key[0]][key]
        self.counts[key[0]] -= 1
        if not self.counts[key[0]]:
            del self.counts[key[0]]
            del self.namespace_order[key[0]]

    def __delitem__(self, key):
        super().__delitem__(key)
        self._removed(key)

    def popitem(self, last=True):
        key, value = super().popitem(last=last)
        self._removed(key)
        return key, value

    def move_to_end(self, key, last=True):
        super().move_to_end(key, last=last)
        self.namespace_order[key[0]].move_to_end(key, last=last)

    def pop_namespace(self, namespace, last=True):
        order = self.namespace_order.get(namespace)
        if not order:
            raise KeyError('empty component cache')
        key = next(reversed(order)) if last else next(iter(order))
        value = self[key]
        del self[key]
        return key, value

    def clear(self):
        super().clear()
        self.counts.clear()
        self.namespace_order.clear()


class _CacheView:
    """OrderedDict subset backed by a single hard-capped cross-kernel LRU."""
    def __init__(self, shared, namespace, cap):
        self.shared, self.namespace, self.cap = shared, namespace, cap

    def __contains__(self, key):
        return (self.namespace, key) in self.shared

    def __getitem__(self, key):
        return self.shared[(self.namespace, key)]

    def __setitem__(self, key, value):
        pair = self.namespace, key
        self.shared[pair] = value
        self.shared.move_to_end(pair)
        while len(self.shared) > self.cap:
            self.shared.popitem(last=False)

    def __len__(self):
        return self.shared.counts.get(self.namespace, 0)

    def move_to_end(self, key):
        self.shared.move_to_end((self.namespace, key))

    def popitem(self, last=True):
        pair, value = self.shared.pop_namespace(self.namespace, last=last)
        return pair[1], value

    def clear(self):
        while self.namespace in self.shared.counts:
            self.shared.pop_namespace(self.namespace)


class PersistentComponentTracker(PersistentForestTracker):
    CONFIG_TYPE = PersistentComponentConfig
    SCHEMA = 'persistent_component_identity_tracker_v1'

    def __init__(self, path, *, sequence_id, config=None):
        configuration = config or self.CONFIG_TYPE()
        if type(configuration) is not self.CONFIG_TYPE:
            raise TypeError('persistent component configuration required')
        super().__init__(path, sequence_id=sequence_id, config=configuration)
        self._setup_store()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.store.initialize()
            self.db.execute('CREATE TABLE component_summaries(component INTEGER PRIMARY KEY,payload BLOB NOT NULL)')
            self._set('schema', self.SCHEMA)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            self.close()
            raise

    def _bounds(self):
        # The owner stores global factors, but all partition bounds are local.
        self.suffix, self.suffix_abs = np.zeros(1), np.zeros(1)

    def _setup_store(self):
        self.store = PersistentComponentStore(self, max_components=self.config.max_components,
                                               max_member_rows=self.config.max_member_rows)
        self.kernels, self.shared_cache = {}, _GlobalLRU()
        self.cache = _CacheView(self.shared_cache, ('owner', 'prefix'), self.config.max_total_cache_entries)
        self.row_cache = _CacheView(self.shared_cache, ('owner', 'row'), self.config.max_total_cache_entries)
        self.kernel_config = PersistentForestConfig(**{k: v for k, v in vars(self.config).items()
                                                     if k in PersistentForestConfig.__dataclass_fields__})

    def _kernel(self, component, *, create=False):
        if component not in self.kernels:
            caches = (_CacheView(self.shared_cache, ('prefix', component), self.config.max_total_cache_entries),
                      _CacheView(self.shared_cache, ('row', component), self.config.max_total_cache_entries))
            kernel = (self.store.create_kernel(component, self.kernel_config, caches=caches) if create
                      else self.store.open_kernel(component, self.kernel_config, caches=caches))
            self.kernels[component] = kernel
        return self.kernels[component]

    def close(self):
        if hasattr(self, 'kernels'):
            self.kernels.clear()
            self.shared_cache.clear()
        return super().close()

    @classmethod
    def open(cls, path, *, expected_prediction_sha256, expected_database_sha256):
        path = _path(path)
        if (not path.is_file() or _file_hash(path) != expected_database_sha256
                or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                       for s in (expected_prediction_sha256, expected_database_sha256))):
            raise ValueError('sealed component database hash required')
        connection = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
        try:
            values = {k: json.loads(v) for k, v in connection.execute('SELECT k,v FROM meta')}
        finally:
            connection.close()
        if (values.get('schema') != cls.SCHEMA
                or values['state']['prediction_sha256'] != expected_prediction_sha256):
            raise ValueError('component schema or expected output head differs')
        instance = cls.__new__(cls)
        instance.path, instance.sequence_id = path, values['sequence_id']
        instance.config = cls.CONFIG_TYPE(**dict(values['config'], state=ForestTrackingConfig(**values['config']['state'])))
        instance._connect()
        instance._load()
        instance._setup_store()
        try:
            if instance.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('component database integrity check failed')
            instance.store._check_limits()
            instance.store.validate()
            if instance.db.execute('SELECT count(*) FROM observations').fetchone()[0] != instance.n:
                raise ValueError('component database raw count differs')
            if instance.meta['events']:
                raw = instance.db.execute('SELECT prediction,audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()
                prediction, audit = (json.loads(v) for v in raw)
                if (prediction['commit_sha256'] != expected_prediction_sha256
                        or digest({k: v for k, v in prediction.items() if k != 'commit_sha256'}) != expected_prediction_sha256
                        or digest(audit) != instance.meta['audit_sha256'] or audit['factor_rows_sha256'] != instance.factor_digest()):
                    raise ValueError('component output or factor receipt differs')
        except BaseException:
            instance.close()
            raise
        return instance

    def _restore_runtime(self):
        self.kernels.clear()
        self.shared_cache.clear()
        self._load()

    def _storage_counts(self):
        live = self.store.live()
        prefix_count = sum(self.db.execute(f'SELECT count(*) FROM pc{c}_prefixes').fetchone()[0]
                           for c, in self.db.execute('SELECT component FROM component_catalog'))
        frontier_count = sum(len(self.kernels[c].frontier) for c in live)
        if prefix_count > self.config.max_total_prefix_nodes or frontier_count > self.config.max_total_frontier:
            raise ValueError('shared component prefix/frontier capacity exhausted; transaction not committed')
        return prefix_count, frontier_count

    def _select_allocation(self, candidates, **state):
        component = min(candidates, key=lambda c: (-state['weights'][c]*state['masses'][c][-1], c))
        return component, {}

    def _allocation_summary(self):
        return dict(allocation_policy='weighted_omitted_mass_deterministic_not_learned')

    def _prepare_allocation_proposals(self, proposals, *, weights, prefix_count, frontier_count):
        """Variant hook after all kernels exist; the original policy is unchanged."""
        return None

    def _next_allocation_work(self, kernel, proposals, remaining, prefix_count, frontier_count):
        base = next((h for h in proposals if kernel.n-kernel._prefix(h).depth <= remaining
            and kernel.prefix_count+kernel.n-kernel._prefix(h).depth <= self.kernel_config.max_prefix_nodes
            and prefix_count+kernel.n-kernel._prefix(h).depth <= self.config.max_total_prefix_nodes
            and len(kernel.frontier)+1 <= self.kernel_config.state.max_frontier
            and frontier_count+1 <= self.config.max_total_frontier), None)
        return dict(base=base, kind='greedy_suffix_proposal' if base is not None else 'recoverable_prefix_refinement',
                    requested_steps=kernel.n-kernel._prefix(base).depth if base is not None else 1)

    def _execute_allocation_work(self, kernel, operation, decision_us, prefix_count, frontier_count):
        # Used unchanged by deterministic/learned schedulers and offline probes.
        base = operation['base']
        if base is not None:
            done, handle = 0, base
            while kernel._prefix(handle).depth < kernel.n:
                choice = min(kernel._choices(handle), key=lambda p: (-kernel.transition_log_weight(handle, p), p))
                handle = kernel._child(handle, choice)
                done += 1
            kernel.seed_complete_action(handle, decision_us)
            return done, False
        return kernel._refine(decision_us, budget=1,
            prefix_cap=kernel.prefix_count+self.config.max_total_prefix_nodes-prefix_count,
            frontier_cap=len(kernel.frontier)+self.config.max_total_frontier-frontier_count)

    def _component_inference(self, old_n, rescored_indices, *, reference_us, decision_us, indices):
        changes = self.store.append_partition(old_n, decision_us=decision_us)
        changed = {c.component: c for c in changes}
        merged = {p for c in changes if c.merge_restart for p in c.predecessors}
        # Archived kernels remain on disk; release their in-memory arrays/cache.
        for component in merged:
            old = self.kernels.pop(component, None)
            if old:
                old.cache.clear(); old.row_cache.clear()
            del old
        contexts, scopes, counts, proposals, active_kernels = {}, {}, {}, {}, self.store.live()
        for component in active_kernels:
            change = changed.get(component)
            fresh = change is not None and len(change.predecessors) != 1
            kernel = self._kernel(component, create=fresh)
            local = dict(self.db.execute('SELECT global_i,local_i FROM component_members WHERE component=?', (component,)))
            scopes[component] = tuple(local[i] for i in indices if i in local)
            counts[component] = len(scopes[component])
            fallback = (self.store.merged_fallback(component, change.predecessors)
                        if change is not None and change.merge_restart else None)
            contexts[component] = self.store.activate_kernel(kernel, decision_us=decision_us,
                rescore=any(i in local for i in rescored_indices), fallback_parents=fallback)
            kernel.seed_complete_action(contexts[component]['fallback'], decision_us)
            if fresh:
                old_depth = kernel.n-len(change.added_indices)
                bases = {kernel.ancestor(contexts[component]['fallback'], old_depth)}
            elif contexts[component]['old_n'] < kernel.n:
                bases = contexts[component]['old_active'] | {kernel.meta['output']}
            else:
                bases = set()
            proposals[component] = sorted(bases, key=lambda h: (-kernel._weight(h), kernel._prefix(h).sha256))
        prefix_count, frontier_count = self._storage_counts()
        total = len(indices)
        weights = {c: counts[c]/total if total else 0. for c in active_kernels}
        masses = {c: self.kernels[c]._mass() for c in active_kernels}
        self._prepare_allocation_proposals(proposals,weights=weights,
            prefix_count=prefix_count,frontier_count=frontier_count)
        remaining, blocked, allocations = self.config.state.expansion_budget, set(), []
        spent, proposal_spent, limited = dict.fromkeys(active_kernels, 0), dict.fromkeys(active_kernels, 0), dict.fromkeys(active_kernels, False)
        while remaining:
            candidates = [c for c in active_kernels if c not in blocked and counts[c] and masses[c][-1] > 0
                          and any(self.kernels[c]._prefix(h).depth < self.kernels[c].n for h in self.kernels[c].frontier)]
            if not candidates:
                break
            component, selection = self._select_allocation(candidates, weights=weights, masses=masses,
                scopes=scopes, contexts=contexts, proposals=proposals, remaining=remaining,
                prefix_count=prefix_count, frontier_count=frontier_count, decision_us=decision_us)
            if component not in candidates:
                raise ValueError('allocation selected an ineligible component')
            kernel = self.kernels[component]
            before_prefix, before_frontier = kernel.prefix_count, len(kernel.frontier)
            risk_before = weights[component]*masses[component][-1]
            operation = self._next_allocation_work(kernel, proposals[component], remaining, prefix_count, frontier_count)
            done, stopped = self._execute_allocation_work(kernel, operation, decision_us, prefix_count, frontier_count)
            if operation['base'] is not None:
                proposals[component].remove(operation['base'])
                proposal_spent[component] += done
            else:
                spent[component] += done
            prefix_count += kernel.prefix_count-before_prefix
            frontier_count += len(kernel.frontier)-before_frontier
            if prefix_count > self.config.max_total_prefix_nodes or frontier_count > self.config.max_total_frontier:
                raise ValueError('shared component proposal/refinement storage cap exceeded')
            remaining -= done
            limited[component] |= stopped
            masses[component] = kernel._mass()
            allocations.append(dict(component=component, weighted_eta_before=risk_before,
                work_kind=operation['kind'], charged_search_steps=done, eta_after=masses[component][-1], **selection))
            if not done:
                blocked.add(component)
        choices, decisions = {}, {}
        for component in active_kernels:
            active, absolute, retained, _, eta = masses[component]
            choices[component], decisions[component] = self.kernels[component]._decode(
                active, absolute, retained, eta, scopes[component], contexts[component]['fallback'], fallback_on_risk=False)
        proposed_bound = math.fsum(weights[c]*decisions[c]['risk_bound'] for c in active_kernels)
        # One is the explicit conditional-Bayes comparison mode. Do not enable
        # fallback there due to floating summation above the unit loss range.
        fallback_used = self.config.state.max_model_regret < 1. and proposed_bound > self.config.state.max_model_regret
        if fallback_used:
            for component in active_kernels:
                active, absolute, retained, _, eta = masses[component]
                choices[component], decisions[component] = self.kernels[component]._decode(
                    active, absolute, retained, eta, scopes[component], contexts[component]['fallback'], force_fallback=True)
        predictions, summaries, state_work = [], [], 0
        for component in active_kernels:
            kernel, chosen = self.kernels[component], choices[component]
            active, absolute, retained, upper, eta = masses[component]
            branch_states = []
            last_time = kernel.db.execute('SELECT max(state_us) FROM observations').fetchone()[0]
            expired = last_time is not None and reference_us-last_time > self.config.state.max_age_us
            for handle in sorted(set(active) | {chosen}):
                state = [] if expired else kernel._predict(handle, reference_us)
                branch_states.append(dict(handle=handle, sha256=kernel._prefix(handle).sha256, state_sha256=digest(state)))
                if handle == chosen:
                    predictions.extend(state)
            state_work += kernel.state_updates
            if state_work > self.config.state.max_replay_operations:
                raise ValueError('shared component state-replay work cap exceeded')
            change = changed.get(component)
            recoveries = self.store.output_recoveries(component, kernel, chosen, contexts[component], change)
            summary = dict(component=component, nodes=kernel.n, indices_count=counts[component], weight=weights[component],
                representation='exact_root_partition_classes_with_seeded_lower_mass_v1',
                active=[dict(handle=h, sha256=kernel._prefix(h).sha256, log_weight=w) for h, w in zip(active, absolute)],
                frontier=[dict(handle=h, sha256=kernel._prefix(h).sha256, depth=kernel._prefix(h).depth,
                               log_upper=kernel._upper(h)) for h in sorted(kernel.frontier)],
                log_retained=retained if active else None, log_partition_upper=upper, eta_upper=eta,
                decision=decisions[component], decision_indices=scopes[component], output_handle=chosen,
                output_sha256=kernel._prefix(chosen).sha256, branches=branch_states,
                prefix_nodes=kernel.prefix_count, state_updates=kernel.state_updates, expansions=spent[component],
                proposal_steps=proposal_spent[component],
                resource_limited=limited[component], expired_output_only=expired,
                merge_restart=bool(change and change.merge_restart),
                predecessors=change.predecessors if change else (component,),
                recovery_events=recoveries,
                complete_raw_support_retained=True, fallback_handle=contexts[component]['fallback'])
            self.store.save_kernel(kernel, chosen=chosen, reference_us=reference_us, decision_us=decision_us)
            self.db.execute('INSERT OR REPLACE INTO component_summaries VALUES(?,?)', (component, canonical(summary)))
            summaries.append(summary)
        if len({p['track_id'] for p in predictions}) != len(predictions):
            raise ValueError('independent component outputs produced duplicate track IDs')
        log_ratio = math.fsum(masses[c][2]-masses[c][3] for c in active_kernels) if all(masses[c][0] for c in active_kernels) else None
        return sorted(predictions, key=lambda p: p['track_id']), dict(components=summaries,
            **self._allocation_summary(), allocation_trace=allocations,
            expansion_budget=self.config.state.expansion_budget, expansions=sum(spent.values()),
            proposal_steps=sum(proposal_spent.values()), search_steps=sum(spent.values())+sum(proposal_spent.values()),
            search_budget=self.config.state.expansion_budget,
            log_partition_upper=math.fsum(masses[c][3] for c in active_kernels),
            product_omitted_mass_upper=1. if log_ratio is None else max(0., min(1., -math.expm1(min(0., log_ratio)))),
            weighted_truncation_risk_upper=math.fsum(weights[c]*masses[c][-1] for c in active_kernels),
            proposed_model_regret_upper=proposed_bound, global_fallback_used=fallback_used,
            model_regret_upper=math.fsum(weights[c]*decisions[c]['risk_bound'] for c in active_kernels),
            decision_indices=indices, total_prefix_nodes=prefix_count, total_frontier=frontier_count,
            state_updates=state_work, shared_cache_entries=len(self.shared_cache),
            live_row_bound_metadata_bytes=sum(k.suffix.nbytes+k.suffix_abs.nbytes for k in self.kernels.values()),
            member_rows=self.db.execute('SELECT count(*) FROM component_members').fetchone()[0])

    def _audit_properties(self):
        return dict(kind='persistent_component_recoverable_forest_v1',
            raw_history_retained_on_disk=True, historical_identity_map_compressed=False,
            exact_equivalent_parent_paths_summed=True, component_allocation_integrated=True,
            learned_allocation_policy=False, formal_numeric_certificate=False,
            true_posterior_or_metric_bound=False)

    def step(self, observations, rows, *, frame_id, reference_us, decision_us, event_id,
             rescored_rows=(), decision_indices=None, cache_ingestion=None, scorer_binding=None):
        new, rows, rescored = tuple(observations), tuple(rows), tuple(rescored_rows)
        if (len(new) != len(rows) or len(new) > self.config.max_new_observations
                or any(type(o) is not RawIdentityDetection for o in new)
                or any(not isinstance(v, str) or not v for v in (frame_id, event_id))
                or type(reference_us) is not int or type(decision_us) is not int or not 0 <= reference_us <= decision_us):
            raise ValueError('invalid persistent component event or raw input')
        if digest(asdict(self.config)) != self.configuration_sha256:
            raise ValueError('persistent component configuration changed')
        request = digest([frame_id, reference_us, decision_us, [asdict(o) for o in new], rows, rescored,
                          decision_indices, cache_ingestion, scorer_binding])
        previous = self.db.execute('SELECT request,prediction,audit FROM events WHERE event_id=?', (event_id,)).fetchone()
        if previous:
            if previous[0] != request:
                raise ValueError('conflicting duplicate persistent component event')
            return PersistentForestCommit(previous[1], previous[2])
        if (reference_us <= self.meta['reference_us'] or decision_us < self.meta['decision_us']
                or self.n+len(new) > self.config.max_observations or self.meta['events'] >= self.config.max_events):
            raise ValueError('nonmonotonic component output or raw/event capacity exceeded')
        previous_arrival = self.db.execute('SELECT max(arrival_us) FROM observations').fetchone()[0] or -1
        for raw in new:
            if (raw.sequence_id != self.sequence_id or raw.node.arrival_us > decision_us
                    or raw.node.arrival_us < max(previous_arrival, self.meta['decision_us'])):
                raise ValueError('future, withheld, reordered or cross-sequence component observation')
            previous_arrival = raw.node.arrival_us
        old_n = self.n
        validated = tuple(self._validate_row(old_n+i, row) for i, row in enumerate(rows))
        if len({i for i, _ in rescored}) != len(rescored):
            raise ValueError('duplicate rescored component row')
        updates = []
        for i, row in rescored:
            if type(i) is not int or not 0 <= i < old_n:
                raise ValueError('rescore outside old component raw history')
            updates.append((i, self._validate_row(i, row, rescore=True)))
        deliveries = ()
        if cache_ingestion is not None:
            from .forest_cache_stream import CacheDelivery
            if (type(cache_ingestion) is not dict or cache_ingestion.get('kind') != 'persistent_cache_ingestion_v1'
                    or not isinstance(cache_ingestion.get('configuration_sha256'), str)):
                raise ValueError('invalid component cache ingestion audit')
            deliveries = tuple(CacheDelivery(**d) for d in cache_ingestion['new_deliveries'])
            receipt_map = {(d.side, d.frame_id): d for d in deliveries}
            if len(receipt_map) != len(deliveries) or any(d.sequence_id != self.sequence_id
                    or d.arrival_us > decision_us or d.arrival_us < self.meta['decision_us'] for d in deliveries):
                raise ValueError('invalid component cache receipt')
            for raw in new:
                key = ('vehicle-side' if raw.node.source_id == 0 else 'infrastructure-side', raw.node.frame_id)
                delivery = receipt_map.get(key)
                if delivery is None or (delivery.arrival_us, delivery.frame_sha256) != (raw.node.arrival_us, raw.source_cache_sha256):
                    raise ValueError('new raw component observation not bound to receipt')
        bindings = dict(cache_binding=cache_ingestion['configuration_sha256'] if cache_ingestion is not None else None,
                        scorer_binding=scorer_binding)
        for key, value in bindings.items():
            previous = self.db.execute('SELECT v FROM meta WHERE k=?', (key,)).fetchone()
            if previous is not None and json.loads(previous[0]) != value:
                raise ValueError('persistent component cache/scoring protocol changed')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for key, value in bindings.items():
                if value is not None:
                    self._set(key, value)
            for delivery in deliveries:
                self.db.execute('INSERT INTO cache_receipts VALUES(?,?,?,?)',
                    (delivery.side, delivery.frame_id, delivery.arrival_us, delivery.frame_sha256))
            for i, (raw, row) in enumerate(zip(new, validated), old_n):
                payload = canonical(asdict(raw))
                self.db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (i, raw.node.node_id, raw.node.source_id, raw.node.frame_id, raw.detection_index,
                     raw.state_us, raw.node.arrival_us, raw.score, payload, hashlib.sha256(payload).hexdigest()))
                self.db.executemany('INSERT INTO potentials VALUES(?,?,?)', ((i, p, w) for p, w in row))
            for i, row in updates:
                self.db.executemany('UPDATE potentials SET w=? WHERE i=? AND p=?', ((w, i, p) for p, w in row))
            self.n += len(new)
            self.revision += bool(updates)
            self.row_cache.clear()
            indices = (tuple(i for i, in self.db.execute('SELECT i FROM observations WHERE state_us>=? ORDER BY i',
                (max(0, reference_us-self.config.state.window_us),))) if decision_indices is None else tuple(decision_indices))
            if (len(indices) > self.config.max_decision_nodes or len(indices) != len(set(indices))
                    or any(type(i) is not int or not 0 <= i < self.n for i in indices)):
                raise ValueError('invalid component decision scope or decision-node capacity exceeded')
            predictions, inference = self._component_inference(old_n, {i for i, _ in updates},
                reference_us=reference_us, decision_us=decision_us, indices=indices)
            prediction = dict(sequence_id=self.sequence_id, frame_id=frame_id,
                box_reference_timestamp_us=reference_us, decision_timestamp_us=decision_us,
                coordinate_frame='world', state_layout='gravity_xyz_length_width_height_yaw_vxy',
                predictions=predictions, previous_commit_sha256=self.meta['prediction_sha256'])
            prediction['commit_sha256'] = digest(prediction)
            audit = dict(sequence_id=self.sequence_id,
                event_id=event_id, configuration_sha256=self.configuration_sha256,
                factor_rows_sha256=self.factor_digest(), observation_count=self.n, new_observations=len(new),
                appended_rows=validated, rescored_rows=updates, **inference,
                **self._audit_properties(), scorer_binding=scorer_binding, cache_ingestion=cache_ingestion,
                previous_audit_sha256=self.meta['audit_sha256'], prediction_sha256=prediction['commit_sha256'])
            result = PersistentForestCommit(canonical(prediction), canonical(audit))
            self.db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                (self.meta['events'], event_id, request, result.prediction_json, result.audit_json))
            self.meta = dict(self.meta, n=self.n, revision=self.revision, decision_us=decision_us, reference_us=reference_us,
                             events=self.meta['events']+1, prediction_sha256=prediction['commit_sha256'], audit_sha256=digest(audit))
            self._set('state', self.meta)
            self.db.execute('COMMIT')
            return result
        except BaseException:
            self.db.execute('ROLLBACK')
            self._restore_runtime()
            raise
