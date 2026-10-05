"""Event-boundary recovery ablation over the original exclusive forest.

At the next event, only extensions of previous retained/output identity classes
remain admissible. Component merges intersect the predecessor restrictions;
neither a root restart nor all-legal action search can resurrect a discarded
class. Raw evidence and old commits stay on disk for audit. Bounds are explicitly
conditional on this pruned support, never guarantees for the original model.
"""
from dataclasses import dataclass, field
import json

from .detection_cache_v2 import canonical
from .exclusive_completion_tracking import (
    ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig,
    _ExclusivePartitionKernel, _ExclusiveStore,
)
from .identity_forest import ForestFactors, digest
from .forest_tracking import PaperForestTrackingConfig
from .paper_decision import conditional_action
from .persistent_component_store import _Namespace
from .persistent_component_tracking import _CacheView


@dataclass(frozen=True)
class RecoveryOffConfig(PersistentExclusiveCompletionConfig):
    state: PaperForestTrackingConfig = field(default_factory=lambda: PaperForestTrackingConfig(
        decision_mode='all-legal-hamming', max_model_regret=1.))
    recovery_off_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if type(self.recovery_off_version) is not int or self.recovery_off_version != 1:
            raise ValueError('unsupported recovery-off policy')
        if getattr(self.state, 'decision_mode', None) != 'all-legal-hamming':
            raise ValueError('recovery ablation requires the original Hamming decoder')


class HistoryRestriction:
    """Conjunction of disjoint predecessor projections, without Cartesian blowup.

    Each clause contains local observation indices and the retained root vectors
    expressed in the CURRENT local component ordering. Prefix tests retain the
    entire row correlation; independent per-node allowed roots are insufficient.
    """
    def __init__(self, clauses=()):
        self.clauses = tuple((tuple(indices), tuple(tuple(v) for v in variants))
                             for indices, variants in clauses)
        used = set()
        for indices, variants in self.clauses:
            if (not indices or tuple(sorted(set(indices))) != indices
                    or any(type(i) is not int or i < 0 for i in indices)
                    or used.intersection(indices) or not variants
                    or len(set(variants)) != len(variants)):
                raise ValueError('invalid disjoint historical support clause')
            for variant in variants:
                if len(variant) != len(indices):
                    raise ValueError('incomplete retained historical class')
                mapping = dict(zip(indices, variant))
                if any(type(root) is not int or root not in mapping or root > i
                       or mapping[root] != root for i, root in mapping.items()):
                    raise ValueError('historical root is not a canonical predecessor root')
            used.update(indices)

    def allows(self, roots):
        for indices, variants in self.clauses:
            projected = tuple((j, roots[i]) for j, i in enumerate(indices) if i < len(roots))
            if not any(all(v[j] == root for j, root in projected) for v in variants):
                return False
        return True

    def kernel_choice(self, kernel, handle, root):
        depth = kernel._prefix(handle).depth
        for indices, variants in self.clauses:
            if depth not in indices:
                continue
            projected = [(j, kernel._prefix(kernel.ancestor(handle, i + 1)).root)
                         for j, i in enumerate(indices) if i < depth]
            projected.append((indices.index(depth), root))
            return any(all(v[j] == value for j, value in projected) for v in variants)
        return True

    def payload(self):
        return dict(recipe='previous_commit_active_union_output_root_classes_v1',
                    clauses=self.clauses, cartesian_product_materialized=False)


class RestrictedFactors:
    def __init__(self, factors, restriction):
        self.factors, self.restriction = factors, restriction
        self.nodes = factors.nodes

    def roots(self, prefix):
        roots = self.factors.roots(prefix)
        if not self.restriction.allows(roots):
            raise ValueError('action would recover an irreversibly pruned history')
        return roots

    def children(self, prefix):
        self.roots(prefix)
        return tuple(child for child in self.factors.children(prefix)
                     if self.restriction.allows(self.factors.roots(child)))


class _RecoveryOffKernel(_ExclusivePartitionKernel):
    def _admitted_handle(self, handle):
        if handle in self.support_cache:
            value = self.support_cache[handle]
            self.support_cache.move_to_end(handle)
            return value
        depth = self._prefix(handle).depth
        roots = tuple(self._prefix(self.ancestor(handle, i + 1)).root for i in range(depth))
        value = self.restriction.allows(roots)
        self.support_cache[handle] = value
        return value

    def residual_regions(self, active, weights):
        regions = super().residual_regions(active, weights)
        for region in regions:
            region.pop('region_sha256')
            region['base_upper_recipe'] = region['region_recipe']
            region['region_recipe'] = 'history_restriction_intersect_prefix_minus_retained_v1'
            region['historical_support_sha256'] = digest(self.restriction.payload())
            region['gross_upper_may_include_pruned_histories'] = True
            region['region_sha256'] = digest(region)
        return regions

    def _choices(self, handle):
        if not self._admitted_handle(handle):
            raise ValueError('archived prefix lies outside committed historical support')
        choices = super()._choices(handle)
        depth = self._prefix(handle).depth
        return tuple(choice for choice in choices if self.restriction.kernel_choice(
            self, handle, depth if choice < 0 else self._prefix(self.ancestor(handle, choice + 1)).root))

    def _child(self, handle, choice):
        if not self._admitted_handle(handle):
            raise ValueError('archived prefix would recover a pruned historical class')
        root = (self._prefix(handle).depth if choice < 0
                else self._prefix(self.ancestor(handle, choice + 1)).root)
        if not self.restriction.kernel_choice(self, handle, root):
            raise ValueError('materialization would recover a pruned historical class')
        child = super()._child(handle, choice)
        self.support_cache[child] = True
        return child

    def _decode(self, active, weights, retained, eta, indices, fallback,
                *, fallback_on_risk=True, force_fallback=False):
        # Keep the original no-work/capacity fallback. It is an extension of the
        # prior output and has already passed the same _child restriction.
        if not active or force_fallback or self.n > self.config.max_decision_nodes:
            handle, audit = super()._decode(active, weights, retained, eta, indices, fallback,
                                           fallback_on_risk=fallback_on_risk, force_fallback=force_fallback)
        else:
            factors = RestrictedFactors(ForestFactors(
                tuple(self._observation(i).node for i in range(self.n)),
                tuple(self._row(i) for i in range(self.n))), self.restriction)
            paths = tuple(tuple(self._prefix(self.ancestor(h, i + 1)).choice
                                for i in range(self.n)) for h in active)
            action, audit = conditional_action(
                factors, paths, weights, indices,
                budget=getattr(self, '_paper_action_remaining', self.config.state.paper_action_budget),
                frontier_limit=self.config.state.paper_action_frontier)
            before = self.prefix_count
            handle = 0
            for choice in action:
                handle = self._child(handle, choice)
            audit = dict(audit, action_materialization_prefixes=self.prefix_count - before,
                         action_materialization_steps=len(action),
                         risk_bound=min(1., eta + audit['optimization_gap']), empty_loss_scope=not indices)
        roots = tuple(self._prefix(self.ancestor(handle, i + 1)).root for i in range(self.n))
        if not self.restriction.allows(roots):
            raise ValueError('output violated the committed historical restriction')
        return handle, dict(audit, action_space='legal_extensions_of_previous_retained_and_output_classes',
                            support_sha256=digest(self.restriction.payload()),
                            risk_scope='conditional_on_irreversibly_pruned_support',
                            original_model_regret_upper=1., original_model_risk_certified=False)


class _RecoveryOffStore(_ExclusiveStore):
    def open_kernel(self, component, config, *, caches=None):
        kernel = super().open_kernel(component, config, caches=caches)
        kernel.__class__ = _RecoveryOffKernel
        kernel.restriction = HistoryRestriction()
        kernel.support_cache = _CacheView(self.owner.shared_cache,
            ('recovery_off_support', component), self.owner.config.max_total_cache_entries)
        return kernel

    def _previous_clause(self, component, current_members):
        sql = _Namespace(self.db, component)
        previous = json.loads(sql.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
        if not previous['n']:
            return None
        old_members = self.members(component)[:previous['n']]
        inverse = {v: i for i, v in enumerate(current_members)}
        indices = tuple(inverse[i] for i in old_members)
        variants = []
        for handle in sorted(set(previous['active']) | {previous['output']}):
            roots = []
            remaining = len(old_members)
            while handle:
                row = sql.execute('SELECT parent,depth,root FROM prefixes WHERE h=?', (handle,)).fetchone()
                if row is None or row[1] != remaining:
                    raise ValueError('retained predecessor has incomplete immutable history')
                roots.append(inverse[old_members[row[2]]])
                handle, remaining = row[0], remaining - 1
            if remaining:
                raise ValueError('retained predecessor has missing prefix history')
            variants.append(tuple(reversed(roots)))
        return indices, tuple(sorted(set(variants)))

    def activate_kernel(self, kernel, *, decision_us, rescore=False, fallback_parents=None):
        component = int(kernel.db.prefix[2:-1])
        catalog = self.db.execute('SELECT predecessors FROM component_catalog WHERE component=?', (component,)).fetchone()
        predecessors = ((component,) if kernel.meta['events'] else tuple(json.loads(catalog[0])))
        members = self.members(component)
        clauses = [self._previous_clause(c, members) for c in predecessors]
        kernel.restriction = HistoryRestriction(c for c in clauses if c is not None)
        kernel.support_cache.clear()
        kernel._set('recovery_off_restriction', kernel.restriction.payload())
        # Restart enumeration inside the restricted domain. Original prefixes and
        # factors remain archived, but no old frontier grants live admission.
        kernel.frontier = {0}
        return super().activate_kernel(kernel, decision_us=decision_us, rescore=rescore,
                                       fallback_parents=fallback_parents)

    def output_recoveries(self, component, kernel, chosen, context, change):
        events = super().output_recoveries(component, kernel, chosen, context, change)
        if events:
            raise ValueError('recovery-off output escaped previous active/output support')
        return []


class _RecoveryOffRegions:
    CONFIG_TYPE = RecoveryOffConfig

    def _setup_store(self):
        super()._setup_store()
        self.store = _RecoveryOffStore(self, max_components=self.config.max_components,
                                      max_member_rows=self.config.max_member_rows)

    def _component_inference(self, *args, **kwargs):
        predictions, inference = super()._component_inference(*args, **kwargs)
        metadata_bytes = 0
        for summary in inference['components']:
            kernel = self.kernels[summary['component']]
            payload = kernel.restriction.payload()
            metadata_bytes += len(canonical(payload))
            summary.update(historical_support=payload, historical_support_sha256=digest(payload),
                           complete_raw_support_retained=False, raw_factors_preserved=True,
                           representation='exclusive_recovery_off_prefix_regions_v1',
                           residual_partition_recipe='history_restriction_intersect_prefix_minus_retained_v1',
                           mass_scope='conditional_on_irreversibly_pruned_support')
            self.db.execute('INSERT OR REPLACE INTO component_summaries VALUES(?,?)',
                            (summary['component'], canonical(summary)))
        # A restricted posterior may give a tight conditional risk bound while
        # almost all original posterior mass was discarded. Never conflate them.
        for key in ('model_regret_upper', 'proposed_model_regret_upper',
                    'weighted_truncation_risk_upper', 'product_omitted_mass_upper'):
            inference['restricted_support_' + key] = inference[key]
            inference[key] = 1.
        inference.update(restriction_serialized_metadata_bytes=metadata_bytes,
                         restriction_metadata_counted_in_database=True,
                         restriction_serialized_bytes_are_not_process_peak_memory=True,
                         partition_bound_scope='current_irreversibly_pruned_support',
                         original_model_risk_certified=False)
        inference['shared_cache_entries'] = len(self.shared_cache)
        return predictions, inference

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA, recovery_enabled=False,
                    recovery_policy='prune_to_previous_commit_active_union_output',
                    historical_raw_evidence_preserved=True,
                    independent_experiment_accepted=False)


class RecoveryOffTracker(_RecoveryOffRegions, ExclusiveCompletionTracker):
    SCHEMA = 'persistent_exclusive_event_boundary_recovery_off_v1'
