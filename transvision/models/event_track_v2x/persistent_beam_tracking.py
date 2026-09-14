"""Irreversible fixed-width identity-class beam on the SAME raw-node factors.

This is a declared local beam baseline, not a reproduced MHT implementation.
Pruning is per arrival-ordered node, not exact full-history ranked assignment.
Merged components use exact top-K products of their retained old beams, never
restart from raw support. Archived prefixes exist for immutable output audits,
but are not inference candidates. Raw/history disk growth is explicitly retained
for this controlled comparison; beam width does not imply constant total memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
import math

from .detection_cache_v2 import canonical
from .forest_tracking import ForestTrackingConfig
from .hypothesis_bank import logsumexp
from .identity_forest import digest
from .persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from .persistent_forest import PersistentForestTracker


@dataclass(frozen=True)
class PersistentBeamConfig(PersistentComponentConfig):
    max_beam_expansions: int = 100_000
    max_candidate_evaluations: int = 1_000_000
    max_merge_prefix_steps: int = 1_000_000

    def __post_init__(self):
        super().__post_init__()
        defaults = ForestTrackingConfig()
        for key in ('expansion_budget', 'max_frontier', 'max_model_regret'):
            if getattr(self.state, key) != getattr(defaults, key):
                raise ValueError('beam does not use '+key+'; configure explicit beam work limits')
        if self.max_total_frontier != PersistentComponentConfig().max_total_frontier:
            raise ValueError('irreversible beam has no recoverable frontier capacity')


@dataclass(frozen=True)
class PersistentRankedBeamConfig(PersistentBeamConfig):
    max_batch_frontier: int = 65536


class _Work:
    def __init__(self, config):
        self.config = config
        self.beam_expansions = self.candidate_evaluations = self.merge_prefix_steps = 0
        self.merge_score_evaluations = 0
        self.merge_state_cache_queries = self.merge_state_cache_copies = 0

    def charge(self, field):
        value = getattr(self, field)+1
        if value > getattr(self.config, 'max_'+field):
            raise ValueError('beam '+field+' capacity exhausted; no partial event committed')
        setattr(self, field, value)


def _top_products(groups, width, work):
    """Exact top-K products without materializing an exponential Cartesian set.

The scores are additive log weights. At each merge, any prefix combination
below its top K is dominated by K retained prefixes paired with the SAME next
component, so it cannot be necessary for the joint top K. Ties use handles.
"""
    retained = [(0., ())]
    full_support = True
    stages = []
    for component, options in groups:
        if not options:
            raise ValueError('predecessor beam is empty')
        left = sorted(retained, key=lambda item: (-item[0], item[1]))
        right = sorted(options, key=lambda item: (-item[0], item[1]))
        heap, visited = [], set()
        def push(i, j):
            if i >= len(left) or j >= len(right) or (i, j) in visited:
                return
            work.charge('candidate_evaluations')
            work.merge_score_evaluations += 1
            visited.add((i, j))
            score = math.fsum([left[i][0], right[j][0]])
            selection = left[i][1]+((component, right[j][1]),)
            heapq.heappush(heap, (-score, selection, i, j))
        push(0, 0)
        retained = []
        while heap and len(retained) < width:
            negative, selection, i, j = heapq.heappop(heap)
            retained.append((-negative, selection))
            push(i+1, j); push(i, j+1)
        candidates = len(left)*len(right)
        full_support &= candidates <= width
        stages.append(dict(component=component, retained_prefix_product_candidates=candidates,
                           kept=len(retained), dropped_at_stage=candidates-len(retained)))
    return retained, full_support, stages


class PersistentBeamTracker(PersistentComponentTracker):
    CONFIG_TYPE = PersistentBeamConfig
    SCHEMA = 'persistent_irreversible_identity_beam_v1'
    PRUNING_POLICY = 'fixed_top_k_root_classes_after_each_arrival_ordered_node'

    def _audit_properties(self):
        return dict(kind='persistent_irreversible_identity_beam_v1',
            raw_history_retained_on_disk=True, historical_identity_map_compressed=False,
            exact_equivalent_parent_paths_summed=True, component_allocation_integrated=False,
            learned_allocation_policy=False, formal_numeric_certificate=False,
            true_posterior_or_metric_bound=False, recovery_enabled=False,
            archived_prefixes_used_for_recovery=False, reproduced_classical_mht=False,
            selected_predecessor_branch_state_cache_reused=True,
            pruning_policy=self.PRUNING_POLICY,
            output_policy='conditional_bayes_over_retained_classes_no_regret_fallback')

    @staticmethod
    def _complete(kernel):
        row = kernel.db.execute("SELECT v FROM meta WHERE k='beam_support_complete'").fetchone()
        if row is None:
            if kernel.n:
                raise ValueError('nonempty beam is missing its pruning provenance')
            return True
        value = json.loads(row[0])
        if type(value) is not bool:
            raise ValueError('invalid beam support provenance')
        return value

    def _lift(self, component, selections, old_depth, *, include_state_sources=False):
        members = self.store.members(component)[:old_depth]
        inverse = {i: j for j, i in enumerate(members)}
        parents, covered = [-1]*old_depth, set()
        state_sources = [None]*old_depth if include_state_sources else None
        for previous, handle in selections:
            old_members = self.store.members(previous)
            depth = len(old_members)
            while handle:
                row = self.db.execute(f'SELECT parent,depth,choice FROM pc{previous}_prefixes WHERE h=?', (handle,)).fetchone()
                if row is None or row[1] != depth:
                    raise ValueError('invalid retained predecessor beam prefix')
                local = inverse[old_members[depth-1]]
                if local in covered:
                    raise ValueError('overlapping predecessor beam membership')
                covered.add(local)
                parents[local] = -1 if row[2] < 0 else inverse[old_members[row[2]]]
                if state_sources is not None:
                    state_sources[local] = (previous, handle)
                handle, depth = row[0], depth-1
            if depth:
                raise ValueError('retained predecessor prefix lost history')
        if len(covered) != old_depth:
            raise ValueError('merged beam does not cover predecessor observations')
        return (tuple(parents), tuple(state_sources)) if include_state_sources else tuple(parents)

    def _extend(self, kernel, beam, start, child, work):
        pruning, complete = [], True
        for depth in range(start, kernel.n):
            candidate_count, generated_log_mass = 0, -math.inf
            def candidates():
                nonlocal candidate_count, generated_log_mass
                for handle in sorted(beam):
                    work.charge('beam_expansions')
                    prefix, base = kernel._prefix(handle), kernel._weight(handle)
                    if prefix.depth != depth:
                        raise ValueError('beam contains mixed prefix depths')
                    for choice in kernel._choices(handle):
                        work.charge('candidate_evaluations')
                        candidate_count += 1
                        score = math.fsum([base, kernel.transition_log_weight(handle, choice)])
                        generated_log_mass = logsumexp([generated_log_mass, score])
                        root = depth if choice < 0 else kernel._prefix(kernel.ancestor(handle, choice+1)).root
                        sha = digest([prefix.sha256, depth, choice, root])
                        yield -score, sha, handle, choice
            selected = heapq.nsmallest(self.config.state.active_limit, candidates())
            next_beam = set()
            for negative, _, handle, choice in selected:
                new_handle = child(kernel, handle, choice)
                kernel.db.execute('INSERT OR REPLACE INTO weights VALUES(?,?,?)', (new_handle, kernel.revision, -negative))
                next_beam.add(new_handle)
            if not next_beam:
                raise ValueError('beam extension unexpectedly empty')
            kept_log_mass = logsumexp(-v[0] for v in selected)
            pruning.append(dict(depth=depth+1, generated_classes=candidate_count, retained_classes=len(next_beam),
                dropped_classes=candidate_count-len(next_beam),
                generated_prefix_discarded_mass=max(0., min(1., -math.expm1(min(0., kept_log_mass-generated_log_mass)))),
                mass_scope='current_generated_prefixes_not_full_posterior'))
            complete &= candidate_count == len(next_beam)
            beam = next_beam
        return beam, complete, pruning

    def _prepare_merge(self, groups, complete, work):
        combined, unpruned, stages = _top_products(groups, self.config.state.active_limit, work)
        return dict(combined=combined, complete=complete and unpruned, stages=stages)

    def _transfer_state(self, kernel, handle, source, work):
        """Copy only a cache for the SAME root-observation history under a lift.

        `_lift` maps every old observation bijectively and preserves each
        selected root chain. State payloads contain no local identity indices.
        Raw observations and state configuration are unchanged; factor rescoring
        does not alter a fixed branch's CI state. Missing cache remains lazy.
        This does not search deleted classes or average different identities.
        Each query is bounded by an already charged merge-prefix step.
        """
        previous, old_handle = source
        work.merge_state_cache_queries += 1
        row = self.db.execute(f'SELECT payload FROM pc{previous}_states WHERE h=?', (old_handle,)).fetchone()
        if row is None:
            return
        existing = kernel.db.execute('SELECT payload FROM states WHERE h=?', (handle,)).fetchone()
        if existing is not None and existing[0] != row[0]:
            raise ValueError('lifted root history has conflicting conditional state caches')
        if existing is None:
            kernel.db.execute('INSERT INTO states VALUES(?,?)', (handle, row[0]))
            work.merge_state_cache_copies += 1

    def _materialize_combination(self, kernel, component, start, prior_weight, selection, child, work):
        for previous, _ in selection:
            stored = self.db.execute(f"SELECT v FROM pc{previous}_meta WHERE k='config'").fetchone()
            if stored is None or json.loads(stored[0]) != asdict(kernel.config):
                raise ValueError('predecessor state configuration differs before cache transfer')
        parents, state_sources = self._lift(component, selection, start, include_state_sources=True)
        handle = 0
        for choice, source in zip(parents, state_sources):
            work.charge('merge_prefix_steps')
            if choice not in kernel._choices(handle):
                raise ValueError('lifted retained beam is not a legal identity class')
            handle = child(kernel, handle, choice)
            self._transfer_state(kernel, handle, source, work)
        # Recompute the class SUM from current raw factors after index remapping.
        if not math.isclose(kernel._weight(handle), prior_weight, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('lifted beam weight differs from predecessor class product')
        return handle

    def _advance_component(self, kernel, component, context, change, merge, child, work):
        beam, complete, start = context['beam'], context['complete'], context['old_n']
        merge_stages = []
        if merge is not None:
            complete, merge_stages = merge['complete'], merge['stages']
            start = kernel.n-len(change.added_indices)
            beam = {self._materialize_combination(kernel, component, start, weight, selection, child, work)
                    for weight, selection in merge['combined']}
        beam, no_new_pruning, pruning = self._extend(kernel, beam, start, child, work)
        return beam, complete and no_new_pruning, pruning, merge_stages

    def _component_inference(self, old_n, rescored_indices, *, reference_us, decision_us, indices):
        work = _Work(self.config)
        changes = self.store.append_partition(old_n, decision_us=decision_us)
        changed, products = {c.component: c for c in changes}, {}
        # Score only predecessor RETAINED classes, even if the current evidence
        # changes their weights. No archive or raw-support branch resurrection.
        for change in changes:
            if not change.merge_restart:
                continue
            groups, complete = [], True
            for previous in change.predecessors:
                kernel = self._kernel(previous)
                complete &= self._complete(kernel)
                if any(i in rescored_indices for i in self.store.members(previous)):
                    kernel.revision += 1
                    kernel.row_cache.clear()
                groups.append((previous, [(kernel._weight(h), h) for h in kernel.active]))
            products[change.component] = self._prepare_merge(groups, complete, work)
        for previous in {p for c in changes if c.merge_restart for p in c.predecessors}:
            kernel = self.kernels.pop(previous, None)
            if kernel is not None:
                kernel.cache.clear(); kernel.row_cache.clear()
            del kernel
        contexts, scopes = {}, {}
        for component in self.store.live():
            change = changed.get(component)
            fresh = change is not None and len(change.predecessors) != 1
            kernel = self._kernel(component, create=fresh)
            local = {i: j for j, i in enumerate(self.store.members(component))}
            scopes[component] = tuple(local[i] for i in indices if i in local)
            complete, previous_n = self._complete(kernel), kernel.n
            kernel.n = len(local)
            rescore = any(i in local for i in rescored_indices)
            kernel.revision += rescore
            if kernel.n != previous_n or rescore:
                kernel.row_cache.clear()
                kernel._bounds()
            kernel.state_updates = 0
            if not fresh and (not kernel.active or any(kernel._prefix(h).depth != previous_n for h in kernel.active)):
                raise ValueError('old retained beam does not cover its raw history')
            contexts[component] = dict(old_n=previous_n, complete=complete,
                                       beam={0} if fresh else set(kernel.active))
        prefix_count = sum(self.db.execute(f'SELECT count(*) FROM pc{c}_prefixes').fetchone()[0]
                           for c, in self.db.execute('SELECT component FROM component_catalog'))
        if prefix_count > self.config.max_total_prefix_nodes:
            raise ValueError('shared beam prefix capacity exhausted')
        def child(kernel, handle, choice):
            nonlocal prefix_count
            exists = kernel.db.execute('SELECT h FROM prefixes WHERE parent=? AND choice=?', (handle, choice)).fetchone()
            if exists is None and prefix_count >= self.config.max_total_prefix_nodes:
                raise ValueError('shared beam prefix capacity exhausted')
            before = kernel.prefix_count
            result = kernel._child(handle, choice)
            prefix_count += kernel.prefix_count-before
            return result
        summaries, predictions, state_work = [], [], 0
        for component in self.store.live():
            kernel, context = self.kernels[component], contexts[component]
            change = changed.get(component)
            beam, complete, pruning, merge_stages = self._advance_component(
                kernel, component, context, change, products.get(component), child, work)
            kernel.active, kernel.frontier = beam, set()
            kernel._set('beam_support_complete', complete)
            active = sorted(beam, key=lambda h: (-kernel._weight(h), kernel._prefix(h).sha256))
            absolute = [kernel._weight(h) for h in active]
            retained = logsumexp(absolute)
            upper = retained if complete else kernel._upper(0)
            eta = 0. if complete else max(0., min(1., -math.expm1(min(0., retained-upper))))
            for h in active:
                kernel.db.execute('INSERT OR IGNORE INTO discovered VALUES(?,?)', (h, decision_us))
            if complete:
                chosen, decision = kernel._decode(active, absolute, retained, 0., scopes[component], active[0], fallback_on_risk=False)
            else:
                chosen, decision = PersistentForestTracker._decode(kernel, active, absolute, retained, eta,
                    scopes[component], active[0], fallback_on_risk=False)
                decision['conditional_bayes_lower_kind'] = 'independent_root_relaxation'
            last_time = kernel.db.execute('SELECT max(state_us) FROM observations').fetchone()[0]
            expired = last_time is not None and reference_us-last_time > self.config.state.max_age_us
            branch_states = []
            for h in active:
                state = [] if expired else kernel._predict(h, reference_us)
                branch_states.append(dict(handle=h, state_sha256=digest(state)))
                if h == chosen:
                    predictions.extend(state)
            state_work += kernel.state_updates
            if state_work > self.config.state.max_replay_operations:
                raise ValueError('shared beam state-replay work capacity exhausted')
            summary = dict(component=component, nodes=kernel.n, decision_indices=scopes[component],
                indices_count=len(scopes[component]), weight=len(scopes[component])/len(indices) if indices else 0.,
                active=[dict(handle=h, sha256=kernel._prefix(h).sha256, log_weight=w) for h, w in zip(active, absolute)],
                frontier=[], log_retained=retained, log_partition_upper=upper, eta_upper=eta,
                full_model_support_still_in_beam=complete, decision=decision, output_handle=chosen,
                output_sha256=kernel._prefix(chosen).sha256, branches=branch_states, prefix_nodes=kernel.prefix_count,
                merge_restart=False, merge_retained_beams_only=component in products,
                predecessors=change.predecessors if change else (component,), merge_pruning_stages=merge_stages,
                pruning=pruning, recovery_events=[], complete_raw_support_retained=True,
                expired_output_only=expired, state_updates=kernel.state_updates)
            self.store.save_kernel(kernel, chosen=chosen, reference_us=reference_us, decision_us=decision_us)
            self.db.execute('INSERT OR REPLACE INTO component_summaries VALUES(?,?)', (component, canonical(summary)))
            summaries.append(summary)
        if len({p['track_id'] for p in predictions}) != len(predictions):
            raise ValueError('beam component outputs produced duplicate track IDs')
        log_ratio = math.fsum(s['log_retained']-s['log_partition_upper'] for s in summaries)
        return sorted(predictions, key=lambda p: p['track_id']), dict(components=summaries,
            beam_width=self.config.state.active_limit, allocation_policy='fixed_width_no_adaptive_search',
            beam_expansions=work.beam_expansions, candidate_evaluations=work.candidate_evaluations,
            merge_score_evaluations=work.merge_score_evaluations, merge_prefix_steps=work.merge_prefix_steps,
            merge_state_cache_queries=work.merge_state_cache_queries,
            merge_state_cache_copies=work.merge_state_cache_copies,
            merge_state_cache_IO_included_in_latency=True,
            merge_state_cache_queries_bounded_by_merge_prefix_steps=True,
            work_caps=dict(beam_expansions=self.config.max_beam_expansions,
                candidate_evaluations=self.config.max_candidate_evaluations, merge_prefix_steps=self.config.max_merge_prefix_steps),
            work_counters_are_latency_equivalence=False, total_prefix_nodes=prefix_count, total_frontier=0,
            shared_cache_entries=len(self.shared_cache), state_updates=state_work,
            member_rows=self.db.execute('SELECT count(*) FROM component_members').fetchone()[0],
            log_partition_upper=math.fsum(s['log_partition_upper'] for s in summaries),
            product_omitted_mass_upper=max(0., min(1., -math.expm1(min(0., log_ratio)))),
            weighted_truncation_risk_upper=math.fsum(s['weight']*s['eta_upper'] for s in summaries),
            model_regret_upper=math.fsum(s['weight']*s['decision']['risk_bound'] for s in summaries),
            decision_indices=indices, global_fallback_used=False)


class PersistentRankedBeamTracker(PersistentBeamTracker):
    """Top-K COMPLETE current-batch extensions, with old pruning irreversible.

When old components merge, the inherited explicit top-K PRIOR product selection
still applies. Ranked search is exact only inside that preselected domain; it
is not claimed to rank extensions of its omitted Cartesian combinations.
"""
    CONFIG_TYPE = PersistentRankedBeamConfig
    SCHEMA = 'persistent_irreversible_batch_ranked_beam_v1'
    PRUNING_POLICY = 'fixed_top_k_complete_batch_extensions_after_prior_merge_selection'

    def _prefix_upper_function(self, kernel, start, work):
        return kernel._upper

    def _extend(self, kernel, beam, start, child, work):
        if start == kernel.n:
            return beam, True, []
        upper = self._prefix_upper_function(kernel, start, work)
        queue = [(-upper(h), kernel._prefix(h).sha256, h) for h in beam]
        heapq.heapify(queue)
        if len(queue) > self.config.max_batch_frontier:
            raise ValueError('batch-ranked frontier capacity exhausted')
        retained, terminal_count, peak = [], 0, len(queue)
        width = self.config.state.active_limit
        initial_expansions, initial_candidates = work.beam_expansions, work.candidate_evaluations
        while queue:
            if len(retained) == width and -queue[0][0] < -retained[-1][0]:
                break
            _, _, handle = heapq.heappop(queue)
            prefix = kernel._prefix(handle)
            if prefix.depth == kernel.n:
                terminal_count += 1
                retained = sorted([*retained, (-kernel._weight(handle), prefix.sha256, handle)])[:width]
                continue
            work.charge('beam_expansions')
            choices = kernel._choices(handle)
            if len(queue)+len(choices) > self.config.max_batch_frontier:
                raise ValueError('batch-ranked frontier capacity exhausted; top-K not certified')
            for choice in choices:
                work.charge('candidate_evaluations')
                new_handle = child(kernel, handle, choice)
                heapq.heappush(queue, (-upper(new_handle), kernel._prefix(new_handle).sha256, new_handle))
            peak = max(peak, len(queue))
        if not retained:
            raise ValueError('batch-ranked beam has no complete class')
        return {v[2] for v in retained}, not queue and terminal_count <= width, [dict(
            kind='complete_batch_extension_ranking', start_depth=start, end_depth=kernel.n,
            exact_arithmetic_top_k_condition_met=True, formal_numeric_certificate=False,
            selection_domain='extensions_of_preselected_retained_prior_beam_not_full_history',
            retained_classes=len(retained), visited_complete_classes=terminal_count,
            permanently_discarded_frontier_regions=len(queue),
            discarded_visited_complete_classes=terminal_count-len(retained),
            frontier_peak=peak, beam_expansions=work.beam_expansions-initial_expansions,
            candidate_evaluations=work.candidate_evaluations-initial_candidates)]
