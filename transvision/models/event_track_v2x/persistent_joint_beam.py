"""Rank a full product of OLD RETAINED beams jointly with current-batch evidence.

No prior-product Top-K preselection is made. A bounded lazy Cartesian cursor
and current-batch prefix search share an upper-bound stopping rule. This is an
irreversible fixed-K factor-model baseline, not recovery of previously deleted
histories, a true posterior guarantee, or a reproduced classical MHT tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .persistent_beam_tracking import PersistentRankedBeamConfig, PersistentRankedBeamTracker


@dataclass(frozen=True)
class PersistentJointBeamConfig(PersistentRankedBeamConfig):
    max_cartesian_states: int = 65536


class RankedCartesianCursor:
    """Lazy full Cartesian enumeration in nonincreasing additive log weight.

Every tuple is reachable from all-zero ranks by coordinate increments. Keeping
its first insertion only removes duplicate paths to that SAME tuple, not any
combination. The queue maximum bounds all not-yet-emitted combinations because
each path to an unseen tuple crosses a queued ancestor with no smaller score.
"""
    def __init__(self, groups, *, state_limit, work):
        if type(state_limit) is not int or state_limit < 1 or not groups:
            raise ValueError('nonempty Cartesian groups and positive capacity required')
        self.groups = tuple((c, tuple(sorted(options, key=lambda v: (-v[0], v[1])))) for c, options in groups)
        components = [c for c, _ in self.groups]
        if (any(type(c) is not int or c < 1 for c in components) or len(set(components)) != len(components)
                or components != sorted(components)):
            raise ValueError('unique ordered component identities required')
        for _, options in self.groups:
            if (not options or len({h for _, h in options}) != len(options)
                    or any(not math.isfinite(w) or type(h) is not int or h < 0 for w, h in options)):
                raise ValueError('finite weights and unique retained handles required')
        self.state_limit, self.work = state_limit, work
        self.queue, self.seen, self.emitted = [], set(), 0
        self.peak = 0
        self._push((0,)*len(self.groups))

    def _push(self, ranks):
        if ranks in self.seen:
            return
        if len(self.seen) >= self.state_limit:
            raise ValueError('Cartesian state capacity exhausted; joint ranking not completed')
        self.work.charge('candidate_evaluations')
        self.work.merge_score_evaluations += 1
        score = math.fsum(options[index][0] for (_, options), index in zip(self.groups, ranks))
        selection = tuple((component, options[index][1]) for (component, options), index in zip(self.groups, ranks))
        self.seen.add(ranks)
        heapq.heappush(self.queue, (-score, selection, ranks))
        self.peak = max(self.peak, len(self.queue))

    def peek(self):
        return -self.queue[0][0] if self.queue else -math.inf

    def pop(self):
        if not self.queue:
            raise StopIteration
        negative, selection, ranks = heapq.heappop(self.queue)
        self.emitted += 1
        for axis, (_, options) in enumerate(self.groups):
            if ranks[axis]+1 < len(options):
                neighbor = ranks[:axis]+(ranks[axis]+1,)+ranks[axis+1:]
                self._push(neighbor)
        return -negative, selection


class PersistentJointBeamTracker(PersistentRankedBeamTracker):
    CONFIG_TYPE = PersistentJointBeamConfig
    SCHEMA = 'persistent_irreversible_joint_cartesian_beam_v1'
    PRUNING_POLICY = 'top_k_complete_batch_extensions_of_full_retained_cartesian_product'

    def _audit_properties(self):
        return dict(super()._audit_properties(), kind=self.SCHEMA,
            prior_cartesian_preselection=False, previous_event_pruning_irreversible=True)

    def _prepare_merge(self, groups, complete, work):
        # Retain all OLD active classes as a factored domain. No enumeration or
        # width truncation before the new component's factors become available.
        return dict(groups=tuple((c, tuple(options)) for c, options in groups), complete=complete)

    def _joint_upper_functions(self, kernel, component, start, groups, cursor, work):
        # The v1 backend deliberately retains its original row-mass relaxation.
        absolute = math.fsum(max(abs(w) for w, _ in options) for _, options in groups)
        margin = 64*np.finfo(float).eps*(kernel.n+len(groups)+1)*max(1., absolute, kernel.suffix_abs[start])
        def product_upper():
            return math.fsum([cursor.peek(), kernel.suffix[start], margin]) if cursor.queue else -math.inf
        def prefix_upper(h):
            return kernel._upper(h)+margin
        return prefix_upper, product_upper

    def _advance_component(self, kernel, component, context, change, merge, child, work):
        if merge is None:
            beam, complete, pruning, stages = super()._advance_component(
                kernel, component, context, change, merge, child, work)
            for row in pruning:
                row['selection_domain'] = 'all_current_batch_extensions_of_previous_retained_classes'
            return beam, complete, pruning, stages
        start = kernel.n-len(change.added_indices)
        groups = merge['groups']
        initial_expansions, initial_candidates = work.beam_expansions, work.candidate_evaluations
        initial_lift_steps = work.merge_prefix_steps
        cursor = RankedCartesianCursor(groups, state_limit=self.config.max_cartesian_states, work=work)
        # One conservative margin is added to BOTH kinds of unresolved bounds.
        # This is not interval arithmetic, so the audit never claims a numeric
        # certificate. In exact arithmetic the extra margin is unnecessary.
        prefix_upper, product_upper = self._joint_upper_functions(kernel, component, start, groups, cursor, work)
        queue, retained, terminal_count, peak = [], [], 0, 0
        width = self.config.state.active_limit
        while queue or cursor.queue:
            prior_bound = product_upper()
            branch_bound = -queue[0][0] if queue else -math.inf
            if len(retained) == width and max(prior_bound, branch_bound) < -retained[-1][0]:
                break
            if not queue or prior_bound > branch_bound:
                if len(queue)+1 > self.config.max_batch_frontier:
                    raise ValueError('joint batch frontier capacity exhausted; event not committed')
                weight, selection = cursor.pop()
                handle = self._materialize_combination(kernel, component, start, weight, selection, child, work)
                heapq.heappush(queue, (-prefix_upper(handle), kernel._prefix(handle).sha256, handle))
                peak = max(peak, len(queue))
                continue
            _, _, handle = heapq.heappop(queue)
            prefix = kernel._prefix(handle)
            if prefix.depth == kernel.n:
                terminal_count += 1
                retained = sorted([*retained, (-kernel._weight(handle), prefix.sha256, handle)])[:width]
                continue
            work.charge('beam_expansions')
            choices = kernel._choices(handle)
            if len(queue)+len(choices) > self.config.max_batch_frontier:
                raise ValueError('joint batch frontier capacity exhausted; event not committed')
            for choice in choices:
                work.charge('candidate_evaluations')
                handle_child = child(kernel, handle, choice)
                heapq.heappush(queue, (-prefix_upper(handle_child), kernel._prefix(handle_child).sha256, handle_child))
            peak = max(peak, len(queue))
        if not retained:
            raise ValueError('joint Cartesian ranking did not produce a complete class')
        remaining_upper = max(product_upper(), -queue[0][0] if queue else -math.inf)
        if (queue or cursor.queue) and (len(retained) != width or remaining_upper >= -retained[-1][0]):
            raise ValueError('joint Cartesian top-K stopping condition is not satisfied')
        complete = merge['complete'] and not queue and not cursor.queue and terminal_count <= width
        return {v[2] for v in retained}, complete, [dict(
            kind='joint_cartesian_complete_batch_ranking', start_depth=start, end_depth=kernel.n,
            selection_domain='full_cartesian_product_of_previous_retained_classes_and_all_current_batch_extensions',
            exact_arithmetic_top_k_condition_met=True, formal_numeric_certificate=False,
            retained_classes=len(retained), visited_complete_classes=terminal_count,
            permanently_discarded_prefix_regions=len(queue),
            discarded_visited_complete_classes=terminal_count-len(retained),
            remaining_log_upper=None if not math.isfinite(remaining_upper) else remaining_upper,
            kth_log_weight=-retained[-1][0], prefix_frontier_peak=peak,
            beam_expansions=work.beam_expansions-initial_expansions,
            candidate_evaluations=work.candidate_evaluations-initial_candidates,
            merge_prefix_steps=work.merge_prefix_steps-initial_lift_steps)], [dict(
            kind='lazy_full_retained_cartesian_domain', prior_preselection=False,
            predecessor_widths=[len(options) for _, options in groups],
            log_prior_combination_count=math.fsum(math.log(len(options)) for _, options in groups),
            cartesian_states_generated=len(cursor.seen), prior_combinations_materialized=cursor.emitted,
            cartesian_queue_remaining=len(cursor.queue), cartesian_queue_peak=cursor.peak,
            cartesian_state_limit=self.config.max_cartesian_states)]
