"""Budgeted complete-class proposals from recoverable frontier regions.

Experimental operation selection, not yet the default allocation policy.
Greedy rollout supplies a legal lower-mass contribution; it is not an upper
bound or a guarantee of finding the best history. Unresolved support remains.
"""
from __future__ import annotations


def frontier_completion_operation(kernel, *, remaining, prefix_count, frontier_count, config, excluded=()):
    if (type(remaining) is not int or remaining < 0 or type(prefix_count) is not int
            or prefix_count < 0 or type(frontier_count) is not int or frontier_count < 0):
        raise ValueError('nonnegative integer compute and storage counts required')
    # Seeding can evict at most one old explicit class into the frontier.
    if (len(kernel.frontier)+1 > kernel.config.state.max_frontier
            or frontier_count+1 > config.max_total_frontier):
        return None
    budget = min(remaining, kernel.config.max_prefix_nodes-kernel.prefix_count,
                 config.max_total_prefix_nodes-prefix_count)
    excluded = frozenset(excluded)
    eligible = [h for h in kernel.frontier if h not in excluded
                and 0 < kernel.n-kernel._prefix(h).depth <= budget]
    if not eligible:
        return None
    base = min(eligible,key=lambda h:(-kernel._upper(h),kernel._prefix(h).sha256))
    return dict(base=base,kind='frontier_completion_proposal',
                requested_steps=kernel.n-kernel._prefix(base).depth)
