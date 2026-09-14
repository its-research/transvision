"""Coverage-based storage admission for already legal complete-class proposals.

This is a conservative capacity calculation, not a posterior confidence or a
new mass bound. It assumes the kernel's complete-support covering invariant.
No factors, explicit classes or unresolved regions are removed here.
"""
from __future__ import annotations


def proposal_frontier_reserve(kernel):
    """A covered old explicit class cannot add a frontier region on eviction.

If every old explicit class is covered, a legal new class is also covered
(otherwise it already belongs to the explicit set). Promotion can only keep
or shrink the frontier. In the general case the existing +1 bound remains.
    """
    return 0 if all(kernel._covered(h) for h in kernel.active) else 1


def covered_proposal_operation(kernel, *, proposals=None, remaining, prefix_count,
                               frontier_count, config, excluded=()):
    """Choose a fully affordable suffix; None means no admissible proposal.

With proposals=None, select the highest-bound unresolved frontier prefix.
Otherwise preserve the supplied history-extension proposal order. Every
suffix step, including cached nodes, is still charged by the common executor.
    """
    if any(type(value) is not int or value < 0 for value in (remaining, prefix_count, frontier_count)):
        raise ValueError('nonnegative integer compute and storage counts required')
    reserve = proposal_frontier_reserve(kernel)
    if (len(kernel.frontier)+reserve > kernel.config.state.max_frontier
            or frontier_count+reserve > config.max_total_frontier):
        return None
    budget = min(remaining, kernel.config.max_prefix_nodes-kernel.prefix_count,
                 config.max_total_prefix_nodes-prefix_count)
    excluded = frozenset(excluded)
    candidates = kernel.frontier if proposals is None else proposals
    eligible = [h for h in candidates if h not in excluded
                and 0 < kernel.n-kernel._prefix(h).depth <= budget]
    if not eligible:
        return None
    base = (min(eligible, key=lambda h: (-kernel._upper(h), kernel._prefix(h).sha256))
            if proposals is None else eligible[0])
    return dict(base=base, kind='coverage_admitted_complete_proposal',
                requested_steps=kernel.n-kernel._prefix(base).depth, frontier_reserve=reserve)
