"""Bounded conditional Hamming action search over ALL legal forest histories.

Posterior support and action support are deliberately separate. A frontier cap returns a valid incumbent and a lower bound; it never calls an unfinished search Bayes optimal.
Bounds are floating-point model diagnostics, not certificates.
"""
import heapq
import math


def conditional_action(factors, active, log_weights, indices, *, budget=4096, frontier_limit=8192):
    active, indices = tuple(active), tuple(indices)
    if (not active or len(active) != len(log_weights) or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(factors.nodes) for i in indices)
            or type(budget) is not int or budget < 0 or type(frontier_limit) is not int or frontier_limit < 1 or any(not math.isfinite(w) for w in log_weights)):
        raise ValueError('invalid conditional action problem')
    for a in active:
        if len(a) != len(factors.nodes):
            raise ValueError('posterior leaves must be complete')
        factors.roots(a)
    maximum = max(log_weights)
    raw = [math.exp(w - maximum) for w in log_weights]
    weights = [w / math.fsum(raw) for w in raw]
    marginal = {i: {} for i in indices}
    for a, weight in zip(active, weights):
        roots = factors.roots(a)
        for i in indices:
            marginal[i][roots[i]] = marginal[i].get(roots[i], 0.) + weight

    def lower(prefix):
        roots = factors.roots(prefix)
        return max(0., math.fsum(1. - (marginal[i].get(roots[i], 0.) if i < len(prefix) else max(marginal[i].values())) for i in indices) / max(1, len(indices)))

    chosen = min(active, key=lambda a: (lower(a), a))
    best = lower(chosen)
    queue, steps = [(lower(()), ())], 0
    while queue and steps < budget:
        bound, prefix = queue[0]
        if bound >= best:
            queue.clear()
            break
        children = factors.children(prefix)
        # Leave parent in queue if its full disjoint replacement will not fit.
        if len(queue) - 1 + len(children) > frontier_limit:
            break
        heapq.heappop(queue)
        steps += 1
        for child in children:
            bound = lower(child)
            if len(child) == len(factors.nodes):
                if (bound, child) < (best, chosen):
                    chosen, best = child, bound
            elif bound < best:
                heapq.heappush(queue, (bound, child))
    low = min(best, queue[0][0]) if queue else best
    return chosen, dict(
        decision_rule='conditional_Hamming',
        loss='mean_identity_root_Hamming',
        conditional_risk=best,
        relaxed_bayes_lower=low,
        optimization_gap=max(0., best - low),
        action_search_steps=steps,
        action_search_complete=not queue,
        action_space='all_supported_legal_histories',
        selected_outside_active=chosen not in active,
        numeric_certificate=False)


def decode_persistent(kernel, active, weights, retained, eta, indices, fallback, *, force_fallback=False, materialize=True):
    from .identity_forest import ForestFactors

    def path(handle):
        return tuple(kernel._prefix(kernel.ancestor(handle, i + 1)).choice for i in range(kernel.n))

    if not active or force_fallback:
        return fallback, dict(
            risk_bound=1.,
            conditional_risk=None,
            relaxed_bayes_lower=None,
            optimization_gap=None,
            empty_loss_scope=not indices,
            action_search_complete=False,
            action_space='all_supported_legal_histories',
            selected_outside_active=False)
    if kernel.n > kernel.config.max_decision_nodes:
        raise ValueError('full action materialization exceeds decision-node capacity')
    factors = ForestFactors(tuple(kernel._observation(i).node for i in range(kernel.n)), tuple(kernel._row(i) for i in range(kernel.n)))
    action, audit = conditional_action(
        factors,
        tuple(path(h) for h in active),
        weights,
        indices,
        budget=(getattr(kernel, '_paper_action_remaining', kernel.config.state.paper_action_budget) if materialize else 0),
        frontier_limit=kernel.config.state.paper_action_frontier)
    before = kernel.prefix_count
    handle = active[0]
    if materialize:
        handle = 0
        for parent in action:
            handle = kernel._child(handle, parent)
    return handle, dict(
        audit,
        action_materialization_prefixes=kernel.prefix_count - before,
        action_materialization_steps=len(action) if materialize else 0,
        risk_bound=min(1., eta + audit['optimization_gap']),
        empty_loss_scope=not indices)
