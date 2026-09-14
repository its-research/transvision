"""Current-action risk from independent-parent messages and collision mass.

Let q0 be the row-normalized independent-parent law before same-source/frame
exclusion. The constrained model is q0 conditioned on no collision. Exact
root marginals and root-equality messages give an upper bound on excluded
mass by the union bound. This bounds MODEL regret, not real tracking metrics.
Floating guards are not interval arithmetic or a formal numeric certificate.
"""
from __future__ import annotations

import math

import numpy as np

from .identity_forest import ForestFactors


def independent_parent_messages(factors, *, maximum_nodes=1024):
    if type(factors) is not ForestFactors or type(maximum_nodes) is not int or maximum_nodes < 1:
        raise ValueError('validated factors and a positive node capacity required')
    n = len(factors.nodes)
    if n > maximum_nodes:
        raise ValueError('independent-parent message capacity exceeded; no risk certificate computed')
    roots = np.zeros((n, n), dtype=np.float64)
    equality = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(factors.rows):
        maximum = max(logit for _, logit in row)
        unnormalized = [math.exp(logit-maximum) for _, logit in row]
        total = math.fsum(unnormalized)
        for (parent, _), value in zip(row, unnormalized):
            probability = value/total
            if parent < 0:
                roots[i, i] += probability
            else:
                roots[i, :i] += probability*roots[parent, :i]
                equality[i, :i] += probability*equality[parent, :i]
        equality[:i, i] = equality[i, :i]
        equality[i, i] = 1.
    return roots, equality


def action_risk_certificate(factors, parents, *, scope=None, maximum_nodes=1024):
    """Bound one complete legal action, using the same legal action space.

The independent-node Bayes floor is only a lower bound; no claim that its
nodewise minimizers jointly form a legal identity configuration is needed.
Storage is O(n^2), work O(n * number_of_parent_edges), explicitly capped.
    """
    if type(factors) is not ForestFactors or type(parents) is not tuple or len(parents) != len(factors.nodes):
        raise ValueError('complete legal parent action and validated factors required')
    action = factors.roots(parents)  # Reject unsupported edges and identity collisions.
    n = len(factors.nodes)
    scope = tuple(range(n)) if scope is None else tuple(scope)
    if (len(scope) != len(set(scope)) or any(type(i) is not int or not 0 <= i < n for i in scope)):
        raise ValueError('unique in-range loss indices required')
    roots, equality = independent_parent_messages(factors, maximum_nodes=maximum_nodes)
    slots = {}
    for i, node in enumerate(factors.nodes):
        if node.source_id != -1:
            slots.setdefault((node.source_id, node.frame_id), []).append(i)
    terms = [float(equality[i, j]) for indices in slots.values()
             for index, i in enumerate(indices) for j in indices[:index]]
    guard = 64*np.finfo(float).eps*(n+1)*(1+sum(map(len, factors.rows))+len(terms))
    excluded_upper = min(1., max(0., math.fsum(terms))+guard)
    if scope:
        probabilities = [float(roots[i, action[i]]) for i in scope]
        risk = math.fsum(1-probability for probability in probabilities)/len(scope)
        floor = math.fsum(1-float(roots[i].max()) for i in scope)/len(scope)
        gap = max(0., risk-floor)
        # (1-delta) Regret_q(a) <= [R_q0(a)-BayesFloor_q0] + delta.
        conditional = 1. if excluded_upper >= 1. else min(1., (gap+guard+excluded_upper)/(1-excluded_upper))
        # The TV comparison is also valid; neither bound uses retained mass.
        bound = min(conditional, min(1., gap+guard+2*excluded_upper))
    else:
        probabilities, risk, floor, gap, conditional, bound = [], 0., 0., 0., 0., 0.
    return dict(kind='independent_parent_current_action_risk_v1', factors_sha256=factors.digest(),
        nodes=n, loss_indices=list(scope), action_roots=[action[i] for i in scope],
        selected_root_probabilities=probabilities, independent_action_risk=risk,
        independent_bayes_lower=floor, independent_optimization_gap=gap,
        collision_pairs=len(terms), collision_union_sum=math.fsum(terms),
        excluded_mass_upper=excluded_upper, conditional_model_regret_upper=conditional,
        model_regret_upper=bound, numeric_guard=guard, formal_numeric_certificate=False,
        message_array_bytes=roots.nbytes+equality.nbytes, maximum_nodes=maximum_nodes,
        model_only=True, true_posterior_or_metric_bound=False)
