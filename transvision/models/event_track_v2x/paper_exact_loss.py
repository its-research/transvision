"""Exact SMALL forest NLL.

Never substitute this name for the row surrogate.
"""
import torch


def exact_forest_nll(nodes, support, logits, target_roots, *, maximum_nodes=8, maximum_histories=50000):
    """Marginalize all parent aliases of a complete supervised identity
    partition.

    The bounded enumeration is exact if it returns; exceeding either cap raises instead of returning a truncated denominator called an exact likelihood. Targets enter only this
    offline loss function, not a network forward call.
    """
    nodes, support, logits, target_roots = tuple(nodes), tuple(map(tuple, support)), tuple(logits), tuple(target_roots)
    n = len(nodes)
    if (not 0 < n <= maximum_nodes or type(maximum_histories) is not int or maximum_histories < 1 or not n == len(support) == len(logits) == len(target_roots)):
        raise ValueError('bounded aligned small-model supervision required')
    for i, (choices, row) in enumerate(zip(support, logits)):
        if (len(set(choices)) != len(choices) or -1 not in choices or any(type(p) is not int or not -1 <= p < i for p in choices) or row.shape != (len(choices), )
                or not torch.isfinite(row).all()):
            raise ValueError('invalid differentiable forest support')
    all_weights, positive_weights = [], []

    def visit(roots, slots, weight):
        i = len(roots)
        if i == n:
            if len(all_weights) >= maximum_histories:
                raise ValueError('exact forest partition enumeration limit exceeded')
            all_weights.append(weight)
            if roots == target_roots:
                positive_weights.append(weight)
            return
        slot = (nodes[i].source_id, nodes[i].frame_id)
        for offset, parent in enumerate(support[i]):
            root = i if parent < 0 else roots[parent]
            if slot in slots.get(root, frozenset()):
                continue
            updated = dict(slots)
            updated[root] = slots.get(root, frozenset()) | {slot}
            visit(roots + (root, ), updated, weight + logits[i][offset])

    visit((), {}, logits[0].sum() * 0.)
    if not positive_weights:
        raise ValueError('target partition is outside legal support')
    logz = torch.logsumexp(torch.stack(all_weights), 0)
    target = torch.logsumexp(torch.stack(positive_weights), 0)
    return dict(
        loss=logz - target,
        objective='exact_small_forest_partition_nll',
        log_partition=logz,
        log_target_partition_mass=target,
        legal_histories=len(all_weights),
        target_parent_aliases=len(positive_weights))
