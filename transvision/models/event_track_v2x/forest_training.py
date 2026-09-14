"""Batched row-context training using the inference network and physical features.

Only the last token is scored as a query. Current and historical embeddings are
computed separately, as in RecoverableIdentityModel.forward (including dropout).
Eval logits must agree with neural_parent_logits; no GT enters this function.
"""
from __future__ import annotations

import math

import torch

from .forest_potentials import geometry_parent_logit
from .forest_row_context import ForestRowContext
from .learned_identity import RecoverableIdentityModel


def batched_row_logits(model, contexts, *, max_nodes=9, max_batch=256,
                       geometry_weight=1., process_noise=.1):
    contexts = tuple(contexts)
    if (not isinstance(model, RecoverableIdentityModel) or not contexts
            or type(max_nodes) is not int or max_nodes < 1
            or type(max_batch) is not int or not 1 <= len(contexts) <= max_batch
            or any(type(c) is not ForestRowContext or len(c.indices) > max_nodes for c in contexts)
            or not math.isfinite(geometry_weight) or geometry_weight < 0
            or not math.isfinite(process_noise) or process_noise < 0):
        raise ValueError('invalid bounded prediction-only training contexts')
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    batch, tokens = len(contexts), max(len(c.indices) for c in contexts)
    features = torch.zeros(batch, tokens, 203, device=device, dtype=dtype)
    source = torch.zeros(batch, tokens, device=device, dtype=torch.long)
    information, arrival = torch.zeros_like(source), torch.zeros_like(source)
    decision = torch.tensor([c.decision_us for c in contexts], device=device, dtype=torch.long)[:, None]
    lengths = torch.tensor([len(c.indices) for c in contexts], device=device, dtype=torch.long)
    geometry = torch.zeros(batch, tokens, device=device, dtype=dtype)
    for i, context in enumerate(contexts):
        raw = context.observations
        n = len(raw)
        features[i, :n] = torch.tensor([o.features for o in raw], device=device, dtype=dtype)
        source[i, :n] = torch.tensor([o.node.source_id for o in raw], device=device)
        information[i, :n] = torch.tensor([o.node.information_us for o in raw], device=device)
        arrival[i, :n] = torch.tensor([o.node.arrival_us for o in raw], device=device)
        geometry[i, 1:n] = torch.tensor([geometry_parent_logit(raw[-1], p, process_noise=process_noise)
                                         for p in raw[:-1]], device=device, dtype=dtype)
    # Integer timestamp differences are taken before floating-point conversion.
    flat_decision = decision.expand(-1, tokens).reshape(-1)
    def embed():
        return model._embed(features.reshape(-1, 203), source.reshape(-1), information.reshape(-1),
                            arrival.reshape(-1), flat_decision).reshape(batch, tokens, -1)
    current, history = embed(), embed()
    padding = torch.arange(tokens, device=device)[None, :] >= lengths[:, None]
    attended, _ = model.temporal_attention(current, history, history, key_padding_mask=padding, need_weights=False)
    context = model.context_norm(current+attended)
    row = torch.arange(batch, device=device)
    query = context[row, lengths-1]
    query_source = source[row, lengths-1]
    unmatched = torch.where(query_source == 0, model.left_unmatched(query).squeeze(-1),
                            model.right_unmatched(query).squeeze(-1))
    birth = unmatched+model.temporal_unmatched(query).squeeze(-1)
    q = query[:, None].expand(-1, tokens, -1)

    def pair(a, b, scorer):
        return scorer(torch.cat((a, b, torch.abs(a-b), a*b), -1)).squeeze(-1)

    temporal = pair(q, history, model.temporal_scorer)
    left = torch.where((query_source == 0)[:, None, None], q, context)
    right = torch.where((query_source == 0)[:, None, None], context, q)
    cross = pair(left, right, model.pair_scorer)
    matches = temporal+torch.where(source != query_source[:, None], cross, unmatched[:, None])
    logits = torch.cat((birth[:, None], matches[:, :-1]), 1)+geometry_weight*geometry
    output = tuple(logits[i, :len(c.indices)] for i, c in enumerate(contexts))
    if any(not torch.isfinite(r).all() for r in output):
        raise ValueError('nonfinite neural row logits')
    return output
