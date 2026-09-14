"""Exact single-source/scan marginalisation conditional on one past root class.

This bridge carries frozen raw forest row potentials into JPDA/PKF without
replacing a sum over equivalent parents by a maximum. Conditioning on one past
identity class is explicit: it is NOT the complete multi-history posterior.
Mixed-source/time batches must not be silently flattened into one matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hypothesis_bank import LogAssociationFactors, logsumexp
from .identity_forest import ForestFactors


@dataclass(frozen=True)
class ConditionalScanFactors:
    factors: LogAssociationFactors
    track_roots: tuple[int, ...]
    detection_indices: tuple[int, ...]
    conditioned_past_roots: tuple[int, ...]
    forest_sha256: str
    source_slot: tuple[int, str]
    decision_us: int


def condition_forest_scan(forest, prefix, *, decision_us):
    """Rows = established identity roots; columns = new scan detections.

For root r and new node j, psi_rj = sum_{p: root(p)=r} exp(row_j[p]).
The track-unmatched potential is one; detection-unmatched is the original
birth potential. Slots already occupied in the past class are forbidden.
The matching posterior equals the aggregate legal forest extensions conditional
on these past roots, but says nothing about alternative past root classes.
"""
    if type(forest) is not ForestFactors:
        raise TypeError('immutable forest factors required')
    roots = forest.roots(prefix)
    start = len(prefix)
    if (start == len(forest.nodes) or type(decision_us) is not int or decision_us < 0
            or any(n.arrival_us > decision_us for n in forest.nodes)):
        raise ValueError('nonempty arrived scan required before conditional inference')
    slots = {(n.source_id, n.frame_id) for n in forest.nodes[start:]}
    if len(slots) != 1 or next(iter(slots))[0] < 0:
        raise ValueError('one real source/frame scan required; mixed scans need explicit sequencing')
    slot = next(iter(slots))
    track_roots = tuple(sorted(set(roots)))
    root_index = {r: i for i, r in enumerate(track_roots)}
    occupied = {roots[p] for p in range(start)
                if (forest.nodes[p].source_id, forest.nodes[p].frame_id) == slot}
    detection_indices = tuple(range(start, len(forest.nodes)))
    pair = np.zeros((len(track_roots), len(detection_indices)))
    allowed = np.zeros(pair.shape, dtype=bool)
    birth = []
    for j, child in enumerate(detection_indices):
        grouped = {}
        for parent, value in forest.rows[child]:
            if parent == -1:
                birth.append(value)
            elif parent < start and roots[parent] not in occupied:
                grouped.setdefault(roots[parent], []).append(value)
            # Parents in the current source/frame create a slot collision and
            # are illegal in the original forest too; they carry zero support.
        for root, weights in grouped.items():
            i = root_index[root]
            pair[i, j], allowed[i, j] = logsumexp(weights), True
    factors = LogAssociationFactors(pair, np.zeros(len(track_roots)), birth, allowed)
    return ConditionalScanFactors(factors, track_roots, detection_indices, roots, forest.digest(), slot, decision_us)
