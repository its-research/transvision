"""Prediction-only candidate context shared by training and persistent inference."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .forest_tracking import ForestTrackingConfig, RawIdentityDetection
from .tracking_v2 import propagate


CONTEXT_RECIPE = 'persistent-forest-arrival-row-parent-context-v1'


@dataclass(frozen=True)
class ForestRowContext:
    indices: tuple
    observations: tuple
    decision_us: int

    def __post_init__(self):
        if (not self.indices or type(self.decision_us) is not int or self.decision_us < 0
                or len(self.indices) != len(self.observations)
                or any(type(i) is not int or i < 0 for i in self.indices)
                or tuple(sorted(set(self.indices))) != self.indices
                or any(type(o) is not RawIdentityDetection or o.node.arrival_us > self.decision_us
                       for o in self.observations)
                or len({o.sequence_id for o in self.observations}) != 1
                or len({o.node.node_id for o in self.observations}) != len(self.observations)):
            raise ValueError('invalid causal prediction-only row context')

    @property
    def support(self):
        return tuple((-1,) for _ in self.indices[:-1])+((-1, *range(len(self.indices)-1)),)


def build_row_contexts(new, *, old_count, older_candidates, config, decision_us, sequence_id):
    """Consume (global index, raw observation) streams; never accepts GT labels.

    The caller must supply all old observations in each requested inclusive
    state-time interval. Only top-parent raw vectors are retained in the heap.
    Current-batch earlier nodes are considered even if their source time is late.
    """
    new = tuple(new)
    if (type(config) is not ForestTrackingConfig or type(old_count) is not int or old_count < 0
            or type(decision_us) is not int or decision_us < 0 or not callable(older_candidates)
            or any(type(o) is not RawIdentityDetection or o.sequence_id != sequence_id
                   or o.node.arrival_us > decision_us for o in new)):
        raise ValueError('invalid row-context cohort or availability')
    contexts = []
    for offset, current in enumerate(new):
        candidates = []

        def consider(index, previous):
            if (type(index) is not int or not 0 <= index < old_count+offset
                    or type(previous) is not RawIdentityDetection or previous.sequence_id != sequence_id
                    or previous.node.arrival_us > decision_us):
                raise ValueError('invalid historical candidate or future evidence')
            if ((current.node.source_id, current.node.frame_id) ==
                    (previous.node.source_id, previous.node.frame_id)
                    or abs(current.state_us-previous.state_us) > config.max_parent_gap_us):
                return
            mean, _ = propagate(np.asarray(previous.mean), np.asarray(previous.covariance),
                (current.state_us-previous.state_us)/1e6, config.process_noise)
            distance = float(np.linalg.norm(np.asarray(current.mean[:2])-mean[:2]))
            if distance <= config.gate_distance_m:
                item = (-distance, -index, previous)
                if len(candidates) < config.parent_limit:
                    heapq.heappush(candidates, item)
                elif item[:2] > candidates[0][:2]:
                    heapq.heapreplace(candidates, item)

        for i, previous in older_candidates(max(0, current.state_us-config.max_parent_gap_us),
                                             current.state_us+config.max_parent_gap_us):
            if type(i) is not int or not 0 <= i < old_count:
                raise ValueError('historical stream contains an uncommitted index')
            consider(i, previous)
        for i in range(offset):
            consider(old_count+i, new[i])
        chosen = sorted((-negative_index, obs) for _, negative_index, obs in candidates)
        contexts.append(ForestRowContext(tuple(i for i, _ in chosen)+(old_count+offset,),
                                        tuple(o for _, o in chosen)+(current,), decision_us))
    return tuple(contexts)
