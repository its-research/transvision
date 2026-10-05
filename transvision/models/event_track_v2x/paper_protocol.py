"""Versioned paper contracts.

Historical car-first runs remain a separate protocol.
"""
import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from .detection_cache_v2 import canonical
from .paper_evaluation_policy import evaluation_class

PAPER = 'rbf-all-class-top64-v1'
LEGACY = 'rbf-car-first-top64-v1'
SEEDS = (1337, 2027, 3407)


@dataclass(frozen=True)
class PaperProtocol:
    dataset: str
    split: str
    candidates: str = PAPER
    minimum_raw_score: float = .05
    maximum_detections: int = 64
    evaluation_class: str = 'car'

    def __post_init__(self):
        allowed = {'spd': {'train', 'val'}, 'v2v4real': {'train', 'official_test'}}
        if self.dataset not in allowed or self.split not in allowed[self.dataset]:
            raise ValueError('dataset/split not authorized by paper protocol')
        evaluation_class(self.dataset, self.evaluation_class)
        if (self.candidates not in {PAPER, LEGACY} or (self.evaluation_class == 'vehicle' and self.candidates != PAPER)
                or self.minimum_raw_score != .05 or type(self.maximum_detections) is not int
                or self.maximum_detections != 64):
            raise ValueError('unsupported frozen candidate/evaluation protocol')

    @property
    def sha256(self):
        return hashlib.sha256(canonical(asdict(self))).hexdigest()

    def require_train(self):
        if self.split != 'train':
            raise ValueError('training, calibration and selection require train')

    def select(self, scores, classes):
        scores, classes = np.asarray(scores), np.asarray(classes)
        if (scores.ndim != 1 or classes.shape != scores.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or not np.issubdtype(classes.dtype, np.integer) or np.any((classes < 0) | (classes > 2))):
            raise ValueError('invalid candidate scores/classes')
        mask = scores >= self.minimum_raw_score
        if self.candidates == LEGACY:
            mask &= classes == 0
        indices = np.flatnonzero(mask)
        return np.asarray(sorted(indices, key=lambda i: (-float(scores[i]), int(i)))[:64], dtype=np.int64)


def require_same_protocol(records):
    """Reject mixed tables, including same name with different ROI/feature
    identities."""
    records = list(records)
    if not records:
        raise ValueError('empty comparison')
    fields = ('protocol', 'detector_sha256', 'embedding_sha256', 'motion', 'roi', 'frequency_hz', 'label_version', 'evaluator_sha256', 'amotp_definition')
    if any(not set(fields) <= set(r) for r in records):
        raise ValueError('missing comparison fields, including explicit AMOTP definition')
    bindings = [canonical({key: r[key] for key in fields}) for r in records]
    if len(set(bindings)) != 1:
        raise ValueError('incomparable protocol/input/evaluator identities')
    return hashlib.sha256(bindings[0]).hexdigest()


def training_partition(sequences, *, holdout_fraction=.2):
    """Stable sequence/recording-level partition; caller groups same-session
    fragments."""
    ids = sorted(set(sequences))
    if len(ids) < 2 or not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError('at least two independent training groups required')
    ordered = sorted(ids, key=lambda s: hashlib.sha256(('rbf-train-holdout-v1:' + s).encode()).hexdigest())
    n = max(1, min(len(ids) - 1, round(len(ids) * holdout_fraction)))
    return {'fit': sorted(ordered[n:]), 'holdout': sorted(ordered[:n])}
