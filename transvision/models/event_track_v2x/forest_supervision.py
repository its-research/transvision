"""Offline, ambiguity-aware identity labels. None of these types enter forward.

Source track IDs are joined only by exact cooperative annotation references.
Repeated IDs within a source frame invalidate the entire joined identity, not
just a convenient duplicate. Missing cross-source links are not negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict

import torch

from .identity_forest import digest


@dataclass(frozen=True)
class AnnotationIdentity:
    sequence_id: str
    source_id: int
    frame_id: str
    track_id: str
    token: str
    class_name: str = 'car'

    def __post_init__(self):
        if (self.source_id not in (0, 1) or type(self.source_id) is not int
                or any(not isinstance(v, str) or not v or v == '-1' for v in
                       (self.sequence_id, self.frame_id, self.track_id, self.token, self.class_name))):
            raise ValueError('invalid exact source annotation identity')

    @property
    def reference(self):
        return self.sequence_id, self.source_id, self.frame_id, self.track_id, self.token

    @property
    def track(self):
        return self.sequence_id, self.source_id, self.track_id


@dataclass(frozen=True)
class DetectionIdentityTarget:
    sequence_id: str
    source_id: int
    identity: str | None
    cross_linked: bool = False
    status: str = 'matched'

    def __post_init__(self):
        if (not self.sequence_id or type(self.source_id) is not int or self.source_id not in (0, 1)
                or type(self.cross_linked) is not bool
                or self.status not in ('matched', 'ambiguous', 'unmatched_prediction', 'non_car')
                or (self.status == 'matched') != (isinstance(self.identity, str) and bool(self.identity))):
            raise ValueError('invalid offline detection target')


class AnnotationIdentityIndex:
    def __init__(self, annotations, cooperative_links):
        annotations = tuple(annotations)
        if any(type(a) is not AnnotationIdentity for a in annotations):
            raise TypeError('exact annotation identities required')
        self.annotations = {a.reference: a for a in annotations}
        if len(self.annotations) != len(annotations):
            raise ValueError('repeated annotation reference')
        tokens = [(a.sequence_id, a.source_id, a.frame_id, a.token) for a in annotations]
        if len(set(tokens)) != len(tokens):
            raise ValueError('annotation tokens must be unique within source frame')
        parents = {a.track: a.track for a in annotations}

        def root(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        linked, used = set(), set()
        for left, right in cooperative_links:
            left, right = tuple(left), tuple(right)
            if left not in self.annotations or right not in self.annotations:
                raise ValueError('cooperative reference missing from converted annotations')
            a, b = self.annotations[left], self.annotations[right]
            if a.source_id == b.source_id or a.sequence_id != b.sequence_id or a.class_name != b.class_name:
                raise ValueError('cooperative identity crosses cohort, source or class contract')
            # A single physical source frame is paired at most once in SPD.
            if left in used or right in used:
                raise ValueError('reused cooperative annotation reference')
            used.update((left, right))
            x, y = root(a.track), root(b.track)
            parents[max(x, y)] = min(x, y)
            linked.update((a.track, b.track))
        members, frames, classes = defaultdict(set), defaultdict(Counter), defaultdict(set)
        for a in annotations:
            r = root(a.track)
            members[r].add(a.track)
            frames[r][(a.source_id, a.frame_id)] += 1
            classes[r].add(a.class_name)
        ambiguous = {r for r in members if max(frames[r].values()) > 1 or len(classes[r]) != 1}
        linked_roots = {root(t) for t in linked}
        self.targets = {}
        for reference, a in self.annotations.items():
            r = root(a.track)
            status = 'non_car' if a.class_name != 'car' else 'ambiguous' if r in ambiguous else 'matched'
            identity = digest(['train_annotation_identity_v1', sorted(members[r])]) if status == 'matched' else None
            self.targets[reference] = DetectionIdentityTarget(a.sequence_id, a.source_id, identity,
                                                            r in linked_roots, status)
        self.audit = dict(annotation_count=len(annotations), identity_components=len(members),
                          ambiguous_components=len(ambiguous), cross_linked_components=len(linked_roots),
                          status_counts=dict(Counter(t.status for t in self.targets.values())))


def relation(query, previous):
    """True/False/None = annotated same/different/unknown identity."""
    if query.sequence_id != previous.sequence_id:
        raise ValueError('cross-sequence target comparison')
    if query.status != 'matched':
        return None
    if previous.status == 'unmatched_prediction':
        return False  # Negative only under the declared prediction-to-GT matching rule.
    if previous.status != 'matched':
        return None
    if query.identity == previous.identity:
        return True
    if query.source_id == previous.source_id or query.cross_linked and previous.cross_linked:
        return False
    return None


@dataclass(frozen=True)
class RowSupervision:
    positives: tuple
    known: tuple
    reason: str

    def __post_init__(self):
        if (not self.known or len(self.positives) != len(self.known)
                or any(type(v) is not bool for v in (*self.positives, *self.known))
                or any(p and not k for p, k in zip(self.positives, self.known))
                or self.reason not in ('parent', 'birth', 'out_of_support', 'uncertain_birth',
                                       'ambiguous', 'unmatched_prediction', 'non_car')
                or any(self.positives) != (self.reason in ('parent', 'birth'))):
            raise ValueError('invalid set-valued partial supervision')


def make_row_supervision(context, labels, prior_identities):
    """All same-identity candidate parents are positive; arbitrary parents are not.

    prior_identities contains targets from ALL previously accepted predictions,
    including those outside this gated context. Birth is supervised only when
    absence of an annotated prior can be established. Unknown rows are audited.
    """
    query = labels[context.indices[-1]]
    for i, observation in zip(context.indices, context.observations):
        label = labels[i]
        if (type(label) is not DetectionIdentityTarget or label.sequence_id != observation.sequence_id
                or label.source_id != observation.node.source_id):
            raise ValueError('offline targets do not align with raw prediction identities')
    count = len(context.indices)
    empty = (False,)*count
    if query.status != 'matched':
        return RowSupervision(empty, empty, query.status)
    rel = [relation(query, labels[i]) for i in context.indices[:-1]]
    if any(r is True for r in rel):
        return RowSupervision((False, *(r is True for r in rel)),
                              (True, *(r is not None for r in rel)), 'parent')
    prior = [relation(query, p) for p in prior_identities]
    if any(r is True for r in prior):
        return RowSupervision(empty, empty, 'out_of_support')
    if any(r is None for r in (*prior, *rel)):
        return RowSupervision(empty, empty, 'uncertain_birth')
    return RowSupervision((True, *(False for _ in rel)), (True,)*(len(rel)+1), 'birth')


def set_valued_parent_loss(logits, targets):
    """Local partial-label surrogate, NOT global forest NLL or calibration.

    log sum exp over known options minus log sum exp over all positives.
    Unknown candidates are not silently treated as negatives. Rows without
    supervision are excluded and counted, never reported as zero loss.
    """
    if len(logits) != len(targets):
        raise ValueError('aligned logits and supervision required')
    terms, counts = [], Counter()
    for row, target in zip(logits, targets):
        if (type(target) is not RowSupervision or row.ndim != 1 or len(row) != len(target.known)
                or not torch.isfinite(row).all()):
            raise ValueError('finite support-aligned logits required')
        counts[target.reason] += 1
        if not any(target.positives):
            continue
        known = torch.tensor(target.known, dtype=torch.bool, device=row.device)
        positive = torch.tensor(target.positives, dtype=torch.bool, device=row.device)
        terms.append(torch.logsumexp(row[known], 0)-torch.logsumexp(row[positive], 0))
    if not terms:
        raise ValueError('no supervised rows; zero loss would be misleading')
    return dict(loss=torch.stack(terms).mean(), supervised_rows=len(terms), reason_counts=dict(counts))
