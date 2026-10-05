"""Offline association targets, separated from prediction-only features.

An unmatched prediction uses the frozen prediction-to-GT matching convention.
A matched annotation missing cooperative identity supervision stays unknown.
Unknown targets never become negatives or forced dustbin assignments.
"""
import numpy as np


def _identities(matches, annotation_tokens, source_bindings, side):
    matches = np.asarray(matches)
    if (matches.ndim != 1 or matches.dtype.kind not in 'iu'
            or np.any(matches < -1) or np.any(matches >= len(annotation_tokens))
            or len(set(annotation_tokens)) != len(annotation_tokens)):
        raise ValueError('invalid prediction-to-annotation matching')
    selected = matches[matches >= 0]
    if len(set(selected.tolist())) != len(selected):
        raise ValueError('prediction-to-GT matching must be one-to-one')
    bindings = {}
    for row in source_bindings:
        if row['side'] != side:
            continue
        token = row['annotation_token']
        if token in bindings or token not in annotation_tokens:
            raise ValueError('duplicate or absent supervised annotation token')
        identity = row['cooperative_identity_id']
        if not isinstance(identity, str) or not identity.isascii() or not identity.isdecimal():
            raise ValueError('invalid cooperative identity')
        bindings[token] = int(identity)
    identities = []
    known = []
    for index in matches:
        if index == -1:
            identities.append(None)
            known.append(True)
        else:
            identity = bindings.get(annotation_tokens[int(index)])
            identities.append(identity)
            known.append(identity is not None)
    return identities, np.asarray(known, dtype=bool)


def association_targets(left_matches, right_matches, left_annotation_tokens,
                        right_annotation_tokens, source_bindings, geometry_gate):
    """Build supervised BCE and bidirectional assignment masks for one pair.

    The gate must be computed exclusively from available predictions and poses.
    The caller binds annotation and mapping provenance before invoking this
    offline function. No value returned here may enter online feature encoding.
    """
    left, left_known = _identities(left_matches, left_annotation_tokens,
                                   source_bindings, 'vehicle-side')
    right, right_known = _identities(right_matches, right_annotation_tokens,
                                     source_bindings, 'infrastructure-side')
    gate = np.asarray(geometry_gate)
    if gate.dtype.kind != 'b' or gate.shape != (len(left), len(right)):
        raise ValueError('prediction-only gate shape or dtype differs')
    targets = np.asarray([
        [int(a is not None and b is not None and a == b) for b in right]
        for a in left
    ], dtype=np.uint8).reshape(gate.shape)
    if np.any(targets.sum(axis=0) > 1) or np.any(targets.sum(axis=1) > 1):
        raise ValueError('annotation identities do not give one-to-one targets')
    pair_mask = gate & left_known[:, None] & right_known[None, :]
    # Skip an assignment row/column if an in-gate alternative has unknown GT
    # identity; otherwise a missing positive could become a false unmatched.
    left_assignment_mask = left_known & np.all(~gate | right_known[None, :], axis=1)
    right_assignment_mask = right_known & np.all(~gate | left_known[:, None], axis=0)
    eligible_targets = targets.astype(bool) & gate
    left_assignment = np.full(len(left), len(right), dtype=np.int64)
    right_assignment = np.full(len(right), len(left), dtype=np.int64)
    rows, columns = np.nonzero(eligible_targets)
    left_assignment[rows] = columns
    right_assignment[columns] = rows
    return dict(targets=targets, supervised_pair_mask=pair_mask,
                left_assignment=left_assignment, right_assignment=right_assignment,
                left_assignment_mask=left_assignment_mask,
                right_assignment_mask=right_assignment_mask,
                left_identity_known=left_known, right_identity_known=right_known)
