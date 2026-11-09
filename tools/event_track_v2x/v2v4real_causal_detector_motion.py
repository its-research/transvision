"""GT-free adjacent-source detector motion preparation, not a fitted 9D model.

All-class candidates remain intact. Matching is mutual unique nearest world
centre within an explicit speed gate and ordinal gap. No GT identity, scores,
future rows, cross-source rows or appearance are consumed. Unavailable velocities
remain NaN with a mask until a separately train-fitted initial model is admitted.
"""
from copy import deepcopy
import numpy as np

SIDES = ('vehicle-side', 'infrastructure-side')
FIELDS = {'sequence_id', 'side', 'frame_ordinal', 'source_to_world',
          'states_source_legacy7', 'class_indices', 'row_sha256'}


def validate(frame):
    if set(frame) != FIELDS:
        raise ValueError('exact GT-free source detector fields required')
    if (not isinstance(frame['sequence_id'], str) or not frame['sequence_id']
            or frame['side'] not in SIDES or type(frame['frame_ordinal']) is not int
            or frame['frame_ordinal'] < 0):
        raise ValueError('source identity and nominal ordinal required')
    sha = frame['row_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('bound original row SHA required')
    s = np.asarray(frame['states_source_legacy7'])
    c = np.asarray(frame['class_indices'])
    p = np.asarray(frame['source_to_world'], dtype=np.float64)
    if (s.ndim != 2 or s.shape[1] != 7 or s.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(s).all() or np.any(s[:, 3:6] <= 0)
            or c.shape != (len(s),) or c.dtype.kind not in 'iu' or np.any(c < 0)):
        raise ValueError('finite detector boxes and integer classes required')
    if (p.shape != (4, 4) or not np.isfinite(p).all() or not np.array_equal(p[3], [0, 0, 0, 1])
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=2e-5, rtol=0)):
        raise ValueError('proper column-vector source pose required')
    return s, c, p


def estimate(current, previous, *, max_speed_mps, max_gap_ordinals):
    """Return candidate-order velocity/mask/predecessor evidence without GT.

    Gates have no defaults and must later be chosen on train development only.
    Tie rejection is exact: equal nearest distances are ambiguous, not silently
    tie-broken. The gate is applied before mutual nearest matching. A valid zero
    denotes actual matched static world displacement; invalid NaN is distinct.
    """
    if (isinstance(max_speed_mps, bool) or not np.isscalar(max_speed_mps)
            or not np.isfinite(max_speed_mps) or max_speed_mps <= 0
            or type(max_gap_ordinals) is not int or max_gap_ordinals <= 0):
        raise ValueError('explicit finite positive speed and integer gap gates required')
    state, classes, pose = validate(current)
    n = len(state)
    velocity = np.full((n, 2), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    predecessor = np.full(n, -1, dtype=np.int64)
    reasons = ['no_predecessor'] * n
    dependencies = [current['row_sha256']]
    if previous is not None:
        old, old_classes, old_pose = validate(previous)
        if (current['sequence_id'], current['side']) != (previous['sequence_id'], previous['side']):
            raise ValueError('cross-source/sequence predecessor forbidden')
        gap = current['frame_ordinal'] - previous['frame_ordinal']
        if gap <= 0:
            raise ValueError('duplicate or future predecessor forbidden')
        dependencies.append(previous['row_sha256'])
        reasons = ['gap_exceeded' if gap > max_gap_ordinals else 'no_allowed_predecessor'] * n
        if gap <= max_gap_ordinals and n and len(old):
            dt = gap / 10.0  # frozen nominal clock, never measured timestamps
            world = state[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
            old_world = old[:, :3].astype(np.float64) @ old_pose[:3, :3].T + old_pose[:3, 3]
            delta = world[:, None, :] - old_world[None, :, :]
            distance = np.sum(delta * delta, axis=2)
            allowed = (classes[:, None] == old_classes[None, :]) & (distance <= (max_speed_mps * dt)**2)
            distance = np.where(allowed, distance, np.inf)
            row_min, col_min = distance.min(axis=1), distance.min(axis=0)
            row_unique = np.sum(distance == row_min[:, None], axis=1) == 1
            col_unique = np.sum(distance == col_min[None, :], axis=0) == 1
            for i in range(n):
                if not np.isfinite(row_min[i]):
                    continue
                j = int(distance[i].argmin())
                if not row_unique[i] or not col_unique[j]:
                    reasons[i] = 'ambiguous_nearest'
                elif distance[i, j] != col_min[j]:
                    reasons[i] = 'not_mutual_nearest'
                else:
                    velocity[i] = (delta[i, j] / dt @ pose[:3, :3])[:2]
                    valid[i], predecessor[i], reasons[i] = True, j, 'matched'
    return dict(velocity_source=velocity, velocity_valid=valid,
                predecessor_candidates=predecessor, reasons=reasons,
                dependencies=dependencies, clock='nominal_10hz',
                initial_state_model_applied=False, covariance_fitted=False)


def stream(frames, *, max_speed_mps, max_gap_ordinals):
    """Adjacent source rows only; an empty row breaks detection predecessors."""
    history = {}
    for frame in frames:
        validate(frame)
        key = frame['sequence_id'], frame['side']
        result = estimate(frame, history.get(key), max_speed_mps=max_speed_mps,
                          max_gap_ordinals=max_gap_ordinals)
        history[key] = deepcopy(frame)
        yield result
