"""Independent NumPy float64 head/appearance reference, without model or GT imports.

This prepares numerical checking; it does not certify a network forward or a
cloud experiment. Candidate ordering uses verified observed float32 scores so
the frozen stable-tie rule is tested separately from arithmetic tolerance.
"""
import numpy as np

ATOL = RTOL = 1e-4


def decode_heads(psm, regression, anchors):
    p, r, a = (np.asarray(x, dtype=np.float64) for x in (psm, regression, anchors))
    if (p.ndim != 4 or p.shape[0] != 1 or min(p.shape[1:]) < 1
            or r.shape != (1, 7*p.shape[1], *p.shape[2:])
            or a.shape != (*p.shape[2:], p.shape[1], 7)
            or not all(np.isfinite(x).all() for x in (p, r, a))
            or np.any(a[..., 3:6] <= 0)):
        raise ValueError('invalid native head or anchor shape/value')
    a = a.reshape(-1, 7)
    d = r.transpose(0, 2, 3, 1).reshape(-1, 7)
    boxes = np.empty_like(a)
    diagonal = np.hypot(a[:, 4], a[:, 5])
    boxes[:, :2] = d[:, :2]*diagonal[:, None]+a[:, :2]
    boxes[:, 2] = d[:, 2]*a[:, 3]+a[:, 2]
    with np.errstate(over='raise', invalid='raise'):
        boxes[:, 3:6] = np.exp(d[:, 3:6])*a[:, 3:6]
    boxes[:, 6] = d[:, 6]+a[:, 6]
    z = p.transpose(0, 2, 3, 1).reshape(-1)
    scores = np.empty_like(z)
    positive = z >= 0
    scores[positive] = 1/(1+np.exp(-z[positive]))
    e = np.exp(z[~positive]); scores[~positive] = e/(1+e)
    if not np.isfinite(boxes).all():
        raise ValueError('nonfinite decoded native box')
    return boxes, scores


def stable_observed_top64(observed_scores, reference_scores):
    observed, reference = (np.asarray(x) for x in (observed_scores, reference_scores))
    if (observed.dtype != np.float32 or observed.ndim != 1
            or reference.shape != observed.shape or not np.isfinite(observed).all()
            or not np.isfinite(reference).all() or np.any((observed < 0) | (observed > 1))):
        raise ValueError('explicit finite observed float32 anchor scores required')
    np.testing.assert_allclose(observed, reference, atol=ATOL, rtol=RTOL)
    order = np.argsort(-observed, kind='stable')
    return order[observed[order] >= .05][:64]


def bev_features(bev, boxes, lidar_range):
    b, boxes, bounds = (np.asarray(x, dtype=np.float64) for x in (bev, boxes, lidar_range))
    if (b.ndim != 4 or b.shape[0] != 1 or min(b.shape[1:]) < 1
            or boxes.ndim != 2 or boxes.shape[1] != 7 or bounds.shape != (6,)
            or not all(np.isfinite(x).all() for x in (b, boxes, bounds))
            or np.any(bounds[3:]-bounds[:3] <= 0)):
        raise ValueError('invalid BEV, boxes or bounds')
    n, channels, height, width = len(boxes), *b.shape[1:]
    if not n:
        return np.empty((0, 128), dtype=np.float64), np.empty(0, dtype=bool)
    fractions = (boxes[:, :2]-bounds[:2])/(bounds[3:5]-bounds[:2])
    valid = ((fractions >= 0) & (fractions <= 1)).all(1)
    sampled = np.zeros((n, channels), dtype=np.float64)
    # align_corners=False and zero padding, including half-pixels at bounds.
    for i in np.flatnonzero(valid):
        x, y = fractions[i]*[width, height]-.5
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        for yy in (y0, y0+1):
            for xx in (x0, x0+1):
                if 0 <= xx < width and 0 <= yy < height:
                    sampled[i] += b[0, :, yy, xx]*(1-abs(x-xx))*(1-abs(y-yy))
    pooled = np.empty((n, 128), dtype=np.float64)
    for k in range(128):
        lo, hi = k*channels//128, ((k+1)*channels+127)//128
        pooled[:, k] = sampled[:, lo:hi].mean(1)
    norms = np.linalg.norm(pooled, axis=1)
    valid &= norms > 1e-12
    output = np.zeros_like(pooled)
    output[valid] = pooled[valid]/norms[valid, None]
    return output, valid


def legacy_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all():
        raise ValueError('finite seven-state boxes required')
    out = boxes[:, [0, 1, 2, 4, 5, 3, 6]].copy()
    out[:, 6] = (-boxes[:, 6]-np.pi/2+np.pi) % (2*np.pi)-np.pi
    return out


def verify_raw_row(*, psm, regression, anchors, bev=None, lidar_range,
                   observed_anchor_scores, states_legacy7, raw_scores,
                   appearance, appearance_valid, class_indices,
                   bev_shape=None, bev_cell_indices=None, bev_cell_values=None):
    """Check a captured row; caller must independently bind its source/frame bytes.

    Full observed anchor scores are mandatory: selected-only outputs cannot
    establish the score threshold, stable ordering or all-anchor competition.
    This verifies post-head math only, not the network that produced the heads.
    """
    boxes, mathematical_scores = decode_heads(psm, regression, anchors)
    selected = stable_observed_top64(observed_anchor_scores, mathematical_scores)
    compact=(bev_shape,bev_cell_indices,bev_cell_values)
    if bev is not None and all(x is None for x in compact):
        features, valid = bev_features(bev, boxes[selected], lidar_range)
    elif bev is None and all(x is not None for x in compact):
        from .v2v4real_compact_bev_trace import compact_features
        features, valid = compact_features(boxes=boxes[selected],lidar_range=lidar_range,
            bev_shape=bev_shape,bev_cell_indices=bev_cell_indices,bev_cell_values=bev_cell_values)
    else:
        raise ValueError('exactly one full or lossless sampling-cell BEV layout required')
    n = len(selected)
    observed = [np.asarray(x) for x in (states_legacy7, raw_scores, appearance,
                                      appearance_valid, class_indices)]
    if ([x.shape for x in observed] != [(n,7),(n,),(n,128),(n,),(n,)]
            or observed[3].dtype != np.bool_ or observed[4].dtype.kind not in 'iu'
            or np.any(observed[4] != 0)):
        raise ValueError('explicit original one-vehicle-class raw row required')
    np.testing.assert_allclose(observed[0], legacy_boxes(boxes[selected]), atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(observed[1], np.asarray(observed_anchor_scores)[selected])
    np.testing.assert_array_equal(observed[3], valid)
    np.testing.assert_allclose(observed[2], features, atol=ATOL, rtol=RTOL)
    return dict(candidates=n,selected_anchor_indices=selected.tolist(),
                independent_post_head_math_verified=True,network_forward_verified=False,
                nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)
