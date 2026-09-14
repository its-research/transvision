"""Prediction-only features, supervised matching, and train-fit calibration.

GT is accepted ONLY by matching/calibration functions. Feature construction has
no label/identity argument. State dimensions follow the frozen MMDet box axes,
not an assumed physical length/width convention.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

CLASSES = ("car", "bicycle", "pedestrian")
RAW_KEYS = {"boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy", "gravity_centers_lidar", "scores",
            "class_indices", "appearance_128", "appearance_valid", "image_rois_xyxy"}
STATE_FIELDS = ["gravity_x", "gravity_y", "gravity_z", "dim_x", "dim_y", "dim_z", "yaw", "vx", "vy"]
FEATURE_GROUPS = [("geometry", 8), ("motion", 2), ("appearance", 128), ("coarse_class", 3),
                  ("source_time", 4), ("covariance_upper", 45), ("source_pose", 6),
                  ("lineage", 4), ("scores_and_appearance_valid", 3)]
FEATURE_DIM = sum(x[1] for x in FEATURE_GROUPS)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("xb") as f:
        f.write(canonical(value))


def regular_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != relative:
        raise ValueError("unsafe or noncanonical input path")
    result = Path(root) / relative
    if not result.is_file() or result.is_symlink():
        raise ValueError("input must be a regular contained file")
    try:
        result.resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise ValueError("input escapes its root")
    return result


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def raw_state(arrays):
    if set(arrays) != RAW_KEYS:
        raise ValueError("raw cache has missing/extra fields")
    state = np.asarray(arrays["boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy"], dtype=np.float64).copy()
    center = arrays["gravity_centers_lidar"]
    if state.ndim != 2 or state.shape[1] != 9 or center.shape != (len(state), 3):
        raise ValueError("invalid predicted state shape")
    expected = state[:, :3].copy()
    expected[:, 2] += state[:, 5] / 2
    if not np.allclose(center, expected, atol=1e-4, rtol=1e-5):
        raise ValueError("gravity/bottom-center box convention differs")
    state[:, :3] = center
    state[:, 6] = wrap_angle(state[:, 6])
    if not np.isfinite(state).all() or np.any(state[:, 3:6] <= 0):
        raise ValueError("invalid predicted state")
    return state


def match_predictions(state, scores, classes, gt_state, gt_classes, distance=2.0):
    """Stable score-ordered same-class XY matching; each GT used at most once."""
    n = len(state)
    if len(scores) != n or len(classes) != n or len(gt_classes) != len(gt_state):
        raise ValueError("matching shapes differ")
    if distance <= 0 or not all(np.isfinite(x).all() for x in [state, scores, gt_state[:, :7]]):
        raise ValueError("invalid matching values")
    matches = np.full(n, -1, dtype=np.int64)
    used = set()
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        candidates = np.flatnonzero(np.asarray(gt_classes) == classes[i])
        candidates = [j for j in candidates if int(j) not in used]
        if not candidates:
            continue
        d = np.linalg.norm(gt_state[candidates, :2] - state[i, :2], axis=1)
        k = int(np.argmin(d))
        if d[k] < distance:
            j = int(candidates[k])
            matches[i] = j
            used.add(j)
    return matches


def fit_score(scores, targets, config):
    from scipy.optimize import minimize
    from scipy.special import expit
    scores, targets = np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if not len(scores) or scores.shape != targets.shape or np.any((targets != 0) & (targets != 1)):
        raise ValueError("invalid calibration examples")
    eps = config["logit_clip"]
    p = np.clip(scores, eps, 1 - eps)
    x = np.log(p / (1 - p))
    prior = (targets.sum() + 1) / (len(targets) + 2)
    def objective(theta):
        z = theta[0] * x + theta[1]
        residual = expit(z) - targets
        loss = np.mean(np.logaddexp(0, z) - targets * z) + config["l2"] * theta[0]**2 / 2
        grad = np.array([np.mean(residual * x) + config["l2"] * theta[0], residual.mean()])
        return loss, grad
    result = minimize(objective, [0.0, np.log(prior / (1 - prior))], jac=True,
                      method="L-BFGS-B", bounds=[(0.0, 20.0), (-30.0, 30.0)],
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("score calibration did not converge: " + str(result.message))
    calibrated = expit(result.x[0] * x + result.x[1])
    return {"slope": float(result.x[0]), "intercept": float(result.x[1]),
            "examples": len(scores), "positives": int(targets.sum()),
            "in_sample_brier_raw": float(np.mean((scores - targets)**2)),
            "in_sample_brier_calibrated": float(np.mean((calibrated - targets)**2)),
            "optimizer_converged": True, "logit_clip": eps}


def apply_score(scores, model):
    p = np.clip(scores, model["logit_clip"], 1 - model["logit_clip"])
    z = np.clip(model["slope"] * np.log(p / (1 - p)) + model["intercept"], -80, 80)
    return 1 / (1 + np.exp(-z))


def fit_covariance(residuals, config):
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 9 or not len(r) or not np.isfinite(r).all():
        raise ValueError("invalid covariance residuals")
    r = r.copy()
    r[:, 6] = wrap_angle(r[:, 6])
    second = r.T @ r / len(r)
    shrink = config["shrinkage"]
    cov = (1 - shrink) * second + shrink * np.diag(np.diag(second)) + np.diag(np.square(config["floor_std"]))
    np.linalg.cholesky(cov)
    return {"matrix": cov.tolist(), "samples": len(r), "mean_residual_not_corrected": r.mean(0).tolist(),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(cov).min()),
            "interpretation": "conditional_on_2m_true_positive; includes_bias_second_moment; not_false_positive_uncertainty"}


def choose_candidates(scores, config):
    order = np.argsort(-np.asarray(scores), kind="stable")
    return order[scores[order] >= config["minimum_raw_score"]][:config["maximum_per_side"]]


def validate_pose(meta):
    r = np.asarray(meta["lidar_to_world_row_rotation"], dtype=np.float64)
    t = np.asarray(meta["lidar_to_world_translation"], dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("invalid pose")
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(r), 1, atol=2e-5):
        raise ValueError("pose is not a proper rotation")
    return r, t


def transform_state_covariance(state, covariance, source_meta, reference_meta):
    sr, st = validate_pose(source_meta)
    rr, rt = validate_pose(reference_meta)
    r = sr @ rr.T  # row-vector source -> reference
    t = (st - rt) @ rr.T
    out = state.copy()
    out[:, :3] = state[:, :3] @ r + t
    headings = np.stack([np.cos(state[:, 6]), np.sin(state[:, 6]), np.zeros(len(state))], 1) @ r
    derivatives = np.stack([-np.sin(state[:, 6]), np.cos(state[:, 6]), np.zeros(len(state))], 1) @ r
    norm2 = (headings[:, :2]**2).sum(1)
    if np.any(norm2 < 1e-8):
        raise ValueError("upright heading is undefined after pose rotation")
    out[:, 6] = np.arctan2(headings[:, 1], headings[:, 0])
    out[:, 7:9] = state[:, 7:9] @ r[:2, :2]
    jac = np.broadcast_to(np.eye(9), (len(state), 9, 9)).copy()
    jac[:, :3, :3] = r.T
    jac[:, 7:9, 7:9] = r[:2, :2].T
    jac[:, 6, 6] = (headings[:, 0] * derivatives[:, 1] - headings[:, 1] * derivatives[:, 0]) / norm2
    cov = jac @ covariance @ jac.transpose(0, 2, 1)
    return out, cov, r, t


def encode_features(state, covariance, arrays, selected, meta, reference_meta, pair_delta_s, origin_us, calibration):
    """Only predicted arrays + public poses/times + frozen fitted parameters."""
    selected = np.asarray(selected, dtype=np.int64)
    s, cov, r, t = transform_state_covariance(state[selected], covariance[selected], meta, reference_meta)
    n = len(s)
    geometry = np.concatenate([s[:, :6] / [100, 100, 20, 20, 20, 10], np.sin(s[:, 6:7]), np.cos(s[:, 6:7])], 1)
    cls = arrays["class_indices"][selected]
    calibrated = np.asarray([apply_score(np.asarray([arrays["scores"][i]], dtype=np.float64), calibration[CLASSES[int(arrays["class_indices"][i])]]["score"])[0] for i in selected])
    time = [(meta["box_reference_timestamp_us"] - origin_us) / 1e8,
            (meta["source_image_timestamp_us"] - meta["box_reference_timestamp_us"]) / 1e6,
            pair_delta_s, float(meta["side"] == "infrastructure-side")]
    # Rotation XYZ angles for the column-vector transform r.T.
    from scipy.spatial.transform import Rotation
    pose = np.r_[t / [100, 100, 20], Rotation.from_matrix(r.T).as_euler("xyz") / np.pi]
    features = np.concatenate([geometry, s[:, 7:9] / 30, arrays["appearance_128"][selected], np.eye(3)[cls],
                               np.tile(time, (n, 1)), cov[:, np.triu_indices(9)[0], np.triu_indices(9)[1]] / 100,
                               np.tile(pose, (n, 1)), np.tile([1, np.log(2) / 8, 0, 0], (n, 1)),
                               np.stack([arrays["scores"][selected], calibrated, arrays["appearance_valid"][selected]], 1)], 1).astype(np.float32)
    if features.shape != (n, FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("feature contract differs")
    return features
