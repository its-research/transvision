"""Train-only motion selection and full residual second moments.

This recipe uses the frozen calibration groups, leaving each group out in turn.
The initial velocity is a population prior, never a measured zero. Full noise
matrices describe uniquely matched training detections with past GT velocity;
they do not assert physical identity correctness or calibrated FP uncertainty.
"""
import numpy as np

from tools.event_track_v2x.v2v4real_causal_detector_motion import stream

SIDES = ('vehicle-side', 'infrastructure-side')
RECIPE = dict(
    kind='v2v4real_train_group_motion_recipe_v1', split='train',
    official_test_used=False, role='calibration_fit',
    speed_candidates_mps=[5., 10., 20., 40.], gap_candidates_ordinals=[1, 3, 10],
    selection='leave_one_frozen_calibration_name_group_out_mean_seed_group_velocity_MSE',
    selection_tie='smaller_speed_then_smaller_gap',
    offline_match='stable_descending_raw_score_greedy_unique_unused_GT_XY_strict_lt_2m',
    initial_policy='per_seed_side_mean_of_fit_group_valid_past_GT_velocity_population_prior',
    covariance_policy='uncentered_full9_residual_second_moment_0.05_diagonal_shrinkage_plus_1e-6_identity',
    residual_layout='xyz_width_length_height_legacy_yaw_vx_vy',
    yaw_policy='wrap_to_minus_pi_inclusive_pi_exclusive',
    minimum_prior_samples=10, minimum_covariance_samples_per_side_status=10,
    shrinkage=.05, variance_floor=1e-6, atol=1e-4, rtol=1e-4,
    first_observation_transfer='same_unavailable_population_model_no_future_first_frame_target',
    false_positive_noise='conditional_true_positive_residual_model_no_FP_calibration_claim',
)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def legacy_targets(states_lwh, velocities):
    s = np.asarray(states_lwh, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 7 or v.shape != (len(s), 2):
        raise ValueError('source lwh/yaw geometry and source velocity required')
    result = np.empty((len(s), 9), dtype=np.float64)
    result[:, :3] = s[:, :3]
    result[:, 3:6] = s[:, [4, 3, 5]]
    result[:, 6] = wrap(-s[:, 6] - np.pi / 2)
    result[:, 7:] = v
    return result


def offline_matches(boxes, scores, gt_boxes):
    """Offline training assignment only; never called by online motion."""
    boxes, scores, gt = map(np.asarray, (boxes, scores, gt_boxes))
    if boxes.shape != (len(scores), 7) or gt.ndim != 2 or gt.shape[1] != 7:
        raise ValueError('box and score shapes differ')
    out = np.full(len(boxes), -1, dtype=np.int64)
    free = np.ones(len(gt), dtype=bool)
    for i in np.argsort(-scores, kind='stable'):
        distance = np.sum((gt[:, :2] - boxes[i, :2]) ** 2, axis=1)
        allowed = free & (distance < 4.)
        if not allowed.any():
            continue
        closest = np.min(distance[allowed])
        js = np.flatnonzero(allowed & (distance == closest))
        if len(js) == 1:
            out[i] = js[0]
            free[js[0]] = False
    return out


def prior(target_rows, groups):
    result = {}
    for side in SIDES:
        v = [r['velocity_source'][r['velocity_valid']] for r in target_rows
             if r['side'] == side and r['group'] in groups]
        a = np.concatenate(v) if v else np.empty((0, 2))
        if len(a) < RECIPE['minimum_prior_samples'] or not np.isfinite(a).all():
            raise ValueError('insufficient finite train-only initial prior samples')
        result[side] = dict(velocity_source=np.mean(a, axis=0).tolist(), samples=len(a))
    return result


def eligible_indices(row):
    ids = row['gt_matches']
    good = ids >= 0
    good[good] &= row['target_velocity_valid'][ids[good]]
    return np.flatnonzero(good)


def velocities(rows, speed, gap):
    frames = [{k: r[k] for k in ('sequence_id', 'side', 'frame_ordinal', 'source_to_world',
              'states_source_legacy7', 'class_indices', 'row_sha256')} for r in rows]
    return list(stream(frames, max_speed_mps=speed, max_gap_ordinals=gap))


def select_gate(seed_rows, target_rows, groups, *, progress=lambda _: None):
    groups = sorted(set(groups))
    if len(groups) < 2:
        raise ValueError('at least two frozen train calibration groups required')
    if (set(r['group'] for r in target_rows) != set(groups)
            or any(set(r['group'] for r in rows) != set(groups) for rows in seed_rows.values())):
        raise ValueError('exact frozen calibration group coverage required')
    priors = {g: prior(target_rows, set(groups) - {g}) for g in groups}
    scores = []
    for speed in RECIPE['speed_candidates_mps']:
        for gap in RECIPE['gap_candidates_ordinals']:
            folds = []
            for seed, rows in sorted(seed_rows.items()):
                predicted = velocities(rows, speed, gap)
                for group in groups:
                    loss, count = 0., 0
                    for row, motion in zip(rows, predicted):
                        if row['group'] != group:
                            continue
                        ids = eligible_indices(row)
                        estimate = motion['velocity_source'][ids].copy()
                        valid = motion['velocity_valid'][ids]
                        estimate[~valid] = priors[group][row['side']]['velocity_source']
                        truth = row['target_velocity_source'][row['gt_matches'][ids]]
                        loss += float(np.sum((estimate - truth) ** 2))
                        count += len(ids)
                    if count == 0:
                        raise ValueError('empty held-out velocity scoring scope')
                    folds.append(dict(seed=seed, group=group, samples=count,
                                      sum_squared_vector_error=loss, vector_MSE=loss/count))
            item = dict(max_speed_mps=speed, max_gap_ordinals=gap, folds=folds,
                        mean_seed_group_vector_MSE=float(np.mean([f['vector_MSE'] for f in folds])))
            scores.append(item)
            progress(item)
    selected = min(scores, key=lambda x: (x['mean_seed_group_vector_MSE'],
                   x['max_speed_mps'], x['max_gap_ordinals']))
    return dict(selected={k: selected[k] for k in ('max_speed_mps', 'max_gap_ordinals')},
                scores=scores, fold_priors=priors, groups=groups, official_test_used=False)


def residuals(row, motion, initial):
    ids = eligible_indices(row)
    state = np.concatenate((row['states_source_legacy7'][ids],
                            motion['velocity_source'][ids]), axis=1)
    valid = motion['velocity_valid'][ids]
    state[~valid, 7:] = initial[row['side']]['velocity_source']
    gt = row['gt_matches'][ids]
    truth = legacy_targets(row['target_states_source'][gt], row['target_velocity_source'][gt])
    error = state - truth
    error[:, 6] = wrap(error[:, 6])
    if not np.isfinite(error).all():
        raise ValueError('finite full9 paired residuals required')
    return ids, valid, error


def fit_model(rows, target_rows, groups, selected):
    if (set(r['group'] for r in rows) != set(groups)
            or set(r['group'] for r in target_rows) != set(groups)):
        raise ValueError('fit must contain exactly the declared training groups')
    initial = prior(target_rows, set(groups))
    motion = velocities(rows, **dict(speed=selected['max_speed_mps'], gap=selected['max_gap_ordinals']))
    samples = {(side, status): [] for side in SIDES for status in ('matched', 'unavailable')}
    for row, pred in zip(rows, motion):
        _, valid, error = residuals(row, pred, initial)
        samples[row['side'], 'matched'].append(error[valid])
        samples[row['side'], 'unavailable'].append(error[~valid])
    models = {}
    for side in SIDES:
        models[side] = dict(initial_prior=initial[side], statuses={})
        for status in ('matched', 'unavailable'):
            a = np.concatenate(samples[side, status])
            if len(a) < RECIPE['minimum_covariance_samples_per_side_status']:
                raise ValueError('insufficient actual full9 residual samples for '+side+'/'+status)
            second = a.T @ a / len(a)
            cov = ((1-RECIPE['shrinkage'])*second + RECIPE['shrinkage']*np.diag(np.diag(second))
                   + RECIPE['variance_floor']*np.eye(9))
            np.linalg.cholesky(cov)
            models[side]['statuses'][status] = dict(samples=len(a),
                residual_mean=a.mean(axis=0).tolist(), residual_second_moment=second.tolist(),
                covariance=cov.tolist(), minimum_eigenvalue=float(np.linalg.eigvalsh(cov)[0]))
    return dict(kind='v2v4real_train_fitted_causal_detector_motion_model_v1',
                gates=selected, sides=models, recipe=RECIPE), motion


def apply_model(row, motion, model):
    """No targets/scores/IDs used. Unavailable flag remains explicit evidence."""
    side = model['sides'][row['side']]
    valid = motion['velocity_valid']
    velocity = np.array(motion['velocity_source'], copy=True)
    velocity[~valid] = side['initial_prior']['velocity_source']
    cov = np.empty((len(valid), 9, 9), dtype=np.float64)
    for status, mask in (('matched', valid), ('unavailable', ~valid)):
        cov[mask] = side['statuses'][status]['covariance']
    return dict(velocity_source=velocity, velocity_measured_valid=valid.copy(),
                covariances=cov, initial_prior_applied=~valid)
