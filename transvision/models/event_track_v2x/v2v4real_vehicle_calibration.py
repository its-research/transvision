"""Explicit native-vehicle train-only V2V4Real existence calibration.

This separately named entry reuses the unchanged 2 m training-only matching
rule and optimizer, not the old strict-Car targets or calibration receipt.

The prediction archive consumed here is GT-free and already expressed in the
current ego LiDAR frame using public poses.  Ground truth is opened only by
this offline calibration stage and never copied into a detection cache.
"""
from __future__ import annotations

from collections import Counter
import re
from dataclasses import asdict

import numpy as np

from .paper_calibration import binary_diagnostics, fit_existence
from .paper_protocol import PaperProtocol
from .prediction_features import match_predictions
from .paper_evaluation_policy import VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, native_vehicle, vehicle_binding
from .v2v4real_ground_truth import VEHICLE_GT_RECIPE


PREDICTION_KIND = "v2v4real_calibration_predictions_v1"
PREDICTION_SHARD_KIND = "v2v4real_calibration_prediction_shard_v1"
CALIBRATION_KIND = "v2v4real_native_vehicle_existence_calibration_v1"
MATCHING_RULE = "raw_score_order_same_class_xy_strict_lt_2m_one_to_one_per_source_frame"
DETECTOR_SEMANTIC_SOURCE_PINS = {
    'materializer': 'aae761c404873ad400c05d511aa760aa33bf9b3b8ce19cec3206bd7cd4c4d7f9',
    'box_utils': '961914a1291f57e86d9a7416e7318a3e911f0de8fb98a0c4492e2a9d5785de86',
    'base_postprocessor': '3ad1d15e410965558661a52eea229b148e9c8a175e3d829154c8e7de02b61089',
}


def _gt_centers(frame):
    states = []
    for row in frame["objects"]:
        corners = np.asarray(row["corners_ego"], dtype=np.float64)
        if corners.shape != (8, 3) or not np.isfinite(corners).all() or row.get("evaluation_class") != "vehicle" or not native_vehicle(row.get("raw_class")):
            raise ValueError("finite native vehicle GT corners and explicit obj_type required")
        state = np.zeros(7, dtype=np.float64)
        state[:3] = corners.mean(axis=0)
        states.append(state)
    return np.asarray(states, dtype=np.float64).reshape(-1, 7)


def fit_vehicle_existence(prediction_manifest, prediction_rows, gt_manifest, gt_frames,
                         partition, *, prediction_sha256, gt_manifest_sha256,
                         partition_sha256, detector_semantics, detector_semantics_sha256, progress=lambda value: None):
    """Fit one monotone calibration from exactly the declared calibration role."""
    for digest in (prediction_sha256, gt_manifest_sha256, partition_sha256, detector_semantics_sha256):
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('explicit SHA-256 input bindings required')
    if (detector_semantics.get('kind') != 'v2v4real_native_vehicle_detector_semantics_v1'
            or detector_semantics.get('native_label_source') != NATIVE_VEHICLE_SELECTION
            or detector_semantics.get('numeric_class_index') != 0
            or type(detector_semantics.get('numeric_class_index')) is not int
            or detector_semantics.get('checkpoint_sha256') != prediction_manifest.get('checkpoint_sha256')
            or detector_semantics.get('source_sha256') != DETECTOR_SEMANTIC_SOURCE_PINS):
        raise ValueError('bound original single-class native vehicle detector evidence required')
    if (prediction_manifest.get("kind") != PREDICTION_KIND
            or prediction_manifest.get("protocol_id") != "v2v4real-nominal-10hz-formal-v1"
            or prediction_manifest.get("split") != "train"
            or prediction_manifest.get("gt_read") is not False
            or prediction_manifest.get("official_test_read") is not False
            or prediction_manifest.get("partition_sha256") != partition_sha256
            or prediction_manifest.get("candidate_protocol") != "rbf-all-class-top64-v1"):
        raise ValueError("formal GT-free calibration prediction contract required")
    if (gt_manifest.get("kind") != "v2v4real_native_vehicle_gt_projection_v1"
            or gt_manifest.get("recipe") != VEHICLE_GT_RECIPE
            or gt_manifest.get("evaluation_protocol") != VEHICLE_PROTOCOL
            or gt_manifest.get("native_label_source") != NATIVE_VEHICLE_SELECTION
            or gt_manifest.get("test_payloads_read") is not False
            or gt_manifest.get("split") != "train"
            or gt_manifest.get("class_scope") != ["vehicle"]
            or gt_manifest.get("inference_input") is not False
            or gt_manifest.get("GT_read") is not True):
        raise ValueError("separate native vehicle train GT projection required")
    if (partition.get("kind") != "v2v4real_formal_nested_training_partition_v1"
            or partition.get("protocol_id") != prediction_manifest["protocol_id"]
            or partition.get("official_split") != "train"
            or partition.get("official_test_used_for_partition") is not False
            or partition.get("gt_used_for_partition") is not False):
        raise ValueError("formal nested partition required")
    groups = tuple(sorted(partition.get("calibration_fit", ())))
    mapping = partition.get("sequence_to_name_group")
    role_groups = [set(partition.get(k, ())) for k in ("detector_fit", "calibration_fit", "identity_selection")]
    if any(role_groups[i] & role_groups[j] for i in range(3) for j in range(i)):
        raise ValueError("detector/calibration/identity role groups must be disjoint")
    if not groups or not isinstance(mapping, dict):
        raise ValueError("nonempty calibration groups required")
    sequences = {sequence for sequence, group in mapping.items() if group in groups}
    if not sequences or set(prediction_manifest.get("sequence_ids", ())) != sequences:
        raise ValueError("prediction archive must cover exactly calibration_fit sequences")

    gt_by_key = {(row["sequence_id"], row["frame_key"]): row for row in gt_frames
                 if row["sequence_id"] in sequences}
    if len(gt_by_key) != sum(row["sequence_id"] in sequences for row in gt_frames):
        raise ValueError("duplicate calibration GT frame")
    rows_by_key = {}
    for row in prediction_rows:
        key = row.get("sequence_id"), row.get("frame_id")
        if key not in gt_by_key:
            raise ValueError("prediction outside calibration_fit GT coverage")
        side = row.get("side")
        if side not in {"vehicle-side", "infrastructure-side"} or (key, side) in rows_by_key:
            raise ValueError("one prediction row per side and frame required")
        rows_by_key[key, side] = row
    expected = {(key, side) for key in gt_by_key
                for side in ("vehicle-side", "infrastructure-side")}
    if set(rows_by_key) != expected:
        raise ValueError("complete two-side calibration prediction coverage required")

    raw_scores, targets, sample_groups = [], [], []
    side_counts, positives = Counter(), Counter()
    for completed, key in enumerate(sorted(gt_by_key), 1):
        gt_state = _gt_centers(gt_by_key[key])
        gt_classes = np.zeros(len(gt_state), dtype=np.int64)
        for side in ("vehicle-side", "infrastructure-side"):
            row = rows_by_key[key, side]
            states = np.asarray(row.get("states_ego"), dtype=np.float64)
            if states.size == 0:
                states = states.reshape(0, 9)
            scores = np.asarray(row.get("raw_scores"), dtype=np.float64)
            classes = np.asarray(row.get("class_indices"))
            if (states.ndim != 2 or states.shape[1] != 9 or scores.ndim != 1 or classes.ndim != 1
                    or (len(classes) and not np.issubdtype(classes.dtype, np.integer))
                    or np.any(states[:, 3:6] <= 0) or len(states) != len(scores)
                    or len(states) != len(classes) or not np.isfinite(states).all()
                    or not np.isfinite(scores).all() or np.any((scores < .05) | (scores > 1))
                    or len(states) > 64 or np.any(classes != 0)):
                raise ValueError("formal raw top64 vehicle predictions required")
            matches = match_predictions(states, scores, classes, gt_state, gt_classes, distance=2.)
            labels = matches >= 0
            group = mapping[key[0]]
            raw_scores.extend(scores.tolist())
            targets.extend(labels.astype(np.int64).tolist())
            sample_groups.extend([group] * len(scores))
            side_counts[side] += len(scores)
            positives[side] += int(labels.sum())
        progress(dict(completed_rows=completed * 2, total_rows=len(gt_by_key) * 2))
    if not raw_scores or len(set(targets)) != 2:
        raise ValueError("calibration requires positive and negative prediction events")
    model = fit_existence(raw_scores, targets, protocol=PaperProtocol("v2v4real", "train", evaluation_class="vehicle"),
                          groups=sample_groups, fit_groups=groups)
    calibrated = model.apply(raw_scores)
    return asdict(model), dict(
        kind=CALIBRATION_KIND,
        protocol_id=VEHICLE_PROTOCOL,
        source_prediction_protocol_id=prediction_manifest["protocol_id"],
        evaluation_class_binding=vehicle_binding(),
        detector_semantics_sha256=detector_semantics_sha256,
        checkpoint_seed=prediction_manifest.get("checkpoint_seed"),
        checkpoint_sha256=prediction_manifest.get("checkpoint_sha256"),
        prediction_sha256=prediction_sha256,
        gt_manifest_sha256=gt_manifest_sha256,
        partition_sha256=partition_sha256,
        fit_split="train",
        fit_groups=list(groups),
        fit_sequences=sorted(sequences),
        matching_rule=MATCHING_RULE,
        target_interpretation="true_positive_indicator_under_current_2m_matching_protocol",
        examples=len(raw_scores),
        positives=int(sum(targets)),
        examples_by_side=dict(side_counts),
        positives_by_side=dict(positives),
        raw_diagnostics=binary_diagnostics(raw_scores, targets),
        calibrated_diagnostics=binary_diagnostics(calibrated, targets),
        official_test_used=False,
        gt_used_only_for_offline_calibration=True,
        gt_written_to_calibration_artifact=False,
        paper_metric=False,
        independent_acceptance_verified=False,
        native_nine_state_cache_admitted=False,
        velocity_covariance_fitted=False,
    )
