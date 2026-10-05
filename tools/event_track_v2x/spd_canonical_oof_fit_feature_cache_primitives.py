"""GT-free raw detector-cache primitives; not a calibrated DetectionCacheV2."""
import hashlib
import json
from pathlib import Path

import numpy as np

FRAME_ARRAYS = {
    "boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy", "gravity_centers_lidar", "scores",
    "class_indices", "appearance_128", "appearance_valid", "image_rois_xyxy",
}
FEATURE_METHOD = "imagenet-r50-c5-roialign3-mean16x128-l2-v1"
PRETRAINED_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
DETECTOR_DECODE_SOURCE = "raw-head-all-queries-no-roi-no-nms-v1"


def install_raw_detector_decode(model):
    """Decode frozen D2 head tensors before tracking masks overwrite low scores.

    Require the uncropped query decoder installed first. This
    adapter is restricted to single-frame detector-only models; tracking output
    and training are intentionally not modified. Every snapshot is consumed
    exactly once within the same inference frame, including on error paths.
    """
    import torch
    if not getattr(model, "_uncropped_query_decode_installed", False):
        raise ValueError("uncropped query decoder must be installed first")
    if (model.training or model.train_det is not True or model.is_motion
            or model.is_cooperation or model.STReasoner.history_reasoning
            or model.STReasoner.future_reasoning):
        raise ValueError("raw detector decode requires an eval-only D2 model")
    if getattr(model, "_raw_detector_decode_installed", False):
        raise ValueError("raw detector decode already installed")
    original_frame = model._forward_single_frame_inference
    original_cache = model.load_detection_output_into_cache
    original_decode = model._det_instances2results
    state = {"active": None, "last_audit": None, "frames": 0}

    def capture(instances, out):
        current = state["active"]
        if current is None or current["captures"] != 0:
            raise RuntimeError("missing frame or duplicate raw detector capture")
        raw = {key: out[key].detach().clone() for key in ("all_cls_scores", "all_bbox_preds")}
        cls, boxes = raw["all_cls_scores"], raw["all_bbox_preds"]
        if cls.ndim != 3 or boxes.ndim != 3 or cls.shape[:2] != boxes.shape[:2]:
            raise ValueError("unexpected single-frame detection head layout")
        if not all(bool(torch.isfinite(value).all()) for value in raw.values()):
            raise FloatingPointError("non-finite raw detector head output")
        current.update(captures=1, raw=raw)
        return original_cache(instances, out)

    def decode(out, img_metas):
        current = state["active"]
        if (current is None or current["captures"] != 1 or current["decodes"] != 0
                or current["token"] != str(img_metas[0]["sample_idx"])):
            raise RuntimeError("stale, missing or duplicate raw detector decode")
        raw = current.pop("raw")
        def zero_queries(values):
            return int(((values["all_cls_scores"][-1] == 0).all(dim=-1)
                       & (values["all_bbox_preds"][-1] == 0).all(dim=-1)).sum().item())
        audit = {"frame_id": current["token"], "head_queries": int(raw["all_cls_scores"].shape[1]),
                 "raw_zero_queries": zero_queries(raw), "summarized_zero_queries": zero_queries(out),
                 "detector_decode_source": DETECTOR_DECODE_SOURCE}
        current["decodes"] = 1
        result = original_decode(dict(out, **raw), img_metas)
        indices = result["query_indices_det"]
        if (len(indices) != audit["head_queries"]
                or not torch.equal(indices.cpu(), torch.arange(audit["head_queries"]))):
            raise ValueError("raw query coverage/order differs")
        state["last_audit"] = audit
        return result

    def frame(*args, **kwargs):
        if state["active"] is not None or model.training:
            raise RuntimeError("nested frame or training use of raw detector adapter")
        meta = kwargs["img_metas"] if "img_metas" in kwargs else args[1]
        if len(meta) != 1:
            raise ValueError("raw detector adapter requires batch size one")
        state["active"] = {"token": str(meta[0]["sample_idx"]), "captures": 0, "decodes": 0}
        state["last_audit"] = None
        try:
            result = original_frame(*args, **kwargs)
            current = state["active"]
            if current["captures"] != 1 or current["decodes"] != 1 or "raw" in current:
                raise RuntimeError("unconsumed detector output in inference frame")
            state["frames"] += 1
            return result
        finally:
            state["active"] = None

    model.load_detection_output_into_cache = capture
    model._det_instances2results = decode
    model._forward_single_frame_inference = frame
    model._raw_detector_decode_installed = True
    return state


def normalize_visible_features(values, visible):
    """Distinguish unavailable zero vectors from invalid non-finite features."""
    values = np.asarray(values, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool).copy()
    indices = np.flatnonzero(visible)
    if values.shape != (len(indices), 128):
        raise ValueError("appearance shape differs from visible boxes")
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite visible appearance values")
    norms = np.linalg.norm(values, axis=1)
    usable = norms > 1e-12
    embeddings = np.zeros((len(visible), 128), dtype=np.float32)
    embeddings[indices[usable]] = values[usable] / norms[usable, None]
    visible[indices[~usable]] = False
    return embeddings, visible, int((~usable).sum())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def project_corners(corners, lidar2image, image_hw):
    """Project visible cuboid vertices/near-plane intersections; retain every box.

    Missing image support is explicit, not a reason to delete a detection.
    MMDetection3D corner order connects vertices differing in one cube bit.
    """
    corners = np.asarray(corners, dtype=np.float64)
    matrix = np.asarray(lidar2image, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3) or matrix.shape != (4, 4):
        raise ValueError("invalid projection input shape")
    if not np.isfinite(corners).all() or not np.isfinite(matrix).all():
        raise ValueError("nonfinite projection input")
    height, width = map(int, image_hw)
    if min(height, width) <= 0:
        raise ValueError("empty image")
    # Exact MMDetection3D corner ordering: 000,001,011,010,100,101,111,110.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    homogeneous = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=-1)
    projected = homogeneous @ matrix.T
    rois = np.zeros((len(corners), 4), dtype=np.float32)
    valid = np.zeros(len(corners), dtype=bool)
    near = 1e-3
    for index, vertices in enumerate(projected):
        points = [v[:3] for v in vertices if v[2] >= near]
        for first, second in edges:
            a, b = vertices[first], vertices[second]
            if (a[2] < near) != (b[2] < near):
                points.append((a + (b - a) * ((near - a[2]) / (b[2] - a[2])))[:3])
        if not points:
            continue
        p = np.asarray(points)
        xy = p[:, :2] / p[:, 2:3]
        low = np.maximum(xy.min(axis=0), [0, 0])
        high = np.minimum(xy.max(axis=0), [width, height])
        if np.all(high - low >= 1.0):
            rois[index] = [low[0], low[1], high[0], high[1]]
            valid[index] = True
    return rois, valid


def validate_arrays(arrays):
    if set(arrays) != FRAME_ARRAYS:
        raise ValueError("unknown/missing raw prediction fields")
    boxes = arrays["boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy"]
    count = len(boxes)
    shapes = {"boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy": (count, 9),
              "gravity_centers_lidar": (count, 3), "scores": (count,),
              "class_indices": (count,), "appearance_128": (count, 128),
              "appearance_valid": (count,), "image_rois_xyxy": (count, 4)}
    for key, shape in shapes.items():
        value = arrays[key]
        if not isinstance(value, np.ndarray) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid raw prediction array: " + key)
        if value.dtype.hasobject:
            raise ValueError("object arrays are forbidden")
    if arrays["class_indices"].dtype.kind not in "iu" or arrays["appearance_valid"].dtype.kind != "b":
        raise ValueError("invalid labels/visibility dtype")
    if np.any(boxes[:, 3:6] <= 0) or np.any((arrays["scores"] < 0) | (arrays["scores"] > 1)):
        raise ValueError("invalid box sizes or scores")
    if np.any((arrays["class_indices"] < 0) | (arrays["class_indices"] >= 3)):
        raise ValueError("unknown class")
    valid = arrays["appearance_valid"]
    if not np.allclose(np.linalg.norm(arrays["appearance_128"][valid], axis=1), 1, atol=1e-5):
        raise ValueError("visible appearance must have unit norm")
    if np.any(arrays["appearance_128"][~valid] != 0):
        raise ValueError("unavailable appearance must be explicitly zero")
    return count


def verify_cache(root, inputs=None, package_manifest=None):
    """Independent process readback of every NPZ and JSON, with strict inventory."""
    root = Path(root)
    manifest = json.loads((root / "raw-cache-manifest.json").read_bytes())
    if manifest["kind"] != "eventtrack_fit_feature_raw_detector_cache_v1" or manifest["formal_v2_ready"]:
        raise ValueError("unexpected cache contract")
    if manifest.get("detector_decode_source") != DETECTOR_DECODE_SOURCE:
        raise ValueError("cache did not decode unmodified raw detector head outputs")
    if inputs is None or package_manifest is None or manifest.get("canonical_oof_fit_feature_export") is not True:
        raise ValueError("canonical OOF verifier requires bound inputs and fit package")
    from spd_canonical_oof_fit_feature_input_gate import validate_inputs
    package = json.loads(Path(package_manifest).read_bytes())
    admitted = validate_inputs(inputs, package, manifest["input_manifest_sha256"], package_manifest_sha256=digest(Path(package_manifest)))
    binding = manifest["training_binding"]
    if (binding.get("kind") != "spd-canonical-oof-fit-feature-export-binding-v1"
            or binding.get("metadata_binding_accepted") is not True
            or binding.get("package_manifest_sha256") != digest(Path(package_manifest))
            or binding.get("fold_id") != manifest.get("fold_id")
            or binding.get("fold_id") != package["fold_id"]
            or binding.get("seed") != manifest.get("seed")
            or binding.get("side") != manifest.get("side")
            or binding.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
            or binding.get("input_manifest_sha256") != manifest["input_manifest_sha256"]
            or binding.get("fit_sequence_ids") != package["fit_sequence_ids"]
            or binding.get("export_sequence_ids") != admitted["fit_sequence_ids"]
            or binding.get("held_out_sequence_ids") != package["held_out_sequence_ids"]
            or binding.get("held_out_selection_scoring_eligible") is not False
            or manifest.get("held_out_selection_scoring_eligible") is not False
            or manifest.get("shard_count") != 2
            or manifest.get("shard_index") not in (0, 1)
            or manifest["sequences"] != sorted(admitted["fit_sequence_ids"])[manifest["shard_index"]::2]):
        raise ValueError("canonical OOF export identity differs")
    launch_path = root / "launch-receipt.json"
    launch = json.loads(launch_path.read_bytes())
    if any(launch.get(k) != manifest.get(k) for k in launch):
        # Completion changes kind and frame representation only.
        differing = {k for k in launch if launch.get(k) != manifest.get(k)}
        if differing != {"kind", "frames"}:
            raise ValueError("launch/completion provenance differs")
    expected = {"raw-cache-manifest.json", "launch-receipt.json", "resolved-cache-config.py"}
    identities = set()
    expected_rows = None
    if inputs is not None:
        inputs = Path(inputs)
        if digest(inputs / "input-manifest.json") != manifest["input_manifest_sha256"]:
            raise ValueError("input manifest identity differs")
        index = json.loads((inputs / manifest["side"] / "frame-index.json").read_bytes())
        expected_rows = {(manifest["side"], r["sequence_id"], r["frame_id"]): r
                         for r in index if r["sequence_id"] in manifest["sequences"]}
    detections = 0
    for item in manifest["frames"]:
        if item.get("raw_query_count") != item["detections"]:
            raise ValueError("raw query coverage is not complete")
        for key in ["arrays", "metadata"]:
            record = item[key]
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("cache path escape")
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("cache payload differs")
            expected.add(relative.as_posix())
        with np.load(root / item["arrays"]["path"], allow_pickle=False) as raw:
            count = validate_arrays({key: raw[key] for key in raw.files})
        meta = json.loads((root / item["metadata"]["path"]).read_bytes())
        if set(meta) != {"sequence_id", "frame_id", "side", "source_image_timestamp_us",
                        "box_reference_timestamp_us", "lidar_to_world_row_rotation",
                        "lidar_to_world_translation", "coordinate_system", "image_sha256"}:
            raise ValueError("unknown metadata, including possible GT/identity leakage")
        identity = (meta["side"], meta["sequence_id"], meta["frame_id"])
        if identity in identities or count != item["detections"]:
            raise ValueError("duplicate frame or altered detection count")
        if expected_rows is not None:
            row = expected_rows.get(identity)
            if row is None or any(meta[key] != row[key] for key in row):
                raise ValueError("cache frame identity/time/image differs from input")
        rotation = np.asarray(meta["lidar_to_world_row_rotation"])
        translation = np.asarray(meta["lidar_to_world_translation"])
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("invalid source pose")
        identities.add(identity)
        detections += count
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("cache contains extra/missing files or symlinks")
    if len(identities) != manifest["frame_count"] or detections != manifest["detection_count"]:
        raise ValueError("cache totals differ")
    if expected_rows is not None and identities != set(expected_rows):
        raise ValueError("cache does not cover exact expected frames")
    return {"frames_verified": len(identities), "detections_verified": detections,
            "detector_decode_source": DETECTOR_DECODE_SOURCE,
            "manifest_sha256": digest(root / "raw-cache-manifest.json"), "all_payloads_read": True,
            "expected_frame_coverage_verified": expected_rows is not None}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_cache(args.root, args.inputs, args.package_manifest)))
