"""Export uncropped fit-only features and frozen appearance, with no GT access.

Recovered from ClearML c39e2280d3894737b82c393dd4117dea and adapted for
the raw all-query paper protocol. Old cached results are not relabeled.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np

from spd_canonical_oof_fit_feature_cache_primitives import DETECTOR_DECODE_SOURCE, FEATURE_METHOD, PRETRAINED_SHA, digest, install_raw_detector_decode, normalize_visible_features, project_corners, validate_arrays, write_json
from spd_canonical_oof_query_decode import install_uncropped_query_decode
from spd_canonical_oof_fit_feature_export_binding import validate_binding
from spd_canonical_oof_fit_feature_input_gate import validate_inputs


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_export_readback(output, inputs, package_manifest, frames, detections):
    verified = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("spd_canonical_oof_fit_feature_cache_primitives.py")),
         str(output), "--inputs", str(inputs), "--package-manifest", str(package_manifest)],
        check=True, capture_output=True, text=True)
    readback = json.loads(verified.stdout)
    if (readback.get("all_payloads_read") is not True
            or readback.get("expected_frame_coverage_verified") is not True
            or readback.get("frames_verified") != frames
            or readback.get("detections_verified") != detections
            or readback.get("manifest_sha256") != digest(output / "raw-cache-manifest.json")):
        raise RuntimeError("independent cache readback differs")
    return readback


def strip_pipeline(pipeline, root):
    out = []
    for source in pipeline:
        op = copy.deepcopy(source)
        if op["type"].startswith("LoadAnnotations"):
            continue
        if op["type"] == "LoadMultiViewImageFromFilesInCeph":
            op["img_root"] = str(root) + "/"
        if "transforms" in op:
            op["transforms"] = strip_pipeline(op["transforms"], root)
        if op["type"] == "CustomCollect3D":
            op["keys"] = [key for key in op["keys"] if not key.startswith("gt_")]
        out.append(op)
    return out


def check_model_inputs(data):
    if set(data) != {"img", "img_metas", "timestamp", "l2g_r_mat", "l2g_t"}:
        raise ValueError("unexpected inference inputs")
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).startswith("gt_") or key in {"ann_info", "next_idx", "future"}:
                    raise ValueError("GT/future field in model inputs")
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
        elif value.__class__.__name__ == "DataContainer":
            inspect(value.data)
    inspect(data)


class AnnotationDatabaseForbidden:
    def __init__(self, **kwargs):
        pass

    def __getattr__(self, name):
        raise RuntimeError("annotation/evaluation access forbidden in raw cache producer: " + name)


def build_inference_dataset(root, side, upstream, shard_index, shard_count, package, expected_manifest_sha256, package_manifest_sha256):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin.datasets.spd_dataset as module
    manifest = validate_inputs(root, package, expected_manifest_sha256, package_manifest_sha256=package_manifest_sha256)
    if not 0 <= shard_index < shard_count or shard_count != 2:
        raise ValueError("expected one of two deterministic sequence shards")
    scenes = sorted(manifest["fit_sequence_ids"])[shard_index::shard_count]
    label = "veh" if side == "vehicle-side" else "inf"
    upstream_cfg = Config.fromfile(str(upstream / ("projects/configs_spd_" + label + "/cooptrack/tiny_det_r50_stream_bs8_24epoch_3cls.py")))
    cfg = copy.deepcopy(upstream_cfg.data.test)
    cfg.data_root = str(root / side) + "/"
    cfg.ann_file = str(root / side / "image-pose-infos.pkl")
    cfg.split_datas_file = str(root / "train-split.json")
    cfg.test_mode = True
    cfg.forecasting = False
    cfg.eval_mod = []
    cfg.filter_empty_gt = False
    cfg.pipeline = strip_pipeline(cfg.pipeline, root)
    module.NuScenes = AnnotationDatabaseForbidden
    module.SPDDataset.get_ann_info = lambda self, index: {}
    original_load = module.SPDDataset.load_annotations
    def load(self, path):
        values = original_load(self, path)
        for value in values:
            if any(key.startswith("gt_") or key in {"anno_tokens", "next_anno_tokens", "prev_anno_tokens"} for key in value):
                raise ValueError("GT fields persisted in cache inputs")
            if value["next"] or value["prev"] or value["sweeps"]:
                raise ValueError("unexpected cross-frame input")
        return sorted([x for x in values if x["scene_token"] in scenes], key=lambda x: (x["scene_token"], x["timestamp"], x["token"]))
    module.SPDDataset.load_annotations = load
    original_info = module.SPDDataset.get_data_info
    def data_info(self, index):
        out = original_info(self, index)
        out.pop("ann_info", None)
        out.pop("next_idx", None)
        return out
    module.SPDDataset.get_data_info = data_info
    dataset = build_dataset(cfg)
    rows = json.loads((root / side / "frame-index.json").read_bytes())
    rows = {r["frame_id"]: r for r in rows if r["sequence_id"] in scenes}
    tokens = [r["token"] for r in dataset.data_infos]
    if len(tokens) != len(rows) or len(tokens) != len(set(tokens)) or set(tokens) != set(rows):
        raise ValueError("shard frame coverage differs")
    if {x["scene_token"] for x in dataset.data_infos} != set(scenes):
        raise ValueError("shard sequence coverage differs")
    return dataset, cfg, rows, scenes, manifest


def main():
    p = argparse.ArgumentParser()
    for name in ["inputs", "upstream", "checkpoint", "appearance-checkpoint", "output",
                 "training-config", "training-launch", "training-startup", "training-completion",
                 "package-manifest", "byte-freeze"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--input-manifest-sha256", required=True)
    p.add_argument("--package-manifest-sha256", required=True)
    p.add_argument("--byte-freeze-sha256", required=True)
    p.add_argument("--seed", type=int, choices=(1337, 2027, 3407), required=True)
    p.add_argument("--side", choices=["vehicle-side", "infrastructure-side"], required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if digest(args.package_manifest) != args.package_manifest_sha256 or digest(args.byte_freeze) != args.byte_freeze_sha256:
        raise ValueError("bound package/byte-freeze bytes differ")
    package = json.loads(args.package_manifest.read_bytes())
    frozen = json.loads(args.byte_freeze.read_bytes())
    if digest(args.inputs / "input-manifest.json") != args.input_manifest_sha256:
        raise ValueError("held-out input manifest identity differs")
    inputs = json.loads((args.inputs / "input-manifest.json").read_bytes())
    evidence = {args.side + suffix: getattr(args, attr).read_bytes()
                for suffix, attr in (("-launch-receipt.json", "training_launch"),
                                     ("-optimizer-startup.json", "training_startup"),
                                     ("-completion", "training_completion"))}
    binding = validate_binding(package, inputs, frozen,
        json.loads(args.training_launch.read_bytes()), json.loads(args.training_startup.read_bytes()),
        json.loads(args.training_completion.read_bytes()), side=args.side, seed=args.seed,
        checkpoint_sha256=args.checkpoint_sha256, config_sha256=digest(args.training_config),
        package_manifest_sha256=args.package_manifest_sha256,
        input_manifest_sha256=args.input_manifest_sha256, evidence_payloads=evidence)
    if digest(args.checkpoint) != args.checkpoint_sha256 or digest(args.appearance_checkpoint) != PRETRAINED_SHA:
        raise ValueError("checkpoint bytes mismatch")
    sys.path.insert(0, str(args.upstream))
    import torch
    from torch import nn
    from torchvision.models import resnet50
    from torchvision.ops import roi_align
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    dataset, cfg, rows, scenes, inputs_manifest = build_inference_dataset(
        args.inputs, args.side, args.upstream, args.shard_index, args.shard_count,
        package, args.input_manifest_sha256, args.package_manifest_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    training = Config.fromfile(str(args.training_config))
    model_cfg = copy.deepcopy(training.model)
    model_cfg.pretrained = None
    model_cfg.batch_size = 1
    model_cfg.train_cfg = None
    if model_cfg.train_det is not True or model_cfg.spatial_temporal_reason.history_reasoning or model_cfg.spatial_temporal_reason.future_reasoning:
        raise ValueError("expected frozen D2 detector")
    resolved = Config(dict(model=model_cfg, data=dict(cache=cfg), appearance_method=FEATURE_METHOD,
                           export_decode_policy=DETECTOR_DECODE_SOURCE))
    with (args.output / "resolved-cache-config.py").open("x") as stream:
        stream.write(resolved.pretty_text)
    for index in sorted({0, len(dataset)//2, len(dataset)-1}):
        check_model_inputs(dataset[index])
    if args.preflight_only:
        print("EVENTTRACK_CACHE_PREFLIGHT " + json.dumps({"side": args.side, "shard": args.shard_index,
              "frames": len(dataset), "sequences": len(scenes), "gt_inputs": False}), flush=True)
        return
    if digest(args.checkpoint) != args.checkpoint_sha256 or digest(args.appearance_checkpoint) != PRETRAINED_SHA:
        raise ValueError("checkpoint bytes mismatch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one physically assigned CUDA device required")
    set_random_seed(args.seed, deterministic=True)
    model = build_model(model_cfg, test_cfg=training.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu", strict=True)
    model.requires_grad_(False).eval()
    detector_state = state_hash(model)
    uncropped_state = install_uncropped_query_decode(model, denormalize_bbox)
    decode_state = install_raw_detector_decode(model)
    captured = {}
    def image_input_hook(module, values):
        captured["image"] = values[0].detach()
    image_hook = model.img_backbone.register_forward_pre_hook(image_input_hook)
    original_forward = model.forward
    def capture_forward(*a, **kw):
        captured["meta"] = copy.deepcopy(kw["img_metas"][0][0])
        return original_forward(*a, **kw)
    model.forward = capture_forward
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    appearance = resnet50(pretrained=False)
    appearance.load_state_dict(torch.load(str(args.appearance_checkpoint), map_location="cpu"), strict=True)
    appearance = nn.Sequential(*list(appearance.children())[:-2]).requires_grad_(False).eval().cuda()
    feature_state = state_hash(appearance)
    launch = {"kind": "eventtrack_fit_feature_raw_cache_launch_v1", "side": args.side,
              "canonical_oof_fit_feature_export": True, "held_out_selection_scoring_eligible": False, "fold_id": package["fold_id"], "seed": args.seed,
              "physical_device": torch.cuda.get_device_name(0),
              "byte_freeze_sha256": args.byte_freeze_sha256,
              "shard_index": args.shard_index, "shard_count": args.shard_count, "sequences": scenes,
              "frames": len(dataset), "checkpoint_sha256": args.checkpoint_sha256,
              "appearance_checkpoint_sha256": PRETRAINED_SHA, "appearance_method": FEATURE_METHOD,
              "detector_state_sha256": detector_state, "appearance_state_sha256": feature_state,
              "input_manifest_sha256": digest(args.inputs / "input-manifest.json"),
              "training_binding": binding,
              "training_evidence_sha256": {name: digest(getattr(args, name)) for name in
                                            ('training_launch', 'training_startup', 'training_completion')},
              "resolved_config_sha256": digest(args.output / "resolved-cache-config.py"),
              "producer_code_sha256": digest(Path(__file__)),
              "verifier_code_sha256": digest(Path(__file__).with_name("spd_canonical_oof_fit_feature_cache_primitives.py")),
              "uncropped_decoder_sha256": digest(Path(__file__).with_name("spd_canonical_oof_query_decode.py")),
              "detector_decode_source": DETECTOR_DECODE_SOURCE,
              "gt_inputs": False, "optimizer_created": False, "test_payloads_read": False,
              "val_payloads_read": False, "score_filter_changed": True,
              "export_candidate_policy": "all-queries; raw-score>=0.05/all-class-top64 downstream",
              "preselection_roi": False, "preselection_nms": False, "preselection_topk": False,
              "formal_v2_ready": False, "covariance_calibrated": False}
    write_json(args.output / "launch-receipt.json", launch)
    print("EVENTTRACK_CACHE_STARTED " + json.dumps(launch), flush=True)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False, seed=args.seed)
    records = []
    counts = {"detections": 0, "appearance_valid": 0, "zero_features": 0}
    started = time.monotonic()
    for index, data in enumerate(loader):
        check_model_inputs(data)
        captured.clear()
        with torch.no_grad():
            output = model(return_loss=False, rescale=True, **data)
        info = dataset.data_infos[index]
        if len(output) != 1 or output[0]["token"] != info["token"]:
            raise ValueError("detector token/order mismatch")
        if decode_state["frames"] != index + 1 or decode_state["last_audit"]["frame_id"] != str(info["token"]):
            raise ValueError("raw detector snapshot/frame identity differs")
        if index == 0:
            print("EVENTTRACK_CACHE_RAW_HEAD_AUDIT " + json.dumps(dict(decode_state["last_audit"],
                  side=args.side, shard=args.shard_index)), flush=True)
        prediction = output[0]
        boxes = prediction["boxes_3d_det"]
        if (len(boxes) != decode_state["last_audit"]["head_queries"]
                or uncropped_state["frames"] != index + 1):
            raise ValueError("export dropped raw queries")
        meta = captured["meta"]
        image = captured["image"]
        if image.ndim != 4 or image.shape[0] != 1 or len(meta["lidar2img"]) != 1:
            raise ValueError("unexpected image layout")
        pad_hw = image.shape[-2:]
        # Pipeline records resized pre-padding shape in ori_shape.
        image_hw = tuple(meta["ori_shape"][0][:2])
        rois, valid = project_corners(boxes.corners.cpu().numpy(), meta["lidar2img"][0], image_hw)
        with torch.no_grad():
            # Normalize the detector's RGB/BGR convention to torchvision ImageNet.
            norm = meta["img_norm_cfg"]
            source_mean = torch.as_tensor(norm["mean"], device=image.device).reshape(1, 3, 1, 1)
            source_std = torch.as_tensor(norm["std"], device=image.device).reshape(1, 3, 1, 1)
            pixels = image * source_std + source_mean
            if not norm["to_rgb"]:
                pixels = pixels[:, [2, 1, 0]]
            mean = image.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            feature = appearance((pixels / 255.0 - mean) / std)
            embeddings = np.zeros((len(boxes), 128), dtype=np.float32)
            if valid.any():
                scaled = torch.as_tensor(rois[valid], device=feature.device)
                scaled[:, [0, 2]] *= feature.shape[-1] / pad_hw[1]
                scaled[:, [1, 3]] *= feature.shape[-2] / pad_hw[0]
                pooled = roi_align(feature, [scaled], output_size=(3, 3), spatial_scale=1.0, sampling_ratio=2, aligned=False)
                values = pooled.mean(dim=(-1, -2)).reshape(-1, 16, 128).mean(dim=1)
                embeddings, valid, zero_features = normalize_visible_features(values.cpu().numpy(), valid)
                if zero_features and counts["zero_features"] == 0:
                    print("EVENTTRACK_CACHE_ZERO_FEATURE " + json.dumps({"side": args.side,
                          "shard": args.shard_index, "frame_id": info["token"],
                          "zero_feature_count": zero_features, "nonfinite_features": False,
                          "handling": "detections retained; appearance unavailable"}), flush=True)
                counts["zero_features"] += zero_features
        arrays = {"boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy": boxes.tensor.cpu().numpy().astype(np.float32),
                  "gravity_centers_lidar": boxes.gravity_center.cpu().numpy().astype(np.float32),
                  "scores": prediction["scores_3d_det"].cpu().numpy().astype(np.float32),
                  "class_indices": prediction["labels_3d_det"].cpu().numpy().astype(np.int64),
                  "appearance_128": embeddings, "appearance_valid": valid, "image_rois_xyxy": rois}
        count = validate_arrays(arrays)
        placeholder = ((arrays["scores"] == 0.5)
                       & (arrays["boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy"] == np.array([0,0,0,1,1,1,0,0,0])).all(axis=1))
        if placeholder.any():
            raise ValueError("legacy zero-summary placeholder detected after raw-head decode")
        row = rows[info["token"]]
        rotation = np.asarray(data["l2g_r_mat"].cpu()).reshape(3, 3)
        translation = np.asarray(data["l2g_t"].cpu()).reshape(3)
        metadata = dict(row, side=args.side, coordinate_system="source_lidar",
                        lidar_to_world_row_rotation=rotation.tolist(), lidar_to_world_translation=translation.tolist())
        folder = args.output / "frames" / info["scene_token"]
        folder.mkdir(parents=True, exist_ok=True)
        array_path, meta_path = folder / (info["token"] + ".npz"), folder / (info["token"] + ".json")
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        write_json(meta_path, metadata)
        def record(path):
            return {"path": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
        records.append({"arrays": record(array_path), "metadata": record(meta_path),
                        "detections": count, "raw_query_count": decode_state["last_audit"]["head_queries"]})
        counts["detections"] += count
        counts["appearance_valid"] += int(valid.sum())
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print("EVENTTRACK_CACHE_PROGRESS " + json.dumps({"side": args.side, "shard": args.shard_index,
                  "frames": index + 1, "total": len(dataset), "elapsed_seconds": time.monotonic() - started,
                  "eta_seconds": (time.monotonic() - started) * (len(dataset) - index - 1) / (index + 1),
                  "detections": counts["detections"], "appearance_valid": counts["appearance_valid"]}), flush=True)
    image_hook.remove()
    if state_hash(model.module) != detector_state or state_hash(appearance) != feature_state:
        raise RuntimeError("frozen model weights changed during export")
    manifest = dict(launch, kind="eventtrack_fit_feature_raw_detector_cache_v1", frames=records, frame_count=len(records),
                    detection_count=counts["detections"], appearance_valid_count=counts["appearance_valid"],
                    zero_feature_count=counts["zero_features"],
                    raw_head_frames_verified=decode_state["frames"], legacy_placeholder_count=0,
                    weights_unchanged=True, elapsed_seconds=time.monotonic() - started,
                    classes=list(dataset.CLASSES), covariance_source=None,
                    cache_role="fit-only feature export; ineligible for held-out selection scoring")
    write_json(args.output / "raw-cache-manifest.json", manifest)
    # Separate CPU process reopens every payload and checks the complete frame
    # index before the producer is allowed to announce completion. Keep its
    # receipt in the job log, not inside the strict cache inventory.
    readback = verify_export_readback(args.output, args.inputs, args.package_manifest, len(records), counts["detections"])
    print("EVENTTRACK_CACHE_READBACK " + json.dumps(readback), flush=True)
    print("EVENTTRACK_CACHE_COMPLETED " + json.dumps({"side": args.side, "shard": args.shard_index,
          "frames": len(records), "detections": counts["detections"], "weights_unchanged": True,
          "manifest_sha256": digest(args.output / "raw-cache-manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
