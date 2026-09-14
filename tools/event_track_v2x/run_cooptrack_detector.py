#!/usr/bin/env python3
"""Launch the pinned R50 single-agent detector on an official-train cohort.

Python 3.8 entry point inside the isolated CoopTrack environment. Invoke with
torch.distributed.launch (one GPU per process). Official val is never configured.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def install_empty_target_normalization_guard(loss_module):
    """Clamp only sample-count denominators in pinned upstream ClipMatcher.

    All three uses of this module's reduce_mean alias reduce normalization
    counts, never predictions or losses. Empty frames retain negative-class
    supervision and zero box loss instead of dividing by zero. Nonempty counts
    of at least one are unchanged, and non-finite predictions still fail closed.
    """
    original_reduce = loss_module.reduce_mean
    state = {"empty_count_calls": 0}

    def guarded_mean(value):
        reduced = original_reduce(value)
        if not reduced.isfinite().all() or (reduced < 0).any():
            raise FloatingPointError("invalid loss normalization count")
        if (reduced < 1).any():
            state["empty_count_calls"] += 1
            if state["empty_count_calls"] == 1:
                print("EVENTTRACK_EMPTY_NORMALIZER_GUARD zero-positive frame retained", flush=True)
        return reduced.clamp(min=1)

    loss_module.reduce_mean = guarded_mean
    return state


def install_unavailable_velocity_guard(box_loss):
    """Do not supervise undefined finite-difference velocities as zero speed.

    Native frame/box/class targets remain intact. Only the two velocity loss
    weights are masked when a derived velocity is non-finite; geometric target
    corruption or non-finite model predictions still raises an error.
    """
    state = {"masked_velocity_targets": 0, "empty_box_calls": 0}
    original_forward = box_loss.forward

    def guarded_forward(prediction, target, weight, *args, **kwargs):
        if target.shape[-1] != 10:
            raise ValueError("expected normalized box target with two velocity components")
        if target.numel() == 0:
            state["empty_box_calls"] += 1
            return prediction.sum() * 0.0
        if not target[..., :8].isfinite().all():
            raise FloatingPointError("non-finite geometric box target")
        valid = target[..., 8:10].isfinite().all(dim=-1)
        if valid.all():
            return original_forward(prediction, target, weight, *args, **kwargs)
        count = int((~valid).sum())
        if state["masked_velocity_targets"] == 0:
            print("EVENTTRACK_VELOCITY_TARGET_MASK undefined derived velocity retained with zero loss weight", flush=True)
        state["masked_velocity_targets"] += count
        safe_target = target.clone()
        safe_weight = weight.clone()
        safe_target[..., 8:10].masked_fill_(~valid.unsqueeze(-1), 0)
        safe_weight[..., 8:10].masked_fill_(~valid.unsqueeze(-1), 0)
        return original_forward(prediction, safe_target, safe_weight, *args, **kwargs)

    box_loss.forward = guarded_forward
    return state


def install_native_empty_annotation_guard(dataset_class):
    """Normalize empty-array shapes/dtypes without adding or dropping boxes."""
    import numpy as np
    original_load = dataset_class.load_annotations
    original_annotations = dataset_class.get_ann_info

    def load_annotations(self, ann_file):
        infos = original_load(self, ann_file)
        for info in infos:
            count = len(info["gt_boxes"])
            info["valid_flag"] = np.asarray(info["valid_flag"], dtype=bool).reshape(count)
            info["num_lidar_pts"] = np.asarray(info["num_lidar_pts"], dtype=np.int64).reshape(count)
            info["gt_inds"] = np.asarray(info["gt_inds"], dtype=np.int64).reshape(count)
            info["gt_velocity"] = np.asarray(info["gt_velocity"], dtype=float).reshape(count, 2)
        return infos

    def get_ann_info(self, index):
        result = original_annotations(self, index)
        result["gt_labels_3d"] = np.asarray(result["gt_labels_3d"], dtype=np.int64)
        result["gt_inds"] = np.asarray(result["gt_inds"], dtype=np.int64)
        return result

    dataset_class.load_annotations = load_annotations
    dataset_class.get_ann_info = get_ann_info


def preflight_training_annotations(cfg):
    from mmdet3d.datasets import build_dataset
    dataset = build_dataset(cfg.data.train)
    empty = []
    for index in range(len(dataset)):
        annotation = dataset.get_ann_info(index)
        if len(annotation["gt_bboxes_3d"]) == 0:
            empty.append(index)
    # Exercise image/formatting pipelines for every native empty frame before
    # the long run. Other image bytes were checked during input preparation.
    for index in empty:
        if dataset.prepare_train_data(index) is None:
            raise ValueError("training pipeline discarded a native empty frame")
    report = {"frame_annotations_checked": len(dataset), "empty_frame_pipelines_checked": len(empty)}
    print("EVENTTRACK_ANNOTATION_PREFLIGHT " + json.dumps(report), flush=True)
    return report


def configure(cfg, *, side, inputs, converted, output, frame_count, batch_size,
              accumulation, epochs, pretrained, world_size=1,
              allow_larger_batch=False, sequence_count=None):
    if min(batch_size, accumulation, world_size, epochs) < 1:
        raise ValueError("batch, accumulation, world size and epochs must be positive")
    global_micro_batch = batch_size * world_size
    effective_batch = global_micro_batch * accumulation
    if effective_batch != 8 and not allow_larger_batch:
        raise ValueError("effective batch change requires explicit larger-batch mode")
    if allow_larger_batch and effective_batch < 8:
        raise ValueError("larger-batch mode must not reduce the effective batch below 8")
    if sequence_count is not None and global_micro_batch > sequence_count:
        raise ValueError("global micro batch exceeds available causal sequence streams")
    cfg.model.batch_size = batch_size
    cfg.model.pretrained = dict(img=str(pretrained))
    cfg.model.train_det = True
    cfg.model.spatial_temporal_reason.history_reasoning = False
    cfg.model.spatial_temporal_reason.future_reasoning = False
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 2
    cfg.data.train.data_root = str(converted / side) + "/"
    cfg.data.train.ann_file = str(converted / side / "spd_infos_temporal_train.pkl")
    cfg.data.train.split_datas_file = str(inputs / "fold-split.json")
    cfg.data.train.forecasting = False
    cfg.data.train.filter_empty_gt = False
    # Keep complete sequence boundaries, including runs shorter than 20 frames.
    cfg.data.train.num_each_seq = 0
    for operation in cfg.data.train.pipeline:
        if operation["type"] == "LoadMultiViewImageFromFilesInCeph":
            operation["img_root"] = str(inputs / side) + "/"
        if operation["type"] == "LoadAnnotations3D_E2E":
            operation["with_forecasting"] = False
        if operation["type"] == "CustomCollect3D":
            operation["keys"] = [key for key in operation["keys"]
                                 if not key.startswith("gt_forecasting_")]
    # Unused official val/test configuration is removed, not redirected to fit.
    cfg.data.pop("val", None)
    cfg.data.pop("test", None)
    cfg.evaluation = dict(interval=10**12)
    cfg.workflow = [("train", 1)]
    per_epoch = math.ceil(frame_count / global_micro_batch)
    cfg.runner.max_iters = per_epoch * epochs
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    if accumulation > 1:
        cfg.optimizer_config.update(type="GradientCumulativeOptimizerHook",
                                    cumulative_iters=accumulation)
    # Preserve the original 4,000-image warmup exposure and base AdamW LR.
    cfg.lr_config.warmup_iters = math.ceil(4000 / global_micro_batch)
    cfg.checkpoint_config = dict(interval=per_epoch * 2, max_keep_ckpts=3)
    cfg.log_config.interval = 10
    cfg.custom_hooks = [dict(type="EventTrackTrainingGuard", priority="LOW")]
    cfg.work_dir = str(output)
    cfg.load_from = None
    cfg.resume_from = None
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--side", choices=("vehicle-side", "infrastructure-side"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-larger-batch", action="store_true")
    parser.add_argument("--require-device", default="")
    parser.add_argument("--batch-probe-iters", type=int, default=0)
    args = parser.parse_args()
    import torch
    from mmcv import Config
    from mmcv.runner import HOOKS, Hook

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.batch_probe_iters and args.batch_probe_iters < 16:
        raise ValueError("a batch probe needs at least 16 training iterations")
    if not args.preflight_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for detector optimization")
        if args.require_device and args.require_device not in torch.cuda.get_device_name(local_rank):
            raise RuntimeError("assigned GPU does not match required device")

    sys.path.insert(0, str(args.upstream_root))
    # Fail on non-finite loss BEFORE upstream's nan_to_num can replace it.
    from projects.mmdet3d_plugin.cooptrack.detectors.cooptrack import CoopTrack
    from projects.mmdet3d_plugin.losses import track_loss
    from projects.mmdet3d_plugin.datasets.spd_dataset import SPDDataset
    install_native_empty_annotation_guard(SPDDataset)
    normalization_guard = install_empty_target_normalization_guard(track_loss)
    original_forward = CoopTrack.forward_track_stream_train

    def finite_forward(self, *positional, **keywords):
        losses = original_forward(self, *positional, **keywords)
        for key, value in losses.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError("non-finite raw detector loss: " + key)
        return losses

    CoopTrack.forward_track_stream_train = finite_forward

    @HOOKS.register_module()
    class EventTrackTrainingGuard(Hook):
        def before_run(self, runner):
            torch.cuda.reset_peak_memory_stats()
            module = runner.model.module
            self.velocity_guard = install_unavailable_velocity_guard(module.criterion.loss_bboxes)
            self.parameter = dict(module.named_parameters())["img_backbone.layer4.0.conv1.weight"]
            self.initial = self.parameter.detach().cpu().clone()

        def after_train_iter(self, runner):
            if not torch.isfinite(runner.outputs["loss"]):
                raise FloatingPointError("non-finite total detector loss")
            if runner.iter + 1 == 16:
                current = self.parameter.detach().cpu()
                delta = float((current - self.initial).abs().max())
                if not math.isfinite(delta) or delta <= 0:
                    raise RuntimeError("backbone did not update in the first optimizer steps")
                proof = {"kind": "detector_optimizer_startup_v1", "micro_iteration": 16,
                         "loss": float(runner.outputs["loss"].detach().cpu()),
                         "backbone_max_abs_update": delta,
                         "cuda_device": torch.cuda.get_device_name(),
                         "optimizer_state_entries": len(runner.optimizer.state),
                         "empty_normalization_count_calls": normalization_guard["empty_count_calls"],
                         "masked_velocity_targets": self.velocity_guard["masked_velocity_targets"],
                         "empty_box_loss_calls": self.velocity_guard["empty_box_calls"],
                         "world_size": world_size,
                         "batch_per_gpu": args.batch_size,
                         "effective_batch_size": args.batch_size * args.accumulation * world_size,
                         "batch_probe_only": bool(args.batch_probe_iters),
                         "resolved_config_sha256": hashlib.sha256(
                             (args.output / "detector.py").read_bytes()).hexdigest()}
                if rank == 0:
                    with (args.output / "optimizer-startup.json").open("xb") as stream:
                        stream.write(canonical(proof))
                    if not args.batch_probe_iters:
                        runner.save_checkpoint(str(args.output), filename_tmpl="startup-iter-16.pth",
                                               create_symlink=False)
                    print("EVENTTRACK_OPTIMIZER_STARTED " + json.dumps(proof), flush=True)

        def after_run(self, runner):
            memory = {"rank": rank, "device": torch.cuda.get_device_name(),
                      "total_memory_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "world_size": world_size, "batch_per_gpu": args.batch_size,
                      "iterations": runner.iter, "batch_probe_only": bool(args.batch_probe_iters)}
            with (args.output / ("gpu-memory-rank-%d.json" % rank)).open("xb") as stream:
                stream.write(canonical(memory))
            print("EVENTTRACK_GPU_MEMORY " + json.dumps(memory), flush=True)

    manifest = json.loads((args.converted / "conversion-manifest.json").read_bytes())
    content_hash = manifest.pop("content_sha256")
    if hashlib.sha256(canonical(manifest)).hexdigest() != content_hash:
        raise ValueError("conversion manifest hash mismatch")
    if manifest["kind"] != "cooptrack_fit_conversion_v1":
        raise ValueError("unexpected conversion manifest")
    input_hash = hashlib.sha256((args.inputs / "input-manifest.json").read_bytes()).hexdigest()
    if input_hash != manifest["input_manifest_sha256"]:
        raise ValueError("fit input does not match conversion")
    for item in manifest["inventory"]:
        path = args.converted / item["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("converted input was modified")
    selected = "veh" if args.side == "vehicle-side" else "inf"
    base = args.upstream_root / ("projects/configs_spd_" + selected +
                                "/cooptrack/tiny_det_r50_stream_bs8_24epoch_3cls.py")
    cfg = configure(Config.fromfile(str(base)), side=args.side, inputs=args.inputs,
                    converted=args.converted, output=args.output,
                    frame_count=manifest["counts"][args.side]["frames"],
                    batch_size=args.batch_size, accumulation=args.accumulation,
                    epochs=args.epochs, pretrained=args.pretrained,
                    world_size=world_size, allow_larger_batch=args.allow_larger_batch,
                    sequence_count=len(manifest["fit_sequence_ids"]))
    if args.batch_probe_iters:
        cfg.runner.max_iters = args.batch_probe_iters
        cfg.checkpoint_config = None
    annotation_preflight = preflight_training_annotations(cfg)
    if args.preflight_only:
        return
    if rank == 0 and args.output.exists():
        raise FileExistsError("training output must be create-once")
    if rank == 0:
        args.output.mkdir(parents=True)
    # Upstream writes a runtime-resolved copy to work_dir/basename(config).
    # Keep the launch input separate so its recorded content hash stays valid.
    config_path = args.output / "input-config" / "detector.py"
    if rank == 0:
        config_path.parent.mkdir()
        cfg.dump(str(config_path))
    else:
        deadline = time.monotonic() + 120
        while not (args.output / "launch-receipt.json").is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("rank zero did not finish the shared launch receipt")
            time.sleep(0.1)
    receipt = {"kind": "cooptrack_detector_training_launch_v1",
               "stage": "D2-detector-fit", "side": args.side,
               "fold_id": manifest["fold_id"], "seed": args.seed,
               "cohort": manifest.get("cohort", "fit-fold"),
               "empty_target_policy": "keep-frame-clamp-ClipMatcher-count-denominators-to-one",
               "unavailable_velocity_policy": "mask-only-nonfinite-derived-velocity-loss-weights",
               "annotation_array_policy": "normalize-empty-shapes-and-index-dtypes-no-box-changes",
               "annotation_preflight": annotation_preflight,
               "fit_sequence_ids": manifest["fit_sequence_ids"],
               "raw_labels_modified": False, "epochs": args.epochs,
               "world_size": world_size, "batch_per_gpu": args.batch_size,
               "effective_batch_size": args.batch_size * args.accumulation * world_size,
               "batch_probe_only": bool(args.batch_probe_iters),
               "required_device": args.require_device,
               "learning_rate_policy": "unchanged-base-AdamW-LR-warmup-4000-images",
               "max_micro_iterations": cfg.runner.max_iters,
               "pretrained_kind": "ImageNet-R50-only-no-SPD-trained-weights",
               "pretrained_sha256": hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
               "config_path": "input-config/detector.py",
               "data_provenance": "user-confirmed-official-download",
               "independent_source_reverification": "waived-by-user-2026-09-11",
               "paper_ranking_eligible": False, "official_val_test_loaded": False}
    if rank == 0:
        (args.output / "launch-receipt.json").write_bytes(canonical(receipt))
    sys.argv = [str(args.upstream_root / "tools/train.py"), str(config_path),
                "--launcher", "pytorch", "--no-validate", "--seed", str(args.seed),
                "--deterministic", "--work-dir", str(args.output)]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
