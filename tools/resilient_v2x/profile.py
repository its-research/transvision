#!/usr/bin/env python3
"""Profile one ResilientV2X batch on GPU and emit canonical evidence JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.evaluation.resilient_v2x_evidence import (  # noqa: E402
    capture_git_state,
    digest_artifact,
    write_document,
)
from transvision.evaluation.resilient_v2x_profile import (  # noqa: E402
    build_profile_document,
    count_parameters,
    profile_runtime,
    profile_torch_flops,
    unavailable_flops,
)


def _parse_role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("artifact must have the form ROLE=PATH")
    return role, Path(path)


def _parse_override(value: str) -> tuple[str, object]:
    key, separator, raw = value.partition("=")
    if not separator or not key or not raw:
        raise argparse.ArgumentTypeError("override must have the form KEY=JSON_VALUE")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"override value is not JSON: {error.msg}"
        ) from error
    return key, decoded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure student-only parameters, supported FLOPs, peak GPU memory, "
            "and synchronized test_step latency on one already-collated batch."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=["teacher"],
        help="module path excluded from paper-facing parameter counts",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_parse_role_path,
        metavar="ROLE=PATH",
        help="additional protocol/environment artifact to hash",
    )
    parser.add_argument(
        "--cfg-option",
        action="append",
        default=[],
        type=_parse_override,
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument(
        "--detach-training-only-teacher",
        action="store_true",
        help=(
            "build the deployment student without its frozen training teacher and "
            "strict-load only non-teacher checkpoint state"
        ),
    )
    parser.add_argument("--no-git-state", action="store_true")
    return parser.parse_args(argv)


def _configured_batch_size(config: object, batch: object) -> int:
    if isinstance(batch, dict):
        samples = batch.get("data_samples")
        if isinstance(samples, (list, tuple)) and samples:
            return len(samples)
        inputs = batch.get("inputs")
        if isinstance(inputs, dict):
            selections = inputs.get("selections")
            if selections is not None and hasattr(selections, "batch_size"):
                value = int(selections.batch_size)
                if value > 0:
                    return value
    configured = config.test_dataloader.get("batch_size", 1)
    if type(configured) is not int or configured <= 0:
        raise RuntimeError("cannot determine a positive profiling batch size")
    return configured


def _device_metadata(torch, device) -> dict[str, object]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deployment_student_state_dict(
    state_dict: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    """Remove only the sealed training teacher prefix from model state."""

    if not isinstance(state_dict, Mapping) or any(
        type(key) is not str for key in state_dict
    ):
        raise RuntimeError("checkpoint state_dict must be a string-keyed mapping")
    student: dict[str, object] = {}
    excluded = 0
    for original_key, value in state_dict.items():
        key = original_key[7:] if original_key.startswith("module.") else original_key
        if key == "teacher" or key.startswith("teacher."):
            excluded += 1
            continue
        if key in student:
            raise RuntimeError("checkpoint key normalization produced a duplicate")
        student[key] = value
    if excluded == 0:
        raise RuntimeError("checkpoint has no training-only teacher state to detach")
    if not student:
        raise RuntimeError("checkpoint contains no deployment student state")
    return student, excluded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.runner import Runner, load_checkpoint
        from mmengine.runner.checkpoint import CheckpointLoader
        from mmdet3d.registry import MODELS
    except ImportError as error:
        raise RuntimeError(
            "profiling requires the controlled ResilientV2X runtime"
        ) from error

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing runtime profiling requires an available CUDA device")

    from transvision import register_all_modules

    register_all_modules()
    config = Config.fromfile(str(args.config.resolve(strict=True)))
    if args.cfg_option:
        config.merge_from_dict(dict(args.cfg_option))
    teacher_configured = config.model.get("teacher") is not None
    if args.detach_training_only_teacher:
        if not teacher_configured:
            raise RuntimeError("config has no training-only teacher to detach")
        config.model["teacher"] = None
        config.model["teacher_checkpoint"] = None
        config.model["distillation"] = None
    init_default_scope(config.get("default_scope", "mmdet3d"))
    model = MODELS.build(config.model)
    excluded_checkpoint_tensors = 0
    checkpoint_path = str(args.checkpoint.resolve(strict=True))
    if args.detach_training_only_teacher:
        checkpoint_payload = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        if not isinstance(checkpoint_payload, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")
        state_dict = checkpoint_payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("checkpoint payload has no state_dict mapping")
        student_state, excluded_checkpoint_tensors = _deployment_student_state_dict(
            state_dict
        )
        model.load_state_dict(student_state, strict=args.strict_checkpoint)
    else:
        load_checkpoint(
            model,
            checkpoint_path,
            map_location="cpu",
            strict=args.strict_checkpoint,
        )
    parameters = count_parameters(
        model,
        excluded_module_paths=args.exclude_module,
    )
    detached_modules: list[str] = []
    if args.detach_training_only_teacher:
        detached_modules.append("teacher")
    elif "teacher" in args.exclude_module and getattr(model, "teacher", None) is not None:
        # The teacher is a training-only module.  Detach it while still on CPU so
        # paper-facing inference memory and latency describe the student alone.
        model.teacher = None
        detached_modules.append("teacher")
    model.to(device)
    model.eval()

    dataloader = Runner.build_dataloader(
        config.test_dataloader,
        seed=0,
        diff_rank_seed=False,
    )
    try:
        batch = next(iter(dataloader))
    except StopIteration as error:
        raise RuntimeError("test dataloader is empty") from error
    batch_size = _configured_batch_size(config, batch)

    def synchronize() -> None:
        torch.cuda.synchronize(device)

    def run_once():
        with torch.inference_mode():
            return model.test_step(batch)

    runtime = profile_runtime(
        run_once,
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
        synchronize=synchronize,
        reset_peak_memory=lambda: torch.cuda.reset_peak_memory_stats(device),
        peak_allocated_bytes=lambda: int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=lambda: int(torch.cuda.max_memory_reserved(device)),
    )
    if args.skip_flops:
        flops = unavailable_flops("disabled by --skip-flops")
    else:
        flops = profile_torch_flops(
            run_once,
            include_cuda_activity=True,
            synchronize=synchronize,
        )

    artifacts = [
        digest_artifact(args.config, "model_config"),
        digest_artifact(args.checkpoint, "model_checkpoint"),
    ]
    artifacts.extend(digest_artifact(path, role) for role, path in args.artifact)
    source_state = {} if args.no_git_state else capture_git_state(ROOT)
    document = build_profile_document(
        parameters=parameters,
        runtime=runtime,
        flops=flops,
        boundary=(
            "model.test_step from one collated dataloader batch through decoded "
            "predictions; dataloader I/O excluded; CUDA synchronized"
        ),
        batch_size=batch_size,
        device={
            **_device_metadata(torch, device),
            "detached_training_only_modules": detached_modules,
            "excluded_checkpoint_tensor_count": excluded_checkpoint_tensors,
        },
        artifacts=artifacts,
        source_state=source_state,
        measured_at_utc=_utc_now(),
    )
    destination = write_document(args.out, document)
    print(
        json.dumps(
            {
                "output": str(destination),
                "content_sha256": document["content_sha256"],
                "parameter_count": parameters.parameter_count,
                "flop_count": flops.flop_count,
                "latency_median_ms": runtime.latency.median_ms,
                "peak_allocated_bytes": runtime.gpu_memory.peak_allocated_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
