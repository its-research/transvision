#!/usr/bin/env python3
"""Train the development-only EventTrack-V2X cross-agent association head."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.event_track_v2x_association import (  # noqa: E402
    FrozenRGBProjectionAppearanceV1,
    FrozenResNet50AppearanceV1,
    NPZAppearanceCacheV1,
    SPDAssociationDataError,
    build_spd_association_cohort_v1,
)
from transvision.models.event_track_v2x.development_split import (  # noqa: E402
    DEVELOPMENT_SEQUENCE_COUNT_V1,
    DevelopmentSplitError,
    decode_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.learned_association import (  # noqa: E402
    train_association_model_v1,
)
from transvision.models.event_track_v2x.learning_contracts import (  # noqa: E402
    ASSOCIATION_CHECKPOINT_NAME_V1,
    ASSOCIATION_FEATURE_DIM_V1,
    ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
    ASSOCIATION_MANIFEST_NAME_V1,
    AssociationLearningContractError,
    AssociationTrainingArtifactManifestV1,
    AssociationTrainingConfigV1,
)


class AssociationTrainingCLIError(ValueError):
    """Raised before training if the development evidence boundary is unclear."""


def _load_json_bytes(data: bytes, name: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise AssociationTrainingCLIError(
                    f"duplicate JSON key in {name}: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AssociationTrainingCLIError(
                    f"non-finite JSON constant in {name}: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssociationTrainingCLIError(f"invalid JSON in {name}") from exc


def _sequence_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AssociationTrainingCLIError(f"{name} must be an array")
    result = tuple(value)
    if any(
        type(item) is not str or len(item) != 4 or not item.isdecimal()
        for item in result
    ):
        raise AssociationTrainingCLIError(f"{name} contains a noncanonical sequence ID")
    if len(set(result)) != len(result):
        raise AssociationTrainingCLIError(f"{name} contains duplicate sequence IDs")
    return result


def _official_train_split(data: bytes) -> tuple[str, ...]:
    document = _load_json_bytes(data, "official split")
    if not isinstance(document, Mapping):
        raise AssociationTrainingCLIError("official split must be an object")
    batch = document.get("batch_split")
    if not isinstance(batch, Mapping):
        raise AssociationTrainingCLIError("official split has no batch_split object")
    required = {"train", "val", "test", "test_A"}
    if set(batch) != required:
        raise AssociationTrainingCLIError(
            "batch_split must contain exactly train/val/test/test_A"
        )
    splits = {
        name: _sequence_array(batch[name], f"batch_split.{name}") for name in required
    }
    train = splits["train"]
    if len(train) != DEVELOPMENT_SEQUENCE_COUNT_V1:
        raise AssociationTrainingCLIError("SPD train must contain exactly 46 sequences")
    sealed = set(splits["val"]) | set(splits["test"]) | set(splits["test_A"])
    overlap = sorted(set(train).intersection(sealed))
    if overlap:
        raise AssociationTrainingCLIError(
            f"official SPD train overlaps sealed cohorts: {overlap}"
        )
    if not set(splits["test_A"]).issubset(splits["test"]):
        raise AssociationTrainingCLIError("batch_split.test_A must be a subset of test")
    return tuple(sorted(train))


def _load_config(path: Path | None) -> AssociationTrainingConfigV1:
    if path is None:
        return AssociationTrainingConfigV1()
    value = _load_json_bytes(path.read_bytes(), "association config")
    if not isinstance(value, Mapping):
        raise AssociationTrainingCLIError("association config must be an object")
    if "kind" in value or "schema_version" in value:
        return AssociationTrainingConfigV1.from_mapping(value)
    allowed = {
        "assignment_weight",
        "dropout",
        "epochs",
        "frames_per_step",
        "gradient_clip_norm",
        "hidden_dim",
        "learning_rate",
        "max_positive_weight",
        "pairwise_bce_weight",
        "top_h",
        "weight_decay",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AssociationTrainingCLIError(
            f"association config contains unknown fields: {unknown}"
        )
    return replace(AssociationTrainingConfigV1(), **dict(value))


def _count_objects(samples: Sequence[object]) -> int:
    return sum(
        len(sample.left_identity_ids) + len(sample.right_identity_ids)  # type: ignore[attr-defined]
        for sample in samples
    )


def _resolved_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise AssociationTrainingCLIError("CUDA was requested but is unavailable")
    return requested


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, required=True, help="official SPD split JSON"
    )
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--frames-per-step", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--top-h", type=int)
    parser.add_argument(
        "--max-frame-pairs-per-sequence",
        type=int,
        default=2,
        help="deterministic development canary cap; use 0 for every train frame",
    )
    appearance = parser.add_mutually_exclusive_group()
    appearance.add_argument("--appearance-cache", type=Path)
    appearance.add_argument("--appearance-checkpoint", type=Path)
    parser.add_argument("--appearance-checkpoint-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed < 0:
        raise AssociationTrainingCLIError("seed must be non-negative")
    if args.output.exists() or args.output.is_symlink():
        raise AssociationTrainingCLIError("output must not already exist")
    split_raw = args.split.read_bytes()
    official_train = _official_train_split(split_raw)
    official_split_sha256 = hashlib.sha256(split_raw).hexdigest()
    development_raw = args.development_manifest.read_bytes()
    development = decode_development_split_manifest_v1(development_raw)
    if development.split_sha256 != official_split_sha256:
        raise AssociationTrainingCLIError(
            "development manifest does not bind the supplied official split"
        )
    if development.sequence_ids != official_train:
        raise AssociationTrainingCLIError(
            "development manifest does not contain exactly official SPD train"
        )
    fold = development.folds[args.fold_id]
    config = _load_config(args.config)
    overrides: dict[str, Any] = {}
    for argument, field in (
        (args.epochs, "epochs"),
        (args.hidden_dim, "hidden_dim"),
        (args.frames_per_step, "frames_per_step"),
        (args.learning_rate, "learning_rate"),
        (args.top_h, "top_h"),
    ):
        if argument is not None:
            overrides[field] = argument
    if overrides:
        config = replace(config, **overrides)
    if args.max_frame_pairs_per_sequence < 0:
        raise AssociationTrainingCLIError(
            "max-frame-pairs-per-sequence must be non-negative"
        )
    frame_cap = args.max_frame_pairs_per_sequence or None
    if args.appearance_cache is not None:
        provider = NPZAppearanceCacheV1(args.appearance_cache)
    elif args.appearance_checkpoint is not None:
        provider = FrozenResNet50AppearanceV1(
            args.appearance_checkpoint,
            expected_checkpoint_sha256=args.appearance_checkpoint_sha256,
            device=_resolved_device(args.device),
        )
    else:
        if args.appearance_checkpoint_sha256 is not None:
            raise AssociationTrainingCLIError(
                "appearance-checkpoint-sha256 requires appearance-checkpoint"
            )
        provider = FrozenRGBProjectionAppearanceV1()

    cohort = build_spd_association_cohort_v1(
        args.dataset_root,
        official_train_sequence_ids=official_train,
        fit_sequence_ids=fold.fit_sequence_ids,
        held_out_sequence_ids=fold.held_out_sequence_ids,
        appearance_provider=provider,
        max_frame_pairs_per_sequence=frame_cap,
    )
    device = _resolved_device(args.device)
    result = train_association_model_v1(
        cohort.train_samples,
        cohort.validation_samples,
        config=config,
        seed=args.seed,
        device=device,
    )

    args.output.mkdir(parents=True, mode=0o700)
    checkpoint_path = args.output / ASSOCIATION_CHECKPOINT_NAME_V1
    manifest_path = args.output / ASSOCIATION_MANIFEST_NAME_V1
    try:
        import torch

        checkpoint = {
            "cohort_sha256": cohort.content_sha256,
            "config": config.to_primitive(),
            "config_sha256": config.content_sha256,
            "development_manifest_sha256": hashlib.sha256(development_raw).hexdigest(),
            "feature_dim": ASSOCIATION_FEATURE_DIM_V1,
            "feature_schema_sha256": ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
            "fold_id": args.fold_id,
            "formal_evidence": False,
            "gt_supervised_development_only": True,
            "history": [
                {
                    "epoch": item.epoch,
                    "train_loss": item.train_loss,
                    "validation_assignment_accuracy": item.validation_assignment_accuracy,
                    "validation_loss": item.validation_loss,
                    "validation_pair_accuracy": item.validation_pair_accuracy,
                }
                for item in result.history
            ],
            "kind": "learned_association_checkpoint_v1",
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in result.model.state_dict().items()
            },
            "paper_registry_writable": False,
            "ranking_eligible": False,
            "schema_version": 1,
            "training_seed": args.seed,
        }
        with checkpoint_path.open("xb") as stream:
            torch.save(checkpoint, stream)
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        manifest = AssociationTrainingArtifactManifestV1(
            fold_id=args.fold_id,
            training_seed=args.seed,
            development_manifest_sha256=hashlib.sha256(development_raw).hexdigest(),
            official_split_sha256=official_split_sha256,
            cohort_sha256=cohort.content_sha256,
            config_sha256=config.content_sha256,
            checkpoint_sha256=checkpoint_sha256,
            appearance_source=cohort.appearance_source,
            covariance_source=cohort.covariance_source,
            train_sequence_count=len(cohort.train_sequence_ids),
            validation_sequence_count=len(cohort.validation_sequence_ids),
            train_frame_count=len(cohort.train_samples),
            validation_frame_count=len(cohort.validation_samples),
            train_object_count=_count_objects(cohort.train_samples),
            validation_object_count=_count_objects(cohort.validation_samples),
            metrics=result.final_metrics,
        )
        with manifest_path.open("xb") as stream:
            stream.write(manifest.canonical_bytes)
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "artifact_manifest_sha256": manifest.content_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "cohort_sha256": cohort.content_sha256,
                "device": device,
                "formal_evidence": False,
                "gt_supervised_development_only": True,
                "output": str(args.output),
                "ranking_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AssociationLearningContractError,
        AssociationTrainingCLIError,
        DevelopmentSplitError,
        SPDAssociationDataError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
