"""Fail-closed contracts for development-only learned association.

The learned association head is deliberately separated from the paper result
registries.  Its first usable dataset is an SPD ``train`` ground-truth canary;
``val``, ``test`` and ``test_A`` are never accepted by these contracts and the
resulting artifacts are never ranking or formal-evidence eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .wire import canonical_json_bytes


ASSOCIATION_LEARNING_SCHEMA_VERSION_V1 = 1
ASSOCIATION_DATASET_ID_V1 = "v2x-seq-spd"
ASSOCIATION_SPLIT_NAME_V1 = "train"
ASSOCIATION_SOURCE_KIND_V1 = "spd_train_ground_truth_pair_canary_v1"
ASSOCIATION_CHECKPOINT_NAME_V1 = "association-checkpoint.pt"
ASSOCIATION_MANIFEST_NAME_V1 = "association-training-manifest.json"
ASSOCIATION_APPEARANCE_DIM_V1 = 128
ASSOCIATION_CLASS_VOCABULARY_V1 = (
    "Trafficcone",
    "Pedestrian",
    "Car",
    "Cyclist",
    "Van",
    "Truck",
    "Bus",
    "Tricyclist",
    "Motorcyclist",
    "Barrowlist",
    "Other",
)

_GEOMETRY_FIELDS = (
    "center_x_over_100m",
    "center_y_over_100m",
    "center_z_over_20m",
    "length_over_20m",
    "width_over_10m",
    "height_over_10m",
    "sin_yaw",
    "cos_yaw",
)
_MOTION_FIELDS = ("velocity_x_over_30mps", "velocity_y_over_30mps")
_APPEARANCE_FIELDS = tuple(
    f"frozen_appearance_{index:03d}" for index in range(ASSOCIATION_APPEARANCE_DIM_V1)
)
_CLASS_FIELDS = tuple(f"class_{name}" for name in ASSOCIATION_CLASS_VOCABULARY_V1)
_SOURCE_TIME_FIELDS = (
    "sequence_relative_event_time_over_100s",
    "image_minus_pointcloud_time_seconds",
    "infrastructure_minus_vehicle_time_seconds",
    "source_is_infrastructure",
)
_COVARIANCE_FIELDS = tuple(
    f"covariance_upper_{row}_{column}" for row in range(9) for column in range(row, 9)
)
_POSE_FIELDS = (
    "pose_x_over_100m",
    "pose_y_over_100m",
    "pose_z_over_20m",
    "pose_roll_over_pi",
    "pose_pitch_over_pi",
    "pose_yaw_over_pi",
)
_LINEAGE_FIELDS = (
    "lineage_complete",
    "log1p_factor_count_over_8",
    "log1p_ancestor_count_over_8",
    "lineage_has_cross_agent_ancestor",
)

ASSOCIATION_FEATURE_GROUPS_V1 = (
    ("geometry", _GEOMETRY_FIELDS),
    ("motion", _MOTION_FIELDS),
    ("frozen_appearance", _APPEARANCE_FIELDS),
    ("class", _CLASS_FIELDS),
    ("source_time", _SOURCE_TIME_FIELDS),
    ("covariance", _COVARIANCE_FIELDS),
    ("pose", _POSE_FIELDS),
    ("lineage", _LINEAGE_FIELDS),
)
ASSOCIATION_FEATURE_DIM_V1 = sum(
    len(fields) for _, fields in ASSOCIATION_FEATURE_GROUPS_V1
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_APPEARANCE_SOURCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}")


class AssociationLearningContractError(ValueError):
    """Raised when a learned-association artifact crosses its evidence boundary."""


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise AssociationLearningContractError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AssociationLearningContractError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise AssociationLearningContractError(f"{name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AssociationLearningContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise AssociationLearningContractError(f"{name} must be finite and {qualifier}")
    return result


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise AssociationLearningContractError(f"{name} must be a string-keyed object")
    actual = frozenset(value)
    if actual != expected:
        raise AssociationLearningContractError(
            f"{name} has missing or unknown fields; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return value


def association_feature_schema_document_v1() -> dict[str, object]:
    """Return the complete, ordered 208-dimensional node-feature contract."""

    offset = 0
    groups: list[dict[str, object]] = []
    for name, fields in ASSOCIATION_FEATURE_GROUPS_V1:
        groups.append(
            {
                "dimension": len(fields),
                "end_exclusive": offset + len(fields),
                "fields": list(fields),
                "name": name,
                "start": offset,
            }
        )
        offset += len(fields)
    return {
        "appearance_contract": {
            "dimension": ASSOCIATION_APPEARANCE_DIM_V1,
            "frozen": True,
            "normalization": "l2",
            "training_gradient_allowed": False,
        },
        "class_vocabulary": list(ASSOCIATION_CLASS_VOCABULARY_V1),
        "dimension": ASSOCIATION_FEATURE_DIM_V1,
        "groups": groups,
        "kind": "association_node_feature_schema_v1",
        "schema_version": ASSOCIATION_LEARNING_SCHEMA_VERSION_V1,
        "state_order": ["x", "y", "z", "length", "width", "height", "yaw", "vx", "vy"],
    }


ASSOCIATION_FEATURE_SCHEMA_SHA256_V1 = hashlib.sha256(
    canonical_json_bytes(association_feature_schema_document_v1())
).hexdigest()


@dataclass(frozen=True, slots=True)
class AssociationTrainingConfigV1:
    """Resolved train configuration whose canonical digest is artifact-bound."""

    hidden_dim: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2
    frames_per_step: int = 8
    pairwise_bce_weight: float = 1.0
    assignment_weight: float = 1.0
    max_positive_weight: float = 20.0
    gradient_clip_norm: float = 5.0
    top_h: int = 3

    _FIELDS = frozenset(
        {
            "assignment_weight",
            "dropout",
            "epochs",
            "frames_per_step",
            "gradient_clip_norm",
            "hidden_dim",
            "kind",
            "learning_rate",
            "max_positive_weight",
            "pairwise_bce_weight",
            "schema_version",
            "top_h",
            "weight_decay",
        }
    )

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "epochs", "frames_per_step", "top_h"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        dropout = _positive_float(self.dropout, "dropout", allow_zero=True)
        if dropout >= 1.0:
            raise AssociationLearningContractError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", dropout)
        for name in (
            "learning_rate",
            "pairwise_bce_weight",
            "assignment_weight",
            "max_positive_weight",
            "gradient_clip_norm",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "weight_decay",
            _positive_float(self.weight_decay, "weight_decay", allow_zero=True),
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "assignment_weight": self.assignment_weight,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "frames_per_step": self.frames_per_step,
            "gradient_clip_norm": self.gradient_clip_norm,
            "hidden_dim": self.hidden_dim,
            "kind": "association_training_config_v1",
            "learning_rate": self.learning_rate,
            "max_positive_weight": self.max_positive_weight,
            "pairwise_bce_weight": self.pairwise_bce_weight,
            "schema_version": ASSOCIATION_LEARNING_SCHEMA_VERSION_V1,
            "top_h": self.top_h,
            "weight_decay": self.weight_decay,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "AssociationTrainingConfigV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if (
            result["kind"] != "association_training_config_v1"
            or result["schema_version"] != ASSOCIATION_LEARNING_SCHEMA_VERSION_V1
        ):
            raise AssociationLearningContractError(
                "unsupported association train config"
            )
        return cls(
            hidden_dim=result["hidden_dim"],
            dropout=result["dropout"],
            learning_rate=result["learning_rate"],
            weight_decay=result["weight_decay"],
            epochs=result["epochs"],
            frames_per_step=result["frames_per_step"],
            pairwise_bce_weight=result["pairwise_bce_weight"],
            assignment_weight=result["assignment_weight"],
            max_positive_weight=result["max_positive_weight"],
            gradient_clip_norm=result["gradient_clip_norm"],
            top_h=result["top_h"],
        )


_METRIC_FIELDS = frozenset(
    {
        "train_loss",
        "validation_assignment_accuracy",
        "validation_loss",
        "validation_pair_accuracy",
    }
)


@dataclass(frozen=True, slots=True)
class AssociationTrainingArtifactManifestV1:
    """Sealed manifest for a non-publishable SPD-train learning canary."""

    fold_id: int
    training_seed: int
    development_manifest_sha256: str
    official_split_sha256: str
    cohort_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    appearance_source: str
    covariance_source: str
    train_sequence_count: int
    validation_sequence_count: int
    train_frame_count: int
    validation_frame_count: int
    train_object_count: int
    validation_object_count: int
    metrics: Mapping[str, float]
    dataset_id: str = ASSOCIATION_DATASET_ID_V1
    split_name: str = ASSOCIATION_SPLIT_NAME_V1
    source_kind: str = ASSOCIATION_SOURCE_KIND_V1
    feature_schema_sha256: str = ASSOCIATION_FEATURE_SCHEMA_SHA256_V1
    checkpoint_relative_path: str = ASSOCIATION_CHECKPOINT_NAME_V1
    gt_supervised_development_only: bool = True
    ranking_eligible: bool = False
    formal_evidence: bool = False
    paper_registry_writable: bool = False

    _PAYLOAD_FIELDS = frozenset(
        {
            "appearance_source",
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "cohort_sha256",
            "config_sha256",
            "covariance_source",
            "dataset_id",
            "development_manifest_sha256",
            "feature_schema_sha256",
            "fold_id",
            "formal_evidence",
            "gt_supervised_development_only",
            "kind",
            "metrics",
            "official_split_sha256",
            "paper_registry_writable",
            "ranking_eligible",
            "schema_version",
            "source_kind",
            "split_name",
            "train_frame_count",
            "train_object_count",
            "train_sequence_count",
            "training_seed",
            "validation_frame_count",
            "validation_object_count",
            "validation_sequence_count",
        }
    )

    def __post_init__(self) -> None:
        if self.dataset_id != ASSOCIATION_DATASET_ID_V1:
            raise AssociationLearningContractError(
                "learned association only accepts V2X-Seq-SPD"
            )
        if self.split_name != ASSOCIATION_SPLIT_NAME_V1:
            raise AssociationLearningContractError(
                "learned association training is fail-closed to SPD train; "
                "val/test/test_A are forbidden"
            )
        if self.source_kind != ASSOCIATION_SOURCE_KIND_V1:
            raise AssociationLearningContractError(
                "unsupported association supervision source"
            )
        if not 0 <= _integer(self.fold_id, "fold_id") < 5:
            raise AssociationLearningContractError("fold_id must be in [0, 5)")
        _integer(self.training_seed, "training_seed")
        for name in (
            "development_manifest_sha256",
            "official_split_sha256",
            "cohort_sha256",
            "config_sha256",
            "checkpoint_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.feature_schema_sha256 != ASSOCIATION_FEATURE_SCHEMA_SHA256_V1:
            raise AssociationLearningContractError("unknown association feature schema")
        if self.checkpoint_relative_path != ASSOCIATION_CHECKPOINT_NAME_V1:
            raise AssociationLearningContractError("checkpoint path is not canonical")
        for name in ("appearance_source", "covariance_source"):
            value = getattr(self, name)
            if type(value) is not str or _APPEARANCE_SOURCE_RE.fullmatch(value) is None:
                raise AssociationLearningContractError(
                    f"{name} is not a canonical identifier"
                )
        for name in (
            "train_sequence_count",
            "validation_sequence_count",
            "train_frame_count",
            "validation_frame_count",
        ):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("train_object_count", "validation_object_count"):
            _integer(getattr(self, name), name)
        if (
            self.gt_supervised_development_only is not True
            or self.ranking_eligible is not False
            or self.formal_evidence is not False
            or self.paper_registry_writable is not False
        ):
            raise AssociationLearningContractError(
                "development artifacts must remain GT-supervised, non-ranking, "
                "non-formal, and outside paper registries"
            )
        metrics = _strict_mapping(self.metrics, _METRIC_FIELDS, "metrics")
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, bool):
                raise AssociationLearningContractError(
                    f"metrics.{name} must be numeric"
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise AssociationLearningContractError(
                    f"metrics.{name} must be numeric"
                ) from exc
            if not math.isfinite(number) or number < 0.0:
                raise AssociationLearningContractError(
                    f"metrics.{name} must be finite and non-negative"
                )
            if name.endswith("accuracy") and number > 1.0:
                raise AssociationLearningContractError(
                    f"metrics.{name} must be in [0, 1]"
                )
            normalized[name] = number
        object.__setattr__(self, "metrics", normalized)

    def payload(self) -> dict[str, object]:
        return {
            "appearance_source": self.appearance_source,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "cohort_sha256": self.cohort_sha256,
            "config_sha256": self.config_sha256,
            "covariance_source": self.covariance_source,
            "dataset_id": self.dataset_id,
            "development_manifest_sha256": self.development_manifest_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "fold_id": self.fold_id,
            "formal_evidence": self.formal_evidence,
            "gt_supervised_development_only": self.gt_supervised_development_only,
            "kind": "association_training_artifact_manifest_v1",
            "metrics": dict(sorted(self.metrics.items())),
            "official_split_sha256": self.official_split_sha256,
            "paper_registry_writable": self.paper_registry_writable,
            "ranking_eligible": self.ranking_eligible,
            "schema_version": ASSOCIATION_LEARNING_SCHEMA_VERSION_V1,
            "source_kind": self.source_kind,
            "split_name": self.split_name,
            "train_frame_count": self.train_frame_count,
            "train_object_count": self.train_object_count,
            "train_sequence_count": self.train_sequence_count,
            "training_seed": self.training_seed,
            "validation_frame_count": self.validation_frame_count,
            "validation_object_count": self.validation_object_count,
            "validation_sequence_count": self.validation_sequence_count,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {**self.payload(), "content_sha256": self.content_sha256}
        )

    @classmethod
    def from_mapping(cls, value: object) -> "AssociationTrainingArtifactManifestV1":
        result = _strict_mapping(
            value,
            cls._PAYLOAD_FIELDS | {"content_sha256"},
            cls.__name__,
        )
        if (
            result["kind"] != "association_training_artifact_manifest_v1"
            or result["schema_version"] != ASSOCIATION_LEARNING_SCHEMA_VERSION_V1
        ):
            raise AssociationLearningContractError(
                "unsupported association artifact schema"
            )
        observed = _sha256(result["content_sha256"], "content_sha256")
        payload = {name: result[name] for name in cls._PAYLOAD_FIELDS}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise AssociationLearningContractError("artifact manifest SHA-256 mismatch")
        return cls(
            fold_id=result["fold_id"],
            training_seed=result["training_seed"],
            development_manifest_sha256=result["development_manifest_sha256"],
            official_split_sha256=result["official_split_sha256"],
            cohort_sha256=result["cohort_sha256"],
            config_sha256=result["config_sha256"],
            checkpoint_sha256=result["checkpoint_sha256"],
            appearance_source=result["appearance_source"],
            covariance_source=result["covariance_source"],
            train_sequence_count=result["train_sequence_count"],
            validation_sequence_count=result["validation_sequence_count"],
            train_frame_count=result["train_frame_count"],
            validation_frame_count=result["validation_frame_count"],
            train_object_count=result["train_object_count"],
            validation_object_count=result["validation_object_count"],
            metrics=result["metrics"],
            dataset_id=result["dataset_id"],
            split_name=result["split_name"],
            source_kind=result["source_kind"],
            feature_schema_sha256=result["feature_schema_sha256"],
            checkpoint_relative_path=result["checkpoint_relative_path"],
            gt_supervised_development_only=result["gt_supervised_development_only"],
            ranking_eligible=result["ranking_eligible"],
            formal_evidence=result["formal_evidence"],
            paper_registry_writable=result["paper_registry_writable"],
        )


def decode_association_training_manifest_v1(
    data: bytes,
) -> AssociationTrainingArtifactManifestV1:
    if type(data) is not bytes:
        raise TypeError("association manifest data must be bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AssociationLearningContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AssociationLearningContractError(
                    f"non-finite JSON constant is forbidden: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssociationLearningContractError(
            "invalid association manifest JSON"
        ) from exc
    manifest = AssociationTrainingArtifactManifestV1.from_mapping(value)
    if manifest.canonical_bytes != data:
        raise AssociationLearningContractError(
            "association manifest JSON is not canonical"
        )
    return manifest


def verify_association_training_artifact(
    directory: str | Path,
) -> AssociationTrainingArtifactManifestV1:
    """Verify the sealed two-file artifact and its non-publication boundary."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise AssociationLearningContractError(
            "artifact directory must be a real directory"
        )
    observed_names = {item.name for item in root.iterdir()}
    expected_names = {
        ASSOCIATION_CHECKPOINT_NAME_V1,
        ASSOCIATION_MANIFEST_NAME_V1,
    }
    if observed_names != expected_names:
        raise AssociationLearningContractError(
            "artifact directory has missing or unexpected files"
        )
    manifest_path = root / ASSOCIATION_MANIFEST_NAME_V1
    checkpoint_path = root / ASSOCIATION_CHECKPOINT_NAME_V1
    for path in (manifest_path, checkpoint_path):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise AssociationLearningContractError(
                "artifact members must be regular files"
            )
    manifest = decode_association_training_manifest_v1(manifest_path.read_bytes())
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != manifest.checkpoint_sha256:
        raise AssociationLearningContractError("checkpoint SHA-256 mismatch")
    return manifest


__all__ = [
    "ASSOCIATION_APPEARANCE_DIM_V1",
    "ASSOCIATION_CHECKPOINT_NAME_V1",
    "ASSOCIATION_CLASS_VOCABULARY_V1",
    "ASSOCIATION_DATASET_ID_V1",
    "ASSOCIATION_FEATURE_DIM_V1",
    "ASSOCIATION_FEATURE_GROUPS_V1",
    "ASSOCIATION_FEATURE_SCHEMA_SHA256_V1",
    "ASSOCIATION_LEARNING_SCHEMA_VERSION_V1",
    "ASSOCIATION_MANIFEST_NAME_V1",
    "ASSOCIATION_SOURCE_KIND_V1",
    "ASSOCIATION_SPLIT_NAME_V1",
    "AssociationLearningContractError",
    "AssociationTrainingArtifactManifestV1",
    "AssociationTrainingConfigV1",
    "association_feature_schema_document_v1",
    "decode_association_training_manifest_v1",
    "verify_association_training_artifact",
]
