"""Real SPD-train pair construction for learned cross-agent association.

Node inputs come independently from each side's point-cloud labels.  The
cooperative ground truth is used only to map ``veh_track_id`` and
``inf_track_id`` into positive edges.  Missing counterparts become explicit
dustbin targets.  No synthetic observations or cooperative-label geometry is
used as an input feature.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping, Protocol, Sequence

import numpy as np

from transvision.models.event_track_v2x.learning_contracts import (
    ASSOCIATION_APPEARANCE_DIM_V1,
    ASSOCIATION_CLASS_VOCABULARY_V1,
    ASSOCIATION_FEATURE_DIM_V1,
    ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes

from .event_track_v2x_spd import SPDPair, load_spd_metadata


class SPDAssociationDataError(ValueError):
    """Raised when an association sample could touch a sealed SPD split."""


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if type(value) is not str or not value or value != value.strip():
        raise SPDAssociationDataError(f"{name} must be a trimmed relative path")
    result = PurePosixPath(value)
    if (
        result.is_absolute()
        or result.as_posix() != value
        or any(part in ("", ".", "..") for part in result.parts)
        or "\\" in value
        or "\x00" in value
    ):
        raise SPDAssociationDataError(f"{name} must be a safe POSIX relative path")
    return result


def _regular_path(root: Path, relative: object, name: str) -> Path:
    pure = _safe_relative_path(relative, name)
    candidate = root.joinpath(*pure.parts)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as exc:
        raise SPDAssociationDataError(f"required file is missing: {pure}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SPDAssociationDataError(
            f"input must be a regular non-symlink file: {pure}"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SPDAssociationDataError(f"input escapes dataset root: {pure}") from exc
    return resolved


def _read_json(path: Path, *, root: Path) -> object:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SPDAssociationDataError("JSON path escapes dataset root") from exc
    path = _regular_path(root, relative, "JSON path")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise SPDAssociationDataError(
                    f"duplicate JSON key in {relative}: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SPDAssociationDataError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SPDAssociationDataError(f"invalid JSON: {relative}") from exc


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise SPDAssociationDataError(f"{name} must be a string-keyed object")
    return value


def _array(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SPDAssociationDataError(f"{name} must be an array")
    return value


def _sequence_id(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 4 or not value.isdecimal():
        raise SPDAssociationDataError(f"{name} must be a four-digit sequence ID")
    return value


def _track_id(value: object, name: str) -> str:
    if type(value) not in (str, int) or isinstance(value, bool):
        raise SPDAssociationDataError(f"{name} must be a string or integer")
    result = str(value)
    if not result or result != result.strip() or "/" in result or "\\" in result:
        raise SPDAssociationDataError(f"{name} is not a canonical track ID")
    return result


def _float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise SPDAssociationDataError(f"{name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SPDAssociationDataError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SPDAssociationDataError(f"{name} must be finite")
    return result


def _readonly_float32(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SPDAssociationDataError(f"{name} must be numeric") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise SPDAssociationDataError(f"{name} must have finite shape {shape}")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float32).reshape(shape)


@dataclass(frozen=True, slots=True)
class AppearanceObservationV1:
    sequence_id: str
    side: str
    frame_id: str
    source_track_id: str
    image_path: Path
    box_xyxy: tuple[float, float, float, float] | None

    @property
    def cache_key(self) -> str:
        return f"{self.sequence_id}/{self.side}/{self.frame_id}/{self.source_track_id}"


class FrozenAppearanceProviderV1(Protocol):
    @property
    def source_id(self) -> str: ...

    def embedding(self, observation: AppearanceObservationV1) -> np.ndarray: ...


def _crop_rgb(observation: AppearanceObservationV1, *, size: tuple[int, int]):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a runtime dependency.
        raise SPDAssociationDataError(
            "Pillow is required for image appearance"
        ) from exc
    path = observation.image_path
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SPDAssociationDataError("appearance image must be a regular file")
    with Image.open(path) as image:
        image = image.convert("RGB")
        if observation.box_xyxy is not None:
            xmin, ymin, xmax, ymax = observation.box_xyxy
            xmin = max(0.0, min(float(image.width), xmin))
            xmax = max(0.0, min(float(image.width), xmax))
            ymin = max(0.0, min(float(image.height), ymin))
            ymax = max(0.0, min(float(image.height), ymax))
            if xmax > xmin and ymax > ymin:
                image = image.crop((xmin, ymin, xmax, ymax))
        return image.resize(size)


class FrozenRGBProjectionAppearanceV1:
    """Deterministic 128-D crop descriptor for a restricted local canary.

    This is a frozen image-derived projection, not a claimed ResNet feature.
    The artifact records the source ID so it cannot be confused with a formal
    appearance backbone.
    """

    source_id = "frozen_rgb_projection_128_canary_v1"

    def __init__(self) -> None:
        rows = np.arange(ASSOCIATION_APPEARANCE_DIM_V1, dtype=np.float64)[:, None]
        columns = np.arange(8 * 8 * 3, dtype=np.float64)[None, :]
        projection = np.cos((rows + 0.5) * (columns + 0.5) * math.pi / (8 * 8 * 3))
        projection /= np.linalg.norm(projection, axis=1, keepdims=True)
        self._projection = projection.astype(np.float32)
        self._cache: dict[str, np.ndarray] = {}

    def embedding(self, observation: AppearanceObservationV1) -> np.ndarray:
        cached = self._cache.get(observation.cache_key)
        if cached is not None:
            return cached
        image = _crop_rgb(observation, size=(8, 8))
        pixels = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0 - 0.5
        embedding = np.einsum("ij,j->i", self._projection, pixels)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        frozen = _readonly_float32(
            embedding,
            (ASSOCIATION_APPEARANCE_DIM_V1,),
            "appearance embedding",
        )
        self._cache[observation.cache_key] = frozen
        return frozen


class NPZAppearanceCacheV1:
    """Strict frozen cache with ``keys`` and 128-D ``embeddings`` arrays."""

    def __init__(self, path: str | Path) -> None:
        cache_path = Path(path)
        metadata = cache_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SPDAssociationDataError("appearance cache must be a regular file")
        raw_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                if set(archive.files) != {"embeddings", "keys"}:
                    raise SPDAssociationDataError(
                        "appearance cache must contain exactly keys and embeddings"
                    )
                keys = np.asarray(archive["keys"])
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        except (OSError, ValueError) as exc:
            if isinstance(exc, SPDAssociationDataError):
                raise
            raise SPDAssociationDataError("invalid appearance NPZ cache") from exc
        if keys.ndim != 1 or keys.dtype.kind not in {"U", "S"}:
            raise SPDAssociationDataError(
                "appearance cache keys must be a string vector"
            )
        if embeddings.shape != (len(keys), ASSOCIATION_APPEARANCE_DIM_V1):
            raise SPDAssociationDataError(
                "appearance cache embeddings have wrong shape"
            )
        if not np.all(np.isfinite(embeddings)):
            raise SPDAssociationDataError("appearance cache contains non-finite values")
        normalized_keys = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in keys.tolist()
        )
        if len(set(normalized_keys)) != len(normalized_keys):
            raise SPDAssociationDataError("appearance cache keys must be unique")
        self._values = {
            key: _readonly_float32(
                embedding / max(float(np.linalg.norm(embedding)), 1e-12),
                (ASSOCIATION_APPEARANCE_DIM_V1,),
                "cached appearance embedding",
            )
            for key, embedding in zip(normalized_keys, embeddings)
        }
        self._source_id = f"frozen_npz_{raw_sha}_v1"

    @property
    def source_id(self) -> str:
        return self._source_id

    def embedding(self, observation: AppearanceObservationV1) -> np.ndarray:
        try:
            return self._values[observation.cache_key]
        except KeyError as exc:
            raise SPDAssociationDataError(
                f"appearance cache misses {observation.cache_key}"
            ) from exc


class FrozenResNet50AppearanceV1:
    """Plug-in frozen ResNet-50 crop generator using a caller-pinned checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        expected_checkpoint_sha256: str | None = None,
        device: str = "cpu",
    ) -> None:
        checkpoint_path = Path(checkpoint)
        metadata = checkpoint_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SPDAssociationDataError("ResNet-50 checkpoint must be a regular file")
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if (
            expected_checkpoint_sha256 is not None
            and digest != expected_checkpoint_sha256
        ):
            raise SPDAssociationDataError("ResNet-50 checkpoint SHA-256 mismatch")
        try:
            import torch
            from torch import nn
            from torchvision.models import resnet50
            from torchvision.transforms import Compose, Normalize, ToTensor
        except ImportError as exc:  # pragma: no cover - optional provider.
            raise SPDAssociationDataError(
                "torch and torchvision are required for ResNet-50 appearance"
            ) from exc
        used_device = torch.device(device)
        if used_device.type == "cuda" and not torch.cuda.is_available():
            raise SPDAssociationDataError(
                "CUDA appearance generation was requested but unavailable"
            )
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0 compatibility.
            state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise SPDAssociationDataError("ResNet-50 checkpoint has no state dict")
        normalized_state = {
            str(key).removeprefix("module."): value for key, value in state.items()
        }
        model = resnet50(weights=None)
        try:
            model.load_state_dict(normalized_state, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise SPDAssociationDataError(
                "checkpoint is not an exact torchvision ResNet-50"
            ) from exc
        model.fc = nn.Identity()
        model.eval().to(used_device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._torch = torch
        self._model = model
        self._device = used_device
        self._transform = Compose(
            (
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )
        self._cache: dict[str, np.ndarray] = {}
        self._source_id = f"frozen_resnet50_{digest}_128_v1"

    @property
    def source_id(self) -> str:
        return self._source_id

    def embedding(self, observation: AppearanceObservationV1) -> np.ndarray:
        cached = self._cache.get(observation.cache_key)
        if cached is not None:
            return cached
        image = _crop_rgb(observation, size=(224, 224))
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model(tensor).reshape(16, ASSOCIATION_APPEARANCE_DIM_V1)
            embedding = features.mean(dim=0)
            embedding = embedding / self._torch.clamp(embedding.norm(), min=1e-12)
        value = _readonly_float32(
            embedding.cpu().numpy(),
            (ASSOCIATION_APPEARANCE_DIM_V1,),
            "ResNet-50 appearance embedding",
        )
        self._cache[observation.cache_key] = value
        return value


@dataclass(frozen=True, slots=True)
class AssociationNodeInputV1:
    geometry_xyz_lwh_yaw: tuple[float, float, float, float, float, float, float]
    motion_xy: tuple[float, float]
    appearance: np.ndarray
    class_label: str
    sequence_relative_event_time_s: float
    image_minus_pointcloud_time_s: float
    infrastructure_minus_vehicle_time_s: float
    source_is_infrastructure: bool
    covariance: np.ndarray
    pose_xyz_rpy: tuple[float, float, float, float, float, float]
    lineage_complete: bool
    lineage_factor_count: int
    lineage_ancestor_count: int
    lineage_has_cross_agent_ancestor: bool


@dataclass(frozen=True, slots=True)
class _SideTrackLabelV1:
    track_id: str
    token: str
    category: str
    center_xyz: tuple[float, float, float]
    dimensions_lwh: tuple[float, float, float]
    yaw: float


def encode_association_node_v1(node: AssociationNodeInputV1) -> np.ndarray:
    """Encode one object exactly according to the frozen 208-D schema."""

    geometry = np.asarray(node.geometry_xyz_lwh_yaw, dtype=np.float64)
    if geometry.shape != (7,) or not np.all(np.isfinite(geometry)):
        raise SPDAssociationDataError(
            "geometry must be a finite x/y/z/l/w/h/yaw vector"
        )
    if np.any(geometry[3:6] <= 0.0):
        raise SPDAssociationDataError("geometry dimensions must be positive")
    yaw = geometry[6]
    geometry_features = np.asarray(
        (
            geometry[0] / 100.0,
            geometry[1] / 100.0,
            geometry[2] / 20.0,
            geometry[3] / 20.0,
            geometry[4] / 10.0,
            geometry[5] / 10.0,
            math.sin(yaw),
            math.cos(yaw),
        ),
        dtype=np.float32,
    )
    motion = np.asarray(node.motion_xy, dtype=np.float64)
    if motion.shape != (2,) or not np.all(np.isfinite(motion)):
        raise SPDAssociationDataError("motion must be a finite vx/vy vector")
    motion_features = (motion / 30.0).astype(np.float32)
    appearance = _readonly_float32(
        node.appearance,
        (ASSOCIATION_APPEARANCE_DIM_V1,),
        "appearance",
    ).astype(np.float32, copy=True)
    appearance_norm = float(np.linalg.norm(appearance))
    if appearance_norm > 0.0:
        appearance /= appearance_norm
    class_features = np.zeros(len(ASSOCIATION_CLASS_VOCABULARY_V1), dtype=np.float32)
    try:
        class_index = ASSOCIATION_CLASS_VOCABULARY_V1.index(node.class_label)
    except ValueError:
        class_index = ASSOCIATION_CLASS_VOCABULARY_V1.index("Other")
    class_features[class_index] = 1.0
    time_values = np.asarray(
        (
            _float(node.sequence_relative_event_time_s, "sequence relative event time")
            / 100.0,
            _float(node.image_minus_pointcloud_time_s, "image/pointcloud time delta"),
            _float(
                node.infrastructure_minus_vehicle_time_s,
                "infrastructure/vehicle time delta",
            ),
            float(node.source_is_infrastructure),
        ),
        dtype=np.float32,
    )
    covariance = np.asarray(node.covariance, dtype=np.float64)
    if covariance.shape != (9, 9) or not np.all(np.isfinite(covariance)):
        raise SPDAssociationDataError("covariance must be a finite 9x9 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        raise SPDAssociationDataError("covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise SPDAssociationDataError("covariance must be positive definite") from exc
    covariance_features = np.asarray(
        [
            covariance[row, column] / 100.0
            for row in range(9)
            for column in range(row, 9)
        ],
        dtype=np.float32,
    )
    pose = np.asarray(node.pose_xyz_rpy, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise SPDAssociationDataError("pose must be a finite xyz/rpy vector")
    pose_features = np.asarray(
        (pose[0] / 100.0, pose[1] / 100.0, pose[2] / 20.0, *(pose[3:] / math.pi)),
        dtype=np.float32,
    )
    if (
        type(node.lineage_complete) is not bool
        or type(node.lineage_has_cross_agent_ancestor) is not bool
        or type(node.lineage_factor_count) is not int
        or type(node.lineage_ancestor_count) is not int
        or node.lineage_factor_count < 0
        or node.lineage_ancestor_count < 0
    ):
        raise SPDAssociationDataError("lineage fields are malformed")
    lineage_features = np.asarray(
        (
            float(node.lineage_complete),
            math.log1p(node.lineage_factor_count) / 8.0,
            math.log1p(node.lineage_ancestor_count) / 8.0,
            float(node.lineage_has_cross_agent_ancestor),
        ),
        dtype=np.float32,
    )
    result = np.concatenate(
        (
            geometry_features,
            motion_features,
            appearance,
            class_features,
            time_values,
            covariance_features,
            pose_features,
            lineage_features,
        )
    )
    if result.shape != (
        ASSOCIATION_FEATURE_DIM_V1,
    ):  # pragma: no cover - schema guard.
        raise AssertionError("association feature encoder drifted from its schema")
    return _readonly_float32(result, result.shape, "encoded association feature")


@dataclass(frozen=True, slots=True, eq=False)
class AssociationFrameSampleV1:
    sequence_id: str
    vehicle_frame_id: str
    infrastructure_frame_id: str
    left_identity_ids: tuple[str, ...]
    right_identity_ids: tuple[str, ...]
    left_features: np.ndarray
    right_features: np.ndarray
    targets: np.ndarray

    def __post_init__(self) -> None:
        left_count = len(self.left_identity_ids)
        right_count = len(self.right_identity_ids)
        if (
            len(set(self.left_identity_ids)) != left_count
            or len(set(self.right_identity_ids)) != right_count
        ):
            raise SPDAssociationDataError("sample identities must be unique per side")
        left = _readonly_float32(
            self.left_features,
            (left_count, ASSOCIATION_FEATURE_DIM_V1),
            "left_features",
        )
        right = _readonly_float32(
            self.right_features,
            (right_count, ASSOCIATION_FEATURE_DIM_V1),
            "right_features",
        )
        targets = np.asarray(self.targets, dtype=np.uint8)
        if targets.shape != (left_count, right_count) or np.any(targets > 1):
            raise SPDAssociationDataError("targets must be a binary left/right matrix")
        if targets.size and (
            np.any(targets.sum(axis=1) > 1) or np.any(targets.sum(axis=0) > 1)
        ):
            raise SPDAssociationDataError("targets must be one-to-one")
        frozen_targets = np.frombuffer(
            targets.tobytes(order="C"), dtype=np.uint8
        ).reshape(targets.shape)
        object.__setattr__(self, "left_features", left)
        object.__setattr__(self, "right_features", right)
        object.__setattr__(self, "targets", frozen_targets)

    @property
    def content_sha256(self) -> str:
        header = {
            "feature_schema_sha256": ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
            "infrastructure_frame_id": self.infrastructure_frame_id,
            "left_identity_ids": list(self.left_identity_ids),
            "right_identity_ids": list(self.right_identity_ids),
            "sequence_id": self.sequence_id,
            "target_shape": list(self.targets.shape),
            "vehicle_frame_id": self.vehicle_frame_id,
        }
        digest = hashlib.sha256(canonical_json_bytes(header))
        digest.update(self.left_features.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(self.right_features.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(self.targets.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SPDAssociationCohortV1:
    train_samples: tuple[AssociationFrameSampleV1, ...]
    validation_samples: tuple[AssociationFrameSampleV1, ...]
    train_sequence_ids: tuple[str, ...]
    validation_sequence_ids: tuple[str, ...]
    appearance_source: str
    covariance_source: str = "spd_gt_diagonal_proxy_v1"

    def __post_init__(self) -> None:
        if not self.train_samples or not self.validation_samples:
            raise SPDAssociationDataError(
                "fit and held-out samples must both be non-empty"
            )
        if set(self.train_sequence_ids).intersection(self.validation_sequence_ids):
            raise SPDAssociationDataError("fit and held-out sequences overlap")
        for role, samples, sequences in (
            ("fit", self.train_samples, set(self.train_sequence_ids)),
            ("held_out", self.validation_samples, set(self.validation_sequence_ids)),
        ):
            if {sample.sequence_id for sample in samples} - sequences:
                raise SPDAssociationDataError(
                    f"{role} samples escape their sequence set"
                )

    @property
    def content_sha256(self) -> str:
        payload = {
            "appearance_source": self.appearance_source,
            "covariance_source": self.covariance_source,
            "feature_schema_sha256": ASSOCIATION_FEATURE_SCHEMA_SHA256_V1,
            "held_out": [
                {
                    "sample_sha256": sample.content_sha256,
                    "sequence_id": sample.sequence_id,
                    "vehicle_frame_id": sample.vehicle_frame_id,
                }
                for sample in self.validation_samples
            ],
            "held_out_sequence_ids": list(self.validation_sequence_ids),
            "kind": "spd_association_training_cohort_v1",
            "fit": [
                {
                    "sample_sha256": sample.content_sha256,
                    "sequence_id": sample.sequence_id,
                    "vehicle_frame_id": sample.vehicle_frame_id,
                }
                for sample in self.train_samples
            ],
            "fit_sequence_ids": list(self.train_sequence_ids),
            "schema_version": 1,
            "split_name": "train",
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_spd_train_only_projection_v1(
    dataset_root: str | Path,
    *,
    official_train_sequence_ids: Sequence[str],
) -> Path:
    """Reject a directory whose metadata exposes any sealed sequence."""

    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SPDAssociationDataError("dataset_root must be a real directory")
    official = tuple(
        _sequence_id(item, "official train sequence")
        for item in official_train_sequence_ids
    )
    if not official or len(set(official)) != len(official):
        raise SPDAssociationDataError(
            "official train sequence IDs must be non-empty and unique"
        )
    allowed = set(official)
    observed: set[str] = set()
    for side in ("vehicle", "infrastructure"):
        path = root / f"{side}-side/data_info.json"
        records = _array(_read_json(path, root=root), path.name)
        for index, raw in enumerate(records):
            record = _mapping(raw, f"{side}-side/data_info.json[{index}]")
            sequence = _sequence_id(record.get("sequence_id"), "side sequence_id")
            observed.add(sequence)
            if sequence not in allowed:
                raise SPDAssociationDataError(
                    f"train-only projection exposes sealed sequence {sequence}"
                )
    cooperative_path = root / "cooperative/data_info.json"
    records = _array(_read_json(cooperative_path, root=root), cooperative_path.name)
    cooperative_sequences: set[str] = set()
    for index, raw in enumerate(records):
        record = _mapping(raw, f"cooperative/data_info.json[{index}]")
        vehicle_sequence = _sequence_id(
            record.get("vehicle_sequence"), "cooperative vehicle_sequence"
        )
        infrastructure_sequence = _sequence_id(
            record.get("infrastructure_sequence"),
            "cooperative infrastructure_sequence",
        )
        if vehicle_sequence != infrastructure_sequence:
            raise SPDAssociationDataError(
                "cooperative record joins different sequences"
            )
        if vehicle_sequence not in allowed:
            raise SPDAssociationDataError(
                f"train-only projection exposes sealed sequence {vehicle_sequence}"
            )
        cooperative_sequences.add(vehicle_sequence)
    missing = sorted(allowed - cooperative_sequences)
    if missing:
        raise SPDAssociationDataError(
            f"train-only projection is missing official train sequences: {missing}"
        )
    return root


def _side_indexes(root: Path, side: str) -> dict[str, Mapping[str, object]]:
    path = root / f"{side}-side/data_info.json"
    records = _array(_read_json(path, root=root), path.name)
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{side}-side/data_info.json[{index}]")
        frame_id = record.get("frame_id")
        if type(frame_id) is not str or len(frame_id) != 6 or not frame_id.isdecimal():
            raise SPDAssociationDataError("side frame_id must be six digits")
        if frame_id in result:
            raise SPDAssociationDataError(f"duplicate {side} frame_id: {frame_id}")
        result[frame_id] = record
    return result


def _box_xyxy(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        values = tuple(
            _float(raw[name], f"2d_box.{name}")
            for name in ("xmin", "ymin", "xmax", "ymax")
        )
    except (KeyError, SPDAssociationDataError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


def _camera_labels(
    root: Path,
    side: str,
    record: Mapping[str, object],
) -> dict[str, tuple[float, float, float, float] | None]:
    relative = record.get("label_camera_std_path")
    if relative is None:
        return {}
    label_path = _regular_path(root, f"{side}-side/{relative}", "camera label path")
    labels = _array(_read_json(label_path, root=root), "camera labels")
    result: dict[str, tuple[float, float, float, float] | None] = {}
    for index, raw in enumerate(labels):
        label = _mapping(raw, f"camera label[{index}]")
        if "track_id" not in label:
            continue
        track_id = _track_id(label["track_id"], "camera label track_id")
        if track_id in result:
            raise SPDAssociationDataError("camera labels contain duplicate track_id")
        result[track_id] = _box_xyxy(label.get("2d_box"))
    return result


def _trimmed_string(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SPDAssociationDataError(f"{name} must be a trimmed non-empty string")
    return value


def _pointcloud_labels(
    root: Path,
    side: str,
    record: Mapping[str, object],
) -> dict[str, _SideTrackLabelV1]:
    relative = record.get("label_lidar_std_path")
    if relative is None:
        raise SPDAssociationDataError(
            f"{side} data_info is missing label_lidar_std_path"
        )
    label_path = _regular_path(
        root,
        f"{side}-side/{relative}",
        f"{side} pointcloud label path",
    )
    labels = _array(_read_json(label_path, root=root), f"{side} pointcloud labels")
    result: dict[str, _SideTrackLabelV1] = {}
    tokens: set[str] = set()
    required = {
        "track_id",
        "token",
        "type",
        "3d_dimensions",
        "3d_location",
        "rotation",
    }
    for index, raw in enumerate(labels):
        context = f"{side} pointcloud label[{index}]"
        label = _mapping(raw, context)
        missing = sorted(required - set(label))
        if missing:
            raise SPDAssociationDataError(f"{context} is missing fields: {missing}")
        track_id = _track_id(label["track_id"], f"{context}.track_id")
        token = _trimmed_string(label["token"], f"{context}.token")
        if track_id == "-1" or token == "-1":
            raise SPDAssociationDataError(
                f"{context} cannot use the cooperative missing-value sentinel"
            )
        if track_id in result:
            raise SPDAssociationDataError(
                f"{side} pointcloud labels contain duplicate track_id {track_id}"
            )
        if token in tokens:
            raise SPDAssociationDataError(
                f"{side} pointcloud labels contain duplicate token {token}"
            )
        dimensions = _mapping(label["3d_dimensions"], f"{context}.3d_dimensions")
        location = _mapping(label["3d_location"], f"{context}.3d_location")
        dimensions_lwh = (
            _float(dimensions.get("l"), f"{context}.3d_dimensions.l"),
            _float(dimensions.get("w"), f"{context}.3d_dimensions.w"),
            _float(dimensions.get("h"), f"{context}.3d_dimensions.h"),
        )
        if any(dimension <= 0.0 for dimension in dimensions_lwh):
            raise SPDAssociationDataError(f"{context}.3d_dimensions must be positive")
        side_label = _SideTrackLabelV1(
            track_id=track_id,
            token=token,
            category=_trimmed_string(label["type"], f"{context}.type"),
            center_xyz=(
                _float(location.get("x"), f"{context}.3d_location.x"),
                _float(location.get("y"), f"{context}.3d_location.y"),
                _float(location.get("z"), f"{context}.3d_location.z"),
            ),
            dimensions_lwh=dimensions_lwh,
            yaw=_float(label["rotation"], f"{context}.rotation"),
        )
        result[track_id] = side_label
        tokens.add(token)
    return result


def _image_path(root: Path, side: str, record: Mapping[str, object]) -> Path:
    relative = record.get("image_path")
    return _regular_path(root, f"{side}-side/{relative}", "image path")


def _proxy_covariance(label: _SideTrackLabelV1) -> np.ndarray:
    length, width, height = label.dimensions_lwh
    standard_deviations = np.asarray(
        (
            1.0,
            1.0,
            0.5,
            max(0.1, 0.05 * length),
            max(0.1, 0.05 * width),
            max(0.1, 0.05 * height),
            0.1,
            2.0,
            2.0,
        ),
        dtype=np.float64,
    )
    return np.diag(standard_deviations**2)


def _feature_with_root(
    *,
    root: Path,
    pair: SPDPair,
    label: _SideTrackLabelV1,
    side: str,
    source_track_id: str,
    side_record: Mapping[str, object],
    camera_boxes: Mapping[str, tuple[float, float, float, float] | None],
    provider: FrozenAppearanceProviderV1,
    velocity_xy: tuple[float, float],
    sequence_origin_us: int,
) -> np.ndarray:
    side_frame = pair.vehicle if side == "vehicle" else pair.infrastructure
    pair_delta_s = (
        pair.infrastructure.pointcloud_timestamp_us
        - pair.vehicle.pointcloud_timestamp_us
    ) / 1_000_000.0
    observation = AppearanceObservationV1(
        sequence_id=pair.sequence_id,
        side=side,
        frame_id=side_frame.frame_id,
        source_track_id=source_track_id,
        image_path=_image_path(root, side, side_record),
        box_xyxy=camera_boxes.get(source_track_id),
    )
    delta_x, delta_y = pair.system_error_offset_xy
    pose = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if side == "vehicle"
        else (delta_x, delta_y, 0.0, 0.0, 0.0, 0.0)
    )
    return encode_association_node_v1(
        AssociationNodeInputV1(
            geometry_xyz_lwh_yaw=(
                *label.center_xyz,
                *label.dimensions_lwh,
                label.yaw,
            ),
            motion_xy=velocity_xy,
            appearance=provider.embedding(observation),
            class_label=label.category,
            sequence_relative_event_time_s=(
                side_frame.pointcloud_timestamp_us - sequence_origin_us
            )
            / 1_000_000.0,
            image_minus_pointcloud_time_s=(
                side_frame.image_timestamp_us - side_frame.pointcloud_timestamp_us
            )
            / 1_000_000.0,
            infrastructure_minus_vehicle_time_s=pair_delta_s,
            source_is_infrastructure=side == "infrastructure",
            covariance=_proxy_covariance(label),
            pose_xyz_rpy=pose,
            lineage_complete=True,
            lineage_factor_count=1,
            lineage_ancestor_count=0,
            lineage_has_cross_agent_ancestor=False,
        )
    )


def _velocity(
    history: dict[tuple[str, str], tuple[int, tuple[float, float, float]]],
    *,
    sequence_id: str,
    identity: str,
    timestamp_us: int,
    center_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    key = (sequence_id, identity)
    previous = history.get(key)
    history[key] = (timestamp_us, center_xyz)
    if previous is None:
        return (0.0, 0.0)
    delta_s = (timestamp_us - previous[0]) / 1_000_000.0
    if delta_s <= 0.0:
        raise SPDAssociationDataError("track history time must be increasing")
    return (
        (center_xyz[0] - previous[1][0]) / delta_s,
        (center_xyz[1] - previous[1][1]) / delta_s,
    )


def _cooperative_matches(
    pair: SPDPair,
    *,
    vehicle_labels: Mapping[str, _SideTrackLabelV1],
    infrastructure_labels: Mapping[str, _SideTrackLabelV1],
) -> tuple[tuple[str, str], ...]:
    """Validate cooperative references and return only positive ID mappings."""

    referenced: dict[str, set[str]] = {
        "vehicle": set(),
        "infrastructure": set(),
    }
    matches: list[tuple[str, str]] = []
    for label in pair.labels:
        side_values = (
            (
                "vehicle",
                label.vehicle_track_id,
                label.vehicle_token,
                vehicle_labels,
            ),
            (
                "infrastructure",
                label.infrastructure_track_id,
                label.infrastructure_token,
                infrastructure_labels,
            ),
        )
        for side, track_id, token, side_labels in side_values:
            if track_id == "-1":
                if token != "-1":
                    raise SPDAssociationDataError(
                        f"{side} cooperative token must be -1 when track_id is -1"
                    )
                continue
            if token == "-1":
                raise SPDAssociationDataError(
                    f"{side} cooperative token is missing for track_id {track_id}"
                )
            if track_id in referenced[side]:
                raise SPDAssociationDataError(
                    f"cooperative labels repeat {side} track_id {track_id}"
                )
            referenced[side].add(track_id)
            try:
                side_label = side_labels[track_id]
            except KeyError as exc:
                raise SPDAssociationDataError(
                    f"{side} cooperative track_id {track_id} is absent from "
                    f"pointcloud labels for frame "
                    f"{pair.vehicle.frame_id if side == 'vehicle' else pair.infrastructure.frame_id}"
                ) from exc
            if side_label.token != token:
                raise SPDAssociationDataError(
                    f"{side} cooperative/pointcloud token mismatch for track_id "
                    f"{track_id}"
                )
        if label.vehicle_track_id != "-1" and label.infrastructure_track_id != "-1":
            matches.append((label.vehicle_track_id, label.infrastructure_track_id))
    return tuple(matches)


def build_spd_association_samples_v1(
    dataset_root: str | Path,
    *,
    official_train_sequence_ids: Sequence[str],
    selected_sequence_ids: Sequence[str],
    appearance_provider: FrozenAppearanceProviderV1,
    max_frame_pairs_per_sequence: int | None = None,
) -> tuple[AssociationFrameSampleV1, ...]:
    """Build deterministic real GT matrices from a strict train-only projection."""

    root = require_spd_train_only_projection_v1(
        dataset_root,
        official_train_sequence_ids=official_train_sequence_ids,
    )
    selected = tuple(
        _sequence_id(item, "selected sequence ID") for item in selected_sequence_ids
    )
    if not selected or len(set(selected)) != len(selected):
        raise SPDAssociationDataError(
            "selected sequence IDs must be non-empty and unique"
        )
    if not set(selected).issubset(set(official_train_sequence_ids)):
        raise SPDAssociationDataError(
            "selected sequences must be a subset of SPD train"
        )
    if max_frame_pairs_per_sequence is not None and (
        type(max_frame_pairs_per_sequence) is not int
        or max_frame_pairs_per_sequence <= 0
    ):
        raise SPDAssociationDataError("max_frame_pairs_per_sequence must be positive")
    metadata = load_spd_metadata(root, sequence_ids=selected)
    vehicle_index = _side_indexes(root, "vehicle")
    infrastructure_index = _side_indexes(root, "infrastructure")
    ordered = sorted(
        metadata.pairs,
        key=lambda item: (
            item.sequence_id,
            item.vehicle.pointcloud_timestamp_us,
            item.vehicle.frame_id,
        ),
    )
    by_sequence: dict[str, list[SPDPair]] = defaultdict(list)
    for pair in ordered:
        by_sequence[pair.sequence_id].append(pair)
    chosen: list[SPDPair] = []
    for sequence_id in selected:
        pairs = by_sequence.get(sequence_id, [])
        if max_frame_pairs_per_sequence is not None:
            pairs = pairs[:max_frame_pairs_per_sequence]
        chosen.extend(pairs)
    camera_cache: dict[
        tuple[str, str], dict[str, tuple[float, float, float, float] | None]
    ] = {}
    pointcloud_cache: dict[tuple[str, str], dict[str, _SideTrackLabelV1]] = {}
    histories: dict[
        str, dict[tuple[str, str], tuple[int, tuple[float, float, float]]]
    ] = {
        "vehicle": {},
        "infrastructure": {},
    }
    origins = {
        sequence_id: min(pair.vehicle.pointcloud_timestamp_us for pair in pairs)
        for sequence_id, pairs in by_sequence.items()
        if pairs
    }
    samples: list[AssociationFrameSampleV1] = []
    for pair in chosen:
        side_records = {
            "vehicle": vehicle_index[pair.vehicle.frame_id],
            "infrastructure": infrastructure_index[pair.infrastructure.frame_id],
        }
        camera_boxes: dict[
            str, dict[str, tuple[float, float, float, float] | None]
        ] = {}
        pointcloud_labels: dict[str, dict[str, _SideTrackLabelV1]] = {}
        for side, frame_id in (
            ("vehicle", pair.vehicle.frame_id),
            ("infrastructure", pair.infrastructure.frame_id),
        ):
            cache_key = (side, frame_id)
            if cache_key not in camera_cache:
                camera_cache[cache_key] = _camera_labels(root, side, side_records[side])
            camera_boxes[side] = camera_cache[cache_key]

            if cache_key not in pointcloud_cache:
                pointcloud_cache[cache_key] = _pointcloud_labels(
                    root,
                    side,
                    side_records[side],
                )
            pointcloud_labels[side] = pointcloud_cache[cache_key]

        matches = _cooperative_matches(
            pair,
            vehicle_labels=pointcloud_labels["vehicle"],
            infrastructure_labels=pointcloud_labels["infrastructure"],
        )

        rows: dict[str, list[tuple[str, np.ndarray]]] = {
            "vehicle": [],
            "infrastructure": [],
        }
        for side in ("vehicle", "infrastructure"):
            side_frame = pair.vehicle if side == "vehicle" else pair.infrastructure
            for source_track_id, side_label in pointcloud_labels[side].items():
                velocity = _velocity(
                    histories[side],
                    sequence_id=pair.sequence_id,
                    identity=source_track_id,
                    timestamp_us=side_frame.pointcloud_timestamp_us,
                    center_xyz=side_label.center_xyz,
                )
                rows[side].append(
                    (
                        source_track_id,
                        _feature_with_root(
                            root=root,
                            pair=pair,
                            label=side_label,
                            side=side,
                            source_track_id=source_track_id,
                            side_record=side_records[side],
                            camera_boxes=camera_boxes[side],
                            provider=appearance_provider,
                            velocity_xy=velocity,
                            sequence_origin_us=origins[pair.sequence_id],
                        ),
                    )
                )
        left_rows = sorted(rows["vehicle"], key=lambda item: item[0])
        right_rows = sorted(rows["infrastructure"], key=lambda item: item[0])
        left_source_ids = [item[0] for item in left_rows]
        right_source_ids = [item[0] for item in right_rows]
        left_identity_ids = tuple(
            f"{pair.sequence_id}:vehicle:{source_id}" for source_id in left_source_ids
        )
        right_identity_ids = tuple(
            f"{pair.sequence_id}:infrastructure:{source_id}"
            for source_id in right_source_ids
        )
        left_features = (
            np.stack([item[1] for item in left_rows])
            if left_rows
            else np.empty((0, ASSOCIATION_FEATURE_DIM_V1), dtype=np.float32)
        )
        right_features = (
            np.stack([item[1] for item in right_rows])
            if right_rows
            else np.empty((0, ASSOCIATION_FEATURE_DIM_V1), dtype=np.float32)
        )
        targets = np.zeros((len(left_rows), len(right_rows)), dtype=np.uint8)
        left_index = {track_id: index for index, track_id in enumerate(left_source_ids)}
        right_index = {
            track_id: index for index, track_id in enumerate(right_source_ids)
        }
        for vehicle_track_id, infrastructure_track_id in matches:
            targets[
                left_index[vehicle_track_id],
                right_index[infrastructure_track_id],
            ] = 1
        if left_rows or right_rows:
            samples.append(
                AssociationFrameSampleV1(
                    sequence_id=pair.sequence_id,
                    vehicle_frame_id=pair.vehicle.frame_id,
                    infrastructure_frame_id=pair.infrastructure.frame_id,
                    left_identity_ids=left_identity_ids,
                    right_identity_ids=right_identity_ids,
                    left_features=left_features,
                    right_features=right_features,
                    targets=targets,
                )
            )
    if not samples:
        raise SPDAssociationDataError("selected SPD train cohort produced no objects")
    return tuple(samples)


def build_spd_association_cohort_v1(
    dataset_root: str | Path,
    *,
    official_train_sequence_ids: Sequence[str],
    fit_sequence_ids: Sequence[str],
    held_out_sequence_ids: Sequence[str],
    appearance_provider: FrozenAppearanceProviderV1,
    max_frame_pairs_per_sequence: int | None = None,
) -> SPDAssociationCohortV1:
    fit = tuple(sorted(fit_sequence_ids))
    held_out = tuple(sorted(held_out_sequence_ids))
    if set(fit).intersection(held_out):
        raise SPDAssociationDataError("fit and held-out sequence IDs overlap")
    train_samples = build_spd_association_samples_v1(
        dataset_root,
        official_train_sequence_ids=official_train_sequence_ids,
        selected_sequence_ids=fit,
        appearance_provider=appearance_provider,
        max_frame_pairs_per_sequence=max_frame_pairs_per_sequence,
    )
    validation_samples = build_spd_association_samples_v1(
        dataset_root,
        official_train_sequence_ids=official_train_sequence_ids,
        selected_sequence_ids=held_out,
        appearance_provider=appearance_provider,
        max_frame_pairs_per_sequence=max_frame_pairs_per_sequence,
    )
    return SPDAssociationCohortV1(
        train_samples=train_samples,
        validation_samples=validation_samples,
        train_sequence_ids=fit,
        validation_sequence_ids=held_out,
        appearance_source=appearance_provider.source_id,
    )


__all__ = [
    "AppearanceObservationV1",
    "AssociationFrameSampleV1",
    "AssociationNodeInputV1",
    "FrozenAppearanceProviderV1",
    "FrozenRGBProjectionAppearanceV1",
    "FrozenResNet50AppearanceV1",
    "NPZAppearanceCacheV1",
    "SPDAssociationCohortV1",
    "SPDAssociationDataError",
    "build_spd_association_cohort_v1",
    "build_spd_association_samples_v1",
    "encode_association_node_v1",
    "require_spd_train_only_projection_v1",
]
