"""Minimal, fail-closed V2X-Seq adapter boundary for CenterPoint.

The pinned upstream CenterPoint tree supports NuScenes and Waymo, not the
V2X-Seq SPD release.  Its public tracker is a parameter-free, NuScenes-specific
post-processor.  Consequently this file does not claim to be an official
V2X-Seq CenterPoint implementation and does not fake a training step.

``create_adapter`` is the narrow integration API for a separately implemented
and provenance-sealed detection backend and evaluator.  ``build_adapter`` is
the factory expected by ``v2xseq_real_canary.py``; it deliberately fails until
those dependencies and the still-pending tracking protocol are frozen.

No code from either upstream repository is copied into this module.  The
tracker below is an independent, single-class greedy closest-center baseline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


CENTERPOINT_REPOSITORY = "https://github.com/tianweiy/CenterPoint"
CENTERPOINT_COMMIT = "3cf7d870537e287c99b43b68636ea392a5e6f519"
V2XSEQ_REPOSITORY = "https://github.com/AIR-THU/DAIR-V2X"
BASELINE_KIND = "local-adapted-centerpoint-style-vehicle-only"
COORDINATE_FRAME = "V2X-Seq SPD vehicle LiDAR at vehicle capture time"
SUPPORTED_SPD_CLASSES = frozenset({"Car", "Van", "Truck", "Bus"})


class AdapterContractError(RuntimeError):
    """Raised when an adapter input would make the baseline ambiguous."""


class IntegrationBlockedError(AdapterContractError):
    """Raised when production evidence dependencies are not frozen."""


class DetectionBackend(Protocol):
    """Required wrapper around a separately sealed CenterPoint installation.

    The wrapper owns all tensors and optimizer state.  ``infer`` must return
    finite JSON-like detection mappings in the schema consumed by
    :class:`CenterPointStyleTracker`.
    """

    def training_forward(self, sample: Mapping[str, Any]) -> Any: ...

    def backward(self, forward_output: Any) -> Mapping[str, Any]: ...

    def optimizer_step(self) -> None: ...

    def infer(self, sample: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...

    def save_checkpoint(self, path: str) -> None: ...

    def runtime_environment(self) -> Mapping[str, Any]: ...


class TrackingEvaluator(Protocol):
    """Required wrapper around a frozen V2X-Seq tracking evaluator."""

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        payloads: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, int | float]: ...


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterContractError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdapterContractError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AdapterContractError(f"{label} must contain {length} numbers")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _track_id(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AdapterContractError(f"{label} must be a string or integer")
    result = str(value)
    if not result:
        raise AdapterContractError(f"{label} must not be empty")
    return result


def convert_spd_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one official SPD cooperative label without inventing velocity.

    SPD stores dimensions as ``l, w, h``.  This function emits ``w, l, h`` in
    a seven-value vehicle-LiDAR box.  The rotation number is preserved exactly;
    a detection backend must separately freeze and validate its yaw convention.
    """

    row = _mapping(label, "SPD label")
    category = _string(row.get("type"), "SPD label type")
    if category not in SUPPORTED_SPD_CLASSES:
        raise AdapterContractError(
            f"unsupported SPD class {category!r}; no implicit filtering is allowed"
        )
    dimensions = _mapping(row.get("3d_dimensions"), "SPD label 3d_dimensions")
    location = _mapping(row.get("3d_location"), "SPD label 3d_location")
    length = _finite(dimensions.get("l"), "SPD label length")
    width = _finite(dimensions.get("w"), "SPD label width")
    height = _finite(dimensions.get("h"), "SPD label height")
    if min(length, width, height) <= 0:
        raise AdapterContractError("SPD label dimensions must be positive")
    x = _finite(location.get("x"), "SPD label x")
    y = _finite(location.get("y"), "SPD label y")
    z = _finite(location.get("z"), "SPD label z")
    yaw = _finite(row.get("rotation"), "SPD label rotation")
    return {
        "track_id": _track_id(row.get("track_id"), "SPD label track_id"),
        "class_name": "vehicle",
        "source_class": category,
        "box_3d": [x, y, z, width, length, height, yaw],
        "box_3d_order": "x,y,z,w,l,h,raw_spd_rotation",
        "velocity_available": False,
    }


class VehicleOnlyFrameConverter:
    """Convert the canary frame payload without using infrastructure geometry.

    This baseline intentionally consumes only the vehicle LiDAR file.  A
    cooperative early-fusion converter additionally needs the infrastructure
    and vehicle calibration chains plus ``system_error_offset``; the present
    canary payload does not expose that frozen metadata.
    """

    def __init__(self, require_velocity_targets: bool = False) -> None:
        if not isinstance(require_velocity_targets, bool):
            raise AdapterContractError("require_velocity_targets must be boolean")
        self.require_velocity_targets = require_velocity_targets

    def convert(self, frame: Mapping[str, Any], *, training: bool) -> dict[str, Any]:
        row = _mapping(frame, "canary frame")
        sequence_id = _string(row.get("sequence_id"), "frame sequence_id")
        frame_id = _string(row.get("frame_id"), "frame frame_id")
        timestamp_raw = row.get("timestamp")
        if isinstance(timestamp_raw, bool) or not isinstance(timestamp_raw, int):
            raise AdapterContractError("frame timestamp must be an integer")
        assets = row.get("assets")
        if not isinstance(assets, (list, tuple)):
            raise AdapterContractError("frame assets must be an array")
        matches: list[Mapping[str, Any]] = []
        for index, value in enumerate(assets):
            asset = _mapping(value, f"frame asset {index}")
            if asset.get("agent") == "vehicle" and asset.get("modality") == "lidar":
                matches.append(asset)
        if len(matches) != 1:
            raise AdapterContractError(
                "vehicle-only conversion requires exactly one vehicle LiDAR asset"
            )
        lidar_path = Path(
            _string(matches[0].get("absolute_path"), "vehicle LiDAR absolute_path")
        )
        if not lidar_path.is_absolute():
            raise AdapterContractError("vehicle LiDAR path must be absolute")

        sample: dict[str, Any] = {
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "timestamp": timestamp_raw,
            "vehicle_lidar_path": str(lidar_path),
            "coordinate_frame": COORDINATE_FRAME,
            "fusion_scope": "vehicle_only",
            "baseline_kind": BASELINE_KIND,
        }
        if not training:
            # Deliberately do not read ``row['labels']``.  This remains safe if
            # a caller accidentally attaches future ground truth to an
            # otherwise observation-only inference payload.
            return sample

        labels = row.get("labels")
        if not isinstance(labels, (list, tuple)):
            raise AdapterContractError("training frame labels must be an array")
        annotations = [
            convert_spd_label(_mapping(value, f"frame label {index}"))
            for index, value in enumerate(labels)
        ]
        if self.require_velocity_targets and annotations:
            raise AdapterContractError(
                "SPD labels contain no velocity target; a velocity head may not be "
                "trained by silently inserting zeros"
            )
        sample["annotations"] = annotations
        sample["annotation_schema"] = "SPD vehicle-LiDAR 7D boxes; velocity absent"
        return sample


class _Track:
    __slots__ = ("track_id", "center_xy", "missed")

    def __init__(
        self, *, track_id: int, center_xy: tuple[float, float], missed: int
    ) -> None:
        self.track_id = track_id
        self.center_xy = center_xy
        self.missed = missed


class CenterPointStyleTracker:
    """Independent single-class greedy closest-center association baseline.

    This is not the upstream NuScenes ``PubTracker``.  Its distance threshold,
    timestamp scale, velocity use, and maximum age are explicit constructor
    arguments so they can be frozen by a future protocol.
    """

    def __init__(
        self,
        *,
        max_distance_m: float,
        timestamp_to_seconds: float,
        max_age_frames: int = 0,
        use_velocity: bool,
    ) -> None:
        self.max_distance_m = _finite(max_distance_m, "max_distance_m")
        self.timestamp_to_seconds = _finite(
            timestamp_to_seconds, "timestamp_to_seconds"
        )
        if self.max_distance_m <= 0 or self.timestamp_to_seconds <= 0:
            raise AdapterContractError("distance and timestamp scale must be positive")
        if isinstance(max_age_frames, bool) or not isinstance(max_age_frames, int):
            raise AdapterContractError("max_age_frames must be an integer")
        if max_age_frames < 0:
            raise AdapterContractError("max_age_frames must be non-negative")
        if not isinstance(use_velocity, bool):
            raise AdapterContractError("use_velocity must be boolean")
        self.max_age_frames = max_age_frames
        self.use_velocity = use_velocity
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_timestamp: int | None = None

    def _normalize_detection(
        self, raw: Mapping[str, Any], index: int
    ) -> dict[str, Any]:
        detection = _mapping(raw, f"detection {index}")
        if detection.get("class_name") != "vehicle":
            raise AdapterContractError(
                f"detection {index}.class_name must be 'vehicle'"
            )
        translation = _vector(
            detection.get("translation"), 3, f"detection {index}.translation"
        )
        score = _finite(detection.get("score"), f"detection {index}.score")
        if not 0 <= score <= 1:
            raise AdapterContractError("detection score must be in [0, 1]")
        normalized: dict[str, Any] = {
            "translation": translation,
            "score": score,
            "class_name": "vehicle",
        }
        if self.use_velocity:
            normalized["velocity"] = _vector(
                detection.get("velocity"), 2, f"detection {index}.velocity"
            )
        elif "velocity" in detection:
            normalized["velocity"] = _vector(
                detection["velocity"], 2, f"detection {index}.velocity"
            )
        for key in ("size", "yaw"):
            if key in detection:
                normalized[key] = (
                    _vector(detection[key], 3, f"detection {index}.size")
                    if key == "size"
                    else _finite(detection[key], f"detection {index}.yaw")
                )
        return normalized

    def step(
        self, detections: Sequence[Mapping[str, Any]], *, timestamp: int
    ) -> list[dict[str, Any]]:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise AdapterContractError("tracker timestamp must be an integer")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise AdapterContractError("tracker timestamps must increase strictly")
        delta_seconds = (
            0.0
            if self._last_timestamp is None
            else (timestamp - self._last_timestamp) * self.timestamp_to_seconds
        )
        current = [
            self._normalize_detection(value, index)
            for index, value in enumerate(detections)
        ]
        unmatched_tracks = set(range(len(self._tracks)))
        assignments: dict[int, int] = {}

        # Match in backend output order, mirroring the simple greedy nature of
        # CenterPoint's public tracker without copying its implementation.
        for detection_index, detection in enumerate(current):
            x, y = detection["translation"][:2]
            if self.use_velocity:
                vx, vy = detection["velocity"]
                comparison = (x - vx * delta_seconds, y - vy * delta_seconds)
            else:
                comparison = (x, y)
            candidates = []
            for track_index in unmatched_tracks:
                track = self._tracks[track_index]
                distance = math.hypot(
                    comparison[0] - track.center_xy[0],
                    comparison[1] - track.center_xy[1],
                )
                candidates.append((distance, track.track_id, track_index))
            if candidates:
                distance, _, track_index = min(candidates)
                if distance <= self.max_distance_m:
                    assignments[detection_index] = track_index
                    unmatched_tracks.remove(track_index)

        next_tracks: list[_Track] = []
        output: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(current):
            if detection_index in assignments:
                track_id = self._tracks[assignments[detection_index]].track_id
            else:
                track_id = self._next_id
                self._next_id += 1
            center = tuple(detection["translation"][:2])
            next_tracks.append(_Track(track_id=track_id, center_xy=center, missed=0))
            output.append({**detection, "track_id": str(track_id)})

        for track_index in sorted(unmatched_tracks):
            track = self._tracks[track_index]
            if track.missed < self.max_age_frames:
                next_tracks.append(
                    _Track(
                        track_id=track.track_id,
                        center_xy=track.center_xy,
                        missed=track.missed + 1,
                    )
                )
        self._tracks = next_tracks
        self._last_timestamp = timestamp
        return output


class CenterPointV2XSeqAdapter:
    """Canary-compatible adapter with injected backend and evaluator."""

    def __init__(
        self,
        *,
        backend: DetectionBackend,
        evaluator: TrackingEvaluator,
        converter: VehicleOnlyFrameConverter,
        tracker: CenterPointStyleTracker,
        expected_metrics: Sequence[str],
    ) -> None:
        if not expected_metrics or len(set(expected_metrics)) != len(expected_metrics):
            raise AdapterContractError("expected_metrics must be non-empty and unique")
        self._backend = backend
        self._evaluator = evaluator
        self._converter = converter
        self._tracker = tracker
        self._expected_metrics = tuple(expected_metrics)
        self._sequence_id: str | None = None

    def forward(self, frame: Mapping[str, Any], training: bool) -> Any:
        if not isinstance(training, bool):
            raise AdapterContractError("training flag must be boolean")
        sample = self._converter.convert(frame, training=training)
        if training:
            return self._backend.training_forward(sample)
        if self._sequence_id != sample["sequence_id"]:
            self._tracker.reset()
            self._sequence_id = sample["sequence_id"]
        detections = self._backend.infer(sample)
        tracks = self._tracker.step(detections, timestamp=sample["timestamp"])
        return {
            "baseline_kind": BASELINE_KIND,
            "coordinate_frame": COORDINATE_FRAME,
            "frame_id": sample["frame_id"],
            "tracks": tracks,
        }

    def backward(self, forward_output: Any) -> Mapping[str, Any]:
        return self._backend.backward(forward_output)

    def optimizer_step(self) -> None:
        self._backend.optimizer_step()

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        payloads: Sequence[Mapping[str, Any]],
    ) -> dict[str, int | float]:
        observed = dict(self._evaluator.evaluate(predictions, payloads))
        if set(observed) != set(self._expected_metrics):
            raise AdapterContractError(
                "evaluator metrics do not match the frozen protocol exactly"
            )
        for name, value in observed.items():
            _finite(value, f"metric {name}")
        return observed

    def save_checkpoint(self, path: str) -> None:
        self._backend.save_checkpoint(path)

    def runtime_environment(self) -> dict[str, str]:
        """Return the exact runtime identity required by the canary runner."""

        observed = _mapping(
            self._backend.runtime_environment(), "backend runtime environment"
        )
        expected_fields = {
            "framework",
            "framework_version",
            "device_name",
        }
        if set(observed) != expected_fields:
            raise AdapterContractError(
                "backend runtime environment must contain framework, "
                "framework_version, and device_name exactly"
            )

        normalized: dict[str, str] = {}
        for field in ("framework", "framework_version", "device_name"):
            value = _string(
                observed.get(field), f"backend runtime environment {field}"
            )
            if not value.strip():
                raise AdapterContractError(
                    f"backend runtime environment {field} must be a non-empty string"
                )
            normalized[field] = value
        return normalized


def create_adapter(
    context: Mapping[str, Any],
    *,
    backend: DetectionBackend,
    evaluator: TrackingEvaluator,
    max_distance_m: float,
    timestamp_to_seconds: float,
    max_age_frames: int,
    use_velocity: bool,
    require_velocity_targets: bool = False,
) -> CenterPointV2XSeqAdapter:
    """Create an adapter after external provenance and protocol gates pass."""

    root = _mapping(context, "adapter context")
    protocol = _mapping(root.get("protocol"), "adapter context protocol")
    if protocol.get("status") != "frozen":
        raise IntegrationBlockedError("tracking protocol is not frozen")
    if protocol.get("coordinate_frame") != COORDINATE_FRAME:
        raise IntegrationBlockedError(
            "protocol coordinate frame does not match adapter"
        )
    if protocol.get("classes") != ["vehicle"]:
        raise IntegrationBlockedError("adapter supports exactly the vehicle class")
    metrics = protocol.get("tracking_metrics")
    if not isinstance(metrics, list) or any(not isinstance(x, str) for x in metrics):
        raise IntegrationBlockedError("tracking metrics are not frozen")
    return CenterPointV2XSeqAdapter(
        backend=backend,
        evaluator=evaluator,
        converter=VehicleOnlyFrameConverter(
            require_velocity_targets=require_velocity_targets
        ),
        tracker=CenterPointStyleTracker(
            max_distance_m=max_distance_m,
            timestamp_to_seconds=timestamp_to_seconds,
            max_age_frames=max_age_frames,
            use_velocity=use_velocity,
        ),
        expected_metrics=metrics,
    )


def build_adapter(context: Mapping[str, Any]) -> CenterPointV2XSeqAdapter:
    """Canary factory; fail until the real backend/evaluator are sealed.

    The current canary verifies only the adapter file at Git HEAD.  Dynamically
    importing a CenterPoint checkout, SPD preprocessing helper, checkpoint, or
    evaluator here would bypass that evidence boundary.  A future enabled
    configuration must use package/source/environment seals and inject wrappers
    through a reviewed factory before this function can return an adapter.
    """

    _mapping(context, "adapter context")
    raise IntegrationBlockedError(
        "CenterPoint SPD backend, official-data calibration parity, checkpoint, "
        "dependency environment, and V2X-Seq evaluator are not provenance-sealed; "
        "this local algorithmic baseline is not official V2X-Seq CenterPoint"
    )
