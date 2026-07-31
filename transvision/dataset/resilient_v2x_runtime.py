from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from transvision.models.resilient_v2x import (
    Agent,
    BranchSelection,
    Modality,
    ResilientBatchSelections,
    SourceCandidate,
    compose_source_to_target,
    latest_arrived_rsu_age_intervals,
    select_causal_source,
)

from .resilient_v2x_manifest import (
    PreparedArtifactRecord,
    RawSliceRecord,
    TemporalManifest,
    TemporalSampleRecord,
    load_temporal_manifest,
)
from .resilient_v2x_schedule import (
    FaultOverlayRecord,
    TransportOverlayRecord,
    augmentation_seed,
    read_overlay,
)


RUNTIME_BRANCH_ORDER = (
    (Agent.EGO, Modality.LIDAR),
    (Agent.RSU, Modality.LIDAR),
    (Agent.EGO, Modality.CAMERA),
    (Agent.RSU, Modality.CAMERA),
)
_TRANSPORT_FIELDS = frozenset(
    {
        "epoch",
        "sample_id",
        "packet_id",
        "n_s",
        "delay_ms",
        "arrival_tau_ms",
    }
)
_FAULT_FIELDS = frozenset(
    {
        "epoch",
        "sample_id",
        "agent",
        "modality",
        "n_s",
        "masked",
        "pre_mask_selected_n_s",
        "fallback_selected_n_s",
    }
)
_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class RuntimeProtocolError(RuntimeError):
    """Raised when a runtime overlay or payload violates the protocol."""


def _point_cloud_range(
    value: Sequence[float] | None,
) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 6:
        raise ValueError("point_cloud_range must contain six finite values")
    limits = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in limits):
        raise ValueError("point_cloud_range must contain six finite values")
    if any(upper <= lower for lower, upper in zip(limits[:3], limits[3:])):
        raise ValueError("point_cloud_range maxima must exceed minima")
    return limits


def _exact_epoch(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise RuntimeProtocolError(f"{name} must be null or a nonnegative integer")
    return value


def _resolve_epoch_record(
    records: Mapping[tuple[object, ...], object],
    exact_key: tuple[object, ...],
    global_key: tuple[object, ...],
    context: str,
) -> object | None:
    exact = records.get(exact_key)
    global_value = records.get(global_key)
    if exact is not None and global_value is not None:
        raise RuntimeProtocolError(
            f"{context} is ambiguous between epoch-specific and global records"
        )
    return exact if exact is not None else global_value


@dataclass(frozen=True)
class RuntimeOverlayIndex:
    transport: Mapping[tuple[int | None, str, str], TransportOverlayRecord]
    faults: Mapping[
        tuple[int | None, str, str, str, int],
        FaultOverlayRecord,
    ]
    transport_sha256: str | None = None
    fault_sha256: str | None = None

    @classmethod
    def build(
        cls,
        manifest: TemporalManifest,
        transport_records: Sequence[Mapping[str, object]] = (),
        fault_records: Sequence[Mapping[str, object]] = (),
        transport_sha256: str | None = None,
        fault_sha256: str | None = None,
    ) -> "RuntimeOverlayIndex":
        if not isinstance(manifest, TemporalManifest):
            raise RuntimeProtocolError("manifest must be TemporalManifest")
        samples = {sample.sample_id: sample for sample in manifest.samples}
        packet_lookup = {
            (sample.sample_id, source.packet_id): source
            for sample in manifest.samples
            for source in sample.source_slices
        }
        position_lookup = {
            (
                sample.sample_id,
                source.agent,
                source.modality,
                source.n_s,
            ): source
            for sample in manifest.samples
            for source in sample.source_slices
        }

        transport: dict[
            tuple[int | None, str, str],
            TransportOverlayRecord,
        ] = {}
        shared_tick_delay: dict[
            tuple[int | None, str, int],
            int,
        ] = {}
        for index, value in enumerate(transport_records):
            if not isinstance(value, Mapping) or frozenset(value) != _TRANSPORT_FIELDS:
                raise RuntimeProtocolError(f"transport record {index} fields mismatch")
            try:
                record = TransportOverlayRecord(**dict(value))
            except (TypeError, ValueError) as error:
                raise RuntimeProtocolError(
                    f"invalid transport record {index}"
                ) from error
            source = packet_lookup.get((record.sample_id, record.packet_id))
            if source is None or source.agent != "rsu":
                raise RuntimeProtocolError(
                    "transport record references an unknown or ego packet"
                )
            if (
                record.n_s != source.n_s
                or record.arrival_tau_ms != source.tau_s_ms + record.delay_ms
            ):
                raise RuntimeProtocolError(
                    "transport record timing provenance mismatch"
                )
            key = (record.epoch, record.sample_id, record.packet_id)
            if key in transport:
                raise RuntimeProtocolError("duplicate transport record")
            transport[key] = record
            tick_key = (record.epoch, record.sample_id, record.n_s)
            prior_delay = shared_tick_delay.setdefault(tick_key, record.delay_ms)
            if prior_delay != record.delay_ms:
                raise RuntimeProtocolError(
                    "RSU LiDAR and camera packets at one tick must share delay"
                )

        faults: dict[
            tuple[int | None, str, str, str, int],
            FaultOverlayRecord,
        ] = {}
        for index, value in enumerate(fault_records):
            if not isinstance(value, Mapping) or frozenset(value) != _FAULT_FIELDS:
                raise RuntimeProtocolError(f"fault record {index} fields mismatch")
            try:
                record = FaultOverlayRecord(**dict(value))
            except (TypeError, ValueError) as error:
                raise RuntimeProtocolError(f"invalid fault record {index}") from error
            if record.sample_id not in samples:
                raise RuntimeProtocolError("fault record references an unknown sample")
            source = position_lookup.get(
                (
                    record.sample_id,
                    record.agent,
                    record.modality,
                    record.n_s,
                )
            )
            if source is None:
                raise RuntimeProtocolError(
                    "fault record references an unknown source position"
                )
            key = (
                record.epoch,
                record.sample_id,
                record.agent,
                record.modality,
                record.n_s,
            )
            if key in faults:
                raise RuntimeProtocolError("duplicate fault record")
            faults[key] = record

        return cls(
            transport=MappingProxyType(transport),
            faults=MappingProxyType(faults),
            transport_sha256=transport_sha256,
            fault_sha256=fault_sha256,
        )

    def arrival_tau_ms(
        self,
        sample_id: str,
        source: RawSliceRecord,
        epoch: int,
        *,
        zero_latency: bool,
    ) -> int | None:
        if source.agent == "ego":
            return None
        if zero_latency:
            if self.transport:
                raise RuntimeProtocolError(
                    "zero_latency cannot be combined with transport records"
                )
            return source.tau_s_ms
        record = _resolve_epoch_record(
            self.transport,
            (epoch, sample_id, source.packet_id),
            (None, sample_id, source.packet_id),
            "transport record",
        )
        if record is None:
            raise RuntimeProtocolError(
                "transport overlay does not cover every RSU source"
            )
        if not isinstance(record, TransportOverlayRecord):
            raise RuntimeProtocolError("transport index contains an invalid record")
        return record.arrival_tau_ms

    def faulted(
        self,
        sample_id: str,
        source: RawSliceRecord,
        epoch: int,
    ) -> bool:
        record = _resolve_epoch_record(
            self.faults,
            (
                epoch,
                sample_id,
                source.agent,
                source.modality,
                source.n_s,
            ),
            (
                None,
                sample_id,
                source.agent,
                source.modality,
                source.n_s,
            ),
            "fault record",
        )
        if record is None:
            return False
        if not isinstance(record, FaultOverlayRecord):
            raise RuntimeProtocolError("fault index contains an invalid record")
        return record.masked


@dataclass(frozen=True)
class ResolvedHistorySlot:
    horizon: int
    source: RawSliceRecord | None
    candidate: SourceCandidate | None
    available: bool
    source_to_target: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 0 <= self.horizon <= 3:
            raise RuntimeProtocolError("history horizon must be in [0,3]")
        if self.source is None:
            if self.candidate is not None or self.available:
                raise RuntimeProtocolError("empty history slot must be neutral")
        elif not isinstance(self.candidate, SourceCandidate):
            raise RuntimeProtocolError("source slot requires a SourceCandidate")


@dataclass(frozen=True)
class ResolvedBranch:
    agent: Agent
    modality: Modality
    slots: tuple[
        ResolvedHistorySlot,
        ResolvedHistorySlot,
        ResolvedHistorySlot,
        ResolvedHistorySlot,
    ]
    selection: BranchSelection

    def __post_init__(self) -> None:
        if type(self.slots) is not tuple or len(self.slots) != 4:
            raise RuntimeProtocolError("resolved branch requires four slots")
        if tuple(slot.horizon for slot in self.slots) != (0, 1, 2, 3):
            raise RuntimeProtocolError("history slots must be ordered by horizon")
        if (
            self.selection.agent is not self.agent
            or self.selection.modality is not self.modality
        ):
            raise RuntimeProtocolError("selection and branch identities differ")
        if self.selection.supported:
            if self.selection.horizon is None:
                raise RuntimeProtocolError("supported selection lacks a horizon")
            if not self.slots[self.selection.horizon].available:
                raise RuntimeProtocolError("selected history slot is unavailable")


@dataclass(frozen=True)
class ResolvedTemporalSample:
    sample: TemporalSampleRecord
    epoch: int
    augmentation_seed: int
    branches: tuple[
        ResolvedBranch,
        ResolvedBranch,
        ResolvedBranch,
        ResolvedBranch,
    ]

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 0:
            raise RuntimeProtocolError("epoch must be a nonnegative integer")
        if type(self.augmentation_seed) is not int or self.augmentation_seed < 0:
            raise RuntimeProtocolError("augmentation seed must be nonnegative")
        if type(self.branches) is not tuple or len(self.branches) != 4:
            raise RuntimeProtocolError("resolved sample requires four branches")
        actual = tuple((branch.agent, branch.modality) for branch in self.branches)
        if actual != RUNTIME_BRANCH_ORDER:
            raise RuntimeProtocolError("resolved branches use the wrong order")

    def branch(self, agent: Agent, modality: Modality) -> ResolvedBranch:
        for branch in self.branches:
            if branch.agent is agent and branch.modality is modality:
                return branch
        raise KeyError((agent, modality))

    @property
    def selections(self) -> tuple[BranchSelection, ...]:
        return tuple(branch.selection for branch in self.branches)


def _matrix_tuple(value: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value.tolist())


def _target_world_from_ego(sample: TemporalSampleRecord) -> RawSliceRecord:
    matches = [
        source
        for source in sample.source_slices
        if source.agent == "ego"
        and source.modality == "lidar"
        and source.n_s == sample.n_t
    ]
    if len(matches) != 1:
        raise RuntimeProtocolError(
            "sample must contain exactly one current ego LiDAR reference"
        )
    target = matches[0]
    if not target.pose_valid or not target.calibration_valid:
        raise RuntimeProtocolError("current ego reference geometry is invalid")
    return target


def resolve_temporal_sample(
    sample: TemporalSampleRecord,
    *,
    delta_t_ms: int,
    history_limit: int,
    overlays: RuntimeOverlayIndex,
    epoch: int,
    global_seed: int,
    zero_latency: bool,
) -> ResolvedTemporalSample:
    if not isinstance(sample, TemporalSampleRecord):
        raise RuntimeProtocolError("sample must be TemporalSampleRecord")
    if type(delta_t_ms) is not int or delta_t_ms <= 0:
        raise RuntimeProtocolError("delta_t_ms must be positive")
    if history_limit != 3:
        raise RuntimeProtocolError("the implemented reproduction profile requires k=3")
    epoch = _exact_epoch(epoch, "epoch")
    if epoch is None:
        raise RuntimeProtocolError("runtime epoch cannot be null")
    if type(global_seed) is not int or global_seed < 0:
        raise RuntimeProtocolError("global_seed must be nonnegative")
    if type(zero_latency) is not bool:
        raise RuntimeProtocolError("zero_latency must be boolean")
    if sample.tau_t_ms != sample.n_t * delta_t_ms:
        raise RuntimeProtocolError("sample decision time violates the temporal grid")

    target = _target_world_from_ego(sample)
    target_world = torch.tensor(target.world_from_agent, dtype=torch.float64)
    by_position = {
        (source.agent, source.modality, source.n_s): source
        for source in sample.source_slices
    }
    branches: list[ResolvedBranch] = []
    for agent, modality in RUNTIME_BRANCH_ORDER:
        agent_text = agent.value
        modality_text = modality.value
        candidates: list[SourceCandidate] = []
        slots: list[ResolvedHistorySlot] = []
        for horizon in range(history_limit + 1):
            source_tick = sample.n_t - horizon
            source = by_position.get((agent_text, modality_text, source_tick))
            if source is None:
                slots.append(
                    ResolvedHistorySlot(
                        horizon=horizon,
                        source=None,
                        candidate=None,
                        available=False,
                        source_to_target=_IDENTITY,
                    )
                )
                continue
            arrival_tau_ms = overlays.arrival_tau_ms(
                sample.sample_id,
                source,
                epoch,
                zero_latency=zero_latency,
            )
            timestamp_valid = (
                source.n_s <= sample.n_t and source.tau_s_ms == source.n_s * delta_t_ms
            )
            faulted = overlays.faulted(sample.sample_id, source, epoch)
            candidate = SourceCandidate(
                packet_id=source.packet_id,
                n_s=source.n_s,
                tau_s_ms=source.tau_s_ms,
                arrival_tau_ms=arrival_tau_ms,
                payload_valid=source.payload_valid,
                timestamp_valid=timestamp_valid,
                pose_valid=source.pose_valid,
                calibration_valid=source.calibration_valid,
                faulted=faulted,
            )
            candidates.append(candidate)
            arrived = agent is Agent.EGO or (
                arrival_tau_ms is not None and arrival_tau_ms <= sample.tau_t_ms
            )
            available = (
                arrived
                and not faulted
                and candidate.payload_valid
                and candidate.timestamp_valid
                and candidate.pose_valid
                and candidate.calibration_valid
            )
            source_world = torch.tensor(
                source.world_from_agent,
                dtype=torch.float64,
            )
            transform = compose_source_to_target(
                source_world.unsqueeze(0),
                target_world.unsqueeze(0),
            )[0]
            slots.append(
                ResolvedHistorySlot(
                    horizon=horizon,
                    source=source,
                    candidate=candidate,
                    available=available,
                    source_to_target=_matrix_tuple(transform),
                )
            )

        selection = select_causal_source(
            agent=agent,
            modality=modality,
            n_t=sample.n_t,
            target_tau_ms=sample.tau_t_ms,
            delta_t_ms=delta_t_ms,
            history_limit=history_limit,
            candidates=candidates,
        )
        branches.append(
            ResolvedBranch(
                agent=agent,
                modality=modality,
                slots=tuple(slots),  # type: ignore[arg-type]
                selection=selection,
            )
        )

    return ResolvedTemporalSample(
        sample=sample,
        epoch=epoch,
        augmentation_seed=augmentation_seed(
            global_seed,
            epoch,
            sample.sample_id,
        ),
        branches=tuple(branches),  # type: ignore[arg-type]
    )


def _safe_payload_path(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise RuntimeProtocolError("payload path escapes data_root") from error
    if not candidate.is_file():
        raise RuntimeProtocolError("payload path is not a regular file")
    return candidate


def _file_bytes(
    path: Path,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeProtocolError("unable to read payload") from error
    if len(raw) != expected_size:
        raise RuntimeProtocolError("payload size does not match manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeProtocolError("payload hash does not match manifest")
    return raw


def _transform_points_to_agent(
    points: Tensor,
    agent_from_sensor: tuple[tuple[float, ...], ...],
) -> Tensor:
    if points.ndim != 2 or points.shape[1] != 4:
        raise RuntimeProtocolError("point tensor must have shape [N,4]")
    transform = points.new_tensor(agent_from_sensor)
    homogeneous = torch.cat(
        (points[:, :3], torch.ones_like(points[:, :1])),
        dim=1,
    )
    xyz = (transform @ homogeneous.t()).t()[:, :3]
    return torch.cat((xyz, points[:, 3:4]), dim=1)


def _normalize_lidar_intensity(
    points: Tensor,
    *,
    allow_legacy_u8: bool = True,
) -> Tensor:
    """Normalize legacy DAIR U8 intensity while preserving unit float data."""

    if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 4:
        raise RuntimeProtocolError("prepared LiDAR payload must have shape [N,4]")
    if not torch.isfinite(points).all().item():
        raise RuntimeProtocolError("prepared LiDAR payload must be finite")
    if not points.shape[0]:
        return points
    intensity = points[:, 3]
    minimum = float(intensity.min().item())
    maximum = float(intensity.max().item())
    if minimum < 0.0 or maximum > 255.0:
        raise RuntimeProtocolError("LiDAR intensity must be in [0,1] or legacy U8")
    if maximum <= 1.0:
        return points
    if not allow_legacy_u8:
        raise RuntimeProtocolError(
            "v2 prepared LiDAR intensity must already be normalized to [0,1]"
        )
    normalized = points.clone()
    normalized[:, 3].div_(255.0)
    return normalized


class ResilientTemporalDataset(Dataset[dict[str, object]]):
    """Strict manifest dataset that performs causal I/O only."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        data_root: str | Path,
        split: Literal["train", "val", "test"],
        expected_split_hash: str,
        allow_fixture: bool = False,
        transport_overlay_path: str | Path | None = None,
        transport_overlay_sha256: str | None = None,
        fault_overlay_path: str | Path | None = None,
        fault_overlay_sha256: str | None = None,
        epoch: int = 0,
        seed: int = 0,
        load_camera: bool = True,
        load_lidar: bool = True,
        camera_image_size: tuple[int, int] | None = None,
        point_cloud_range: Sequence[float] | None = None,
        include_clean_teacher: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if type(allow_fixture) is not bool:
            raise ValueError("allow_fixture must be boolean")
        if type(load_camera) is not bool or type(load_lidar) is not bool:
            raise ValueError("load flags must be boolean")
        if not load_camera and not load_lidar:
            raise ValueError("at least one modality must be loaded")
        if camera_image_size is not None and (
            type(camera_image_size) is not tuple
            or len(camera_image_size) != 2
            or any(type(value) is not int or value <= 0 for value in camera_image_size)
        ):
            raise ValueError(
                "camera_image_size must be null or positive (height,width)"
            )
        if type(include_clean_teacher) is not bool:
            raise ValueError("include_clean_teacher must be boolean")
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise RuntimeProtocolError("data_root must be a directory")
        self.manifest = load_temporal_manifest(
            Path(manifest_path),
            expected_split_hash=expected_split_hash,
            allow_fixture=allow_fixture,
        )
        if self.manifest.history_limit != 3:
            raise RuntimeProtocolError(
                "the implemented reproduction profile requires history_limit=3"
            )
        if self.manifest.delta_t_ms != 100:
            raise RuntimeProtocolError(
                "the implemented reproduction profile requires delta_t_ms=100"
            )
        split_samples = tuple(
            sample for sample in self.manifest.samples if sample.split == split
        )
        if not split_samples:
            raise RuntimeProtocolError("requested split has no samples")
        self.epoch = _exact_epoch(epoch, "epoch")
        if self.epoch is None:
            raise ValueError("epoch cannot be null")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = seed
        self.load_camera = load_camera
        self.load_lidar = load_lidar
        self.camera_image_size = camera_image_size
        self.point_cloud_range = _point_cloud_range(point_cloud_range)
        self.include_clean_teacher = include_clean_teacher

        transport_records = self._read_optional_overlay(
            transport_overlay_path,
            transport_overlay_sha256,
            "transport",
        )
        fault_records = self._read_optional_overlay(
            fault_overlay_path,
            fault_overlay_sha256,
            "fault",
        )
        self.zero_latency = transport_overlay_path is None
        self.overlays = RuntimeOverlayIndex.build(
            self.manifest,
            transport_records=transport_records,
            fault_records=fault_records,
            transport_sha256=transport_overlay_sha256,
            fault_sha256=fault_overlay_sha256,
        )
        transport_sample_ids = {
            sample_id for _, sample_id, _ in self.overlays.transport
        }
        fault_sample_ids = {sample_id for _, sample_id, _, _, _ in self.overlays.faults}
        if (
            transport_sample_ids
            and fault_sample_ids
            and transport_sample_ids != fault_sample_ids
        ):
            raise RuntimeProtocolError(
                "transport and fault overlays must cover identical sample sets"
            )
        overlay_sample_ids = transport_sample_ids or fault_sample_ids
        if overlay_sample_ids:
            self.samples = tuple(
                sample
                for sample in split_samples
                if sample.sample_id in overlay_sample_ids
            )
            if len(self.samples) != len(overlay_sample_ids):
                raise RuntimeProtocolError(
                    "overlay sample set must exactly belong to the requested split"
                )
        else:
            self.samples = split_samples
        prepared = {
            item.source_relative_path: item for item in self.manifest.prepared_artifacts
        }
        if len(prepared) != len(self.manifest.prepared_artifacts):
            raise RuntimeProtocolError("duplicate prepared artifact source")
        self._prepared = prepared
        self._inventory = {
            item.relative_path: item for item in self.manifest.release_inventory
        }

    @staticmethod
    def _read_optional_overlay(
        path: str | Path | None,
        digest: str | None,
        name: str,
    ) -> tuple[Mapping[str, object], ...]:
        if (path is None) != (digest is None):
            raise ValueError(f"{name} overlay path and digest must be paired")
        if path is None:
            return ()
        return read_overlay(Path(path), digest)  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        normalized = _exact_epoch(epoch, "epoch")
        if normalized is None:
            raise ValueError("epoch cannot be null")
        self.epoch = normalized

    def __len__(self) -> int:
        return len(self.samples)

    def _resolved_index(self, index: object) -> tuple[int, int]:
        epoch = self.epoch
        sample_index = index
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("compound index must be (epoch, sample_index)")
            epoch, sample_index = index
        if type(epoch) is not int or epoch < 0:
            raise IndexError("epoch index must be nonnegative")
        if type(sample_index) is not int:
            raise IndexError("sample index must be an integer")
        if sample_index < 0:
            sample_index += len(self.samples)
        if not 0 <= sample_index < len(self.samples):
            raise IndexError("sample index out of range")
        return epoch, sample_index

    def resolve(self, index: object) -> ResolvedTemporalSample:
        epoch, sample_index = self._resolved_index(index)
        return resolve_temporal_sample(
            self.samples[sample_index],
            delta_t_ms=self.manifest.delta_t_ms,
            history_limit=self.manifest.history_limit,
            overlays=self.overlays,
            epoch=epoch,
            global_seed=self.seed,
            zero_latency=self.zero_latency,
        )

    def _load_lidar(self, source: RawSliceRecord) -> Tensor:
        artifact = self._prepared.get(source.relative_path)
        if not isinstance(artifact, PreparedArtifactRecord):
            raise RuntimeProtocolError("LiDAR source has no verified prepared artifact")
        path = _safe_payload_path(
            self.data_root,
            artifact.prepared_relative_path,
        )
        raw = _file_bytes(path, artifact.size, artifact.sha256)
        array = np.frombuffer(raw, dtype="<f4")
        if array.size != artifact.point_count * 4:
            raise RuntimeProtocolError("prepared point count mismatch")
        points = torch.from_numpy(array.copy()).view(-1, 4)
        points = _normalize_lidar_intensity(
            points,
            allow_legacy_u8=not artifact.prepared_relative_path.startswith(
                "prepared/resilient_v2x_v2/"
            ),
        )
        return _transform_points_to_agent(points, source.agent_from_sensor)

    def _load_camera(self, source: RawSliceRecord) -> Tensor:
        inventory = self._inventory.get(source.relative_path)
        if inventory is None:
            raise RuntimeProtocolError("camera source is absent from inventory")
        path = _safe_payload_path(self.data_root, source.relative_path)
        raw = _file_bytes(path, inventory.size, inventory.sha256)
        try:
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(raw)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except Exception as error:
            raise RuntimeProtocolError(
                "unable to decode verified camera image"
            ) from error
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _materialize(
        self,
        resolved: ResolvedTemporalSample,
    ) -> dict[str, object]:
        lidar_payloads: list[Tensor] = []
        lidar_owner: list[tuple[int, int]] = []
        camera_images: list[Tensor] = []
        camera_owner: list[tuple[int, int]] = []
        camera_intrinsics: list[Tensor] = []
        camera_agent_from_sensor: list[Tensor] = []

        availability = torch.zeros(2, 2, 4, dtype=torch.bool)
        transforms = (
            torch.eye(4, dtype=torch.float32)
            .view(
                1,
                1,
                1,
                4,
                4,
            )
            .repeat(2, 2, 4, 1, 1)
        )
        for branch in resolved.branches:
            modality_index = 0 if branch.modality is Modality.LIDAR else 1
            agent_index = 0 if branch.agent is Agent.EGO else 1
            for slot in branch.slots:
                transforms[
                    modality_index,
                    agent_index,
                    slot.horizon,
                ] = torch.tensor(slot.source_to_target, dtype=torch.float32)
                availability[
                    modality_index,
                    agent_index,
                    slot.horizon,
                ] = slot.available
                if not slot.available:
                    continue
                source = slot.source
                if source is None:
                    raise RuntimeProtocolError("available slot lacks a source")
                if branch.modality is Modality.LIDAR and self.load_lidar:
                    lidar_payloads.append(self._load_lidar(source))
                    lidar_owner.append((agent_index, slot.horizon))
                elif branch.modality is Modality.CAMERA and self.load_camera:
                    image = self._load_camera(source)
                    camera_owner.append((agent_index, slot.horizon))
                    if source.camera_intrinsic is None:
                        raise RuntimeProtocolError(
                            "camera source lacks intrinsic calibration"
                        )
                    intrinsic = torch.tensor(
                        source.camera_intrinsic,
                        dtype=torch.float32,
                    )
                    if self.camera_image_size is not None:
                        source_height, source_width = image.shape[-2:]
                        target_height, target_width = self.camera_image_size
                        image = (
                            F.interpolate(
                                image.unsqueeze(0).float(),
                                size=(target_height, target_width),
                                mode="bilinear",
                                align_corners=False,
                            )
                            .round()
                            .clamp(0, 255)
                            .to(torch.uint8)[0]
                        )
                        intrinsic[0] *= target_width / source_width
                        intrinsic[1] *= target_height / source_height
                    camera_images.append(image)
                    camera_intrinsics.append(intrinsic)
                    camera_agent_from_sensor.append(
                        torch.tensor(source.agent_from_sensor, dtype=torch.float32)
                    )

        gt = torch.tensor(
            [
                (
                    box.x,
                    box.y,
                    box.z_bottom,
                    box.length,
                    box.width,
                    box.height,
                    box.yaw,
                )
                for box in resolved.sample.ground_truth
            ],
            dtype=torch.float32,
        ).reshape(-1, 7)
        if self.point_cloud_range is not None and gt.shape[0]:
            centers = gt[:, :3].clone()
            centers[:, 2] += gt[:, 5] / 2.0
            lower = centers.new_tensor(self.point_cloud_range[:3])
            upper = centers.new_tensor(self.point_cloud_range[3:])
            gt = gt[((centers >= lower) & (centers < upper)).all(dim=1)]
        return {
            "resolved": resolved,
            "lidar_points": tuple(lidar_payloads),
            "lidar_owner": torch.tensor(lidar_owner, dtype=torch.long).reshape(-1, 2),
            "camera_images": tuple(camera_images),
            "camera_owner": torch.tensor(camera_owner, dtype=torch.long).reshape(-1, 2),
            "camera_intrinsics": tuple(camera_intrinsics),
            "camera_agent_from_sensor": tuple(camera_agent_from_sensor),
            "availability": availability,
            "source_to_target": transforms,
            "gt_bboxes_3d": gt,
            "gt_labels_3d": torch.zeros(gt.shape[0], dtype=torch.long),
        }

    def __getitem__(self, index: object) -> dict[str, object]:
        epoch, sample_index = self._resolved_index(index)
        resolved = self.resolve((epoch, sample_index))
        result = self._materialize(resolved)
        if self.include_clean_teacher:
            clean = resolve_temporal_sample(
                self.samples[sample_index],
                delta_t_ms=self.manifest.delta_t_ms,
                history_limit=self.manifest.history_limit,
                overlays=RuntimeOverlayIndex.build(self.manifest),
                epoch=epoch,
                global_seed=self.seed,
                zero_latency=True,
            )
            result["teacher_clean"] = self._materialize(clean)
        return result


def collate_resilient_samples(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("samples must be a sequence")
    if not samples:
        raise ValueError("samples must not be empty")
    resolved_values: list[ResolvedTemporalSample] = []
    lidar_points: list[Tensor] = []
    lidar_owner: list[tuple[int, int, int]] = []
    camera_images: list[Tensor] = []
    camera_owner: list[tuple[int, int, int]] = []
    camera_intrinsics: list[Tensor] = []
    camera_agent_from_sensor: list[Tensor] = []
    availability: list[Tensor] = []
    transforms: list[Tensor] = []
    gt_boxes: list[Tensor] = []
    gt_labels: list[Tensor] = []

    for batch_index, sample in enumerate(samples):
        resolved = sample.get("resolved")
        if not isinstance(resolved, ResolvedTemporalSample):
            raise ValueError("every sample requires resolved metadata")
        resolved_values.append(resolved)
        owners = sample["lidar_owner"]
        payloads = sample["lidar_points"]
        if not isinstance(owners, Tensor) or not isinstance(payloads, tuple):
            raise ValueError("invalid lidar payload representation")
        for owner, payload in zip(owners.tolist(), payloads):
            lidar_points.append(payload)
            lidar_owner.append((batch_index, int(owner[0]), int(owner[1])))

        image_owners = sample["camera_owner"]
        images = sample["camera_images"]
        intrinsics = sample["camera_intrinsics"]
        extrinsics = sample["camera_agent_from_sensor"]
        if (
            not isinstance(image_owners, Tensor)
            or not isinstance(images, tuple)
            or not isinstance(intrinsics, tuple)
            or not isinstance(extrinsics, tuple)
        ):
            raise ValueError("invalid camera payload representation")
        for position, (owner, image) in enumerate(zip(image_owners.tolist(), images)):
            camera_images.append(image)
            camera_owner.append((batch_index, int(owner[0]), int(owner[1])))
            camera_intrinsics.append(intrinsics[position])
            camera_agent_from_sensor.append(extrinsics[position])

        for key, destination in (
            ("availability", availability),
            ("source_to_target", transforms),
            ("gt_bboxes_3d", gt_boxes),
            ("gt_labels_3d", gt_labels),
        ):
            value = sample.get(key)
            if not isinstance(value, Tensor):
                raise ValueError(f"sample {key} must be a tensor")
            destination.append(value)

    if camera_images:
        max_height = max(image.shape[-2] for image in camera_images)
        max_width = max(image.shape[-1] for image in camera_images)
        padded_images = camera_images[0].new_zeros(
            len(camera_images),
            3,
            max_height,
            max_width,
        )
        image_shapes = torch.empty(len(camera_images), 2, dtype=torch.long)
        for index, image in enumerate(camera_images):
            height, width = image.shape[-2:]
            padded_images[index, :, :height, :width] = image
            image_shapes[index] = torch.tensor((height, width))
    else:
        padded_images = torch.empty(0, 3, 1, 1, dtype=torch.uint8)
        image_shapes = torch.empty(0, 2, dtype=torch.long)

    rsu_delay_intervals: list[float] = []
    for resolved in resolved_values:
        rsu_candidates = [
            slot.candidate
            for branch in resolved.branches
            if branch.agent is Agent.RSU
            for slot in branch.slots
            if slot.candidate is not None
        ]
        rsu_delay_intervals.append(
            latest_arrived_rsu_age_intervals(
                rsu_candidates,
                target_tau_ms=resolved.sample.tau_t_ms,
                history_limit=3,
                delta_t_ms=100,
            )
        )

    selections = ResilientBatchSelections(
        sample_ids=tuple(value.sample.sample_id for value in resolved_values),
        lidar_ego=tuple(value.branches[0].selection for value in resolved_values),
        lidar_rsu=tuple(value.branches[1].selection for value in resolved_values),
        camera_ego=tuple(value.branches[2].selection for value in resolved_values),
        camera_rsu=tuple(value.branches[3].selection for value in resolved_values),
        rsu_delay_intervals=tuple(rsu_delay_intervals),
    )
    result = {
        "lidar_points": tuple(lidar_points),
        "lidar_owner": torch.tensor(lidar_owner, dtype=torch.long).reshape(-1, 3),
        "camera_images": padded_images,
        "camera_image_shapes": image_shapes,
        "camera_owner": torch.tensor(camera_owner, dtype=torch.long).reshape(-1, 3),
        "camera_intrinsics": (
            torch.stack(camera_intrinsics)
            if camera_intrinsics
            else torch.empty(0, 3, 3)
        ),
        "camera_agent_from_sensor": (
            torch.stack(camera_agent_from_sensor)
            if camera_agent_from_sensor
            else torch.empty(0, 4, 4)
        ),
        "availability": torch.stack(availability),
        "source_to_target": torch.stack(transforms),
        "selections": selections,
        "resolved": tuple(resolved_values),
        "gt_bboxes_3d": tuple(gt_boxes),
        "gt_labels_3d": tuple(gt_labels),
    }
    teacher_values = [sample.get("teacher_clean") for sample in samples]
    if any(value is not None for value in teacher_values):
        if not all(isinstance(value, Mapping) for value in teacher_values):
            raise ValueError("teacher_clean must be present for the whole batch")
        result["teacher_clean"] = collate_resilient_samples(
            teacher_values  # type: ignore[arg-type]
        )
    return result


class EpochIndexSampler(Sampler[tuple[int, int]]):
    """Epoch-carrying deterministic sampler safe for persistent workers."""

    def __init__(
        self,
        dataset: Dataset[object],
        *,
        shuffle: bool,
        seed: int,
        rank: int | None = None,
        world_size: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise ValueError("shuffle and drop_last must be boolean")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be nonnegative")
        if rank is None or world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                detected_rank = torch.distributed.get_rank()
                detected_world_size = torch.distributed.get_world_size()
            else:
                detected_rank, detected_world_size = 0, 1
            rank = detected_rank if rank is None else rank
            world_size = detected_world_size if world_size is None else world_size
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be positive")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        normalized = _exact_epoch(epoch, "epoch")
        if normalized is None:
            raise ValueError("epoch cannot be null")
        self.epoch = normalized

    def _global_indices(self) -> list[int]:
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))
        if self.drop_last:
            total = size - size % self.world_size
            return indices[:total]
        total = math.ceil(size / self.world_size) * self.world_size
        if total > size and size:
            indices += (indices * math.ceil((total - size) / size))[: total - size]
        return indices

    def __iter__(self):
        indices = self._global_indices()[self.rank :: self.world_size]
        return iter((self.epoch, index) for index in indices)

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.world_size
        return math.ceil(size / self.world_size)


__all__ = (
    "RUNTIME_BRANCH_ORDER",
    "RuntimeProtocolError",
    "RuntimeOverlayIndex",
    "ResolvedHistorySlot",
    "ResolvedBranch",
    "ResolvedTemporalSample",
    "resolve_temporal_sample",
    "ResilientTemporalDataset",
    "collate_resilient_samples",
    "EpochIndexSampler",
)
