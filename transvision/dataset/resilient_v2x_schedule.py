from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import zstandard

from .resilient_v2x_manifest import (
    RawSliceRecord,
    TemporalManifest,
    TemporalSampleRecord,
    canonical_json_bytes,
)


__all__ = (
    "ScheduleError",
    "TransportPlan",
    "FaultPlan",
    "ArrivalRelativeFaultPlan",
    "CausalFaultPlan",
    "OverlayDigest",
    "TransportOverlayRecord",
    "FaultOverlayRecord",
    "stable_uint64",
    "bernoulli_from_hash",
    "delay_from_hash",
    "training_condition_from_hash",
    "TRAINING_CONDITION_HASH_DOMAIN",
    "TRAINING_CONDITION_MODE",
    "write_transport_overlay",
    "write_fault_overlay",
    "write_arrival_relative_fault_overlay",
    "write_causal_fault_overlay",
    "read_overlay",
    "augmentation_seed",
)


class ScheduleError(ValueError):
    """A protocol plan, overlay, or immutable publication is invalid."""


_SPLITS = ("train", "val", "test")
_EVALUATION_SPLITS = ("val", "test")
_AGENTS = ("ego", "rsu")
_MODALITIES = ("lidar", "camera")
_DELAY_VALUES_MS = (0, 100, 200, 300)
_CONDITIONS = ("Full", "L-Fail", "C-Fail")
TRAINING_CONDITION_MODE = "train_condition_matrix"
TRAINING_CONDITION_HASH_DOMAIN = "training-condition-matrix-v1"
_TRAINING_CONDITION_MATRIX = tuple(
    (delay_ms, condition)
    for delay_ms in _DELAY_VALUES_MS
    for condition in _CONDITIONS
)
_TRANSPORT_FIELDS = frozenset(
    {"epoch", "sample_id", "packet_id", "n_s", "delay_ms", "arrival_tau_ms"}
)


def _exact_int(value: object, context: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ScheduleError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ScheduleError(f"{context} must be >= {minimum}")
    return value


def _optional_int(
    value: object,
    context: str,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _exact_int(value, context, minimum)


def _nonempty_string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ScheduleError(f"{context} must be a non-empty string")
    return value


def _literal(value: object, choices: tuple[str, ...], context: str) -> str:
    result = _nonempty_string(value, context)
    if result not in choices:
        raise ScheduleError(f"{context} must be one of: {', '.join(choices)}")
    return result


def _sha256(value: object, context: str) -> str:
    result = _nonempty_string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ScheduleError(f"{context} must be 64 lowercase hexadecimal characters")
    return result


def _sequence(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScheduleError(f"{context} must be a sequence")
    return tuple(value)


def _samples(
    value: object,
    split: str,
    context: str,
) -> tuple[TemporalSampleRecord, ...]:
    samples = _sequence(value, context)
    if not samples:
        raise ScheduleError(f"{context} must not be empty")
    if not all(isinstance(sample, TemporalSampleRecord) for sample in samples):
        raise ScheduleError(f"{context} must contain TemporalSampleRecord")
    if any(sample.split != split for sample in samples):
        raise ScheduleError(f"{context} must match the plan split")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ScheduleError(f"{context} sample IDs must be unique")
    return samples


def _epochs(value: object, context: str) -> tuple[int, ...]:
    raw = _sequence(value, context)
    epochs = tuple(_exact_int(item, context, 0) for item in raw)
    if len(epochs) != len(set(epochs)):
        raise ScheduleError(f"{context} must not contain duplicates")
    return epochs


def _agents(value: object, context: str) -> tuple[Literal["ego", "rsu"], ...]:
    raw = _sequence(value, context)
    agents = tuple(_literal(item, _AGENTS, context) for item in raw)
    if len(agents) != len(set(agents)):
        raise ScheduleError(f"{context} must not contain duplicates")
    return tuple(agent for agent in _AGENTS if agent in agents)  # type: ignore[return-value]


def _probability(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ScheduleError(f"{context} must be a number in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ScheduleError(f"{context} must be a number in [0, 1]")
    return result


@dataclass(frozen=True)
class TransportPlan:
    temporal_manifest_sha256: str
    split: Literal["train", "val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    mode: Literal[
        "train_random",
        "train_condition_matrix",
        "fixed_evaluation",
    ]
    protocol_seed: int | None
    epochs: tuple[int, ...]
    delay_values_ms: tuple[int, ...]
    fixed_delay_ms: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_manifest_sha256",
            _sha256(self.temporal_manifest_sha256, "temporal manifest digest"),
        )
        split = _literal(self.split, _SPLITS, "transport split")
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "samples",
            _samples(self.samples, split, "transport samples"),
        )
        mode = _literal(
            self.mode,
            ("train_random", TRAINING_CONDITION_MODE, "fixed_evaluation"),
            "transport mode",
        )
        object.__setattr__(self, "mode", mode)
        seed = _optional_int(self.protocol_seed, "transport protocol seed", 0)
        epochs = _epochs(self.epochs, "transport epochs")
        delays = tuple(
            _exact_int(item, "transport delay value", 0)
            for item in _sequence(
                self.delay_values_ms,
                "transport delay values",
            )
        )
        fixed_delay = _optional_int(
            self.fixed_delay_ms,
            "transport fixed delay",
            0,
        )
        if mode in ("train_random", TRAINING_CONDITION_MODE):
            if split != "train":
                raise ScheduleError("epoch-indexed transport is train-only")
            if seed is None or not epochs:
                raise ScheduleError(
                    "epoch-indexed transport requires a seed and non-empty epochs"
                )
            if delays != _DELAY_VALUES_MS or fixed_delay is not None:
                raise ScheduleError(
                    "epoch-indexed transport requires protocol delays and no fixed delay"
                )
        else:
            if split not in _EVALUATION_SPLITS:
                raise ScheduleError("fixed transport is evaluation-only")
            if seed is not None or epochs or delays:
                raise ScheduleError(
                    "fixed transport forbids seed, epochs, and random delays"
                )
            if fixed_delay not in _DELAY_VALUES_MS:
                raise ScheduleError("fixed transport requires one protocol fixed delay")
        object.__setattr__(self, "protocol_seed", seed)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "delay_values_ms", delays)
        object.__setattr__(self, "fixed_delay_ms", fixed_delay)


@dataclass(frozen=True)
class FaultPlan:
    temporal_manifest_sha256: str
    split: Literal["train", "val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    mode: Literal[
        "train_random",
        "train_condition_matrix",
        "global_target",
        "continuous",
    ]
    protocol_seed: int | None
    epochs: tuple[int, ...]
    condition: Literal["Full", "L-Fail", "C-Fail"] | None
    p_lidar: float | None
    p_camera: float | None
    agents: tuple[Literal["ego", "rsu"], ...]
    modality: Literal["lidar", "camera"] | None
    duration: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_manifest_sha256",
            _sha256(self.temporal_manifest_sha256, "temporal manifest digest"),
        )
        split = _literal(self.split, _SPLITS, "fault split")
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "samples",
            _samples(self.samples, split, "fault samples"),
        )
        mode = _literal(
            self.mode,
            (
                "train_random",
                TRAINING_CONDITION_MODE,
                "global_target",
                "continuous",
            ),
            "fault mode",
        )
        object.__setattr__(self, "mode", mode)
        seed = _optional_int(self.protocol_seed, "fault protocol seed", 0)
        epochs = _epochs(self.epochs, "fault epochs")
        condition = (
            None
            if self.condition is None
            else _literal(
                self.condition,
                ("Full", "L-Fail", "C-Fail"),
                "fault condition",
            )
        )
        p_lidar = (
            None
            if self.p_lidar is None
            else _probability(self.p_lidar, "LiDAR fault probability")
        )
        p_camera = (
            None
            if self.p_camera is None
            else _probability(self.p_camera, "Camera fault probability")
        )
        agents = _agents(self.agents, "fault agents")
        modality = (
            None
            if self.modality is None
            else _literal(self.modality, _MODALITIES, "fault modality")
        )
        duration = _optional_int(self.duration, "fault duration", 1)
        if mode == "train_random":
            if split != "train":
                raise ScheduleError("random faults are train-only")
            if seed is None or not epochs:
                raise ScheduleError("random faults require a seed and non-empty epochs")
            if p_lidar is None or p_camera is None:
                raise ScheduleError(
                    "random faults require LiDAR and Camera probabilities"
                )
            if (
                condition is not None
                or agents != _AGENTS
                or modality is not None
                or duration is not None
            ):
                raise ScheduleError("random faults target both agents at n_t only")
        elif mode == TRAINING_CONDITION_MODE:
            if split != "train":
                raise ScheduleError("condition-matrix faults are train-only")
            if seed is None or not epochs:
                raise ScheduleError(
                    "condition-matrix faults require a seed and non-empty epochs"
                )
            if (
                condition is not None
                or p_lidar is not None
                or p_camera is not None
                or agents != _AGENTS
                or modality is not None
                or duration is not None
            ):
                raise ScheduleError(
                    "condition-matrix faults require only both agents and the shared hash"
                )
        elif mode == "global_target":
            if split not in _EVALUATION_SPLITS:
                raise ScheduleError("global target faults are evaluation-only")
            if (
                seed is not None
                or epochs
                or condition is None
                or p_lidar is not None
                or p_camera is not None
                or agents != _AGENTS
                or modality is not None
                or duration is not None
            ):
                raise ScheduleError(
                    "global target faults require only condition and both agents"
                )
        else:
            if split not in _EVALUATION_SPLITS:
                raise ScheduleError("continuous faults are evaluation-only")
            if (
                seed is not None
                or epochs
                or condition is not None
                or p_lidar is not None
                or p_camera is not None
                or not agents
                or modality is None
                or duration is None
                or duration > 4
            ):
                raise ScheduleError(
                    "continuous faults require explicit agents, modality, and duration 1..4"
                )
        object.__setattr__(self, "protocol_seed", seed)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "p_lidar", p_lidar)
        object.__setattr__(self, "p_camera", p_camera)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True)
class ArrivalRelativeFaultPlan:
    temporal_manifest_sha256: str
    transport_overlay_sha256: str
    split: Literal["val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    scope: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    fixed_delay_ms: Literal[0, 300]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_manifest_sha256",
            _sha256(self.temporal_manifest_sha256, "temporal manifest digest"),
        )
        object.__setattr__(
            self,
            "transport_overlay_sha256",
            _sha256(self.transport_overlay_sha256, "transport overlay digest"),
        )
        split = _literal(self.split, _EVALUATION_SPLITS, "arrival-relative split")
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "samples",
            _samples(self.samples, split, "arrival-relative samples"),
        )
        object.__setattr__(
            self,
            "scope",
            _literal(self.scope, _AGENTS, "arrival-relative scope"),
        )
        object.__setattr__(
            self,
            "modality",
            _literal(
                self.modality,
                _MODALITIES,
                "arrival-relative modality",
            ),
        )
        delay = _exact_int(
            self.fixed_delay_ms,
            "arrival-relative fixed delay",
            0,
        )
        if delay not in (0, 300):
            raise ScheduleError("arrival-relative fixed delay must be 0 or 300")
        object.__setattr__(self, "fixed_delay_ms", delay)


@dataclass(frozen=True)
class CausalFaultPlan:
    """Evaluation faults anchored to each branch's causal endpoint.

    Unlike ``FaultPlan(mode='global_target')``, this plan binds the RSU mask
    to the latest packet that actually arrived under one immutable transport
    overlay.  It therefore implements the paper's joint latency--fault and
    E-only/R-only diagnostic contracts.
    """

    temporal_manifest_sha256: str
    transport_overlay_sha256: str
    split: Literal["val", "test"]
    samples: tuple[TemporalSampleRecord, ...]
    condition: Literal["Full", "L-Fail", "C-Fail"]
    agents: tuple[Literal["ego", "rsu"], ...]
    duration: int
    fixed_delay_ms: Literal[0, 100, 200, 300]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_manifest_sha256",
            _sha256(self.temporal_manifest_sha256, "temporal manifest digest"),
        )
        object.__setattr__(
            self,
            "transport_overlay_sha256",
            _sha256(self.transport_overlay_sha256, "transport overlay digest"),
        )
        split = _literal(self.split, _EVALUATION_SPLITS, "causal fault split")
        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "samples",
            _samples(self.samples, split, "causal fault samples"),
        )
        condition = _literal(
            self.condition,
            ("Full", "L-Fail", "C-Fail"),
            "causal fault condition",
        )
        agents = _agents(self.agents, "causal fault agents")
        if not agents:
            raise ScheduleError("causal faults require at least one agent")
        duration = _exact_int(self.duration, "causal fault duration", 1)
        if condition == "Full" and duration != 1:
            raise ScheduleError("Full condition requires duration 1")
        delay = _exact_int(self.fixed_delay_ms, "causal fault fixed delay", 0)
        if delay not in _DELAY_VALUES_MS:
            raise ScheduleError(
                "causal fault fixed delay must be 0, 100, 200, or 300"
            )
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "fixed_delay_ms", delay)


@dataclass(frozen=True)
class OverlayDigest:
    path: Path
    record_count: int
    uncompressed_size: int
    uncompressed_sha256: str
    compressed_size: int
    compressed_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.name:
            raise ScheduleError("overlay path must name a file")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "record_count",
            _exact_int(self.record_count, "overlay record count", 0),
        )
        object.__setattr__(
            self,
            "uncompressed_size",
            _exact_int(self.uncompressed_size, "uncompressed size", 0),
        )
        object.__setattr__(
            self,
            "uncompressed_sha256",
            _sha256(self.uncompressed_sha256, "uncompressed digest"),
        )
        object.__setattr__(
            self,
            "compressed_size",
            _exact_int(self.compressed_size, "compressed size", 0),
        )
        object.__setattr__(
            self,
            "compressed_sha256",
            _sha256(self.compressed_sha256, "compressed digest"),
        )


@dataclass(frozen=True)
class TransportOverlayRecord:
    epoch: int | None
    sample_id: str
    packet_id: str
    n_s: int
    delay_ms: int
    arrival_tau_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _optional_int(self.epoch, "transport record epoch", 0),
        )
        object.__setattr__(
            self,
            "sample_id",
            _nonempty_string(self.sample_id, "transport record sample ID"),
        )
        object.__setattr__(
            self,
            "packet_id",
            _nonempty_string(self.packet_id, "transport record packet ID"),
        )
        object.__setattr__(
            self,
            "n_s",
            _exact_int(self.n_s, "transport record n_s", 0),
        )
        object.__setattr__(
            self,
            "delay_ms",
            _exact_int(self.delay_ms, "transport record delay", 0),
        )
        object.__setattr__(
            self,
            "arrival_tau_ms",
            _exact_int(self.arrival_tau_ms, "transport record arrival", 0),
        )


@dataclass(frozen=True)
class FaultOverlayRecord:
    epoch: int | None
    sample_id: str
    agent: Literal["ego", "rsu"]
    modality: Literal["lidar", "camera"]
    n_s: int
    masked: bool
    pre_mask_selected_n_s: int | None
    fallback_selected_n_s: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _optional_int(self.epoch, "fault record epoch", 0),
        )
        object.__setattr__(
            self,
            "sample_id",
            _nonempty_string(self.sample_id, "fault record sample ID"),
        )
        object.__setattr__(
            self,
            "agent",
            _literal(self.agent, _AGENTS, "fault record agent"),
        )
        object.__setattr__(
            self,
            "modality",
            _literal(self.modality, _MODALITIES, "fault record modality"),
        )
        object.__setattr__(
            self,
            "n_s",
            _exact_int(self.n_s, "fault record n_s", 0),
        )
        if type(self.masked) is not bool:
            raise ScheduleError("fault record masked must be a boolean")
        object.__setattr__(
            self,
            "pre_mask_selected_n_s",
            _optional_int(
                self.pre_mask_selected_n_s,
                "fault record pre-mask selection",
                0,
            ),
        )
        object.__setattr__(
            self,
            "fallback_selected_n_s",
            _optional_int(
                self.fallback_selected_n_s,
                "fault record fallback selection",
                0,
            ),
        )


def stable_uint64(domain: str, key: Mapping[str, object]) -> int:
    domain_value = _nonempty_string(domain, "hash domain")
    if not isinstance(key, Mapping):
        raise ScheduleError("hash key must be a mapping")
    try:
        payload = domain_value.encode("utf-8") + b"\x00" + canonical_json_bytes(key)
    except (TypeError, ValueError, UnicodeError) as error:
        raise ScheduleError("hash key is outside the canonical JSON domain") from error
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bernoulli_from_hash(
    domain: str,
    key: Mapping[str, object],
    probability: float,
) -> bool:
    probability_value = _probability(probability, "probability")
    x = stable_uint64(domain, key)
    u = (x + 0.5) / 2**64
    return u < probability_value


def delay_from_hash(
    domain: str,
    key: Mapping[str, object],
    values_ms: Sequence[int],
) -> int:
    values = tuple(
        _exact_int(value, "delay choice", 0)
        for value in _sequence(values_ms, "delay choices")
    )
    if not values:
        raise ScheduleError("delay choices must not be empty")
    return values[stable_uint64(domain, key) % len(values)]


def training_condition_from_hash(
    protocol_seed: int,
    epoch: int,
    sample_id: str,
) -> tuple[int, str]:
    """Select one of the formal 4-delay by 3-condition cells."""

    index = stable_uint64(
        TRAINING_CONDITION_HASH_DOMAIN,
        {
            "seed": _exact_int(protocol_seed, "training protocol seed", 0),
            "epoch": _exact_int(epoch, "training condition epoch", 0),
            "sample_id": _nonempty_string(
                sample_id,
                "training condition sample ID",
            ),
        },
    ) % len(_TRAINING_CONDITION_MATRIX)
    return _TRAINING_CONDITION_MATRIX[index]


def augmentation_seed(seed: int, epoch: int, sample_id: str) -> int:
    return stable_uint64(
        "augmentation-seed-v1",
        {
            "seed": _exact_int(seed, "augmentation seed", 0),
            "epoch": _exact_int(epoch, "augmentation epoch", 0),
            "sample_id": _nonempty_string(
                sample_id,
                "augmentation sample ID",
            ),
        },
    )


def _transport_sort_key(
    record: TransportOverlayRecord,
) -> tuple[bool, int, str, str]:
    return (
        record.epoch is None,
        -1 if record.epoch is None else record.epoch,
        record.sample_id,
        record.packet_id,
    )


def _fault_sort_key(
    record: FaultOverlayRecord,
) -> tuple[bool, int, str, str, str, int]:
    return (
        record.epoch is None,
        -1 if record.epoch is None else record.epoch,
        record.sample_id,
        record.agent,
        record.modality,
        record.n_s,
    )


def _transport_records(plan: TransportPlan) -> tuple[TransportOverlayRecord, ...]:
    if any(
        not any(source.agent == "rsu" for source in sample.source_slices)
        for sample in plan.samples
    ):
        raise ScheduleError("transport samples require RSU packet coverage")
    records: list[TransportOverlayRecord] = []
    epochs: tuple[int | None, ...] = (
        (None,) if plan.mode == "fixed_evaluation" else tuple(plan.epochs)
    )
    for epoch in epochs:
        for sample in plan.samples:
            shared_delay: int | None = None
            if plan.mode == TRAINING_CONDITION_MODE:
                assert plan.protocol_seed is not None
                assert epoch is not None
                shared_delay, _ = training_condition_from_hash(
                    plan.protocol_seed,
                    epoch,
                    sample.sample_id,
                )
            delay_by_tick: dict[int, int] = {}
            for source in sample.source_slices:
                if source.agent != "rsu":
                    continue
                if source.n_s not in delay_by_tick:
                    if plan.mode == "train_random":
                        assert plan.protocol_seed is not None
                        assert epoch is not None
                        delay_by_tick[source.n_s] = delay_from_hash(
                            "transport-delay-choice-v1",
                            {
                                "seed": plan.protocol_seed,
                                "epoch": epoch,
                                "sample_id": sample.sample_id,
                                "n_s": source.n_s,
                            },
                            plan.delay_values_ms,
                        )
                    elif plan.mode == TRAINING_CONDITION_MODE:
                        assert shared_delay is not None
                        delay_by_tick[source.n_s] = shared_delay
                    else:
                        assert plan.fixed_delay_ms is not None
                        delay_by_tick[source.n_s] = plan.fixed_delay_ms
                delay = delay_by_tick[source.n_s]
                records.append(
                    TransportOverlayRecord(
                        epoch=epoch,
                        sample_id=sample.sample_id,
                        packet_id=source.packet_id,
                        n_s=source.n_s,
                        delay_ms=delay,
                        arrival_tau_ms=source.tau_s_ms + delay,
                    )
                )
    records.sort(key=_transport_sort_key)
    return tuple(records)


def _fault_records(plan: FaultPlan) -> tuple[FaultOverlayRecord, ...]:
    records: list[FaultOverlayRecord] = []
    if plan.mode == "train_random":
        assert plan.protocol_seed is not None
        assert plan.p_lidar is not None
        assert plan.p_camera is not None
        for epoch in plan.epochs:
            for sample in plan.samples:
                for agent in _AGENTS:
                    for modality in _MODALITIES:
                        probability = (
                            plan.p_lidar if modality == "lidar" else plan.p_camera
                        )
                        records.append(
                            FaultOverlayRecord(
                                epoch=epoch,
                                sample_id=sample.sample_id,
                                agent=agent,  # type: ignore[arg-type]
                                modality=modality,  # type: ignore[arg-type]
                                n_s=sample.n_t,
                                masked=bernoulli_from_hash(
                                    "target-fault-v1",
                                    {
                                        "seed": plan.protocol_seed,
                                        "epoch": epoch,
                                        "sample_id": sample.sample_id,
                                        "agent": agent,
                                        "modality": modality,
                                        "n_t": sample.n_t,
                                    },
                                    probability,
                                ),
                                pre_mask_selected_n_s=None,
                                fallback_selected_n_s=None,
                            )
                        )
    elif plan.mode == TRAINING_CONDITION_MODE:
        assert plan.protocol_seed is not None
        for epoch in plan.epochs:
            for sample in plan.samples:
                delay_ms, condition = training_condition_from_hash(
                    plan.protocol_seed,
                    epoch,
                    sample.sample_id,
                )
                failed_modality = {
                    "Full": None,
                    "L-Fail": "lidar",
                    "C-Fail": "camera",
                }[condition]
                arrival_by_packet = {
                    source.packet_id: source.tau_s_ms + delay_ms
                    for source in sample.source_slices
                    if source.agent == "rsu"
                }
                for agent in _AGENTS:
                    for modality in _MODALITIES:
                        sources = _causal_branch_sources(
                            sample,
                            agent,
                            modality,
                            arrival_by_packet,
                        )
                        if not sources:
                            raise ScheduleError(
                                "condition-matrix branch has no causal source selection"
                            )
                        selected = sources[0].n_s
                        masked = modality == failed_modality
                        fallback = (
                            sources[1].n_s if masked and len(sources) > 1 else None
                        )
                        records.append(
                            FaultOverlayRecord(
                                epoch=epoch,
                                sample_id=sample.sample_id,
                                agent=agent,  # type: ignore[arg-type]
                                modality=modality,  # type: ignore[arg-type]
                                n_s=selected,
                                masked=masked,
                                pre_mask_selected_n_s=selected,
                                fallback_selected_n_s=(
                                    fallback if masked else selected
                                ),
                            )
                        )
    elif plan.mode == "global_target":
        assert plan.condition is not None
        failed_modality = {
            "Full": None,
            "L-Fail": "lidar",
            "C-Fail": "camera",
        }[plan.condition]
        for sample in plan.samples:
            for agent in _AGENTS:
                for modality in _MODALITIES:
                    records.append(
                        FaultOverlayRecord(
                            epoch=None,
                            sample_id=sample.sample_id,
                            agent=agent,  # type: ignore[arg-type]
                            modality=modality,  # type: ignore[arg-type]
                            n_s=sample.n_t,
                            masked=modality == failed_modality,
                            pre_mask_selected_n_s=None,
                            fallback_selected_n_s=None,
                        )
                    )
    else:
        assert plan.modality is not None
        assert plan.duration is not None
        for sample in plan.samples:
            start = max(0, sample.n_t - plan.duration + 1)
            for agent in plan.agents:
                for n_s in range(start, sample.n_t + 1):
                    records.append(
                        FaultOverlayRecord(
                            epoch=None,
                            sample_id=sample.sample_id,
                            agent=agent,
                            modality=plan.modality,
                            n_s=n_s,
                            masked=True,
                            pre_mask_selected_n_s=None,
                            fallback_selected_n_s=None,
                        )
                    )
    records.sort(key=_fault_sort_key)
    return tuple(records)


def _encode_records(
    records: Sequence[TransportOverlayRecord | FaultOverlayRecord],
) -> bytes:
    return b"".join(canonical_json_bytes(asdict(record)) + b"\n" for record in records)


def _compress(raw: bytes) -> bytes:
    return zstandard.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=True,
    ).compress(raw)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short staging write")
        view = view[written:]


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _publish_no_replace(staging: Path, output: Path) -> None:
    os.link(staging, output, follow_symlinks=False)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging(staging: Path) -> None:
    os.unlink(staging)


def _read_regular_file(path: Path, context: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ScheduleError(f"unable to inspect {context}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ScheduleError(f"{context} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ScheduleError(f"unable to read {context}") from error


def _decompress(compressed: bytes) -> bytes:
    try:
        return zstandard.ZstdDecompressor().decompress(compressed)
    except zstandard.ZstdError as error:
        raise ScheduleError("invalid zstd overlay") from error


def _verify_encoded(compressed: bytes, raw: bytes) -> None:
    if hashlib.sha256(compressed).digest() != hashlib.sha256(_compress(raw)).digest():
        raise ScheduleError("compressed overlay verification failed")
    if _decompress(compressed) != raw:
        raise ScheduleError("uncompressed overlay verification failed")


def _existing_matches(output: Path, compressed: bytes, raw: bytes) -> bool:
    existing = _read_regular_file(output, "existing overlay")
    if existing != compressed:
        return False
    return _decompress(existing) == raw


def _digest(
    output: Path,
    records: Sequence[TransportOverlayRecord | FaultOverlayRecord],
    raw: bytes,
    compressed: bytes,
) -> OverlayDigest:
    return OverlayDigest(
        path=output,
        record_count=len(records),
        uncompressed_size=len(raw),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_size=len(compressed),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
    )


def _publish_overlay(
    records: Sequence[TransportOverlayRecord | FaultOverlayRecord],
    output: Path,
) -> OverlayDigest:
    output = Path(output)
    parent = output.parent
    if not output.name or not parent.is_dir():
        raise ScheduleError("overlay output parent must be an existing directory")
    raw = _encode_records(records)
    compressed = _compress(raw)
    staging = parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    staging_exists = False
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        staging_exists = True
        try:
            _write_all(descriptor, compressed)
            _fsync_file(descriptor)
        finally:
            os.close(descriptor)
        staged = _read_regular_file(staging, "staging overlay")
        _verify_encoded(staged, raw)
        try:
            output.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ScheduleError("unable to inspect overlay destination") from error
        else:
            if not _existing_matches(output, compressed, raw):
                raise ScheduleError("overlay destination conflict")
            _remove_staging(staging)
            staging_exists = False
            return _digest(output, records, raw, compressed)
        try:
            _publish_no_replace(staging, output)
        except FileExistsError:
            if not _existing_matches(output, compressed, raw):
                raise ScheduleError("overlay destination conflict")
        _fsync_directory(parent)
        _remove_staging(staging)
        staging_exists = False
        return _digest(output, records, raw, compressed)
    except BaseException as error:
        cleanup_error: BaseException | None = None
        if staging_exists:
            try:
                _remove_staging(staging)
            except BaseException as caught:
                cleanup_error = caught
        if isinstance(error, ScheduleError):
            raise
        raise ScheduleError("overlay publish failed") from (
            cleanup_error if cleanup_error is not None else error
        )


def write_transport_overlay(
    plan: TransportPlan,
    output: Path,
) -> OverlayDigest:
    if not isinstance(plan, TransportPlan):
        raise ScheduleError("transport plan must be TransportPlan")
    return _publish_overlay(_transport_records(plan), output)


def write_fault_overlay(plan: FaultPlan, output: Path) -> OverlayDigest:
    if not isinstance(plan, FaultPlan):
        raise ScheduleError("fault plan must be FaultPlan")
    return _publish_overlay(_fault_records(plan), output)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleError(f"duplicate overlay JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ScheduleError(f"non-finite overlay JSON constant: {value}")


def _decode_record(
    raw_line: bytes,
    line_number: int,
) -> Mapping[str, object]:
    try:
        text = raw_line.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ScheduleError(f"invalid overlay JSON at line {line_number}") from error
    if not isinstance(value, Mapping):
        raise ScheduleError(f"overlay line {line_number} must be an object")
    if canonical_json_bytes(value) != raw_line:
        raise ScheduleError(f"overlay line {line_number} is not canonical JSON")
    return MappingProxyType(dict(value))


def _decode_overlay_bytes(
    compressed: bytes,
    expected_uncompressed_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    expected = _sha256(expected_uncompressed_sha256, "expected overlay digest")
    raw = _decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ScheduleError("uncompressed overlay digest mismatch")
    if not raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ScheduleError("overlay must be non-empty newline-terminated JSONL")
    return tuple(
        _decode_record(line, line_number)
        for line_number, line in enumerate(raw[:-1].split(b"\n"), 1)
    )


def read_overlay(
    path: Path,
    expected_uncompressed_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    return _decode_overlay_bytes(
        _read_regular_file(Path(path), "overlay"),
        expected_uncompressed_sha256,
    )


def _transport_from_mappings(
    plan: ArrivalRelativeFaultPlan | CausalFaultPlan,
    transport_records: Sequence[Mapping[str, object]],
) -> tuple[TransportOverlayRecord, ...]:
    if isinstance(transport_records, (str, bytes)) or not isinstance(
        transport_records,
        Sequence,
    ):
        raise ScheduleError("transport records must be a sequence")
    expected_sources = {
        (sample.sample_id, source.packet_id): source
        for sample in plan.samples
        for source in sample.source_slices
        if source.agent == "rsu"
    }
    parsed: list[TransportOverlayRecord] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(transport_records):
        if not isinstance(value, Mapping) or frozenset(value) != _TRANSPORT_FIELDS:
            raise ScheduleError(f"transport record {index} fields mismatch")
        try:
            record = TransportOverlayRecord(**dict(value))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ScheduleError(f"invalid transport record {index}") from error
        key = (record.sample_id, record.packet_id)
        source = expected_sources.get(key)
        if source is None:
            raise ScheduleError("transport overlay contains an ego or unknown packet")
        if key in seen:
            raise ScheduleError("transport overlay contains a duplicate packet")
        seen.add(key)
        if (
            record.epoch is not None
            or record.n_s != source.n_s
            or record.delay_ms != plan.fixed_delay_ms
            or record.arrival_tau_ms != source.tau_s_ms + record.delay_ms
        ):
            raise ScheduleError("transport overlay record provenance mismatch")
        parsed.append(record)
    if seen != set(expected_sources):
        raise ScheduleError("transport overlay does not exactly cover RSU packets")
    raw = _encode_records(parsed)
    if hashlib.sha256(raw).hexdigest() != plan.transport_overlay_sha256:
        raise ScheduleError("transport overlay digest mismatch")
    return tuple(parsed)


def _select_latest_source(
    sample: TemporalSampleRecord,
    agent: str,
    modality: str,
    arrival_by_packet: Mapping[str, int],
    excluded_n_s: int | None = None,
) -> int | None:
    selected: int | None = None
    for source in sample.source_slices:
        if (source.agent, source.modality) != (agent, modality):
            continue
        if source.n_s == excluded_n_s:
            continue
        if agent == "rsu":
            arrival = arrival_by_packet.get(source.packet_id)
            if arrival is None or arrival > sample.tau_t_ms:
                continue
        if selected is None or source.n_s > selected:
            selected = source.n_s
    return selected


def write_arrival_relative_fault_overlay(
    plan: ArrivalRelativeFaultPlan,
    temporal_manifest: TemporalManifest,
    transport_records: Sequence[Mapping[str, object]],
    output: Path,
) -> OverlayDigest:
    if not isinstance(plan, ArrivalRelativeFaultPlan):
        raise ScheduleError("arrival-relative plan must be ArrivalRelativeFaultPlan")
    if not isinstance(temporal_manifest, TemporalManifest):
        raise ScheduleError("temporal manifest must be TemporalManifest")
    if temporal_manifest.content_sha256 != plan.temporal_manifest_sha256:
        raise ScheduleError("temporal manifest digest mismatch")
    manifest_samples = {
        sample.sample_id: sample for sample in temporal_manifest.samples
    }
    if any(manifest_samples.get(sample.sample_id) != sample for sample in plan.samples):
        raise ScheduleError("arrival-relative samples are absent from manifest")
    parsed = _transport_from_mappings(plan, transport_records)
    arrival_by_packet = {record.packet_id: record.arrival_tau_ms for record in parsed}
    records: list[FaultOverlayRecord] = []
    for sample in plan.samples:
        selected = _select_latest_source(
            sample,
            plan.scope,
            plan.modality,
            arrival_by_packet,
        )
        if selected is None:
            raise ScheduleError(
                "arrival-relative branch has no no-fault source selection"
            )
        fallback = _select_latest_source(
            sample,
            plan.scope,
            plan.modality,
            arrival_by_packet,
            excluded_n_s=selected,
        )
        records.append(
            FaultOverlayRecord(
                epoch=None,
                sample_id=sample.sample_id,
                agent=plan.scope,
                modality=plan.modality,
                n_s=selected,
                masked=True,
                pre_mask_selected_n_s=selected,
                fallback_selected_n_s=fallback,
            )
        )
    records.sort(key=_fault_sort_key)
    return _publish_overlay(records, output)


def _causal_branch_sources(
    sample: TemporalSampleRecord,
    agent: str,
    modality: str,
    arrival_by_packet: Mapping[str, int],
) -> tuple[RawSliceRecord, ...]:
    sources = []
    for source in sample.source_slices:
        if (source.agent, source.modality) != (agent, modality):
            continue
        if source.n_s > sample.n_t:
            continue
        if agent == "rsu":
            arrival = arrival_by_packet.get(source.packet_id)
            if arrival is None or arrival > sample.tau_t_ms:
                continue
        sources.append(source)
    sources.sort(key=lambda source: source.n_s, reverse=True)
    return tuple(sources)


def write_causal_fault_overlay(
    plan: CausalFaultPlan,
    temporal_manifest: TemporalManifest,
    transport_records: Sequence[Mapping[str, object]],
    output: Path,
) -> OverlayDigest:
    """Write endpoint-relative Full/L-Fail/C-Fail evaluation records."""

    if not isinstance(plan, CausalFaultPlan):
        raise ScheduleError("causal fault plan must be CausalFaultPlan")
    if not isinstance(temporal_manifest, TemporalManifest):
        raise ScheduleError("temporal manifest must be TemporalManifest")
    if temporal_manifest.content_sha256 != plan.temporal_manifest_sha256:
        raise ScheduleError("temporal manifest digest mismatch")
    if plan.duration > temporal_manifest.history_limit + 1:
        raise ScheduleError(
            "causal fault duration exceeds the represented history"
        )
    manifest_samples = {
        sample.sample_id: sample for sample in temporal_manifest.samples
    }
    if any(manifest_samples.get(sample.sample_id) != sample for sample in plan.samples):
        raise ScheduleError("causal fault samples are absent from manifest")

    parsed = _transport_from_mappings(plan, transport_records)
    arrival_by_packet = {record.packet_id: record.arrival_tau_ms for record in parsed}
    failed_modality = {
        "Full": None,
        "L-Fail": "lidar",
        "C-Fail": "camera",
    }[plan.condition]
    records: list[FaultOverlayRecord] = []
    for sample in plan.samples:
        for agent in _AGENTS:
            for modality in _MODALITIES:
                sources = _causal_branch_sources(
                    sample,
                    agent,
                    modality,
                    arrival_by_packet,
                )
                if not sources:
                    raise ScheduleError(
                        "causal fault branch has no no-fault source selection"
                    )
                masked = agent in plan.agents and modality == failed_modality
                count = plan.duration if masked else 1
                if len(sources) < count:
                    raise ScheduleError(
                        "causal fault sample lacks the requested history duration"
                    )
                selected = sources[0].n_s
                fallback = (
                    sources[count].n_s if masked and len(sources) > count else None
                )
                for source in sources[:count]:
                    records.append(
                        FaultOverlayRecord(
                            epoch=None,
                            sample_id=sample.sample_id,
                            agent=agent,  # type: ignore[arg-type]
                            modality=modality,  # type: ignore[arg-type]
                            n_s=source.n_s,
                            masked=masked,
                            pre_mask_selected_n_s=selected,
                            fallback_selected_n_s=fallback if masked else selected,
                        )
                    )
    records.sort(key=_fault_sort_key)
    return _publish_overlay(records, output)
