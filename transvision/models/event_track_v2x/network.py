"""Deterministic synthetic V2X network traces and byte accounting.

The legacy :class:`NetworkTrace` API remains intentionally small.  The
versioned V1 objects below freeze the paper-facing condition, perturbation and
wire-accounting contract so that traces can be hashed and shared across
methods without relying on mutable experiment configuration.

Any attempt timeline emitted here is a deterministic simulator ledger.  It is
not a measured packet capture and cannot substantiate a real-link claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq
import json
import math
from typing import Any

import numpy as np

from .schema import WireMessage
from .wire import wire_size


class LossModel(str, Enum):
    NONE = "none"
    IID = "iid"
    GILBERT_ELLIOTT = "gilbert_elliott"


class JitterModel(str, Enum):
    """Delay perturbation distribution used by a network condition."""

    NONE = "none"
    UNIFORM = "uniform"
    TRUNCATED_NORMAL = "truncated_normal"


@dataclass(frozen=True, slots=True)
class GilbertElliott:
    probability_good_to_good: float
    probability_bad_to_good: float
    success_probability_good: float
    success_probability_bad: float
    initial_good_probability: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "probability_good_to_good",
            "probability_bad_to_good",
            "success_probability_good",
            "success_probability_bad",
            "initial_good_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    fixed_delay: float
    jitter: float = 0.0
    loss_model: LossModel = LossModel.NONE
    iid_loss_probability: float = 0.0
    gilbert_elliott: GilbertElliott | None = None
    reorder_probability: float = 0.0
    reorder_extra_delay: float = 0.0
    outage_intervals: tuple[tuple[float, float], ...] = ()
    jitter_model: JitterModel = JitterModel.UNIFORM
    jitter_clip_sigma: float = 3.0

    def __post_init__(self) -> None:
        for name in ("fixed_delay", "jitter", "reorder_extra_delay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        jitter_clip_sigma = float(self.jitter_clip_sigma)
        if not np.isfinite(jitter_clip_sigma) or jitter_clip_sigma <= 0.0:
            raise ValueError("jitter_clip_sigma must be finite and positive")
        object.__setattr__(self, "jitter_clip_sigma", jitter_clip_sigma)
        jitter_model = JitterModel(self.jitter_model)
        if self.jitter == 0.0:
            jitter_model = JitterModel.NONE
        elif jitter_model is JitterModel.NONE:
            raise ValueError("non-zero jitter requires a jitter distribution")
        object.__setattr__(self, "jitter_model", jitter_model)
        for name in ("iid_loss_probability", "reorder_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        loss_model = LossModel(self.loss_model)
        object.__setattr__(self, "loss_model", loss_model)
        if loss_model is LossModel.GILBERT_ELLIOTT and self.gilbert_elliott is None:
            raise ValueError("Gilbert-Elliott loss requires gilbert_elliott parameters")
        if loss_model is not LossModel.GILBERT_ELLIOTT and self.gilbert_elliott is not None:
            raise ValueError("gilbert_elliott parameters require the matching loss model")
        intervals: list[tuple[float, float]] = []
        for start, end in self.outage_intervals:
            start = float(start)
            end = float(end)
            if not np.isfinite(start) or not np.isfinite(end) or start > end:
                raise ValueError("outage intervals must be finite and ordered")
            intervals.append((start, end))
        object.__setattr__(self, "outage_intervals", tuple(sorted(intervals)))


@dataclass(frozen=True, slots=True)
class PacketRequest:
    """A channel request timed only in the network's monotonic clock domain.

    ``transmitted_at`` must never be populated from
    :class:`~event_track_v2x.schema.LocalTimestamps`; sender-local timestamps
    remain message payload data used by the clock estimator.
    """

    message_id: str
    source: str
    sequence: int
    transmitted_at: float
    deadline: float
    encoded_bytes: int

    def __post_init__(self) -> None:
        if not self.message_id or not self.source:
            raise ValueError("packet identity fields must be non-empty")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        for name in ("transmitted_at", "deadline"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if isinstance(self.encoded_bytes, bool) or not isinstance(self.encoded_bytes, int):
            raise TypeError("encoded_bytes must be an integer")
        if self.encoded_bytes <= 0:
            raise ValueError("encoded_bytes must be positive")

    @classmethod
    def from_message(
        cls,
        message: WireMessage,
        *,
        transmitted_at: float,
        deadline: float | None = None,
    ) -> "PacketRequest":
        return cls(
            message_id=message.message_id,
            source=message.source,
            sequence=message.sequence,
            transmitted_at=transmitted_at,
            deadline=message.deadline if deadline is None else deadline,
            encoded_bytes=wire_size(message),
        )


@dataclass(frozen=True, slots=True)
class NetworkEvent:
    packet: PacketRequest
    arrival_time: float | None
    dropped: bool
    channel_state: str | None

    @property
    def on_time(self) -> bool:
        return self.arrival_time is not None and self.arrival_time <= self.packet.deadline


@dataclass(frozen=True, slots=True)
class NetworkTrace:
    seed: int
    events: tuple[NetworkEvent, ...]

    @property
    def arrival_order(self) -> tuple[NetworkEvent, ...]:
        return tuple(
            sorted(
                (event for event in self.events if event.arrival_time is not None),
                key=lambda event: (
                    event.arrival_time,
                    event.packet.source,
                    event.packet.sequence,
                    event.packet.message_id,
                ),
            )
        )


def _in_outage(time: float, intervals: tuple[tuple[float, float], ...]) -> bool:
    return any(start <= time <= end for start, end in intervals)


def _truncated_normal(
    rng: np.random.Generator,
    *,
    standard_deviation: float,
    clip_sigma: float,
) -> float:
    limit = standard_deviation * clip_sigma
    while True:
        sample = float(rng.normal(0.0, standard_deviation))
        if -limit <= sample <= limit:
            return sample


def generate_network_trace(
    packets: list[PacketRequest] | tuple[PacketRequest, ...],
    config: NetworkConfig,
    *,
    seed: int,
) -> NetworkTrace:
    """Generate a reproducible trace independent of input container order."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ordered = sorted(
        packets,
        key=lambda packet: (
            packet.transmitted_at,
            packet.source,
            packet.sequence,
            packet.message_id,
        ),
    )
    identities = [packet.message_id for packet in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("packet message_id values must be unique")
    rng = np.random.default_rng(seed)
    good: bool | None = None
    ge = config.gilbert_elliott
    if config.loss_model is LossModel.GILBERT_ELLIOTT:
        assert ge is not None
        good = bool(rng.random() < ge.initial_good_probability)

    events: list[NetworkEvent] = []
    for packet in ordered:
        state_name: str | None = None
        dropped = _in_outage(packet.transmitted_at, config.outage_intervals)
        if config.loss_model is LossModel.IID:
            dropped = dropped or bool(rng.random() < config.iid_loss_probability)
        elif config.loss_model is LossModel.GILBERT_ELLIOTT:
            assert ge is not None and good is not None
            state_name = "good" if good else "bad"
            success_probability = (
                ge.success_probability_good if good else ge.success_probability_bad
            )
            dropped = dropped or bool(rng.random() >= success_probability)
            if good:
                good = bool(rng.random() < ge.probability_good_to_good)
            else:
                good = bool(rng.random() < ge.probability_bad_to_good)

        if dropped:
            arrival = None
        else:
            if config.jitter_model is JitterModel.UNIFORM:
                jitter = rng.uniform(-config.jitter, config.jitter)
            elif config.jitter_model is JitterModel.TRUNCATED_NORMAL:
                jitter = _truncated_normal(
                    rng,
                    standard_deviation=config.jitter,
                    clip_sigma=config.jitter_clip_sigma,
                )
            else:
                jitter = 0.0
            delay = max(0.0, config.fixed_delay + jitter)
            if config.reorder_probability and rng.random() < config.reorder_probability:
                delay += config.reorder_extra_delay
            arrival = packet.transmitted_at + delay
        events.append(
            NetworkEvent(
                packet=packet,
                arrival_time=arrival,
                dropped=dropped,
                channel_state=state_name,
            )
        )
    return NetworkTrace(seed=seed, events=tuple(events))


NETWORK_TRACE_SCHEMA_V1 = "eventtrack-v2x.network-trace.v1"
NETWORK_CONDITION_PLAN_SCHEMA_V1 = "eventtrack-v2x.network-condition-plan.v1"
DEFAULT_NETWORK_SEEDS_V1 = tuple(range(1001, 1011))


class NetworkTraceError(ValueError):
    """Raised when a serialized NetworkTraceV1 is unsafe or inconsistent."""


class NetworkConditionId(str, Enum):
    """Preregistered synthetic conditions; C9 is reserved for measured traces."""

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    C6 = "C6"
    C7 = "C7"
    C8 = "C8"


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class WireAccountingConfigV1:
    """Explicit byte-accounting assumptions for one link direction.

    ``max_frame_bytes`` and ``per_fragment_header_bytes`` are both on-wire
    sizes.  ``ack_frame_bytes`` is already a complete on-wire frame size and
    is therefore not fragmented again.  ``max_retransmissions`` counts
    additional data sends after attempt zero, and each attempt's ACK deadline
    is measured from that attempt's send time.
    """

    max_frame_bytes: int = 1500
    per_fragment_header_bytes: int = 48
    per_message_header_bytes: int = 0
    ack_frame_bytes: int = 64
    ack_on_success: bool = True
    ack_timeout_seconds: float = 0.100
    max_retransmissions: int = 0

    def __post_init__(self) -> None:
        maximum = _integer(self.max_frame_bytes, "max_frame_bytes", minimum=1)
        fragment_header = _integer(
            self.per_fragment_header_bytes,
            "per_fragment_header_bytes",
        )
        message_header = _integer(
            self.per_message_header_bytes,
            "per_message_header_bytes",
        )
        ack = _integer(self.ack_frame_bytes, "ack_frame_bytes")
        ack_timeout = float(self.ack_timeout_seconds)
        if not np.isfinite(ack_timeout) or ack_timeout <= 0.0:
            raise ValueError("ack_timeout_seconds must be finite and positive")
        max_retransmissions = _integer(
            self.max_retransmissions,
            "max_retransmissions",
        )
        if fragment_header >= maximum:
            raise ValueError("per_fragment_header_bytes must be smaller than max_frame_bytes")
        if not isinstance(self.ack_on_success, bool):
            raise TypeError("ack_on_success must be bool")
        if max_retransmissions and not self.ack_on_success:
            raise ValueError("retransmissions require ack_on_success")
        if max_retransmissions and ack == 0:
            raise ValueError("retransmissions require a non-zero ACK frame")
        object.__setattr__(self, "max_frame_bytes", maximum)
        object.__setattr__(self, "per_fragment_header_bytes", fragment_header)
        object.__setattr__(self, "per_message_header_bytes", message_header)
        object.__setattr__(self, "ack_frame_bytes", ack)
        object.__setattr__(self, "ack_timeout_seconds", ack_timeout)
        object.__setattr__(self, "max_retransmissions", max_retransmissions)

    @property
    def fragment_payload_bytes(self) -> int:
        return self.max_frame_bytes - self.per_fragment_header_bytes

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "ack_frame_bytes": self.ack_frame_bytes,
            "ack_on_success": self.ack_on_success,
            "ack_timeout_seconds": self.ack_timeout_seconds,
            "max_frame_bytes": self.max_frame_bytes,
            "max_retransmissions": self.max_retransmissions,
            "per_fragment_header_bytes": self.per_fragment_header_bytes,
            "per_message_header_bytes": self.per_message_header_bytes,
        }


@dataclass(frozen=True, slots=True)
class ByteAccountV1:
    """Exact application and on-wire byte totals for one logical message."""

    application_bytes: int
    data_transmissions: int
    fragments_per_transmission: int
    data_on_wire_bytes: int
    ack_transmissions: int
    ack_on_wire_bytes: int
    total_on_wire_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "application_bytes",
            "data_transmissions",
            "fragments_per_transmission",
            "data_on_wire_bytes",
            "ack_transmissions",
            "ack_on_wire_bytes",
            "total_on_wire_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.application_bytes <= 0:
            raise ValueError("application_bytes must be positive")
        if self.data_transmissions > 0 and self.fragments_per_transmission <= 0:
            raise ValueError("a data transmission must contain at least one fragment")
        if self.data_transmissions == 0 and (
            self.fragments_per_transmission != 0 or self.data_on_wire_bytes != 0
        ):
            raise ValueError("zero data transmissions must account for zero fragments and bytes")
        if self.data_transmissions > 0 and self.data_on_wire_bytes < (
            self.application_bytes * self.data_transmissions
        ):
            raise ValueError(
                "data_on_wire_bytes cannot be smaller than transmitted application bytes"
            )
        if self.ack_transmissions == 0 and self.ack_on_wire_bytes != 0:
            raise ValueError("zero ACK transmissions must account for zero ACK bytes")
        if self.total_on_wire_bytes != self.data_on_wire_bytes + self.ack_on_wire_bytes:
            raise ValueError("total_on_wire_bytes must equal data plus ACK bytes")

    @property
    def retransmissions(self) -> int:
        return max(0, self.data_transmissions - 1)

    def to_dict(self) -> dict[str, int]:
        return {
            "ack_on_wire_bytes": self.ack_on_wire_bytes,
            "ack_transmissions": self.ack_transmissions,
            "application_bytes": self.application_bytes,
            "data_on_wire_bytes": self.data_on_wire_bytes,
            "data_transmissions": self.data_transmissions,
            "fragments_per_transmission": self.fragments_per_transmission,
            "retransmissions": self.retransmissions,
            "total_on_wire_bytes": self.total_on_wire_bytes,
        }


SYNTHETIC_LEDGER_SOURCE_V1 = "synthetic_state_machine_not_packet_capture"


@dataclass(frozen=True, slots=True)
class SyntheticTransmissionAttemptV1:
    """One auditable attempt emitted by the deterministic simulator.

    This is a synthetic protocol timeline, not evidence from a packet capture.
    Attempt zero is the initial transmission; positive indices are bounded
    retransmissions.
    """

    attempt_index: int
    data_sent_at: float
    data_dropped: bool
    data_arrival_time: float | None
    data_channel_state: str | None
    ack_sent_at: float | None
    ack_dropped: bool
    ack_arrival_time: float | None
    ack_channel_state: str | None
    ack_timeout_at: float | None
    ack_timed_out: bool
    retry_exhausted: bool
    data_on_wire_bytes: int
    ack_on_wire_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_index",
            _integer(self.attempt_index, "attempt_index"),
        )
        sent = float(self.data_sent_at)
        if not np.isfinite(sent):
            raise ValueError("data_sent_at must be finite")
        object.__setattr__(self, "data_sent_at", sent)
        for name in ("data_dropped", "ack_dropped", "ack_timed_out", "retry_exhausted"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        data_arrival = self._optional_time(
            self.data_arrival_time,
            "data_arrival_time",
            lower_bound=sent,
        )
        if self.data_dropped == (data_arrival is not None):
            raise ValueError(
                "exactly one data outcome is required: dropped or arrival"
            )
        ack_sent = self._optional_time(
            self.ack_sent_at,
            "ack_sent_at",
            lower_bound=data_arrival,
        )
        ack_arrival = self._optional_time(
            self.ack_arrival_time,
            "ack_arrival_time",
            lower_bound=ack_sent,
        )
        timeout = self._optional_time(
            self.ack_timeout_at,
            "ack_timeout_at",
            lower_bound=sent,
            strict=True,
        )
        if ack_sent is None:
            if ack_arrival is not None or self.ack_dropped:
                raise ValueError("an unsent ACK cannot arrive or be dropped")
        elif self.ack_dropped == (ack_arrival is not None):
            raise ValueError(
                "exactly one ACK outcome is required: dropped or arrival"
            )
        if self.ack_timed_out and timeout is None:
            raise ValueError("an ACK timeout requires ack_timeout_at")
        if (
            ack_arrival is not None
            and timeout is not None
            and ack_arrival <= timeout
            and self.ack_timed_out
        ):
            raise ValueError("an on-time ACK cannot also time out")
        if self.retry_exhausted and not self.ack_timed_out:
            raise ValueError("retry exhaustion requires an ACK timeout")
        data_bytes = _integer(self.data_on_wire_bytes, "data_on_wire_bytes", minimum=1)
        ack_bytes = _integer(self.ack_on_wire_bytes, "ack_on_wire_bytes")
        if ack_sent is None and ack_bytes:
            raise ValueError("an unsent ACK must account for zero bytes")
        object.__setattr__(self, "data_arrival_time", data_arrival)
        object.__setattr__(self, "ack_sent_at", ack_sent)
        object.__setattr__(self, "ack_arrival_time", ack_arrival)
        object.__setattr__(self, "ack_timeout_at", timeout)
        object.__setattr__(self, "data_on_wire_bytes", data_bytes)
        object.__setattr__(self, "ack_on_wire_bytes", ack_bytes)
        for name in ("data_channel_state", "ack_channel_state"):
            if getattr(self, name) not in (None, "good", "bad"):
                raise ValueError(f"{name} must be None, good, or bad")

    @staticmethod
    def _optional_time(
        value: float | None,
        name: str,
        *,
        lower_bound: float | None,
        strict: bool = False,
    ) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if lower_bound is None:
            raise ValueError(f"{name} requires its preceding protocol event")
        invalid = value <= lower_bound if strict else value < lower_bound
        if invalid:
            relation = "after" if strict else "not precede"
            raise ValueError(f"{name} must {relation} its preceding protocol event")
        return value

    @property
    def is_retransmission(self) -> bool:
        return self.attempt_index > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ack_arrival_time": self.ack_arrival_time,
            "ack_channel_state": self.ack_channel_state,
            "ack_dropped": self.ack_dropped,
            "ack_on_wire_bytes": self.ack_on_wire_bytes,
            "ack_sent_at": self.ack_sent_at,
            "ack_timed_out": self.ack_timed_out,
            "ack_timeout_at": self.ack_timeout_at,
            "attempt_index": self.attempt_index,
            "data_arrival_time": self.data_arrival_time,
            "data_channel_state": self.data_channel_state,
            "data_dropped": self.data_dropped,
            "data_on_wire_bytes": self.data_on_wire_bytes,
            "data_sent_at": self.data_sent_at,
            "is_retransmission": self.is_retransmission,
            "ledger_source": SYNTHETIC_LEDGER_SOURCE_V1,
            "retry_exhausted": self.retry_exhausted,
        }


def account_wire_bytes_v1(
    application_bytes: int,
    config: WireAccountingConfigV1,
    *,
    data_transmissions: int = 1,
    ack_transmissions: int = 0,
) -> ByteAccountV1:
    """Account bytes without estimating or hiding protocol overhead."""

    application_bytes = _integer(application_bytes, "application_bytes", minimum=1)
    data_transmissions = _integer(data_transmissions, "data_transmissions")
    ack_transmissions = _integer(ack_transmissions, "ack_transmissions")
    logical_payload = application_bytes + config.per_message_header_bytes
    fragments = math.ceil(logical_payload / config.fragment_payload_bytes)
    one_data_transmission = (
        logical_payload + fragments * config.per_fragment_header_bytes
    )
    data_bytes = one_data_transmission * data_transmissions
    ack_bytes = config.ack_frame_bytes * ack_transmissions
    return ByteAccountV1(
        application_bytes=application_bytes,
        data_transmissions=data_transmissions,
        fragments_per_transmission=fragments if data_transmissions else 0,
        data_on_wire_bytes=data_bytes,
        ack_transmissions=ack_transmissions,
        ack_on_wire_bytes=ack_bytes,
        total_on_wire_bytes=data_bytes + ack_bytes,
    )


def bounded_wire_reservation_v1(
    application_bytes: int,
    config: WireAccountingConfigV1,
) -> ByteAccountV1:
    """Return the causal worst-case reservation for a bounded retry policy.

    Before transmission, neither data delivery nor ACK delivery is known.  A
    capped scheduler therefore reserves every permitted data attempt and, when
    enabled, one possible ACK frame for every attempt.
    """

    if not isinstance(config, WireAccountingConfigV1):
        raise TypeError("config must be WireAccountingConfigV1")
    data_transmissions = config.max_retransmissions + 1
    ack_transmissions = data_transmissions if config.ack_on_success else 0
    return account_wire_bytes_v1(
        application_bytes,
        config,
        data_transmissions=data_transmissions,
        ack_transmissions=ack_transmissions,
    )


@dataclass(frozen=True, slots=True)
class NetworkConditionPlanV1:
    """Immutable, canonicalisable definition of one synthetic condition."""

    condition_id: NetworkConditionId
    seeds: tuple[int, ...] = DEFAULT_NETWORK_SEEDS_V1
    fixed_delay_seconds: float = 0.0
    jitter_seconds: float = 0.0
    jitter_model: JitterModel = JitterModel.NONE
    jitter_clip_sigma: float = 3.0
    loss_model: LossModel = LossModel.NONE
    iid_loss_probability: float = 0.0
    gilbert_elliott: GilbertElliott | None = None
    outage_period_seconds: float = 0.0
    outage_duration_seconds: float = 0.0
    clock_offset_limit_seconds: float = 0.0
    clock_drift_limit_ppm: float = 0.0
    translation_noise_std_metres: float = 0.0
    yaw_noise_std_degrees: float = 0.0
    wire_accounting: WireAccountingConfigV1 = WireAccountingConfigV1()
    schema_version: str = NETWORK_CONDITION_PLAN_SCHEMA_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", NetworkConditionId(self.condition_id))
        seeds = tuple(_integer(seed, "seed") for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        object.__setattr__(self, "seeds", seeds)
        for name in (
            "fixed_delay_seconds",
            "jitter_seconds",
            "outage_period_seconds",
            "outage_duration_seconds",
            "clock_offset_limit_seconds",
            "clock_drift_limit_ppm",
            "translation_noise_std_metres",
            "yaw_noise_std_degrees",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_finite(getattr(self, name), name),
            )
        clip = float(self.jitter_clip_sigma)
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError("jitter_clip_sigma must be finite and positive")
        object.__setattr__(self, "jitter_clip_sigma", clip)
        jitter_model = JitterModel(self.jitter_model)
        if self.jitter_seconds == 0.0:
            jitter_model = JitterModel.NONE
        elif jitter_model is JitterModel.NONE:
            raise ValueError("non-zero jitter requires a jitter model")
        object.__setattr__(self, "jitter_model", jitter_model)
        loss_model = LossModel(self.loss_model)
        object.__setattr__(self, "loss_model", loss_model)
        probability = float(self.iid_loss_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("iid_loss_probability must be in [0, 1]")
        object.__setattr__(self, "iid_loss_probability", probability)
        if loss_model is LossModel.GILBERT_ELLIOTT and self.gilbert_elliott is None:
            raise ValueError("Gilbert-Elliott loss requires parameters")
        if loss_model is not LossModel.GILBERT_ELLIOTT and self.gilbert_elliott is not None:
            raise ValueError("Gilbert-Elliott parameters require the matching loss model")
        if bool(self.outage_period_seconds) != bool(self.outage_duration_seconds):
            raise ValueError("periodic outages require both period and duration")
        if self.outage_duration_seconds > self.outage_period_seconds:
            raise ValueError("outage duration must not exceed its period")
        if self.schema_version != NETWORK_CONDITION_PLAN_SCHEMA_V1:
            raise ValueError("unsupported network condition plan schema")
        if not isinstance(self.wire_accounting, WireAccountingConfigV1):
            raise TypeError("wire_accounting must be WireAccountingConfigV1")

    def to_dict(self) -> dict[str, Any]:
        ge = self.gilbert_elliott
        return {
            "clock_drift_limit_ppm": self.clock_drift_limit_ppm,
            "clock_offset_limit_seconds": self.clock_offset_limit_seconds,
            "condition_id": self.condition_id.value,
            "fixed_delay_seconds": self.fixed_delay_seconds,
            "gilbert_elliott": None
            if ge is None
            else {
                "initial_good_probability": ge.initial_good_probability,
                "probability_bad_to_good": ge.probability_bad_to_good,
                "probability_good_to_good": ge.probability_good_to_good,
                "success_probability_bad": ge.success_probability_bad,
                "success_probability_good": ge.success_probability_good,
            },
            "iid_loss_probability": self.iid_loss_probability,
            "jitter_clip_sigma": self.jitter_clip_sigma,
            "jitter_model": self.jitter_model.value,
            "jitter_seconds": self.jitter_seconds,
            "loss_model": self.loss_model.value,
            "outage_duration_seconds": self.outage_duration_seconds,
            "outage_period_seconds": self.outage_period_seconds,
            "schema_version": self.schema_version,
            "seeds": list(self.seeds),
            "translation_noise_std_metres": self.translation_noise_std_metres,
            "wire_accounting": self.wire_accounting.to_dict(),
            "yaw_noise_std_degrees": self.yaw_noise_std_degrees,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ge_loss_with_bad_fraction(
    *, bad_fraction: float, mean_bad_run_packets: float
) -> GilbertElliott:
    bad_to_good = 1.0 / mean_bad_run_packets
    good_to_bad = bad_fraction * bad_to_good / (1.0 - bad_fraction)
    return GilbertElliott(
        probability_good_to_good=1.0 - good_to_bad,
        probability_bad_to_good=bad_to_good,
        success_probability_good=1.0,
        success_probability_bad=0.0,
        initial_good_probability=1.0 - bad_fraction,
    )


def default_condition_plans_v1(
    *, wire_accounting: WireAccountingConfigV1 | None = None
) -> tuple[NetworkConditionPlanV1, ...]:
    """Return the frozen C0--C8 paper condition matrix."""

    wire = WireAccountingConfigV1() if wire_accounting is None else wire_accounting
    ge = _ge_loss_with_bad_fraction(bad_fraction=0.30, mean_bad_run_packets=5.0)
    common = {"wire_accounting": wire}
    return (
        NetworkConditionPlanV1(NetworkConditionId.C0, **common),
        NetworkConditionPlanV1(
            NetworkConditionId.C1,
            fixed_delay_seconds=0.100,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C2,
            fixed_delay_seconds=0.300,
            jitter_seconds=0.050,
            jitter_model=JitterModel.TRUNCATED_NORMAL,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C3,
            loss_model=LossModel.IID,
            iid_loss_probability=0.30,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C4,
            loss_model=LossModel.GILBERT_ELLIOTT,
            gilbert_elliott=ge,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C5,
            outage_period_seconds=10.0,
            outage_duration_seconds=1.0,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C6,
            clock_offset_limit_seconds=0.100,
            clock_drift_limit_ppm=50.0,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C7,
            translation_noise_std_metres=0.5,
            yaw_noise_std_degrees=1.0,
            **common,
        ),
        NetworkConditionPlanV1(
            NetworkConditionId.C8,
            fixed_delay_seconds=0.200,
            jitter_seconds=0.050,
            jitter_model=JitterModel.TRUNCATED_NORMAL,
            loss_model=LossModel.GILBERT_ELLIOTT,
            gilbert_elliott=ge,
            outage_period_seconds=10.0,
            outage_duration_seconds=1.0,
            clock_offset_limit_seconds=0.100,
            clock_drift_limit_ppm=50.0,
            translation_noise_std_metres=0.5,
            yaw_noise_std_degrees=1.0,
            **common,
        ),
    )


def condition_plan_v1(
    condition_id: NetworkConditionId | str,
    *,
    wire_accounting: WireAccountingConfigV1 | None = None,
) -> NetworkConditionPlanV1:
    requested = NetworkConditionId(condition_id)
    return next(
        plan
        for plan in default_condition_plans_v1(wire_accounting=wire_accounting)
        if plan.condition_id is requested
    )


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((seed, stream)))


def _periodic_outages(
    packets: tuple[PacketRequest, ...],
    plan: NetworkConditionPlanV1,
    *,
    seed: int,
    upper_padding_seconds: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    if not packets or plan.outage_period_seconds == 0.0:
        return ()
    lower = min(packet.transmitted_at for packet in packets)
    upper = max(packet.transmitted_at for packet in packets) + _nonnegative_finite(
        upper_padding_seconds,
        "upper_padding_seconds",
    )
    period = plan.outage_period_seconds
    phase = float(_rng(seed, 11).uniform(0.0, period))
    start = math.floor((lower - phase) / period) * period + phase
    while start + plan.outage_duration_seconds <= lower:
        start += period
    intervals: list[tuple[float, float]] = []
    while start <= upper:
        # NetworkConfig historically uses inclusive interval ends.  nextafter
        # represents the preregistered half-open [start, start + duration).
        end = float(
            np.nextafter(start + plan.outage_duration_seconds, -np.inf)
        )
        intervals.append((float(start), end))
        start += period
    return tuple(intervals)


@dataclass(frozen=True, slots=True)
class _SyntheticChannelOutcomeV1:
    dropped: bool
    arrival_time: float | None
    channel_state: str | None


class _SyntheticChannelRuntimeV1:
    """Deterministic bidirectional simulator state for retransmission traces."""

    def __init__(self, config: NetworkConfig, *, seed: int) -> None:
        self._config = config
        self._rng = _rng(seed, 47)
        self._ge = config.gilbert_elliott
        self._good: bool | None = None
        if config.loss_model is LossModel.GILBERT_ELLIOTT:
            assert self._ge is not None
            self._good = bool(
                self._rng.random() < self._ge.initial_good_probability
            )

    def transmit(self, sent_at: float) -> _SyntheticChannelOutcomeV1:
        state_name: str | None = None
        dropped = _in_outage(sent_at, self._config.outage_intervals)
        if self._config.loss_model is LossModel.IID:
            dropped = dropped or bool(
                self._rng.random() < self._config.iid_loss_probability
            )
        elif self._config.loss_model is LossModel.GILBERT_ELLIOTT:
            ge = self._ge
            good = self._good
            assert ge is not None and good is not None
            state_name = "good" if good else "bad"
            success_probability = (
                ge.success_probability_good if good else ge.success_probability_bad
            )
            dropped = dropped or bool(self._rng.random() >= success_probability)
            if good:
                self._good = bool(
                    self._rng.random() < ge.probability_good_to_good
                )
            else:
                self._good = bool(
                    self._rng.random() < ge.probability_bad_to_good
                )
        if dropped:
            return _SyntheticChannelOutcomeV1(True, None, state_name)

        if self._config.jitter_model is JitterModel.UNIFORM:
            jitter = float(
                self._rng.uniform(-self._config.jitter, self._config.jitter)
            )
        elif self._config.jitter_model is JitterModel.TRUNCATED_NORMAL:
            jitter = _truncated_normal(
                self._rng,
                standard_deviation=self._config.jitter,
                clip_sigma=self._config.jitter_clip_sigma,
            )
        else:
            jitter = 0.0
        delay = max(0.0, self._config.fixed_delay + jitter)
        if (
            self._config.reorder_probability
            and self._rng.random() < self._config.reorder_probability
        ):
            delay += self._config.reorder_extra_delay
        return _SyntheticChannelOutcomeV1(
            False,
            sent_at + delay,
            state_name,
        )


@dataclass(slots=True)
class _AttemptRuntimeV1:
    attempt_index: int
    data_sent_at: float
    ack_timeout_at: float
    data_dropped: bool
    data_arrival_time: float | None
    data_channel_state: str | None
    ack_sent_at: float | None = None
    ack_dropped: bool = False
    ack_arrival_time: float | None = None
    ack_channel_state: str | None = None
    ack_timed_out: bool = False
    retry_exhausted: bool = False


@dataclass(slots=True)
class _MessageProtocolRuntimeV1:
    packet: PacketRequest
    attempts: list[_AttemptRuntimeV1] = field(default_factory=list)
    acknowledged: bool = False


@dataclass(frozen=True, slots=True)
class _SyntheticProtocolResultV1:
    event: NetworkEvent
    attempts: tuple[SyntheticTransmissionAttemptV1, ...]
    byte_account: ByteAccountV1


_PROTOCOL_QUEUE_PRIORITY_V1 = {
    "ack_arrival": 0,
    "data_arrival": 1,
    "ack_send": 2,
    "data_send": 2,
    "ack_timeout": 3,
}


def _queue_protocol_event_v1(
    queue: list[tuple[float, int, str, int, str, int, str]],
    *,
    time: float,
    kind: str,
    packet: PacketRequest,
    attempt_index: int,
) -> None:
    heapq.heappush(
        queue,
        (
            float(time),
            _PROTOCOL_QUEUE_PRIORITY_V1[kind],
            packet.source,
            packet.sequence,
            packet.message_id,
            attempt_index,
            kind,
        ),
    )


def _simulate_retransmissions_v1(
    packets: tuple[PacketRequest, ...],
    config: NetworkConfig,
    wire: WireAccountingConfigV1,
    *,
    seed: int,
) -> dict[str, _SyntheticProtocolResultV1]:
    """Run a finite synthetic ACK/retransmission state machine.

    The returned attempt ledger is simulator output.  It must not be described
    or used as measured packet-capture evidence.
    """

    if wire.max_retransmissions <= 0:  # pragma: no cover - caller guards this.
        raise ValueError("the retransmission simulator requires a positive bound")
    runtimes = {
        packet.message_id: _MessageProtocolRuntimeV1(packet)
        for packet in packets
    }
    if len(runtimes) != len(packets):
        raise ValueError("packet message_id values must be unique")
    channel = _SyntheticChannelRuntimeV1(config, seed=seed)
    queue: list[tuple[float, int, str, int, str, int, str]] = []
    for packet in packets:
        _queue_protocol_event_v1(
            queue,
            time=packet.transmitted_at,
            kind="data_send",
            packet=packet,
            attempt_index=0,
        )

    while queue:
        time, _, _, _, message_id, attempt_index, kind = heapq.heappop(queue)
        runtime = runtimes[message_id]
        packet = runtime.packet
        if kind == "data_send":
            if runtime.acknowledged:
                continue
            if attempt_index != len(runtime.attempts):
                raise RuntimeError("non-contiguous synthetic retransmission attempt")
            if attempt_index > wire.max_retransmissions:
                raise RuntimeError("synthetic retransmission bound exceeded")
            outcome = channel.transmit(time)
            attempt = _AttemptRuntimeV1(
                attempt_index=attempt_index,
                data_sent_at=time,
                ack_timeout_at=time + wire.ack_timeout_seconds,
                data_dropped=outcome.dropped,
                data_arrival_time=outcome.arrival_time,
                data_channel_state=outcome.channel_state,
            )
            runtime.attempts.append(attempt)
            if outcome.arrival_time is not None:
                _queue_protocol_event_v1(
                    queue,
                    time=outcome.arrival_time,
                    kind="data_arrival",
                    packet=packet,
                    attempt_index=attempt_index,
                )
            _queue_protocol_event_v1(
                queue,
                time=attempt.ack_timeout_at,
                kind="ack_timeout",
                packet=packet,
                attempt_index=attempt_index,
            )
            continue

        attempt = runtime.attempts[attempt_index]
        if kind == "data_arrival":
            _queue_protocol_event_v1(
                queue,
                time=time,
                kind="ack_send",
                packet=packet,
                attempt_index=attempt_index,
            )
        elif kind == "ack_send":
            attempt.ack_sent_at = time
            outcome = channel.transmit(time)
            attempt.ack_dropped = outcome.dropped
            attempt.ack_arrival_time = outcome.arrival_time
            attempt.ack_channel_state = outcome.channel_state
            if outcome.arrival_time is not None:
                _queue_protocol_event_v1(
                    queue,
                    time=outcome.arrival_time,
                    kind="ack_arrival",
                    packet=packet,
                    attempt_index=attempt_index,
                )
        elif kind == "ack_arrival":
            runtime.acknowledged = True
        elif kind == "ack_timeout":
            if runtime.acknowledged:
                continue
            attempt.ack_timed_out = True
            if attempt_index < wire.max_retransmissions:
                _queue_protocol_event_v1(
                    queue,
                    time=time,
                    kind="data_send",
                    packet=packet,
                    attempt_index=attempt_index + 1,
                )
            else:
                attempt.retry_exhausted = True
        else:  # pragma: no cover - all queue insertions are internal.
            raise RuntimeError(f"unsupported synthetic protocol event: {kind}")

    results: dict[str, _SyntheticProtocolResultV1] = {}
    one_data_bytes: dict[int, int] = {}
    for runtime in runtimes.values():
        application_bytes = runtime.packet.encoded_bytes
        if application_bytes not in one_data_bytes:
            one_data_bytes[application_bytes] = account_wire_bytes_v1(
                application_bytes,
                wire,
                data_transmissions=1,
                ack_transmissions=0,
            ).data_on_wire_bytes
        attempts = tuple(
            SyntheticTransmissionAttemptV1(
                attempt_index=attempt.attempt_index,
                data_sent_at=attempt.data_sent_at,
                data_dropped=attempt.data_dropped,
                data_arrival_time=attempt.data_arrival_time,
                data_channel_state=attempt.data_channel_state,
                ack_sent_at=attempt.ack_sent_at,
                ack_dropped=attempt.ack_dropped,
                ack_arrival_time=attempt.ack_arrival_time,
                ack_channel_state=attempt.ack_channel_state,
                ack_timeout_at=attempt.ack_timeout_at,
                ack_timed_out=attempt.ack_timed_out,
                retry_exhausted=attempt.retry_exhausted,
                data_on_wire_bytes=one_data_bytes[application_bytes],
                ack_on_wire_bytes=(
                    wire.ack_frame_bytes if attempt.ack_sent_at is not None else 0
                ),
            )
            for attempt in runtime.attempts
        )
        arrivals = tuple(
            attempt.data_arrival_time
            for attempt in attempts
            if attempt.data_arrival_time is not None
        )
        byte_account = account_wire_bytes_v1(
            application_bytes,
            wire,
            data_transmissions=len(attempts),
            ack_transmissions=sum(
                attempt.ack_sent_at is not None for attempt in attempts
            ),
        )
        results[runtime.packet.message_id] = _SyntheticProtocolResultV1(
            event=NetworkEvent(
                packet=runtime.packet,
                arrival_time=min(arrivals) if arrivals else None,
                dropped=not arrivals,
                channel_state=(
                    attempts[0].data_channel_state if attempts else None
                ),
            ),
            attempts=attempts,
            byte_account=byte_account,
        )
    return results


@dataclass(frozen=True, slots=True)
class NetworkTraceEventV1:
    packet: PacketRequest
    arrival_time: float | None
    dropped: bool
    channel_state: str | None
    clock_offset_seconds: float
    clock_drift_ppm: float
    clock_error_at_transmit_seconds: float
    translation_noise_metres: tuple[float, float, float]
    yaw_noise_degrees: float
    byte_account: ByteAccountV1
    wire_accounting_config: WireAccountingConfigV1 = WireAccountingConfigV1()
    attempt_timeline: tuple[SyntheticTransmissionAttemptV1, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.packet, PacketRequest):
            raise TypeError("packet must be PacketRequest")
        if not isinstance(self.dropped, bool):
            raise TypeError("dropped must be bool")
        if self.arrival_time is None:
            if not self.dropped:
                raise ValueError("a delivered event requires arrival_time")
        else:
            arrival = float(self.arrival_time)
            if not np.isfinite(arrival) or arrival < self.packet.transmitted_at:
                raise ValueError("arrival_time must be finite and not precede transmission")
            if self.dropped:
                raise ValueError("a dropped event cannot have arrival_time")
            object.__setattr__(self, "arrival_time", arrival)
        if self.channel_state not in (None, "good", "bad"):
            raise ValueError("channel_state must be None, good, or bad")
        for name in (
            "clock_offset_seconds",
            "clock_drift_ppm",
            "clock_error_at_transmit_seconds",
            "yaw_noise_degrees",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        translation = tuple(float(value) for value in self.translation_noise_metres)
        if len(translation) != 3 or not all(np.isfinite(value) for value in translation):
            raise ValueError("translation_noise_metres must contain three finite values")
        object.__setattr__(self, "translation_noise_metres", translation)
        if not isinstance(self.byte_account, ByteAccountV1):
            raise TypeError("byte_account must be ByteAccountV1")
        if not isinstance(self.wire_accounting_config, WireAccountingConfigV1):
            raise TypeError(
                "wire_accounting_config must be WireAccountingConfigV1"
            )
        wire = self.wire_accounting_config
        if self.byte_account.application_bytes != self.packet.encoded_bytes:
            raise ValueError("byte account must match packet encoded_bytes")
        if self.dropped and self.byte_account.ack_transmissions:
            raise ValueError("a dropped data message cannot be acknowledged")
        timeline = tuple(self.attempt_timeline)
        if not all(
            isinstance(attempt, SyntheticTransmissionAttemptV1)
            for attempt in timeline
        ):
            raise TypeError(
                "attempt_timeline must contain SyntheticTransmissionAttemptV1 values"
            )
        if timeline:
            if len(timeline) > wire.max_retransmissions + 1:
                raise ValueError("attempt timeline exceeds the retransmission bound")
            if tuple(attempt.attempt_index for attempt in timeline) != tuple(
                range(len(timeline))
            ):
                raise ValueError("attempt_timeline indices must be contiguous from zero")
            if timeline[0].data_sent_at != self.packet.transmitted_at:
                raise ValueError(
                    "initial attempt must use packet.transmitted_at"
                )
            for previous, current in zip(timeline, timeline[1:]):
                if (
                    not previous.ack_timed_out
                    or previous.ack_timeout_at is None
                    or current.data_sent_at != previous.ack_timeout_at
                ):
                    raise ValueError(
                        "each retransmission must follow the preceding ACK timeout"
                    )
                if previous.retry_exhausted:
                    raise ValueError("retry exhaustion must end the attempt timeline")
            one_data_account = account_wire_bytes_v1(
                self.packet.encoded_bytes,
                wire,
                data_transmissions=1,
                ack_transmissions=0,
            )
            for index, attempt in enumerate(timeline):
                expected_timeout = attempt.data_sent_at + wire.ack_timeout_seconds
                if not np.isclose(
                    attempt.ack_timeout_at,
                    expected_timeout,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "attempt ACK timeout does not match wire accounting config"
                    )
                if attempt.data_on_wire_bytes != one_data_account.data_on_wire_bytes:
                    raise ValueError(
                        "attempt data bytes omit or alter configured wire overhead"
                    )
                expected_ack_sent = (
                    wire.ack_on_success and attempt.data_arrival_time is not None
                )
                if (attempt.ack_sent_at is not None) != expected_ack_sent:
                    raise ValueError(
                        "attempt ACK emission disagrees with data delivery and config"
                    )
                if expected_ack_sent and attempt.ack_sent_at != attempt.data_arrival_time:
                    raise ValueError("ACK must be sent when the data attempt arrives")
                expected_ack_bytes = wire.ack_frame_bytes if expected_ack_sent else 0
                if attempt.ack_on_wire_bytes != expected_ack_bytes:
                    raise ValueError(
                        "attempt ACK bytes omit or alter configured wire overhead"
                    )
                expected_timeout_flag = (
                    attempt.ack_arrival_time is None
                    or (
                        attempt.ack_arrival_time > expected_timeout
                        and not np.isclose(
                            attempt.ack_arrival_time,
                            expected_timeout,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                )
                if attempt.ack_timed_out != expected_timeout_flag:
                    raise ValueError(
                        "attempt ACK timeout flag disagrees with the ACK timeline"
                    )
                expected_exhausted = (
                    expected_timeout_flag and index == wire.max_retransmissions
                )
                if attempt.retry_exhausted != expected_exhausted:
                    raise ValueError(
                        "retry_exhausted disagrees with the configured retry bound"
                    )
                if (
                    index == len(timeline) - 1
                    and expected_timeout_flag
                    and not expected_exhausted
                ):
                    raise ValueError(
                        "a timed-out attempt below the retry bound cannot end the timeline"
                    )
            data_arrivals = tuple(
                attempt.data_arrival_time
                for attempt in timeline
                if attempt.data_arrival_time is not None
            )
            expected_arrival = min(data_arrivals) if data_arrivals else None
            if self.arrival_time != expected_arrival:
                raise ValueError(
                    "logical arrival_time must equal the first attempt arrival"
                )
            if self.dropped != (not data_arrivals):
                raise ValueError("logical dropped flag disagrees with attempt timeline")
            if self.channel_state != timeline[0].data_channel_state:
                raise ValueError(
                    "logical channel_state must equal the initial attempt state"
                )
            if self.byte_account.data_transmissions != len(timeline):
                raise ValueError(
                    "byte account data transmissions disagree with attempt timeline"
                )
            ack_transmissions = sum(
                attempt.ack_sent_at is not None for attempt in timeline
            )
            if self.byte_account.ack_transmissions != ack_transmissions:
                raise ValueError(
                    "byte account ACK transmissions disagree with attempt timeline"
                )
            if self.byte_account.data_on_wire_bytes != sum(
                attempt.data_on_wire_bytes for attempt in timeline
            ):
                raise ValueError(
                    "byte account data bytes disagree with attempt timeline"
                )
            if self.byte_account.ack_on_wire_bytes != sum(
                attempt.ack_on_wire_bytes for attempt in timeline
            ):
                raise ValueError(
                    "byte account ACK bytes disagree with attempt timeline"
                )
        else:
            if wire.max_retransmissions:
                raise ValueError(
                    "enabled retransmissions require an auditable attempt timeline"
                )
            expected_ack_transmissions = int(
                wire.ack_on_success and not self.dropped
            )
            if (
                self.byte_account.data_transmissions != 1
                or self.byte_account.ack_transmissions
                != expected_ack_transmissions
            ):
                raise ValueError(
                    "legacy event transmissions disagree with delivery and config"
                )
        expected_account = account_wire_bytes_v1(
            self.packet.encoded_bytes,
            wire,
            data_transmissions=self.byte_account.data_transmissions,
            ack_transmissions=self.byte_account.ack_transmissions,
        )
        if self.byte_account != expected_account:
            raise ValueError(
                "byte account does not exactly match configured wire overhead"
            )
        object.__setattr__(self, "attempt_timeline", timeline)

    @property
    def on_time(self) -> bool:
        return self.arrival_time is not None and self.arrival_time <= self.packet.deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "arrival_time": self.arrival_time,
            "attempt_timeline": [
                attempt.to_dict() for attempt in self.attempt_timeline
            ],
            "byte_account": self.byte_account.to_dict(),
            "channel_state": self.channel_state,
            "clock_drift_ppm": self.clock_drift_ppm,
            "clock_error_at_transmit_seconds": self.clock_error_at_transmit_seconds,
            "clock_offset_seconds": self.clock_offset_seconds,
            "deadline": self.packet.deadline,
            "dropped": self.dropped,
            "encoded_bytes": self.packet.encoded_bytes,
            "message_id": self.packet.message_id,
            "sequence": self.packet.sequence,
            "source": self.packet.source,
            "transmitted_at": self.packet.transmitted_at,
            "translation_noise_metres": list(self.translation_noise_metres),
            "wire_accounting_config": self.wire_accounting_config.to_dict(),
            "yaw_noise_degrees": self.yaw_noise_degrees,
        }


@dataclass(frozen=True, slots=True)
class NetworkTraceV1:
    condition_id: NetworkConditionId
    seed: int
    condition_plan: NetworkConditionPlanV1
    condition_plan_sha256: str
    events: tuple[NetworkTraceEventV1, ...]
    schema_version: str = NETWORK_TRACE_SCHEMA_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", NetworkConditionId(self.condition_id))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if not isinstance(self.condition_plan, NetworkConditionPlanV1):
            raise TypeError("condition_plan must be NetworkConditionPlanV1")
        if self.condition_plan.condition_id is not self.condition_id:
            raise ValueError("condition_plan does not match condition_id")
        if self.seed not in self.condition_plan.seeds:
            raise ValueError("trace seed is not registered by condition_plan")
        if len(self.condition_plan_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.condition_plan_sha256
        ):
            raise ValueError("condition_plan_sha256 must be a lowercase SHA-256")
        if self.condition_plan_sha256 != self.condition_plan.content_sha256:
            raise ValueError("condition_plan_sha256 does not match condition_plan content")
        if self.schema_version != NETWORK_TRACE_SCHEMA_V1:
            raise ValueError("unsupported network trace schema")
        raw_events = tuple(self.events)
        if not all(isinstance(event, NetworkTraceEventV1) for event in raw_events):
            raise TypeError("events must contain NetworkTraceEventV1 values")
        events = tuple(
            sorted(
                raw_events,
                key=lambda event: (
                    event.packet.transmitted_at,
                    event.packet.source,
                    event.packet.sequence,
                    event.packet.message_id,
                ),
            )
        )
        object.__setattr__(self, "events", events)
        if any(
            event.wire_accounting_config != self.condition_plan.wire_accounting
            for event in events
        ):
            raise ValueError(
                "trace event wire accounting config does not match condition_plan"
            )
        ids = [event.packet.message_id for event in events]
        if len(set(ids)) != len(ids):
            raise ValueError("trace event message IDs must be unique")

    @property
    def arrival_order(self) -> tuple[NetworkTraceEventV1, ...]:
        return tuple(
            sorted(
                (event for event in self.events if event.arrival_time is not None),
                key=lambda event: (
                    event.arrival_time,
                    event.packet.source,
                    event.packet.sequence,
                    event.packet.message_id,
                ),
            )
        )

    @property
    def application_bytes(self) -> int:
        return sum(event.byte_account.application_bytes for event in self.events)

    @property
    def on_wire_bytes(self) -> int:
        return sum(event.byte_account.total_on_wire_bytes for event in self.events)

    @property
    def delivered_application_bytes(self) -> int:
        return sum(
            event.byte_account.application_bytes
            for event in self.events
            if not event.dropped
        )

    @property
    def on_time_application_bytes(self) -> int:
        return sum(
            event.byte_account.application_bytes for event in self.events if event.on_time
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id.value,
            "condition_plan": self.condition_plan.to_dict(),
            "condition_plan_sha256": self.condition_plan_sha256,
            "events": [event.to_dict() for event in self.events],
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _strict_trace_fields_v1(
    value: object,
    expected: frozenset[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise NetworkTraceError(f"{name} must be a string-keyed JSON object")
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise NetworkTraceError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _strict_trace_array_v1(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise NetworkTraceError(f"{name} must be a JSON array")
    return value


def _wire_accounting_from_mapping_v1(value: object) -> WireAccountingConfigV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "ack_frame_bytes",
                "ack_on_success",
                "ack_timeout_seconds",
                "max_frame_bytes",
                "max_retransmissions",
                "per_fragment_header_bytes",
                "per_message_header_bytes",
            }
        ),
        "WireAccountingConfigV1",
    )
    return WireAccountingConfigV1(
        max_frame_bytes=item["max_frame_bytes"],
        per_fragment_header_bytes=item["per_fragment_header_bytes"],
        per_message_header_bytes=item["per_message_header_bytes"],
        ack_frame_bytes=item["ack_frame_bytes"],
        ack_on_success=item["ack_on_success"],
        ack_timeout_seconds=item["ack_timeout_seconds"],
        max_retransmissions=item["max_retransmissions"],
    )


def _gilbert_elliott_from_mapping_v1(value: object) -> GilbertElliott:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "initial_good_probability",
                "probability_bad_to_good",
                "probability_good_to_good",
                "success_probability_bad",
                "success_probability_good",
            }
        ),
        "GilbertElliott",
    )
    return GilbertElliott(
        probability_good_to_good=item["probability_good_to_good"],
        probability_bad_to_good=item["probability_bad_to_good"],
        success_probability_good=item["success_probability_good"],
        success_probability_bad=item["success_probability_bad"],
        initial_good_probability=item["initial_good_probability"],
    )


def _condition_plan_from_mapping_v1(value: object) -> NetworkConditionPlanV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "clock_drift_limit_ppm",
                "clock_offset_limit_seconds",
                "condition_id",
                "fixed_delay_seconds",
                "gilbert_elliott",
                "iid_loss_probability",
                "jitter_clip_sigma",
                "jitter_model",
                "jitter_seconds",
                "loss_model",
                "outage_duration_seconds",
                "outage_period_seconds",
                "schema_version",
                "seeds",
                "translation_noise_std_metres",
                "wire_accounting",
                "yaw_noise_std_degrees",
            }
        ),
        "NetworkConditionPlanV1",
    )
    seeds = _strict_trace_array_v1(item["seeds"], "condition_plan.seeds")
    raw_ge = item["gilbert_elliott"]
    ge = None if raw_ge is None else _gilbert_elliott_from_mapping_v1(raw_ge)
    return NetworkConditionPlanV1(
        condition_id=item["condition_id"],
        seeds=tuple(seeds),
        fixed_delay_seconds=item["fixed_delay_seconds"],
        jitter_seconds=item["jitter_seconds"],
        jitter_model=item["jitter_model"],
        jitter_clip_sigma=item["jitter_clip_sigma"],
        loss_model=item["loss_model"],
        iid_loss_probability=item["iid_loss_probability"],
        gilbert_elliott=ge,
        outage_period_seconds=item["outage_period_seconds"],
        outage_duration_seconds=item["outage_duration_seconds"],
        clock_offset_limit_seconds=item["clock_offset_limit_seconds"],
        clock_drift_limit_ppm=item["clock_drift_limit_ppm"],
        translation_noise_std_metres=item["translation_noise_std_metres"],
        yaw_noise_std_degrees=item["yaw_noise_std_degrees"],
        wire_accounting=_wire_accounting_from_mapping_v1(
            item["wire_accounting"]
        ),
        schema_version=item["schema_version"],
    )


def _byte_account_from_mapping_v1(value: object) -> ByteAccountV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "ack_on_wire_bytes",
                "ack_transmissions",
                "application_bytes",
                "data_on_wire_bytes",
                "data_transmissions",
                "fragments_per_transmission",
                "retransmissions",
                "total_on_wire_bytes",
            }
        ),
        "ByteAccountV1",
    )
    account = ByteAccountV1(
        application_bytes=item["application_bytes"],
        data_transmissions=item["data_transmissions"],
        fragments_per_transmission=item["fragments_per_transmission"],
        data_on_wire_bytes=item["data_on_wire_bytes"],
        ack_transmissions=item["ack_transmissions"],
        ack_on_wire_bytes=item["ack_on_wire_bytes"],
        total_on_wire_bytes=item["total_on_wire_bytes"],
    )
    if type(item["retransmissions"]) is not int or (
        item["retransmissions"] != account.retransmissions
    ):
        raise NetworkTraceError(
            "byte_account.retransmissions does not match data_transmissions"
        )
    return account


def _attempt_from_mapping_v1(value: object) -> SyntheticTransmissionAttemptV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "ack_arrival_time",
                "ack_channel_state",
                "ack_dropped",
                "ack_on_wire_bytes",
                "ack_sent_at",
                "ack_timed_out",
                "ack_timeout_at",
                "attempt_index",
                "data_arrival_time",
                "data_channel_state",
                "data_dropped",
                "data_on_wire_bytes",
                "data_sent_at",
                "is_retransmission",
                "ledger_source",
                "retry_exhausted",
            }
        ),
        "SyntheticTransmissionAttemptV1",
    )
    attempt = SyntheticTransmissionAttemptV1(
        attempt_index=item["attempt_index"],
        data_sent_at=item["data_sent_at"],
        data_dropped=item["data_dropped"],
        data_arrival_time=item["data_arrival_time"],
        data_channel_state=item["data_channel_state"],
        ack_sent_at=item["ack_sent_at"],
        ack_dropped=item["ack_dropped"],
        ack_arrival_time=item["ack_arrival_time"],
        ack_channel_state=item["ack_channel_state"],
        ack_timeout_at=item["ack_timeout_at"],
        ack_timed_out=item["ack_timed_out"],
        retry_exhausted=item["retry_exhausted"],
        data_on_wire_bytes=item["data_on_wire_bytes"],
        ack_on_wire_bytes=item["ack_on_wire_bytes"],
    )
    if type(item["is_retransmission"]) is not bool or (
        item["is_retransmission"] is not attempt.is_retransmission
    ):
        raise NetworkTraceError(
            "attempt is_retransmission disagrees with attempt_index"
        )
    if item["ledger_source"] != SYNTHETIC_LEDGER_SOURCE_V1:
        raise NetworkTraceError("attempt ledger_source is not synthetic V1")
    return attempt


def _trace_event_from_mapping_v1(value: object) -> NetworkTraceEventV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "arrival_time",
                "attempt_timeline",
                "byte_account",
                "channel_state",
                "clock_drift_ppm",
                "clock_error_at_transmit_seconds",
                "clock_offset_seconds",
                "deadline",
                "dropped",
                "encoded_bytes",
                "message_id",
                "sequence",
                "source",
                "transmitted_at",
                "translation_noise_metres",
                "wire_accounting_config",
                "yaw_noise_degrees",
            }
        ),
        "NetworkTraceEventV1",
    )
    if type(item["message_id"]) is not str or type(item["source"]) is not str:
        raise NetworkTraceError("packet identity fields must be strings")
    translation = _strict_trace_array_v1(
        item["translation_noise_metres"],
        "event.translation_noise_metres",
    )
    raw_timeline = _strict_trace_array_v1(
        item["attempt_timeline"],
        "event.attempt_timeline",
    )
    packet = PacketRequest(
        message_id=item["message_id"],
        source=item["source"],
        sequence=item["sequence"],
        transmitted_at=item["transmitted_at"],
        deadline=item["deadline"],
        encoded_bytes=item["encoded_bytes"],
    )
    return NetworkTraceEventV1(
        packet=packet,
        arrival_time=item["arrival_time"],
        dropped=item["dropped"],
        channel_state=item["channel_state"],
        clock_offset_seconds=item["clock_offset_seconds"],
        clock_drift_ppm=item["clock_drift_ppm"],
        clock_error_at_transmit_seconds=item[
            "clock_error_at_transmit_seconds"
        ],
        translation_noise_metres=tuple(translation),
        yaw_noise_degrees=item["yaw_noise_degrees"],
        byte_account=_byte_account_from_mapping_v1(item["byte_account"]),
        wire_accounting_config=_wire_accounting_from_mapping_v1(
            item["wire_accounting_config"]
        ),
        attempt_timeline=tuple(
            _attempt_from_mapping_v1(attempt) for attempt in raw_timeline
        ),
    )


def _network_trace_from_mapping_v1(value: object) -> NetworkTraceV1:
    item = _strict_trace_fields_v1(
        value,
        frozenset(
            {
                "condition_id",
                "condition_plan",
                "condition_plan_sha256",
                "events",
                "schema_version",
                "seed",
            }
        ),
        "NetworkTraceV1",
    )
    if type(item["condition_plan_sha256"]) is not str:
        raise NetworkTraceError("condition_plan_sha256 must be a string")
    raw_events = _strict_trace_array_v1(item["events"], "trace.events")
    return NetworkTraceV1(
        condition_id=item["condition_id"],
        seed=item["seed"],
        condition_plan=_condition_plan_from_mapping_v1(item["condition_plan"]),
        condition_plan_sha256=item["condition_plan_sha256"],
        events=tuple(_trace_event_from_mapping_v1(event) for event in raw_events),
        schema_version=item["schema_version"],
    )


def decode_network_trace_v1(data: bytes) -> NetworkTraceV1:
    """Decode exact canonical NetworkTraceV1 JSON and replay its invariants."""

    if type(data) is not bytes:
        raise TypeError("network trace data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise NetworkTraceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                NetworkTraceError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NetworkTraceError("invalid NetworkTraceV1 JSON") from exc
    try:
        trace = _network_trace_from_mapping_v1(value)
    except NetworkTraceError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise NetworkTraceError(f"invalid NetworkTraceV1 content: {exc}") from exc
    if trace.canonical_bytes != data:
        raise NetworkTraceError("NetworkTraceV1 JSON is not canonical")
    return trace


def generate_condition_trace_v1(
    packets: list[PacketRequest] | tuple[PacketRequest, ...],
    condition: NetworkConditionPlanV1 | NetworkConditionId | str,
    *,
    seed: int,
) -> NetworkTraceV1:
    """Generate one order-invariant C0--C8 trace with sealed byte totals."""

    plan = (
        condition
        if isinstance(condition, NetworkConditionPlanV1)
        else condition_plan_v1(condition)
    )
    seed = _integer(seed, "seed")
    if seed not in plan.seeds:
        raise ValueError(f"seed {seed} is not registered by {plan.condition_id.value}")
    ordered_packets = tuple(
        sorted(
            packets,
            key=lambda packet: (
                packet.transmitted_at,
                packet.source,
                packet.sequence,
                packet.message_id,
            ),
        )
    )
    maximum_jitter = plan.jitter_seconds
    if plan.jitter_model is JitterModel.TRUNCATED_NORMAL:
        maximum_jitter *= plan.jitter_clip_sigma
    retransmission_padding = 0.0
    if plan.wire_accounting.max_retransmissions:
        retransmission_padding = (
            (plan.wire_accounting.max_retransmissions + 1)
            * plan.wire_accounting.ack_timeout_seconds
            + 2.0 * (plan.fixed_delay_seconds + maximum_jitter)
        )
    config = NetworkConfig(
        fixed_delay=plan.fixed_delay_seconds,
        jitter=plan.jitter_seconds,
        jitter_model=plan.jitter_model,
        jitter_clip_sigma=plan.jitter_clip_sigma,
        loss_model=plan.loss_model,
        iid_loss_probability=plan.iid_loss_probability,
        gilbert_elliott=plan.gilbert_elliott,
        outage_intervals=_periodic_outages(
            ordered_packets,
            plan,
            seed=seed,
            upper_padding_seconds=retransmission_padding,
        ),
    )
    protocol_results: dict[str, _SyntheticProtocolResultV1] = {}
    if plan.wire_accounting.max_retransmissions:
        protocol_results = _simulate_retransmissions_v1(
            ordered_packets,
            config,
            plan.wire_accounting,
            seed=seed,
        )
        base_events = tuple(
            protocol_results[packet.message_id].event for packet in ordered_packets
        )
    else:
        base_events = generate_network_trace(
            ordered_packets,
            config,
            seed=seed,
        ).events

    sources = sorted({packet.source for packet in ordered_packets})
    clock_rng = _rng(seed, 23)
    source_clock: dict[str, tuple[float, float, float]] = {}
    for source in sources:
        source_times = [
            packet.transmitted_at for packet in ordered_packets if packet.source == source
        ]
        offset = (
            float(
                clock_rng.uniform(
                    -plan.clock_offset_limit_seconds,
                    plan.clock_offset_limit_seconds,
                )
            )
            if plan.clock_offset_limit_seconds
            else 0.0
        )
        drift = (
            float(
                clock_rng.uniform(
                    -plan.clock_drift_limit_ppm,
                    plan.clock_drift_limit_ppm,
                )
            )
            if plan.clock_drift_limit_ppm
            else 0.0
        )
        source_clock[source] = (offset, drift, min(source_times))

    pose_rng = _rng(seed, 37)
    events: list[NetworkTraceEventV1] = []
    for event in base_events:
        offset, drift, clock_reference = source_clock[event.packet.source]
        clock_error = offset + drift * 1e-6 * (
            event.packet.transmitted_at - clock_reference
        )
        if plan.translation_noise_std_metres:
            translation = tuple(
                float(value)
                for value in pose_rng.normal(
                    0.0,
                    plan.translation_noise_std_metres,
                    size=3,
                )
            )
        else:
            translation = (0.0, 0.0, 0.0)
        yaw = (
            float(pose_rng.normal(0.0, plan.yaw_noise_std_degrees))
            if plan.yaw_noise_std_degrees
            else 0.0
        )
        protocol_result = protocol_results.get(event.packet.message_id)
        if protocol_result is None:
            ack_transmissions = int(
                plan.wire_accounting.ack_on_success and not event.dropped
            )
            account = account_wire_bytes_v1(
                event.packet.encoded_bytes,
                plan.wire_accounting,
                data_transmissions=1,
                ack_transmissions=ack_transmissions,
            )
            # Preserve the legacy V1 event shape when retransmission is disabled.
            # Enabling the bounded state machine emits the strict timeline below.
            attempt_timeline = ()
        else:
            account = protocol_result.byte_account
            attempt_timeline = protocol_result.attempts
        events.append(
            NetworkTraceEventV1(
                packet=event.packet,
                arrival_time=event.arrival_time,
                dropped=event.dropped,
                channel_state=event.channel_state,
                clock_offset_seconds=offset,
                clock_drift_ppm=drift,
                clock_error_at_transmit_seconds=clock_error,
                translation_noise_metres=translation,
                yaw_noise_degrees=yaw,
                byte_account=account,
                wire_accounting_config=plan.wire_accounting,
                attempt_timeline=attempt_timeline,
            )
        )
    return NetworkTraceV1(
        condition_id=plan.condition_id,
        seed=seed,
        condition_plan=plan,
        condition_plan_sha256=plan.content_sha256,
        events=tuple(events),
    )


__all__ = [
    "ByteAccountV1",
    "DEFAULT_NETWORK_SEEDS_V1",
    "GilbertElliott",
    "JitterModel",
    "LossModel",
    "NETWORK_CONDITION_PLAN_SCHEMA_V1",
    "NETWORK_TRACE_SCHEMA_V1",
    "NetworkConditionId",
    "NetworkConditionPlanV1",
    "NetworkConfig",
    "NetworkEvent",
    "NetworkTrace",
    "NetworkTraceError",
    "NetworkTraceEventV1",
    "NetworkTraceV1",
    "PacketRequest",
    "SYNTHETIC_LEDGER_SOURCE_V1",
    "SyntheticTransmissionAttemptV1",
    "WireAccountingConfigV1",
    "account_wire_bytes_v1",
    "bounded_wire_reservation_v1",
    "condition_plan_v1",
    "decode_network_trace_v1",
    "default_condition_plans_v1",
    "generate_condition_trace_v1",
    "generate_network_trace",
]
