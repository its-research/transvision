"""Deterministic synthetic V2X network traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .schema import WireMessage
from .wire import wire_size


class LossModel(str, Enum):
    NONE = "none"
    IID = "iid"
    GILBERT_ELLIOTT = "gilbert_elliott"


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

    def __post_init__(self) -> None:
        for name in ("fixed_delay", "jitter", "reorder_extra_delay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
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
            jitter = rng.uniform(-config.jitter, config.jitter) if config.jitter else 0.0
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


__all__ = [
    "GilbertElliott",
    "LossModel",
    "NetworkConfig",
    "NetworkEvent",
    "NetworkTrace",
    "PacketRequest",
    "generate_network_trace",
]
