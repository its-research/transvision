from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x import (
    AffineClockMap,
    CorrelatedTrackBelief,
    EventTimes,
    GilbertElliott,
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    LossModel,
    NetworkConfig,
    PacketRequest,
    ReceivedMessage,
    TimeEstimate,
    WireMessage,
    conservative_no_future_gate,
    decode_message,
    encode_message,
    generate_network_trace,
    wire_digest,
    wire_size,
)


def increment_message(
    message_id: str = "m-001",
    *,
    sequence: int = 1,
    factor_id: str = "sensor-a/frame-1",
    source: str = "roadside-a",
    generated: float = 10.2,
) -> WireMessage:
    return WireMessage(
        message_id=message_id,
        source=source,
        sequence=sequence,
        timestamps=LocalTimestamps(10.0, 10.1, generated, generated + 0.1),
        deadline=11.0,
        coordinate_frame="world",
        ttl=2.0,
        payload=IndependentIncrement(
            target_id="track-7",
            measurement=np.array([4.0]),
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.25]]),
            lineage=Lineage((factor_id,), True),
            log_normalizer=-0.5,
        ),
    )


def test_strict_six_time_and_payload_schema() -> None:
    times = EventTimes(1.0, 1.1, 1.2, 1.3, 1.6, 1.5)
    assert times.received == 1.6
    with pytest.raises(ValueError, match="mapped times"):
        EventTimes(1.1, 1.0, 1.2, 1.3, 1.4, 1.5)
    with pytest.raises(ValueError, match="local timestamps"):
        LocalTimestamps(2.0, 1.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="complete"):
        IndependentIncrement(
            target_id="x",
            measurement=np.array([0.0]),
            measurement_matrix=np.array([[1.0]]),
            measurement_covariance=np.array([[1.0]]),
            lineage=Lineage(("factor",), False),
        )
    with pytest.raises(ValueError, match="sum to one"):
        CorrelatedTrackBelief(
            track_id="x",
            existence_probability=0.8,
            mean=np.array([0.0]),
            covariance=np.array([[1.0]]),
            identity_probabilities=(("id-a", 0.4), ("id-b", 0.4)),
            lineage=Lineage(("shared",), False),
        )
    with pytest.raises(TypeError, match="sequence"):
        WireMessage(
            message_id="m",
            source="s",
            sequence=True,
            timestamps=LocalTimestamps(1.0, 1.0, 1.0, 1.0),
            deadline=2.0,
            coordinate_frame="world",
            ttl=1.0,
            payload=increment_message().payload,
        )


def test_schema_objects_are_frozen_and_arrays_are_read_only() -> None:
    message = increment_message()
    with pytest.raises(FrozenInstanceError):
        message.sequence = 8  # type: ignore[misc]
    with pytest.raises(ValueError):
        message.payload.measurement[0] = 99.0


def test_lineage_is_canonical_and_only_complete_disjoint_sets_prove_independence() -> None:
    first = Lineage(("z", "a"), True, ("parent-b", "parent-a"))
    second = Lineage(("b",), True)
    incomplete = Lineage(("c",), False)
    assert first.factor_ids == ("a", "z")
    assert first.proves_independent_from(second)
    assert not first.proves_independent_from(incomplete)
    assert not first.proves_independent_from(Lineage(("z",), True))
    with pytest.raises(ValueError, match="unique"):
        Lineage(("same", "same"), True)


def test_canonical_wire_encoding_round_trip_and_real_byte_length() -> None:
    message = increment_message(message_id="消息-001")
    first = encode_message(message)
    second = encode_message(message)
    assert first == second
    assert first.startswith(b'{"coordinate_frame"')
    assert wire_size(message) == len(first)
    assert wire_digest(message) == __import__("hashlib").sha256(first).hexdigest()
    decoded = decode_message(first)
    assert encode_message(decoded) == first
    assert decoded.message_id == "消息-001"
    assert wire_size(message) > len(first.decode("utf-8"))

    noncanonical = json.dumps(json.loads(first), ensure_ascii=False).encode("utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        decode_message(noncanonical)


def test_correlated_belief_has_distinct_wire_discriminator() -> None:
    message = WireMessage(
        message_id="belief-1",
        source="vehicle",
        sequence=2,
        timestamps=LocalTimestamps(1.0, 1.1, 1.2, 1.3),
        deadline=2.0,
        coordinate_frame="world",
        ttl=2.0,
        payload=CorrelatedTrackBelief(
            track_id="remote-7",
            existence_probability=0.9,
            mean=np.array([1.0, 2.0]),
            covariance=np.eye(2),
            identity_probabilities=(("id-7", 1.0),),
            lineage=Lineage(("shared-factor",), False, ("old-message",)),
        ),
    )
    encoded = encode_message(message)
    assert b'"kind":"correlated_track_belief"' in encoded
    decoded = decode_message(encoded)
    assert isinstance(decoded.payload, CorrelatedTrackBelief)


def test_affine_clock_mapping_propagates_parameter_uncertainty() -> None:
    # local = 2 * receiver + 4; local timestamp 14 maps to receiver time 5.
    clock = AffineClockMap(
        scale=2.0,
        offset=4.0,
        parameter_covariance=np.diag([0.04, 0.09]),
        timestamp_variance=0.16,
    )
    estimate = clock.map_timestamp(14.0)
    gradient = np.array([-2.5, -0.5])
    expected_variance = gradient @ np.diag([0.04, 0.09]) @ gradient + 0.16 / 4.0
    assert estimate.mean == pytest.approx(5.0)
    assert estimate.variance == pytest.approx(expected_variance)

    received = ReceivedMessage(increment_message(), received_at=6.0)
    mapped = AffineClockMap(1.0, 5.0, np.zeros((2, 2))).map_message(received)
    assert mapped.information_cutoff.mean == pytest.approx(5.0)
    assert mapped.at_means().received == 6.0


def test_no_future_gate_uses_upper_bound_not_point_mean() -> None:
    uncertain = TimeEstimate(mean=9.9, variance=0.04)
    assert not conservative_no_future_gate(uncertain, 10.0, confidence=0.999)
    assert conservative_no_future_gate(TimeEstimate(8.0, 0.0), 10.0)


def packet(index: int, *, transmit: float | None = None) -> PacketRequest:
    message = increment_message(
        message_id=f"m-{index}",
        sequence=index,
        factor_id=f"factor-{index}",
        generated=10.2 + index * 0.01,
    )
    return PacketRequest.from_message(
        message,
        transmitted_at=float(index) if transmit is None else transmit,
        deadline=100.0,
    )


def test_fixed_and_iid_network_traces_are_deterministic_and_order_invariant() -> None:
    packets = [packet(index) for index in range(8)]
    fixed = generate_network_trace(
        list(reversed(packets)), NetworkConfig(fixed_delay=0.25), seed=17
    )
    assert [event.packet.message_id for event in fixed.events] == [
        f"m-{index}" for index in range(8)
    ]
    assert [event.arrival_time for event in fixed.events] == [
        index + 0.25 for index in range(8)
    ]

    config = NetworkConfig(
        fixed_delay=0.1,
        jitter=0.04,
        loss_model=LossModel.IID,
        iid_loss_probability=0.35,
    )
    first = generate_network_trace(packets, config, seed=1337)
    second = generate_network_trace(tuple(reversed(packets)), config, seed=1337)
    assert first == second
    assert any(event.dropped for event in first.events)
    assert any(not event.dropped for event in first.events)


def test_gilbert_elliott_trace_preserves_burst_memory() -> None:
    config = NetworkConfig(
        fixed_delay=0.0,
        loss_model=LossModel.GILBERT_ELLIOTT,
        gilbert_elliott=GilbertElliott(
            probability_good_to_good=0.0,
            probability_bad_to_good=0.0,
            success_probability_good=1.0,
            success_probability_bad=0.0,
            initial_good_probability=1.0,
        ),
    )
    trace = generate_network_trace([packet(index) for index in range(5)], config, seed=9)
    assert [event.channel_state for event in trace.events] == [
        "good",
        "bad",
        "bad",
        "bad",
        "bad",
    ]
    assert [event.dropped for event in trace.events] == [False, True, True, True, True]


def test_jitter_reordering_and_outage_are_explicit_and_reproducible() -> None:
    packets = [packet(index, transmit=index * 0.1) for index in range(10)]
    config = NetworkConfig(
        fixed_delay=0.05,
        jitter=0.02,
        reorder_probability=0.5,
        reorder_extra_delay=0.8,
        outage_intervals=((0.4, 0.4),),
    )
    first = generate_network_trace(packets, config, seed=3)
    second = generate_network_trace(packets, config, seed=3)
    assert first == second
    assert first.events[4].dropped
    arrival_sequences = [event.packet.sequence for event in first.arrival_order]
    assert arrival_sequences != sorted(arrival_sequences)
