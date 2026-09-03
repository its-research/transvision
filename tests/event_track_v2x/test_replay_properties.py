"""High-volume deterministic properties preregistered for EventTrack-V2X."""

from __future__ import annotations

import numpy as np

from transvision.models.event_track_v2x.clock import AffineClockMap
from transvision.models.event_track_v2x.commit import AppendOnlyCommitLog
from transvision.models.event_track_v2x.replay import (
    FixedLagReplay,
    IngestStatus,
    constant_velocity_dynamics,
)
from transvision.models.event_track_v2x.schema import (
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    ReceivedMessage,
    WireMessage,
)


PROPERTY_CASES_V1 = 10_000
_CLOCK = AffineClockMap(1.0, 0.0, np.zeros((2, 2)))


def _tracker() -> FixedLagReplay:
    return FixedLagReplay(
        track_id="track",
        initial_time=0.0,
        initial_mean=np.array([0.0, 1.0]),
        initial_covariance=np.diag([1.0, 0.5]),
        dynamics=constant_velocity_dynamics(
            1, acceleration_spectral_density=0.1
        ),
        lag=5.0,
    )


def _message(case: int, index: int, event_time: float) -> ReceivedMessage:
    message_id = f"case-{case:05d}-message-{index}"
    return ReceivedMessage(
        WireMessage(
            message_id=message_id,
            source=f"agent-{index % 2}",
            sequence=index,
            timestamps=LocalTimestamps(
                event_time,
                event_time,
                event_time + 0.001,
                event_time + 0.002,
            ),
            deadline=3.0,
            coordinate_frame="world",
            ttl=5.0,
            payload=IndependentIncrement(
                target_id="track",
                measurement=np.array([event_time + index * 0.01]),
                measurement_matrix=np.array([[1.0, 0.0]]),
                measurement_covariance=np.array([[0.2 + index * 0.01]]),
                lineage=Lineage((f"factor-{message_id}",), True),
            ),
        ),
        received_at=2.0,
    )


def test_ten_thousand_out_of_order_duplicate_and_loss_cases_preserve_history() -> None:
    """Exercise 10,000 fixed-seed message sets without treating frames as IID."""

    rng = np.random.default_rng(3407)
    commit_log = AppendOnlyCommitLog()
    for case in range(PROPERTY_CASES_V1):
        event_times = rng.integers(1, 1900, size=4).astype(float) / 1000.0
        messages = tuple(
            _message(case, index, float(event_time))
            for index, event_time in enumerate(event_times)
        )
        retained = tuple(
            message for message in messages if bool(rng.integers(0, 2))
        )
        arrival_order = tuple(
            retained[index] for index in rng.permutation(len(retained))
        )
        canonical_order = tuple(
            sorted(
                retained,
                key=lambda item: (
                    item.message.timestamps.state_reference,
                    item.message.source,
                    item.message.sequence,
                    item.message.message_id,
                ),
            )
        )

        observed = _tracker()
        reference = _tracker()
        for received in arrival_order:
            assert observed.ingest_increment(
                received, clock_map=_CLOCK, decision_time=2.0
            ).status is IngestStatus.APPLIED
        if retained:
            duplicate = retained[int(rng.integers(0, len(retained)))]
            assert observed.ingest_increment(
                duplicate, clock_map=_CLOCK, decision_time=2.0
            ).status is IngestStatus.DUPLICATE
        for received in canonical_order:
            assert reference.ingest_increment(
                received, clock_map=_CLOCK, decision_time=2.0
            ).status is IngestStatus.APPLIED

        actual = observed.snapshot()
        expected = reference.snapshot()
        assert actual.mean.tobytes() == expected.mean.tobytes()
        assert actual.covariance.tobytes() == expected.covariance.tobytes()
        assert actual.processed_message_ids == expected.processed_message_ids

        prefix_before = commit_log.prefix_hash()
        committed = commit_log.commit(
            decision_time=float(case + 1),
            snapshot={
                "covariance": actual.covariance,
                "mean": actual.mean,
                "processed_message_ids": actual.processed_message_ids,
            },
            message_ids=actual.processed_message_ids,
        )
        assert committed.previous_hash == prefix_before
        assert commit_log.prefix_hash(case) == prefix_before

    assert len(commit_log.records) == PROPERTY_CASES_V1
    assert commit_log.verify()
