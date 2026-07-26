from dataclasses import FrozenInstanceError

import pytest

import transvision.models.resilient_v2x as resilient_v2x
from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    RejectedCandidate,
    SourceCandidate,
    UnsupportedReason,
)


def _candidate() -> SourceCandidate:
    return SourceCandidate(
        packet_id="seq-1:10",
        n_s=10,
        tau_s_ms=1000,
        arrival_tau_ms=1100,
        payload_valid=True,
        timestamp_valid=True,
        pose_valid=True,
        calibration_valid=True,
        faulted=False,
    )


def test_protocol_enum_values_are_stable() -> None:
    assert [agent.value for agent in Agent] == ["ego", "rsu"]
    assert [modality.value for modality in Modality] == ["lidar", "camera"]
    assert [reason.value for reason in UnsupportedReason] == [
        "empty_arrival_set",
        "empty_modality_history",
        "unsupported_horizon",
        "invalid_timestamp",
        "invalid_pose",
        "invalid_calibration",
        "missing_payload",
        "method_not_applicable",
    ]


def test_source_candidate_is_immutable() -> None:
    candidate = _candidate()

    with pytest.raises(FrozenInstanceError):
        candidate.n_s = 11


def test_rejected_candidate_is_immutable() -> None:
    rejected = RejectedCandidate(
        packet_id="seq-1:10",
        n_s=10,
        reason=UnsupportedReason.INVALID_TIMESTAMP,
    )

    with pytest.raises(FrozenInstanceError):
        rejected.n_s = 11


def test_branch_selection_is_immutable() -> None:
    selection = BranchSelection(
        agent=Agent.EGO,
        modality=Modality.LIDAR,
        supported=True,
        source=_candidate(),
        horizon=0,
        observed=True,
        propagated=False,
        rejected=(),
        reason=None,
    )

    with pytest.raises(FrozenInstanceError):
        selection.horizon = 1


@pytest.mark.parametrize(
    ("observed", "propagated", "horizon"),
    [(True, False, 0), (False, True, 2)],
    ids=["observed", "propagated"],
)
def test_supported_selection_accepts_each_valid_mode(
    observed: bool,
    propagated: bool,
    horizon: int,
) -> None:
    source = _candidate()

    selection = BranchSelection(
        agent=Agent.EGO,
        modality=Modality.LIDAR,
        supported=True,
        source=source,
        horizon=horizon,
        observed=observed,
        propagated=propagated,
        rejected=(),
        reason=None,
    )

    assert selection.source is source
    assert selection.horizon == horizon
    assert selection.observed is observed
    assert selection.propagated is propagated


def test_supported_selection_requires_source_and_horizon() -> None:
    with pytest.raises(ValueError, match="supported branch requires"):
        BranchSelection(
            agent=Agent.EGO,
            modality=Modality.LIDAR,
            supported=True,
            source=None,
            horizon=None,
            observed=False,
            propagated=False,
            rejected=(),
            reason=None,
        )


def test_supported_selection_rejects_negative_horizon() -> None:
    with pytest.raises(ProtocolInvariantError, match="horizon must be non-negative"):
        BranchSelection(
            agent=Agent.EGO,
            modality=Modality.LIDAR,
            supported=True,
            source=_candidate(),
            horizon=-1,
            observed=True,
            propagated=False,
            rejected=(),
            reason=None,
        )


@pytest.mark.parametrize(
    ("observed", "propagated"),
    [(False, False), (True, True)],
    ids=["neither", "both"],
)
def test_supported_selection_requires_exactly_one_mode(
    observed: bool,
    propagated: bool,
) -> None:
    with pytest.raises(
        ProtocolInvariantError,
        match="exactly one observed/propagated flag",
    ):
        BranchSelection(
            agent=Agent.EGO,
            modality=Modality.LIDAR,
            supported=True,
            source=_candidate(),
            horizon=0,
            observed=observed,
            propagated=propagated,
            rejected=(),
            reason=None,
        )


def test_supported_selection_forbids_reason() -> None:
    with pytest.raises(
        ProtocolInvariantError,
        match="supported branch requires source and horizon and forbids reason",
    ):
        BranchSelection(
            agent=Agent.EGO,
            modality=Modality.LIDAR,
            supported=True,
            source=_candidate(),
            horizon=0,
            observed=True,
            propagated=False,
            rejected=(),
            reason=UnsupportedReason.METHOD_NOT_APPLICABLE,
        )


def test_unsupported_selection_requires_reason_and_null_source() -> None:
    selection = BranchSelection.unsupported(
        agent=Agent.RSU,
        modality=Modality.CAMERA,
        reason=UnsupportedReason.EMPTY_ARRIVAL_SET,
        rejected=(),
    )

    assert selection.source is None
    assert selection.horizon is None
    assert selection.reason is UnsupportedReason.EMPTY_ARRIVAL_SET


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", _candidate()),
        ("horizon", 0),
        ("reason", None),
        ("observed", True),
        ("propagated", True),
    ],
)
def test_unsupported_selection_rejects_supported_shape_fields(
    field: str,
    value: object,
) -> None:
    values = {
        "agent": Agent.RSU,
        "modality": Modality.CAMERA,
        "supported": False,
        "source": None,
        "horizon": None,
        "observed": False,
        "propagated": False,
        "rejected": (),
        "reason": UnsupportedReason.EMPTY_ARRIVAL_SET,
    }
    values[field] = value

    with pytest.raises(
        ProtocolInvariantError,
        match=(
            "unsupported branch requires null source/horizon, false flags, and reason"
        ),
    ):
        BranchSelection(**values)


def test_package_exports_only_stable_task_one_api() -> None:
    assert resilient_v2x.__all__ == (
        "Agent",
        "Modality",
        "UnsupportedReason",
        "SourceCandidate",
        "RejectedCandidate",
        "BranchSelection",
        "ProtocolInvariantError",
    )
