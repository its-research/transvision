import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import transvision.models.resilient_v2x as resilient_v2x
from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchDiagnostics,
    BranchSelection,
    Modality,
    ProtocolInvariantError,
    RejectedCandidate,
    RepairedBranch,
    SourceCandidate,
    UnsupportedReason,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_branch_diagnostics_is_immutable() -> None:
    diagnostics = BranchDiagnostics(
        agent=Agent.EGO,
        modality=Modality.LIDAR,
        supported=False,
        source_tick=None,
        source_tau_ms=None,
        horizon=None,
        observed=False,
        propagated=False,
        gamma=None,
        reliability=0.0,
        ptf_queried=False,
        displacement=None,
        confidence=None,
        reason=UnsupportedReason.EMPTY_MODALITY_HISTORY,
    )

    with pytest.raises(FrozenInstanceError):
        diagnostics.reliability = 1.0


def test_repaired_branch_keeps_tensor_and_diagnostics_contract() -> None:
    diagnostics = BranchDiagnostics(
        agent=Agent.RSU,
        modality=Modality.CAMERA,
        supported=False,
        source_tick=None,
        source_tau_ms=None,
        horizon=None,
        observed=False,
        propagated=False,
        gamma=None,
        reliability=0.0,
        ptf_queried=False,
        displacement=None,
        confidence=None,
        reason=UnsupportedReason.EMPTY_ARRIVAL_SET,
    )
    output = RepairedBranch(
        feature=torch.zeros(1, 256, 2, 2),
        reliability=torch.zeros(1),
        support=torch.zeros(1, dtype=torch.bool),
        observed=torch.zeros(1, dtype=torch.bool),
        propagated=torch.zeros(1, dtype=torch.bool),
        normalized_age=torch.zeros(1),
        displacement=None,
        confidence=None,
        diagnostics=(diagnostics,),
    )

    assert output.feature.shape == (1, 256, 2, 2)
    assert output.diagnostics == (diagnostics,)


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


def test_package_exports_only_stable_core_api() -> None:
    assert resilient_v2x.__all__ == (
        "Agent",
        "Modality",
        "UnsupportedReason",
        "SourceCandidate",
        "RejectedCandidate",
        "BranchSelection",
        "ProtocolInvariantError",
        "BEVGridSpec",
        "compose_source_to_target",
        "build_backward_grid",
        "align_bev_to_target",
        "warp_with_displacement",
        "arrived_candidates",
        "select_causal_source",
        "age_decay",
        "branch_reliability",
        "normalized_selected_rsu_delay",
        "PTFOutput",
        "HorizonConditionedPTF",
        "BranchDiagnostics",
        "RepairedBranch",
        "CausalBranchRepair",
        "DepthwiseSeparableResidualBlock",
        "ModalityAggregate",
        "ModalityAggregator",
        "RoutingOutput",
        "DynamicExpertRouter",
        "DistillationLosses",
        "freeze_teacher",
        "assert_teacher_frozen",
        "bernoulli_kl_from_logits",
        "distillation_losses",
    )
    assert len(resilient_v2x.__all__) == 32
    assert len(set(resilient_v2x.__all__)) == 32


def test_task_five_package_import_does_not_load_custom_ops() -> None:
    code = (
        "import sys; "
        "from transvision.models.resilient_v2x import CausalBranchRepair; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
