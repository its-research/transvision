import itertools
import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from enum import Enum
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
    ResilientFeatureBatch,
    RoutingDiagnostics,
    SampleInferenceDiagnostics,
    SourceCandidate,
    UnsupportedReason,
)

ROOT = Path(__file__).resolve().parents[2]

CORE_PUBLIC_API = (
    "Agent",
    "Modality",
    "UnsupportedReason",
    "ProtocolInvariantError",
    "SourceCandidate",
    "RejectedCandidate",
    "BranchSelection",
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
    "RoutingDiagnostics",
    "SampleInferenceDiagnostics",
    "DistillationLosses",
    "freeze_teacher",
    "assert_teacher_frozen",
    "bernoulli_kl_from_logits",
    "distillation_losses",
    "ResilientFeatureBatch",
)

BRANCH_ORDER = (
    (Agent.EGO, Modality.LIDAR),
    (Agent.RSU, Modality.LIDAR),
    (Agent.EGO, Modality.CAMERA),
    (Agent.RSU, Modality.CAMERA),
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


def _branch_diagnostics(
    agent: Agent,
    modality: Modality,
    *,
    supported: bool = True,
    ptf_queried: bool = True,
    missing_modality: bool = False,
) -> BranchDiagnostics:
    if not supported:
        return BranchDiagnostics(
            agent=agent,
            modality=modality,
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
            reason=(
                UnsupportedReason.METHOD_NOT_APPLICABLE
                if missing_modality
                else UnsupportedReason.EMPTY_MODALITY_HISTORY
            ),
        )
    observed = agent is Agent.EGO
    if not ptf_queried:
        return BranchDiagnostics(
            agent=agent,
            modality=modality,
            supported=True,
            source_tick=10,
            source_tau_ms=1000,
            horizon=0 if observed else 1,
            observed=observed,
            propagated=not observed,
            gamma=None,
            reliability=1.0 if observed else 0.8,
            ptf_queried=False,
            displacement=None,
            confidence=None,
            reason=None,
        )
    return BranchDiagnostics(
        agent=agent,
        modality=modality,
        supported=True,
        source_tick=10,
        source_tau_ms=1000,
        horizon=0 if observed else 1,
        observed=observed,
        propagated=not observed,
        gamma=1.0 if observed else 0.9,
        reliability=1.0 if observed else 0.8,
        ptf_queried=True,
        displacement=torch.zeros(2, 2, 3),
        confidence=torch.full((1, 2, 3), 0.5),
        reason=None,
    )


def _branches(
    *,
    supported: bool = True,
    ptf_queried: bool = True,
) -> tuple[
    BranchDiagnostics,
    BranchDiagnostics,
    BranchDiagnostics,
    BranchDiagnostics,
]:
    return tuple(
        _branch_diagnostics(
            agent,
            modality,
            supported=supported,
            ptf_queried=ptf_queried,
            missing_modality=not supported,
        )
        for agent, modality in BRANCH_ORDER
    )


def _routing_diagnostics(
    *,
    all_invalid: bool = False,
) -> RoutingDiagnostics:
    return RoutingDiagnostics(
        descriptor=torch.zeros(783),
        expert_support=torch.tensor(
            [False, False, False]
            if all_invalid
            else [True, False, False]
        ),
        weights=torch.tensor(
            [0.0, 0.0, 0.0] if all_invalid else [1.0, 0.0, 0.0]
        ),
        not_applicable_reason=None,
    )


def _sample_diagnostics(
    sample_id: str,
    *,
    all_invalid: bool = False,
    routing: RoutingDiagnostics | None = None,
) -> SampleInferenceDiagnostics:
    return SampleInferenceDiagnostics(
        schema_version=1,
        sample_id=sample_id,
        method="resilient",
        branches=_branches(supported=not all_invalid),
        routing=(
            _routing_diagnostics(all_invalid=all_invalid)
            if routing is None
            else routing
        ),
    )


def _feature_batch() -> ResilientFeatureBatch:
    fused = torch.zeros(2, 256, 2, 3, requires_grad=True)
    routing_weights = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    routing_descriptor = torch.zeros(2, 783, requires_grad=True)
    first = _sample_diagnostics("sample-1")
    second = _sample_diagnostics("sample-2", all_invalid=True)
    first = replace(
        first,
        routing=replace(
            first.routing,
            descriptor=routing_descriptor.detach()[0],
            weights=routing_weights.detach()[0],
        ),
    )
    second = replace(
        second,
        routing=replace(
            second.routing,
            descriptor=routing_descriptor.detach()[1],
            weights=routing_weights.detach()[1],
        ),
    )
    return ResilientFeatureBatch(
        fused=fused,
        overall_support=torch.tensor([True, False]),
        routing_weights=routing_weights,
        routing_descriptor=routing_descriptor,
        branch_features={
            name: torch.zeros(2, 256, 2, 3)
            for name in ("lidar_ego", "lidar_rsu", "camera_ego", "camera_rsu")
        },
        diagnostics=(first, second),
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


def test_routing_diagnostics_is_frozen() -> None:
    routing = _routing_diagnostics()

    with pytest.raises(FrozenInstanceError):
        routing.not_applicable_reason = "changed"


def test_routing_diagnostics_accepts_applicable_and_non_applicable_states() -> None:
    applicable = _routing_diagnostics()
    all_invalid = _routing_diagnostics(all_invalid=True)
    not_applicable = RoutingDiagnostics(
        descriptor=None,
        expert_support=None,
        weights=None,
        not_applicable_reason="baseline has no DER",
    )

    assert applicable.weights.tolist() == [1.0, 0.0, 0.0]
    assert torch.count_nonzero(all_invalid.expert_support).item() == 0
    assert torch.count_nonzero(all_invalid.weights).item() == 0
    assert not_applicable.descriptor is None
    assert not_applicable.not_applicable_reason == "baseline has no DER"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("descriptor", None),
        ("expert_support", None),
        ("weights", None),
        ("not_applicable_reason", "unexpected"),
        ("descriptor", object()),
        ("descriptor", torch.zeros(782)),
        ("descriptor", torch.zeros(783, dtype=torch.int64)),
        ("descriptor", torch.full((783,), torch.nan)),
        ("descriptor", torch.zeros(783, requires_grad=True)),
        ("expert_support", object()),
        ("expert_support", torch.zeros(2, dtype=torch.bool)),
        ("expert_support", torch.zeros(3)),
        ("weights", object()),
        ("weights", torch.zeros(2)),
        ("weights", torch.zeros(3, dtype=torch.int64)),
        ("weights", torch.tensor([torch.nan, 0.0, 0.0])),
        ("weights", torch.tensor([-0.1, 1.1, 0.0])),
        ("weights", torch.zeros(3, requires_grad=True)),
    ],
)
def test_routing_diagnostics_rejects_invalid_applicable_fields(
    field: str,
    value: object,
) -> None:
    values = {
        "descriptor": torch.zeros(783),
        "expert_support": torch.tensor([True, False, False]),
        "weights": torch.tensor([1.0, 0.0, 0.0]),
        "not_applicable_reason": None,
    }
    values[field] = value

    with pytest.raises(ProtocolInvariantError):
        RoutingDiagnostics(**values)


@pytest.mark.parametrize("reason", [None, "", "   ", 1])
def test_non_applicable_routing_requires_nonempty_reason(
    reason: object,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="reason|applicable"):
        RoutingDiagnostics(
            descriptor=None,
            expert_support=None,
            weights=None,
            not_applicable_reason=reason,
        )


@pytest.mark.parametrize(
    ("expert_support", "weights"),
    [
        (
            torch.tensor([True, False, False]),
            torch.tensor([0.5, 0.5, 0.0]),
        ),
        (
            torch.tensor([True, True, False]),
            torch.tensor([0.4, 0.4, 0.0]),
        ),
        (
            torch.tensor([False, False, False]),
            torch.tensor([1.0, 0.0, 0.0]),
        ),
    ],
)
def test_routing_diagnostics_rejects_support_weight_inconsistency(
    expert_support: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="weight|support"):
        RoutingDiagnostics(
            descriptor=torch.zeros(783),
            expert_support=expert_support,
            weights=weights,
            not_applicable_reason=None,
        )


def test_sample_diagnostics_is_frozen_and_accepts_exact_branch_order() -> None:
    sample = _sample_diagnostics("sample-1")

    assert tuple(
        (branch.agent, branch.modality) for branch in sample.branches
    ) == BRANCH_ORDER
    with pytest.raises(FrozenInstanceError):
        sample.sample_id = "changed"


@pytest.mark.parametrize("schema_version", [0, 2, True, 1.0, "1"])
def test_sample_diagnostics_requires_exact_schema_version(
    schema_version: object,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="schema"):
        replace(
            _sample_diagnostics("sample-1"),
            schema_version=schema_version,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", ""),
        ("sample_id", "   "),
        ("sample_id", 1),
        ("method", ""),
        ("method", "\t"),
        ("method", 1),
        ("branches", list(_branches())),
        ("branches", _branches()[:3]),
        ("routing", object()),
    ],
)
def test_sample_diagnostics_rejects_invalid_top_level_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ProtocolInvariantError):
        replace(_sample_diagnostics("sample-1"), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", " sample-1"),
        ("sample_id", "sample-1 "),
        ("method", " resilient"),
        ("method", "resilient "),
    ],
)
def test_sample_diagnostics_rejects_noncanonical_trimmed_text(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ProtocolInvariantError, match=field):
        replace(_sample_diagnostics("sample-1"), **{field: value})


INCORRECT_BRANCH_PERMUTATIONS = tuple(
    permutation
    for permutation in itertools.permutations(range(4))
    if permutation != (0, 1, 2, 3)
)


@pytest.mark.parametrize(
    "permutation",
    INCORRECT_BRANCH_PERMUTATIONS,
    ids=lambda value: "".join(str(index) for index in value),
)
def test_sample_diagnostics_rejects_every_incorrect_branch_permutation(
    permutation: tuple[int, int, int, int],
) -> None:
    branches = _branches()

    with pytest.raises(ProtocolInvariantError, match="order|branch"):
        replace(
            _sample_diagnostics("sample-1"),
            branches=tuple(branches[index] for index in permutation),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_tick", 10),
        ("source_tau_ms", 1000),
        ("horizon", 0),
        ("observed", True),
        ("propagated", True),
        ("gamma", 1.0),
        ("reliability", 0.1),
        ("ptf_queried", True),
        ("displacement", torch.zeros(2, 2, 3)),
        ("confidence", torch.zeros(1, 2, 3)),
        ("reason", None),
    ],
)
def test_missing_modality_branch_requires_exact_null_false_zero_shape(
    field: str,
    value: object,
) -> None:
    branches = list(_branches(supported=False))
    branches[0] = replace(branches[0], **{field: value})

    with pytest.raises(ProtocolInvariantError, match="unsupported|branch"):
        replace(
            _sample_diagnostics("sample-1", all_invalid=True),
            branches=tuple(branches),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_tick", None),
        ("source_tick", True),
        ("source_tau_ms", None),
        ("horizon", None),
        ("horizon", 4),
        ("observed", False),
        ("reason", UnsupportedReason.EMPTY_ARRIVAL_SET),
        ("gamma", math.nan),
        ("gamma", 1.1),
        ("reliability", math.nan),
        ("reliability", -0.1),
        ("displacement", None),
        ("displacement", torch.zeros(1, 2, 3)),
        ("displacement", torch.zeros(2, 2, 3, requires_grad=True)),
        ("confidence", None),
        ("confidence", torch.zeros(2, 2, 3)),
        ("confidence", torch.zeros(1, 2, 3, requires_grad=True)),
    ],
)
def test_supported_ptf_branch_rejects_invalid_contract(
    field: str,
    value: object,
) -> None:
    branches = list(_branches())
    branches[0] = replace(branches[0], **{field: value})

    with pytest.raises(ProtocolInvariantError, match="branch|PTF|supported"):
        replace(
            _sample_diagnostics("sample-1"),
            branches=tuple(branches),
        )


def test_supported_method_without_ptf_and_der_uses_explicit_nulls() -> None:
    routing = RoutingDiagnostics(
        descriptor=None,
        expert_support=None,
        weights=None,
        not_applicable_reason="baseline has no DER",
    )
    sample = SampleInferenceDiagnostics(
        schema_version=1,
        sample_id="baseline-1",
        method="baseline",
        branches=_branches(ptf_queried=False),
        routing=routing,
    )

    assert all(branch.supported for branch in sample.branches)
    assert all(not branch.ptf_queried for branch in sample.branches)
    assert all(branch.gamma is None for branch in sample.branches)
    assert all(branch.displacement is None for branch in sample.branches)
    assert all(branch.confidence is None for branch in sample.branches)
    assert sample.routing.not_applicable_reason == "baseline has no DER"


def _json_normalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, torch.Tensor):
        raise AssertionError("available tensors are not serialized in Task 8")
    return value


def test_test_local_json_normalizer_preserves_unavailable_nulls() -> None:
    sample = SampleInferenceDiagnostics(
        schema_version=1,
        sample_id="baseline-1",
        method="baseline",
        branches=_branches(supported=False),
        routing=RoutingDiagnostics(
            descriptor=None,
            expert_support=None,
            weights=None,
            not_applicable_reason="baseline has no DER",
        ),
    )

    payload = json.loads(json.dumps(_json_normalize(asdict(sample))))

    assert payload["routing"]["descriptor"] is None
    assert payload["routing"]["expert_support"] is None
    assert payload["routing"]["weights"] is None
    assert payload["branches"][0]["source_tick"] is None
    assert payload["branches"][0]["displacement"] is None
    assert payload["branches"][0]["confidence"] is None
    assert (
        payload["branches"][0]["reason"]
        == UnsupportedReason.METHOD_NOT_APPLICABLE.value
    )


def test_resilient_feature_batch_accepts_training_tensors_without_mutation() -> None:
    batch = _feature_batch()
    fused_before = batch.fused.detach().clone()
    weights_before = batch.routing_weights.detach().clone()
    descriptor_before = batch.routing_descriptor.detach().clone()

    assert batch.fused.shape == (2, 256, 2, 3)
    assert batch.overall_support.tolist() == [True, False]
    assert batch.fused.requires_grad
    assert batch.routing_weights.requires_grad
    assert batch.routing_descriptor.requires_grad
    assert all(
        diagnostic.routing.descriptor is None
        or not diagnostic.routing.descriptor.requires_grad
        for diagnostic in batch.diagnostics
    )
    assert all(
        diagnostic.routing.weights is None
        or not diagnostic.routing.weights.requires_grad
        for diagnostic in batch.diagnostics
    )
    torch.testing.assert_close(batch.fused.detach(), fused_before)
    torch.testing.assert_close(batch.routing_weights.detach(), weights_before)
    torch.testing.assert_close(batch.routing_descriptor.detach(), descriptor_before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fused", object()),
        ("fused", torch.zeros(2, 256, 2)),
        ("fused", torch.zeros(0, 256, 2, 3)),
        ("fused", torch.zeros(2, 255, 2, 3)),
        ("fused", torch.zeros(2, 256, 0, 3)),
        ("fused", torch.zeros(2, 256, 2, 3, dtype=torch.int64)),
        ("fused", torch.full((2, 256, 2, 3), torch.nan)),
        ("overall_support", object()),
        ("overall_support", torch.zeros(2, 1, dtype=torch.bool)),
        ("overall_support", torch.zeros(2)),
        ("routing_weights", object()),
        ("routing_weights", torch.zeros(2, 2)),
        ("routing_weights", torch.zeros(2, 3, dtype=torch.int64)),
        ("routing_weights", torch.full((2, 3), torch.nan)),
        ("routing_descriptor", object()),
        ("routing_descriptor", torch.zeros(2, 782)),
        ("routing_descriptor", torch.zeros(2, 783, dtype=torch.int64)),
        ("routing_descriptor", torch.full((2, 783), torch.inf)),
        ("diagnostics", list((_sample_diagnostics("sample-1"),))),
        ("diagnostics", (_sample_diagnostics("sample-1"),)),
        ("branch_features", object()),
    ],
)
def test_resilient_feature_batch_rejects_invalid_top_level_contract(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ProtocolInvariantError):
        replace(_feature_batch(), **{field: value})


def test_resilient_feature_batch_rejects_tensor_device_mismatch() -> None:
    batch = _feature_batch()

    with pytest.raises(ProtocolInvariantError, match="device"):
        replace(
            batch,
            overall_support=torch.empty(
                2,
                dtype=torch.bool,
                device="meta",
            ),
        )
    with pytest.raises(ProtocolInvariantError, match="device"):
        replace(
            batch,
            routing_weights=torch.empty(2, 3, device="meta"),
        )
    with pytest.raises(ProtocolInvariantError, match="device"):
        replace(
            batch,
            routing_descriptor=torch.empty(2, 783, device="meta"),
        )


@pytest.mark.parametrize(
    ("row", "weights"),
    [
        (0, torch.tensor([0.5, 0.0, 0.0])),
        (0, torch.tensor([-0.1, 1.1, 0.0])),
        (1, torch.tensor([1.0, 0.0, 0.0])),
    ],
)
def test_resilient_feature_batch_rejects_invalid_row_weights(
    row: int,
    weights: torch.Tensor,
) -> None:
    batch = _feature_batch()
    changed = batch.routing_weights.detach().clone()
    changed[row] = weights

    with pytest.raises(ProtocolInvariantError, match="weight|support"):
        replace(batch, routing_weights=changed)


def test_resilient_feature_batch_requires_unique_sample_ids() -> None:
    batch = _feature_batch()
    duplicate = replace(batch.diagnostics[1], sample_id="sample-1")

    with pytest.raises(ProtocolInvariantError, match="unique|sample"):
        replace(batch, diagnostics=(batch.diagnostics[0], duplicate))


@pytest.mark.parametrize("noncanonical_duplicate", [" id", "id "])
def test_resilient_feature_batch_rejects_canonical_duplicate_sample_ids(
    noncanonical_duplicate: str,
) -> None:
    batch = _feature_batch()
    first = replace(batch.diagnostics[0], sample_id="id")
    second = batch.diagnostics[1]
    object.__setattr__(second, "sample_id", noncanonical_duplicate)

    with pytest.raises(ProtocolInvariantError, match="sample_id|unique"):
        replace(batch, diagnostics=(first, second))


@pytest.mark.parametrize("field", ["descriptor", "weights"])
def test_resilient_feature_batch_matches_diagnostic_routing_rows(
    field: str,
) -> None:
    batch = _feature_batch()
    routing = batch.diagnostics[0].routing
    value = getattr(routing, field)
    assert isinstance(value, torch.Tensor)
    changed = value.clone()
    changed[0] = 0.5
    if field == "weights":
        changed_routing = replace(
            routing,
            expert_support=torch.tensor([True, True, True]),
            weights=torch.tensor([0.2, 0.3, 0.5]),
        )
    else:
        changed_routing = replace(routing, descriptor=changed)
    diagnostic = replace(batch.diagnostics[0], routing=changed_routing)

    with pytest.raises(ProtocolInvariantError, match="diagnostic|routing"):
        replace(batch, diagnostics=(diagnostic, batch.diagnostics[1]))


@pytest.mark.parametrize("field", ["descriptor", "weights"])
def test_resilient_feature_batch_requires_matching_diagnostic_dtype(
    field: str,
) -> None:
    batch = _feature_batch()
    routing = batch.diagnostics[0].routing
    value = getattr(routing, field)
    assert isinstance(value, torch.Tensor)
    diagnostic = replace(
        batch.diagnostics[0],
        routing=replace(routing, **{field: value.to(torch.float64)}),
    )

    with pytest.raises(ProtocolInvariantError, match="dtype|diagnostic"):
        replace(batch, diagnostics=(diagnostic, batch.diagnostics[1]))


def test_resilient_feature_batch_requires_applicable_routing_diagnostics() -> None:
    batch = _feature_batch()
    diagnostic = replace(
        batch.diagnostics[0],
        routing=RoutingDiagnostics(
            descriptor=None,
            expert_support=None,
            weights=None,
            not_applicable_reason="baseline",
        ),
    )

    with pytest.raises(ProtocolInvariantError, match="applicable|routing"):
        replace(batch, diagnostics=(diagnostic, batch.diagnostics[1]))


def test_resilient_feature_batch_matches_overall_and_expert_support() -> None:
    batch = _feature_batch()
    changed_support = batch.overall_support.clone()
    changed_support[0] = False

    with pytest.raises(ProtocolInvariantError, match="support|diagnostic"):
        replace(batch, overall_support=changed_support)


@pytest.mark.parametrize(
    "branch_features",
    [
        {"branch": object()},
        {1: torch.zeros(2, 256, 2, 3)},
        {"branch": torch.zeros(1, 256, 2, 3)},
        {
            name: torch.zeros(2, 255, 2, 3)
            for name in ("a", "b", "c", "d")
        },
        {
            name: torch.zeros(2, 256, 2, 3, dtype=torch.float64)
            for name in ("a", "b", "c", "d")
        },
        {
            name: torch.zeros(2, 256, 3, 3)
            for name in ("a", "b", "c", "d")
        },
    ],
)
def test_resilient_feature_batch_rejects_invalid_branch_features(
    branch_features: object,
) -> None:
    with pytest.raises(ProtocolInvariantError, match="branch"):
        replace(_feature_batch(), branch_features=branch_features)


def test_core_api_document_covers_all_symbols_and_required_semantics() -> None:
    path = ROOT / "docs" / "resilient_v2x" / "core-api.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")

    for name in CORE_PUBLIC_API:
        assert f"`{name}`" in text
    for phrase in (
        "frozen 35-name prefix",
        "append-only",
        "unsupported-reason precedence",
        "[B,C,Y,X]",
        "cell-center",
        "backward sampling",
        "[D_x,D_y]",
        "arrival cutoff",
        "metadata order",
        "horizon `0..3`",
        "observed",
        "propagated",
        "age decay",
        "confidence",
        "reliability",
        "PTF",
        "`h=0`",
        "linear",
        "nonlinear",
        "neutral scatter",
        "diagnostic `None`",
        "`[L_E,L_R,C_E,C_R]`",
        "dynamic",
        "static",
        "uniform",
        "all-invalid",
        "`3*256 + 2 + 8 + 4 + 1 = 783`",
        "teacher freeze",
        "feature MSE",
        "Bernoulli KL",
        "valid mask",
        "`T^2`",
        "`RoutingDiagnostics`",
        "`SampleInferenceDiagnostics`",
        "`ResilientFeatureBatch`",
        "JSON `null`",
        "`ProtocolInvariantError`",
        "unsupported outcome",
        "runtime failure",
        "Mac arm64 CPU",
        "locked Linux",
    ):
        assert phrase in text


def test_package_exports_only_stable_core_api() -> None:
    assert resilient_v2x.__all__ == CORE_PUBLIC_API
    assert len(resilient_v2x.__all__) == 35
    assert len(set(resilient_v2x.__all__)) == 35


def test_public_core_api_imports_and_frozen_prefix() -> None:
    assert resilient_v2x.__all__[:35] == CORE_PUBLIC_API
    for name in CORE_PUBLIC_API:
        assert getattr(resilient_v2x, name) is not None

    assert resilient_v2x.Agent.EGO.value == "ego"
    assert resilient_v2x.Modality.CAMERA.value == "camera"
    assert (
        resilient_v2x.UnsupportedReason.UNSUPPORTED_HORIZON.value
        == "unsupported_horizon"
    )
    assert resilient_v2x.DynamicExpertRouter.descriptor_dim == 783
    assert ResilientFeatureBatch.__name__ == "ResilientFeatureBatch"
    assert callable(resilient_v2x.compose_source_to_target)
    assert callable(resilient_v2x.distillation_losses)


def test_all_public_core_imports_do_not_load_custom_ops() -> None:
    names = ", ".join(CORE_PUBLIC_API)
    code = (
        "import sys; "
        f"from transvision.models.resilient_v2x import {names}; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_task_five_package_import_does_not_load_custom_ops() -> None:
    code = (
        "import sys; "
        "from transvision.models.resilient_v2x import CausalBranchRepair; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
