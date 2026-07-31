from __future__ import annotations

import pytest
import torch

from transvision.models.resilient_v2x import (
    Agent,
    BEVGridSpec,
    BranchSelection,
    ControlledBaselineInputSelector,
    Modality,
    ProtocolInvariantError,
    ResilientBatchSelections,
    SourceCandidate,
    UnsupportedReason,
)


def _supported(
    agent: Agent,
    modality: Modality,
    horizon: int,
    *,
    target_tick: int = 10,
) -> BranchSelection:
    source_tick = target_tick - horizon
    source_tau_ms = source_tick * 100
    endpoint_tick = target_tick if agent is Agent.EGO else source_tick
    return BranchSelection(
        agent=agent,
        modality=modality,
        supported=True,
        source=SourceCandidate(
            packet_id=f"{agent.value}-{modality.value}-{source_tick}",
            n_s=source_tick,
            tau_s_ms=source_tau_ms,
            arrival_tau_ms=(
                source_tau_ms + horizon * 100 if agent is Agent.RSU else None
            ),
            payload_valid=True,
            timestamp_valid=True,
            pose_valid=True,
            calibration_valid=True,
            faulted=False,
        ),
        horizon=horizon,
        endpoint_tick=endpoint_tick,
        observed=source_tick == endpoint_tick,
        propagated=source_tick != endpoint_tick,
        rejected=(),
        reason=None,
    )


def _unsupported(agent: Agent, modality: Modality) -> BranchSelection:
    return BranchSelection.unsupported(
        agent=agent,
        modality=modality,
        reason=UnsupportedReason.EMPTY_MODALITY_HISTORY,
        rejected=(),
    )


def _mixed_selections() -> ResilientBatchSelections:
    return ResilientBatchSelections(
        sample_ids=("sample-a", "sample-b"),
        lidar_ego=(
            _supported(Agent.EGO, Modality.LIDAR, 0),
            _unsupported(Agent.EGO, Modality.LIDAR),
        ),
        lidar_rsu=(
            _unsupported(Agent.RSU, Modality.LIDAR),
            _supported(Agent.RSU, Modality.LIDAR, 3),
        ),
        camera_ego=(
            _supported(Agent.EGO, Modality.CAMERA, 2),
            _supported(Agent.EGO, Modality.CAMERA, 0),
        ),
        camera_rsu=(
            _supported(Agent.RSU, Modality.CAMERA, 1),
            _unsupported(Agent.RSU, Modality.CAMERA),
        ),
        rsu_delay_intervals=(1.0, 3.0),
    )


def _poisoned_inputs(
    selections: ResilientBatchSelections,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lidar = torch.full((2, 2, 4, 2, 3, 3), float("nan"))
    camera = torch.full_like(lidar, float("nan"))
    transforms = torch.full((2, 2, 2, 4, 4, 4), float("nan"))
    availability = torch.zeros(2, 2, 2, 4, dtype=torch.bool)
    histories = (lidar, camera)
    layout = (
        ("lidar_ego", 0, 0, 1.0),
        ("lidar_rsu", 0, 1, 2.0),
        ("camera_ego", 1, 0, 3.0),
        ("camera_rsu", 1, 1, 4.0),
    )
    for branch_index, (key, modality, agent, value) in enumerate(layout):
        for batch_index, selection in enumerate(selections.branch(key)):
            if not selection.supported:
                continue
            assert selection.horizon is not None
            histories[modality][
                batch_index,
                agent,
                selection.horizon,
            ] = value + 10.0 * batch_index
            transforms[
                batch_index,
                modality,
                agent,
                selection.horizon,
            ] = torch.eye(4)
            availability[
                batch_index,
                modality,
                agent,
                selection.horizon,
            ] = True
    return (
        lidar.requires_grad_(),
        camera.requires_grad_(),
        transforms,
        availability,
    )


def test_selector_uses_fixed_branch_order_and_only_aligns_supported_slots() -> None:
    selections = _mixed_selections()
    lidar, camera, transforms, availability = _poisoned_inputs(selections)
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 3, 3),
        channels=2,
    )

    selected = selector(
        lidar_history=lidar,
        camera_history=camera,
        source_to_target=transforms,
        availability=availability,
        selections=selections,
    )

    assert selected.branches.shape == (2, 4, 2, 3, 3)
    assert torch.equal(
        selected.support,
        torch.tensor(
            [
                [True, False, True, True],
                [False, True, True, False],
            ]
        ),
    )
    assert torch.equal(
        selected.ages,
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 3.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        selected.branches[0, 0],
        torch.full((2, 3, 3), 1.0),
    )
    assert torch.equal(
        selected.branches[0, 2],
        torch.full((2, 3, 3), 3.0),
    )
    assert torch.equal(
        selected.branches[0, 3],
        torch.full((2, 3, 3), 4.0),
    )
    assert torch.equal(
        selected.branches[1, 1],
        torch.full((2, 3, 3), 12.0),
    )
    assert torch.equal(
        selected.branches[1, 2],
        torch.full((2, 3, 3), 13.0),
    )
    assert (
        torch.count_nonzero(
            selected.branches.masked_select(
                (~selected.support)[:, :, None, None, None].expand_as(selected.branches)
            )
        ).item()
        == 0
    )
    assert torch.isfinite(selected.branches).all()

    selected.branches.sum().backward()
    assert lidar.grad is not None
    assert camera.grad is not None
    assert lidar.grad[0, 0, 0].abs().sum().item() > 0
    assert lidar.grad[1, 1, 3].abs().sum().item() > 0
    assert camera.grad[0, 0, 2].abs().sum().item() > 0
    assert camera.grad[0, 1, 1].abs().sum().item() > 0
    assert camera.grad[1, 0, 0].abs().sum().item() > 0


def test_selector_rejects_selected_but_unavailable_slot() -> None:
    selections = _mixed_selections()
    lidar, camera, transforms, availability = _poisoned_inputs(selections)
    availability[0, 0, 0, 0] = False
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 3, 3),
        channels=2,
    )

    with pytest.raises(
        ProtocolInvariantError,
        match="selected source slot must be available",
    ):
        selector(
            lidar,
            camera,
            transforms,
            availability,
            selections,
        )


def test_selector_rejects_supported_horizon_outside_materialized_history() -> None:
    selections = ResilientBatchSelections(
        sample_ids=("sample-a",),
        lidar_ego=(_supported(Agent.EGO, Modality.LIDAR, 4),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 2, 2),
        channels=2,
    )

    with pytest.raises(
        ProtocolInvariantError,
        match=r"horizon must be in \[0,3\]",
    ):
        selector(
            torch.zeros(1, 2, 4, 2, 2, 2),
            torch.zeros(1, 2, 4, 2, 2, 2),
            torch.eye(4).view(1, 1, 1, 1, 4, 4).repeat(1, 2, 2, 4, 1, 1),
            torch.zeros(1, 2, 2, 4, dtype=torch.bool),
            selections,
        )


def test_selector_keeps_all_unsupported_batch_exactly_neutral() -> None:
    selections = ResilientBatchSelections(
        sample_ids=("sample-a",),
        lidar_ego=(_unsupported(Agent.EGO, Modality.LIDAR),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 2, 2),
        channels=2,
    )

    selected = selector(
        torch.full((1, 2, 4, 2, 2, 2), float("nan")),
        torch.full((1, 2, 4, 2, 2, 2), float("nan")),
        torch.full((1, 2, 2, 4, 4, 4), float("nan")),
        torch.zeros(1, 2, 2, 4, dtype=torch.bool),
        selections,
    )

    assert not selected.support.any()
    assert torch.count_nonzero(selected.branches).item() == 0
    assert torch.count_nonzero(selected.ages).item() == 0


def test_selector_applies_source_to_target_translation() -> None:
    selections = ResilientBatchSelections(
        sample_ids=("translated",),
        lidar_ego=(_supported(Agent.EGO, Modality.LIDAR, 0),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    lidar = torch.zeros(1, 2, 4, 1, 3, 3)
    lidar[0, 0, 0, 0, 1, 0] = 1.0
    camera = torch.full_like(lidar, float("nan"))
    transforms = torch.full((1, 2, 2, 4, 4, 4), float("nan"))
    translated = torch.eye(4)
    translated[0, 3] = 1.0
    transforms[0, 0, 0, 0] = translated
    availability = torch.zeros(1, 2, 2, 4, dtype=torch.bool)
    availability[0, 0, 0, 0] = True
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 3, 3),
        channels=1,
    )

    selected = selector(
        lidar,
        camera,
        transforms,
        availability,
        selections,
    )

    expected = torch.zeros(3, 3)
    expected[1, 1] = 1.0
    assert torch.allclose(selected.branches[0, 0, 0], expected, atol=1e-6)
    assert torch.count_nonzero(selected.branches[0, 1:]).item() == 0


def test_selector_uses_fp32_geometry_for_fp16_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections = ResilientBatchSelections(
        sample_ids=("amp",),
        lidar_ego=(_supported(Agent.EGO, Modality.LIDAR, 0),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    lidar = torch.zeros(1, 2, 4, 1, 2, 2, dtype=torch.float16)
    lidar[0, 0, 0] = 1.0
    camera = torch.full_like(lidar, float("nan"))
    transforms = torch.full(
        (1, 2, 2, 4, 4, 4),
        float("nan"),
        dtype=torch.float32,
    )
    transforms[0, 0, 0, 0] = torch.eye(4, dtype=torch.float32)
    availability = torch.zeros(1, 2, 2, 4, dtype=torch.bool)
    availability[0, 0, 0, 0] = True
    observed: dict[str, torch.dtype] = {}

    def capture_alignment(
        source: torch.Tensor,
        source_to_target: torch.Tensor,
        _spec: BEVGridSpec,
    ) -> torch.Tensor:
        observed["feature"] = source.dtype
        observed["transform"] = source_to_target.dtype
        return source

    monkeypatch.setattr(
        "transvision.models.resilient_v2x.baseline_inputs.align_bev_to_target",
        capture_alignment,
    )
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 2, 2),
        channels=1,
    )

    selected = selector(
        lidar,
        camera,
        transforms,
        availability,
        selections,
    )

    assert observed == {
        "feature": torch.float16,
        "transform": torch.float32,
    }
    assert selected.branches.dtype is torch.float16
    assert selected.ages.dtype is torch.float16


def test_selector_rejects_fp16_geometry_for_fp16_features() -> None:
    selections = ResilientBatchSelections(
        sample_ids=("bad-geometry",),
        lidar_ego=(_unsupported(Agent.EGO, Modality.LIDAR),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    selector = ControlledBaselineInputSelector(
        grid_spec=BEVGridSpec(0.0, 0.0, 1.0, 2, 2),
        channels=1,
    )

    with pytest.raises(ValueError, match="required geometry dtype"):
        selector(
            torch.zeros(1, 2, 4, 1, 2, 2, dtype=torch.float16),
            torch.zeros(1, 2, 4, 1, 2, 2, dtype=torch.float16),
            torch.eye(4, dtype=torch.float16)
            .view(1, 1, 1, 1, 4, 4)
            .repeat(1, 2, 2, 4, 1, 1),
            torch.zeros(1, 2, 2, 4, dtype=torch.bool),
            selections,
        )
