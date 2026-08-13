from __future__ import annotations

import pytest
import torch

from transvision.models.resilient_v2x import (
    Agent,
    BEVGridSpec,
    BranchSelection,
    Modality,
    ResilientBatchSelections,
    ResilientV2XFeatureFusion,
    SourceCandidate,
    UnsupportedReason,
)


def _supported(
    agent: Agent,
    modality: Modality,
    horizon: int,
    *,
    target_tick: int = 3,
    delay_ms: int = 0,
    endpoint_tick: int | None = None,
) -> BranchSelection:
    source_tick = target_tick - horizon
    source_tau_ms = source_tick * 100
    source = SourceCandidate(
        packet_id=f"{agent.value}-{modality.value}-{source_tick}",
        n_s=source_tick,
        tau_s_ms=source_tau_ms,
        arrival_tau_ms=(source_tau_ms + delay_ms if agent is Agent.RSU else None),
        payload_valid=True,
        timestamp_valid=True,
        pose_valid=True,
        calibration_valid=True,
        faulted=False,
    )
    if endpoint_tick is None:
        endpoint_tick = target_tick if agent is Agent.EGO else source_tick
    observed = source_tick == endpoint_tick
    return BranchSelection(
        agent=agent,
        modality=modality,
        supported=True,
        source=source,
        horizon=horizon,
        endpoint_tick=endpoint_tick,
        observed=observed,
        propagated=not observed,
        rejected=(),
        reason=None,
    )


def _unsupported(agent: Agent, modality: Modality) -> BranchSelection:
    return BranchSelection.unsupported(
        agent,
        modality,
        UnsupportedReason.EMPTY_MODALITY_HISTORY,
        (),
    )


def _all_supported() -> ResilientBatchSelections:
    return ResilientBatchSelections(
        sample_ids=("sample-a", "sample-b"),
        lidar_ego=(
            _supported(Agent.EGO, Modality.LIDAR, 0),
            _supported(Agent.EGO, Modality.LIDAR, 2),
        ),
        lidar_rsu=(
            _supported(Agent.RSU, Modality.LIDAR, 0),
            _supported(Agent.RSU, Modality.LIDAR, 1, delay_ms=100),
        ),
        camera_ego=(
            _supported(Agent.EGO, Modality.CAMERA, 0),
            _supported(Agent.EGO, Modality.CAMERA, 3),
        ),
        camera_rsu=(
            _supported(Agent.RSU, Modality.CAMERA, 0),
            _supported(Agent.RSU, Modality.CAMERA, 1, delay_ms=100),
        ),
        rsu_delay_intervals=(0.0, 1.0),
    )


def _inputs(selections: ResilientBatchSelections):
    batch = selections.batch_size
    lidar = torch.randn(batch, 2, 4, 256, 4, 4, requires_grad=True)
    camera = torch.randn(batch, 2, 4, 256, 4, 4, requires_grad=True)
    transforms = (
        torch.eye(4)
        .view(1, 1, 1, 4, 4)
        .repeat(
            batch,
            2,
            4,
            1,
            1,
        )
    )
    lidar_available = torch.zeros(batch, 2, 4, dtype=torch.bool)
    camera_available = torch.zeros(batch, 2, 4, dtype=torch.bool)
    for row in range(batch):
        for key, availability in (
            ("lidar_ego", lidar_available),
            ("lidar_rsu", lidar_available),
            ("camera_ego", camera_available),
            ("camera_rsu", camera_available),
        ):
            selection = selections.branch(key)[row]
            if selection.supported:
                agent = 0 if selection.agent is Agent.EGO else 1
                availability[row, agent, selection.horizon] = True
    return lidar, camera, transforms, lidar_available, camera_available


def test_feature_fusion_runs_full_paper_path_and_backpropagates() -> None:
    selections = _all_supported()
    lidar, camera, transforms, lidar_available, camera_available = _inputs(selections)
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, -2.0, 1.0, 4, 4),
        ptf_mode="none",
        routing_mode="uniform",
    )

    output = module(
        lidar,
        camera,
        transforms,
        transforms,
        lidar_available,
        camera_available,
        selections,
    )

    assert output.fused.shape == (2, 256, 4, 4)
    assert output.routing_descriptor.shape == (2, 783)
    assert torch.equal(output.overall_support, torch.tensor([True, True]))
    assert torch.allclose(
        output.routing_weights,
        torch.full((2, 3), 1.0 / 3.0),
    )
    assert set(output.branch_features) == {
        "lidar_ego",
        "lidar_rsu",
        "camera_ego",
        "camera_rsu",
    }
    assert output.diagnostics[0].branches[0].observed
    assert output.diagnostics[0].branches[1].observed
    assert output.routing_descriptor[1, 782].item() == pytest.approx(1.0)

    output.fused.square().mean().backward()
    assert lidar.grad is not None and torch.isfinite(lidar.grad).all()
    assert camera.grad is not None and torch.isfinite(camera.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_feature_fusion_supports_amp_features_with_fp32_geometry() -> None:
    selections = _all_supported()
    lidar, camera, transforms, lidar_available, camera_available = _inputs(selections)
    lidar = lidar.detach().cuda().half().requires_grad_()
    camera = camera.detach().cuda().half().requires_grad_()
    transforms = transforms.cuda()
    lidar_available = lidar_available.cuda()
    camera_available = camera_available.cuda()
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, -2.0, 1.0, 4, 4),
        ptf_mode="nonlinear",
        routing_mode="uniform",
    ).cuda()

    with torch.autocast("cuda", dtype=torch.float16):
        output = module(
            lidar,
            camera,
            transforms,
            transforms,
            lidar_available,
            camera_available,
            selections,
        )
        loss = output.fused.float().square().mean()
    loss.backward()

    assert output.fused.dtype is torch.float16
    assert lidar.grad is not None and torch.isfinite(lidar.grad).all()
    assert camera.grad is not None and torch.isfinite(camera.grad).all()


def test_feature_fusion_all_invalid_is_exactly_neutral() -> None:
    selections = ResilientBatchSelections(
        sample_ids=("invalid",),
        lidar_ego=(_unsupported(Agent.EGO, Modality.LIDAR),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    poison = torch.full((1, 2, 4, 256, 4, 4), float("nan"))
    poison_transform = torch.full((1, 2, 4, 4, 4), float("nan"))
    availability = torch.zeros(1, 2, 4, dtype=torch.bool)
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, 0.0, 1.0, 4, 4),
        ptf_mode="none",
    )

    output = module(
        poison,
        poison.clone(),
        poison_transform,
        poison_transform.clone(),
        availability,
        availability.clone(),
        selections,
    )

    assert not output.overall_support.item()
    assert torch.count_nonzero(output.fused).item() == 0
    assert torch.count_nonzero(output.routing_weights).item() == 0
    assert all(not branch.supported for branch in output.diagnostics[0].branches)


def test_no_ptf_ablation_keeps_full_ptf_parameter_capacity() -> None:
    grid = BEVGridSpec(0.0, 0.0, 1.0, 4, 4)
    full = ResilientV2XFeatureFusion(grid, ptf_mode="nonlinear")
    no_ptf = ResilientV2XFeatureFusion(grid, ptf_mode="none")

    assert sum(parameter.numel() for parameter in no_ptf.parameters()) == sum(
        parameter.numel() for parameter in full.parameters()
    )


def test_support_residual_threads_without_state_schema_change() -> None:
    grid = BEVGridSpec(0.0, 0.0, 1.0, 4, 4)
    legacy = ResilientV2XFeatureFusion(grid, ptf_mode="none")
    legacy_schema = tuple(
        (name, tuple(value.shape)) for name, value in legacy.state_dict().items()
    )
    candidate = ResilientV2XFeatureFusion(
        grid,
        ptf_mode="none",
        support_residual_weight=0.5,
    )
    candidate_schema = tuple(
        (name, tuple(value.shape)) for name, value in candidate.state_dict().items()
    )

    assert candidate.router.support_residual_weight == 0.5
    assert candidate_schema == legacy_schema


def test_explicit_latest_arrival_age_is_not_replaced_by_fallback_packet_delay() -> None:
    base = _all_supported()
    selections = ResilientBatchSelections(
        sample_ids=base.sample_ids,
        lidar_ego=base.lidar_ego,
        lidar_rsu=base.lidar_rsu,
        camera_ego=base.camera_ego,
        camera_rsu=base.camera_rsu,
        rsu_delay_intervals=(2.0, 0.0),
    )
    lidar, camera, transforms, lidar_available, camera_available = _inputs(selections)
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, 0.0, 1.0, 4, 4),
        ptf_mode="none",
    )

    output = module(
        lidar,
        camera,
        transforms,
        transforms,
        lidar_available,
        camera_available,
        selections,
    )

    assert output.routing_descriptor[0, 782].item() == pytest.approx(2.0)
    assert output.routing_descriptor[1, 782].item() == 0.0


def test_feature_fusion_rejects_selected_but_unavailable_slot() -> None:
    selections = _all_supported()
    lidar, camera, transforms, lidar_available, camera_available = _inputs(selections)
    lidar_available[0, 0, 0] = False
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, 0.0, 1.0, 4, 4),
        ptf_mode="none",
    )

    with pytest.raises(
        ValueError,
        match="selected source slot must be available",
    ):
        module(
            lidar,
            camera,
            transforms,
            transforms,
            lidar_available,
            camera_available,
            selections,
        )


@pytest.mark.parametrize("ptf_mode", ("nonlinear", "linear"))
def test_learned_ptf_modes_smoke(ptf_mode: str) -> None:
    selections = _all_supported()
    lidar, camera, transforms, lidar_available, camera_available = _inputs(selections)
    module = ResilientV2XFeatureFusion(
        BEVGridSpec(0.0, 0.0, 1.0, 4, 4),
        ptf_mode=ptf_mode,
        routing_mode="static",
    )

    output = module(
        lidar,
        camera,
        transforms,
        transforms,
        lidar_available,
        camera_available,
        selections,
    )

    assert torch.isfinite(output.fused).all()
    assert torch.allclose(output.routing_weights.sum(dim=1), torch.ones(2))
