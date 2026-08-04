from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

pytest.importorskip("mmdet3d")

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from torch import nn

from transvision.models.detectors.controlled_v2x_baseline import (
    ControlledCooperativeBaselineNet,
)
from transvision.models.detectors.resilient_v2x import (
    VehiclePointPillarsPretrainNet,
)
from transvision.models.resilient_v2x import (
    Agent,
    BranchSelection,
    Modality,
    ResilientBatchSelections,
    SourceCandidate,
    UnsupportedReason,
)


class _NeverCalledEncoder(nn.Module):
    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("precomputed histories must bypass encoders")


class _RecordingFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_support: torch.Tensor | None = None
        self.last_ages: torch.Tensor | None = None

    def forward(
        self,
        branches: torch.Tensor,
        support: torch.Tensor,
        ages: torch.Tensor,
    ) -> torch.Tensor:
        if not support.any(dim=1).all().item():
            raise ValueError("each sample requires at least one supported branch")
        self.last_support = support.detach().clone()
        self.last_ages = ages.detach().clone()
        mask = support[:, :, None, None, None].to(dtype=branches.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (branches * mask).sum(dim=1) / denominator


class _DummyHead(nn.Module):
    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"feature": features[0]}

    def loss(
        self,
        features: list[torch.Tensor],
        _samples: list[Det3DDataSample],
    ) -> dict[str, torch.Tensor]:
        return {"loss_dummy": features[0].mean()}

    def predict(
        self,
        features: list[torch.Tensor],
        samples: list[Det3DDataSample],
    ) -> list[InstanceData]:
        assert features[0].shape[0] == len(samples)
        return [InstanceData() for _ in samples]


class _RecordingLidarEncoder(nn.Module):
    def __init__(self, encoded: torch.Tensor) -> None:
        super().__init__()
        self.encoded = encoded
        self.calls = 0

    def forward(self, points: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.calls += 1
        assert len(points) == self.encoded.shape[0]
        return self.encoded


class _RecordingCameraEncoder(nn.Module):
    def __init__(self, encoded: torch.Tensor) -> None:
        super().__init__()
        self.encoded = encoded
        self.calls = 0

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_agent: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        assert images.shape[0] == intrinsics.shape[0] == camera_to_agent.shape[0]
        return self.encoded


def _supported(
    agent: Agent,
    modality: Modality,
    horizon: int,
    *,
    target_tick: int = 8,
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


def _selections() -> ResilientBatchSelections:
    return ResilientBatchSelections(
        sample_ids=("sample-a",),
        lidar_ego=(_supported(Agent.EGO, Modality.LIDAR, 2),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_supported(Agent.RSU, Modality.CAMERA, 1),),
        rsu_delay_intervals=(1.0,),
    )


def _install_fake_factory(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, int, dict[str, object]]],
) -> None:
    module = ModuleType("transvision.models.resilient_v2x.baselines")

    def build(
        name: str,
        *,
        channels: int,
        **cfg: object,
    ) -> nn.Module:
        calls.append((name, channels, cfg))
        return _RecordingFusion()

    module.build_controlled_baseline_fusion = build  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "transvision.models.resilient_v2x.baselines",
        module,
    )


def _precomputed_inputs() -> dict[str, object]:
    selections = _selections()
    lidar = torch.full((1, 2, 4, 256, 2, 2), float("nan"))
    camera = torch.full_like(lidar, float("nan"))
    lidar[0, 0, 2] = 2.0
    camera[0, 1, 1] = 4.0
    availability = torch.zeros(1, 2, 2, 4, dtype=torch.bool)
    availability[0, 0, 0, 2] = True
    availability[0, 1, 1, 1] = True
    transforms = torch.full((1, 2, 2, 4, 4, 4), float("nan"))
    transforms[0, 0, 0, 2] = torch.eye(4)
    transforms[0, 1, 1, 1] = torch.eye(4)
    return {
        "lidar_history_features": lidar,
        "camera_history_features": camera,
        "availability": availability,
        "source_to_target": transforms,
        "selections": selections,
    }


def test_detector_is_registered_and_delegates_only_controlled_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls: list[tuple[str, int, dict[str, object]]] = []
    _install_fake_factory(monkeypatch, factory_calls)
    model = ControlledCooperativeBaselineNet(
        grid_spec=dict(
            x_min=0.0,
            y_min=0.0,
            resolution=1.0,
            height=2,
            width=2,
        ),
        lidar_encoder=_NeverCalledEncoder(),
        camera_encoder=_NeverCalledEncoder(),
        bbox_head=_DummyHead(),
        baseline_name="unit-test-fusion",
        baseline_cfg={"temperature": 0.5},
    )
    inputs = _precomputed_inputs()

    fused, selected = model.extract_controlled_feature(inputs)

    assert MODELS.get("ControlledCooperativeBaselineNet") is (
        ControlledCooperativeBaselineNet
    )
    assert factory_calls == [("unit-test-fusion", 256, {"temperature": 0.5})]
    assert fused.shape == (1, 256, 2, 2)
    assert torch.equal(fused, torch.full_like(fused, 3.0))
    assert torch.equal(
        selected.support,
        torch.tensor([[True, False, False, True]]),
    )
    assert torch.equal(
        selected.ages,
        torch.tensor([[2.0, 0.0, 0.0, 1.0]]),
    )
    assert isinstance(model.fusion, _RecordingFusion)
    assert torch.equal(model.fusion.last_support, selected.support)
    assert torch.equal(model.fusion.last_ages, selected.ages)

    raw = model._forward(inputs)
    assert isinstance(raw, dict)
    assert torch.equal(raw["feature"], fused)
    losses = model.loss(inputs, [Det3DDataSample()])
    assert set(losses) == {"loss_dummy"}
    predictions = model.predict(inputs, [Det3DDataSample()])
    diagnostic = predictions[0].metainfo["controlled_baseline_diagnostics"]
    assert diagnostic["sample_id"] == "sample-a"
    assert diagnostic["method"] == "unit-test-fusion"
    assert "routing" not in diagnostic
    assert diagnostic["support"] == {
        "lidar_ego": True,
        "lidar_rsu": False,
        "camera_ego": False,
        "camera_rsu": True,
    }
    assert diagnostic["age_intervals"] == {
        "lidar_ego": 2.0,
        "lidar_rsu": None,
        "camera_ego": None,
        "camera_rsu": 1.0,
    }


def test_detector_reuses_sparse_shared_encoder_input_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls: list[tuple[str, int, dict[str, object]]] = []
    _install_fake_factory(monkeypatch, factory_calls)
    lidar_encoder = _RecordingLidarEncoder(torch.full((1, 256, 2, 2), 5.0))
    camera_encoder = _RecordingCameraEncoder(torch.full((1, 256, 2, 2), 7.0))
    model = ControlledCooperativeBaselineNet(
        grid_spec=dict(
            x_min=0.0,
            y_min=0.0,
            resolution=1.0,
            height=2,
            width=2,
        ),
        lidar_encoder=lidar_encoder,
        camera_encoder=camera_encoder,
        bbox_head=_DummyHead(),
        baseline_name="unit-test-fusion",
    )
    availability = torch.zeros(1, 2, 2, 4, dtype=torch.bool)
    availability[0, 0, 0, 2] = True
    availability[0, 1, 1, 1] = True
    transforms = torch.eye(4).view(1, 1, 1, 1, 4, 4).repeat(1, 2, 2, 4, 1, 1)
    inputs = {
        "lidar_points": (torch.zeros(1, 4),),
        "lidar_owner": torch.tensor([[0, 0, 2]]),
        "camera_images": torch.zeros(1, 3, 2, 2),
        "camera_owner": torch.tensor([[0, 1, 1]]),
        "camera_intrinsics": torch.eye(3).unsqueeze(0),
        "camera_agent_from_sensor": torch.eye(4).unsqueeze(0),
        "availability": availability,
        "source_to_target": transforms,
        "selections": _selections(),
    }

    selected = model.select_baseline_inputs(inputs)

    assert lidar_encoder.calls == 1
    assert camera_encoder.calls == 1
    assert torch.equal(
        selected.branches[0, 0],
        torch.full((256, 2, 2), 5.0),
    )
    assert torch.equal(
        selected.branches[0, 3],
        torch.full((256, 2, 2), 7.0),
    )


def test_detector_rejects_resilient_only_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, [])

    with pytest.raises(
        ValueError,
        match="unexpected controlled baseline options: teacher",
    ):
        ControlledCooperativeBaselineNet(
            grid_spec=dict(
                x_min=0.0,
                y_min=0.0,
                resolution=1.0,
                height=2,
                width=2,
            ),
            lidar_encoder=_NeverCalledEncoder(),
            camera_encoder=_NeverCalledEncoder(),
            bbox_head=_DummyHead(),
            baseline_name="unit-test-fusion",
            teacher={"type": "forbidden"},
        )


def test_all_missing_batch_reaches_fusion_and_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_factory(monkeypatch, [])
    model = ControlledCooperativeBaselineNet(
        grid_spec=dict(
            x_min=0.0,
            y_min=0.0,
            resolution=1.0,
            height=2,
            width=2,
        ),
        lidar_encoder=_NeverCalledEncoder(),
        camera_encoder=_NeverCalledEncoder(),
        bbox_head=_DummyHead(),
        baseline_name="unit-test-fusion",
    )
    selections = ResilientBatchSelections(
        sample_ids=("all-missing",),
        lidar_ego=(_unsupported(Agent.EGO, Modality.LIDAR),),
        lidar_rsu=(_unsupported(Agent.RSU, Modality.LIDAR),),
        camera_ego=(_unsupported(Agent.EGO, Modality.CAMERA),),
        camera_rsu=(_unsupported(Agent.RSU, Modality.CAMERA),),
        rsu_delay_intervals=(0.0,),
    )
    inputs = {
        "lidar_history_features": torch.full(
            (1, 2, 4, 256, 2, 2),
            float("nan"),
        ),
        "camera_history_features": torch.full(
            (1, 2, 4, 256, 2, 2),
            float("nan"),
        ),
        "availability": torch.zeros(1, 2, 2, 4, dtype=torch.bool),
        "source_to_target": torch.full(
            (1, 2, 2, 4, 4, 4),
            float("nan"),
        ),
        "selections": selections,
    }

    with pytest.raises(
        ValueError,
        match="at least one supported branch",
    ):
        model.extract_controlled_feature(inputs)


def test_vehicle_pretrain_selects_only_current_ego_lidar() -> None:
    encoded = torch.stack(
        (
            torch.full((256, 2, 2), 3.0),
            torch.full((256, 2, 2), 7.0),
        )
    )
    lidar_encoder = _RecordingLidarEncoder(encoded)
    model = VehiclePointPillarsPretrainNet(
        lidar_encoder=lidar_encoder,
        bbox_head=_DummyHead(),
    )
    points = tuple(torch.full((1, 4), float(index)) for index in range(4))
    inputs = {
        "lidar_points": points,
        "lidar_owner": torch.tensor(
            (
                (0, 1, 0),
                (0, 0, 0),
                (1, 0, 1),
                (1, 0, 0),
            ),
            dtype=torch.long,
        ),
        "availability": torch.ones(2, 2, 2, 4, dtype=torch.bool),
    }
    samples = [Det3DDataSample(), Det3DDataSample()]
    samples[0].set_metainfo({"sample_id": "vehicle-a"})
    samples[1].set_metainfo({"sample_id": "vehicle-b"})

    features = model.extract_feat(inputs)
    assert lidar_encoder.calls == 1
    assert torch.equal(features[0], encoded)
    assert set(model.loss(inputs, samples)) == {"loss_dummy"}
    predictions = model.predict(inputs, samples)
    assert [
        value.metainfo["controlled_baseline_diagnostics"]["sample_id"]
        for value in predictions
    ] == ["vehicle-a", "vehicle-b"]
    for value in predictions:
        diagnostic = value.metainfo["controlled_baseline_diagnostics"]
        assert diagnostic["method"] == "vehicle_pointpillars_pretrain"
        assert diagnostic["support"] == {
            "lidar_ego": True,
            "lidar_rsu": False,
            "camera_ego": False,
            "camera_rsu": False,
        }
        assert diagnostic["age_intervals"]["lidar_ego"] == 0.0


def test_vehicle_pretrain_rejects_missing_current_ego_payload() -> None:
    model = VehiclePointPillarsPretrainNet(
        lidar_encoder=_RecordingLidarEncoder(torch.zeros(1, 256, 2, 2)),
        bbox_head=_DummyHead(),
    )
    inputs = {
        "lidar_points": (torch.zeros(1, 4),),
        "lidar_owner": torch.tensor(((0, 1, 0),), dtype=torch.long),
        "availability": torch.ones(1, 2, 2, 4, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="exactly one current ego"):
        model.extract_feat(inputs)
