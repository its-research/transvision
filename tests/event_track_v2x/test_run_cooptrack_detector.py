from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.event_track_v2x.run_cooptrack_detector import (
    configure, install_empty_target_normalization_guard, install_unavailable_velocity_guard,
    install_native_empty_annotation_guard,
)


class Config(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def base():
    return Config(model=Config(spatial_temporal_reason=Config()),
                  data=Config(train=Config(pipeline=[
                      {"type": "LoadMultiViewImageFromFilesInCeph", "img_root": "/official"},
                      {"type": "LoadAnnotations3D_E2E", "with_forecasting": True},
                      {"type": "CustomCollect3D", "keys": ["img", "gt_bboxes_3d", "gt_forecasting_locs"]},
                  ]), val={"ann_file": "/official/val"}, test={"ann_file": "/official/test"}),
                  runner=Config(), lr_config=Config(), log_config=Config())


def configured(cfg, batch=1, accumulation=8):
    return configure(cfg, side="vehicle-side", inputs=Path("/fit"),
                     converted=Path("/converted"), output=Path("/run"),
                     frame_count=6375, batch_size=batch, accumulation=accumulation,
                     epochs=24, pretrained=Path("/ImageNet.pth"))


def test_config_has_only_fit_data_and_exact_budget():
    cfg = configured(base())
    assert set(cfg.data).isdisjoint({"val", "test"})
    assert cfg.data.train.data_root == "/converted/vehicle-side/"
    assert cfg.data.train.pipeline[0]["img_root"] == "/fit/vehicle-side/"
    assert cfg.data.train.pipeline[1]["with_forecasting"] is False
    assert cfg.data.train.pipeline[2]["keys"] == ["img", "gt_bboxes_3d"]
    assert cfg.runner.max_iters == 153000
    assert cfg.optimizer_config["cumulative_iters"] == 8
    assert cfg.lr_config.warmup_iters == 4000
    assert cfg.model.pretrained == {"img": "/ImageNet.pth"}
    assert cfg.load_from is None and cfg.resume_from is None


def test_batch_change_preserves_effective_batch_and_epoch_count():
    cfg = configured(base(), batch=2, accumulation=4)
    assert cfg.runner.max_iters == 3188 * 24
    assert cfg.optimizer_config["cumulative_iters"] == 4


def test_rejects_effective_batch_drift():
    with pytest.raises(ValueError, match="effective batch"):
        configured(base(), batch=2, accumulation=8)


def test_a100_global_batch_and_epoch_budget_use_all_ranks():
    cfg = configure(base(), side="vehicle-side", inputs=Path("/fit"),
                    converted=Path("/converted"), output=Path("/run"),
                    frame_count=8504, batch_size=8, accumulation=1,
                    epochs=24, pretrained=Path("/ImageNet.pth"), world_size=4,
                    allow_larger_batch=True, sequence_count=46)
    assert cfg.runner.max_iters == 266 * 24
    assert cfg.lr_config.warmup_iters == 125
    assert "cumulative_iters" not in cfg.optimizer_config
    assert cfg.data.samples_per_gpu == 8
    assert cfg.model.batch_size == 8


def test_a100_batch_cannot_exceed_causal_sequence_streams():
    with pytest.raises(ValueError, match="sequence streams"):
        configure(base(), side="vehicle-side", inputs=Path("/fit"),
                  converted=Path("/converted"), output=Path("/run"),
                  frame_count=8504, batch_size=16, accumulation=1,
                  epochs=24, pretrained=Path("/ImageNet.pth"), world_size=4,
                  allow_larger_batch=True, sequence_count=46)


def test_empty_target_count_is_safe_without_changing_positive_counts():
    torch = pytest.importorskip("torch")
    module = SimpleNamespace(reduce_mean=lambda tensor: tensor)
    state = install_empty_target_normalization_guard(module)
    assert module.reduce_mean(torch.tensor([0.0])).item() == 1.0
    assert module.reduce_mean(torch.tensor([3.0])).item() == 3.0
    assert state["empty_count_calls"] == 1
    predictions = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = predictions[:0].sum() / module.reduce_mean(torch.tensor([0.0]))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(predictions.grad).all()
    with pytest.raises(FloatingPointError):
        module.reduce_mean(torch.tensor([float("nan")]))


def test_undefined_velocity_has_no_gradient_but_geometry_is_still_trained():
    torch = pytest.importorskip("torch")

    class BoxLoss(torch.nn.Module):
        def forward(self, prediction, target, weight):
            assert target.numel() > 0  # Matches legacy MMDetection L1Loss.
            return ((prediction - target).abs() * weight).sum()

    loss_fn = BoxLoss()
    state = install_unavailable_velocity_guard(loss_fn)
    prediction = torch.zeros((1, 10), requires_grad=True)
    target = torch.ones((1, 10))
    target[0, 8] = float("inf")
    loss = loss_fn(prediction, target, torch.ones_like(target))
    loss.backward()
    assert loss.item() == 8.0
    assert prediction.grad[0, :8].abs().sum().item() == 8.0
    assert prediction.grad[0, 8:].abs().sum().item() == 0.0
    assert target[0, 8].isinf()  # The source target was not rewritten.
    assert state["masked_velocity_targets"] == 1
    empty_prediction = torch.empty((0, 10), requires_grad=True)
    empty_loss = loss_fn(empty_prediction, torch.empty((0, 10)), torch.empty((0, 10)))
    empty_loss.backward()
    assert empty_loss.item() == 0.0 and state["empty_box_calls"] == 1
    target[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="geometric"):
        loss_fn(prediction, target, torch.ones_like(target))


def test_native_empty_annotation_arrays_remain_empty_and_indexable():
    np = pytest.importorskip("numpy")

    class Dataset:
        def load_annotations(self, path):
            return [{"gt_boxes": np.empty((0, 7)), "valid_flag": np.array([]),
                     "num_lidar_pts": np.array([]), "gt_inds": np.array([]),
                     "gt_velocity": np.array([])}]

        def get_ann_info(self, index):
            info = self.infos[index]
            mask = info["valid_flag"]
            assert info["gt_boxes"][mask].shape == (0, 7)
            assert info["gt_velocity"][mask].shape == (0, 2)
            return {"gt_labels_3d": np.array([]), "gt_inds": info["gt_inds"][mask]}

    install_native_empty_annotation_guard(Dataset)
    dataset = Dataset()
    dataset.infos = dataset.load_annotations("unused")
    annotation = dataset.get_ann_info(0)
    assert annotation["gt_labels_3d"].dtype == np.int64
    assert annotation["gt_inds"].dtype == np.int64
    assert annotation["gt_inds"].size == 0
