from __future__ import annotations

import copy
import re
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "resilient_v2x"
MAIN = CONFIG_ROOT / "dair_resilient_v2x.py"
TEACHER = CONFIG_ROOT / "dair_clean_teacher.py"


def _merge(base: object, update: object) -> object:
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    if update.get("_delete_") is True:
        return {
            key: copy.deepcopy(value)
            for key, value in update.items()
            if key != "_delete_"
        }
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key == "_delete_":
            continue
        merged[key] = _merge(merged.get(key), value)
    return merged


def _load_config(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, object]:
    path = path.resolve()
    if path in stack:
        raise AssertionError(f"cyclic config inheritance: {path}")
    namespace = runpy.run_path(str(path))
    bases = namespace.get("_base_", ())
    if isinstance(bases, str):
        bases = (bases,)
    assert isinstance(bases, (list, tuple))
    merged: dict[str, object] = {}
    for base in bases:
        assert isinstance(base, str)
        resolved = (path.parent / base).resolve()
        assert resolved.is_file(), f"missing config base: {resolved}"
        merged = _merge(
            merged,
            _load_config(resolved, (*stack, path)),
        )  # type: ignore[assignment]
    own = {key: value for key, value in namespace.items() if not key.startswith("_")}
    return _merge(merged, own)  # type: ignore[return-value]


def _dataset(config: dict[str, object], split: str = "test") -> dict[str, object]:
    dataloader = config[f"{split}_dataloader"]
    assert isinstance(dataloader, dict)
    dataset = dataloader["dataset"]
    assert isinstance(dataset, dict)
    return dataset


def test_every_resilient_v2x_python_config_compiles_and_inherits() -> None:
    names: set[str] = set()
    paths = sorted(CONFIG_ROOT.rglob("*.py"))
    assert paths
    for path in paths:
        compile(path.read_text(), str(path), "exec")
        config = _load_config(path)
        experiment = config.get("experiment")
        if experiment is None:
            continue
        assert isinstance(experiment, dict)
        name = experiment.get("name")
        assert isinstance(name, str) and name
        assert name not in names
        names.add(name)
        assert _dataset(config)["split"] == "val"


def test_official_mmengine_loader_accepts_every_config_when_available() -> None:
    mmengine = pytest.importorskip("mmengine")
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        config = mmengine.Config.fromfile(path)
        if "experiment" in config:
            assert config.model.type == "ResilientV2XNet"
            assert config.test_dataloader.dataset.type == ("ResilientTemporalDataset")
            assert config.test_dataloader.dataset.split == "val"


def test_main_model_matches_paper_architecture_contract() -> None:
    config = _load_config(MAIN)
    model = config["model"]
    assert isinstance(model, dict)
    assert set(model) == {
        "type",
        "grid_spec",
        "lidar_encoder",
        "camera_encoder",
        "bbox_head",
        "data_preprocessor",
        "ptf_mode",
        "routing_mode",
        "use_reliability",
        "use_delay_metadata",
        "delta_t_ms",
        "teacher",
        "teacher_checkpoint",
        "distillation",
    }
    assert model["type"] == "ResilientV2XNet"
    assert model["grid_spec"] == {
        "x_min": 0.0,
        "y_min": -40.0,
        "resolution": 0.8,
        "height": 100,
        "width": 100,
    }
    assert model["ptf_mode"] == "nonlinear"
    assert model["routing_mode"] == "dynamic"
    assert model["use_reliability"] is True
    assert model["use_delay_metadata"] is True
    assert model["delta_t_ms"] == 100

    lidar = model["lidar_encoder"]
    assert isinstance(lidar, dict)
    assert lidar["type"] == "SharedPointPillarsBEVEncoder"
    assert lidar["voxel_encoder"]["type"] == "PillarFeatureNet"
    assert lidar["middle_encoder"] == {
        "type": "PointPillarsScatter",
        "in_channels": 64,
        "output_shape": (200, 200),
    }
    assert sum(lidar["neck"]["out_channels"]) == 256
    assert (lidar["output_height"], lidar["output_width"]) == (100, 100)

    camera = model["camera_encoder"]
    assert isinstance(camera, dict)
    assert camera["type"] == "SharedResNetLSSBEVEncoder"
    assert camera["image_backbone"]["depth"] == 50
    assert camera["image_backbone"]["out_indices"] == (1, 2, 3)
    assert camera["view_transform"]["type"] == "LSSTransform"
    assert camera["view_transform"]["out_channels"] == 256
    assert camera["view_transform"]["xbound"] == (0.0, 80.0, 0.8)
    assert camera["view_transform"]["ybound"] == (-40.0, 40.0, 0.8)
    assert (camera["output_height"], camera["output_width"]) == (100, 100)
    assert camera["view_transform_output_order"] == "xy"

    head = model["bbox_head"]
    assert isinstance(head, dict)
    assert head["type"] == "Anchor3DHead"
    assert head["num_classes"] == 1
    assert head["in_channels"] == head["feat_channels"] == 256
    assert "train_cfg" in head and "test_cfg" in head

    teacher = model["teacher"]
    assert isinstance(teacher, dict)
    for key in (
        "type",
        "grid_spec",
        "lidar_encoder",
        "camera_encoder",
        "bbox_head",
        "ptf_mode",
        "routing_mode",
        "use_reliability",
        "use_delay_metadata",
        "delta_t_ms",
    ):
        assert teacher[key] == model[key]
    assert "teacher" not in teacher
    assert model["distillation"] == {
        "temperature": 2.0,
        "lambda_feature": 1.0,
        "lambda_logit": 1.0,
        "head_type": "bernoulli",
        "logit_path": (0, 0),
    }


def test_dataset_and_runtime_are_strict_deterministic_and_runner_compatible() -> None:
    config = _load_config(MAIN)
    expected_dataset_keys = {
        "type",
        "manifest_path",
        "data_root",
        "expected_split_hash",
        "allow_fixture",
        "seed",
        "load_camera",
        "load_lidar",
        "camera_image_size",
        "split",
        "include_clean_teacher",
        "transport_overlay_path",
        "transport_overlay_sha256",
        "fault_overlay_path",
        "fault_overlay_sha256",
    }
    for split in ("train", "val", "test"):
        dataloader = config[f"{split}_dataloader"]
        assert isinstance(dataloader, dict)
        assert dataloader["collate_fn"] == {"type": "collate_resilient_samples"}
        assert dataloader["sampler"]["type"] == "EpochIndexSampler"
        assert dataloader["sampler"]["seed"] == 20250218
        dataset = _dataset(config, split)
        assert set(dataset) == expected_dataset_keys
        assert dataset["type"] == "ResilientTemporalDataset"
        expected_dataset_split = "val" if split == "test" else split
        assert dataset["split"] == expected_dataset_split
        assert dataset["allow_fixture"] is False
        assert dataset["load_camera"] is dataset["load_lidar"] is True
        assert dataset["camera_image_size"] == (256, 704)
    train = _dataset(config, "train")
    assert train["include_clean_teacher"] is True
    assert isinstance(train["transport_overlay_path"], str)
    assert train["transport_overlay_sha256"] is None
    assert isinstance(train["fault_overlay_path"], str)
    assert train["fault_overlay_sha256"] is None
    assert _dataset(config, "val")["transport_overlay_path"] is None
    assert _dataset(config, "test")["fault_overlay_path"] is None

    assert config["randomness"] == {
        "seed": 20250218,
        "deterministic": True,
        "diff_rank_seed": False,
    }
    assert config["env_cfg"]["cudnn_benchmark"] is False
    assert config["train_cfg"]["max_epochs"] == 50
    assert config["default_hooks"]["checkpoint"]["save_best"] == (
        "resilient_v2x/car_3d_ap_r40_0.70"
    )
    assert config["optim_wrapper"]["optimizer"] == {
        "type": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.01,
    }
    assert config["val_evaluator"] == config["test_evaluator"]
    assert config["test_evaluator"] == {
        "type": "ResilientV2XMetric",
        "iou_thresholds": (0.5, 0.7),
        "max_detections": 100,
        "point_cloud_range": [0.0, -40.0, -3.0, 80.0, 40.0, 1.0],
        "prediction_output": None,
    }


def test_clean_teacher_is_zero_latency_and_has_no_recursive_teacher() -> None:
    config = _load_config(TEACHER)
    model = config["model"]
    assert isinstance(model, dict)
    assert model["teacher"] is None
    assert model["teacher_checkpoint"] is None
    assert model["distillation"] is None
    train = _dataset(config, "train")
    assert train["include_clean_teacher"] is False
    assert train["transport_overlay_path"] is None
    assert train["transport_overlay_sha256"] is None
    assert train["fault_overlay_path"] is None
    assert train["fault_overlay_sha256"] is None
    assert config["experiment"]["stage"] == "clean_teacher"


def test_paper_delay_fault_table_is_complete_and_causally_ordered() -> None:
    found: dict[tuple[int, str], dict[str, object]] = {}
    for path in sorted((CONFIG_ROOT / "conditions").glob("*.py")):
        if path.name == "causal_fault_diagnostic.py":
            continue
        config = _load_config(path)
        condition = config["experiment"]["condition"]
        key = (condition["delay_ms"], condition["fault"])
        assert key not in found
        found[key] = config
    assert set(found) == {
        (delay, fault)
        for delay in (0, 100, 200, 300)
        for fault in ("Full", "L-Fail", "C-Fail")
    }

    for (delay, fault), config in found.items():
        condition = config["experiment"]["condition"]
        dataset = _dataset(config)
        assert dataset["transport_overlay_path"] == (
            f"artifacts/resilient_v2x/dair/val_transport_delay_{delay:03d}.jsonl.zst"
        )
        assert dataset["transport_overlay_sha256"] is None
        if fault == "Full":
            assert condition["type"] == "fixed_delay"
            assert dataset["fault_overlay_path"] is None
            assert dataset["fault_overlay_sha256"] is None
        else:
            assert condition["type"] == "causal_endpoint"
            assert condition["scope"] == "E+R"
            assert condition["duration_ticks"] == 1
            assert condition["order"] == ("arrival selection before fault masking")
            fault_slug = fault.lower().replace("-", "_")
            assert dataset["fault_overlay_path"] == (
                "artifacts/resilient_v2x/dair/"
                f"val_causal_delay_{delay:03d}_{fault_slug}.jsonl.zst"
            )
            assert dataset["fault_overlay_sha256"] is None


def test_causal_diagnostic_exposes_all_supported_protocol_axes() -> None:
    path = CONFIG_ROOT / "conditions" / "causal_fault_diagnostic.py"
    config = _load_config(path)
    matrix = config["experiment"]["supported_matrix"]
    assert matrix == {
        "delay_ms": (0, 100, 200, 300),
        "scope": ("E+R", "E-only", "R-only"),
        "modality": ("lidar", "camera"),
        "duration_ticks": (1, 2, 3, 4),
    }
    condition = config["experiment"]["condition"]
    assert condition["type"] == "causal_endpoint_continuous"
    assert condition["order"] == "arrival selection before fault masking"
    dataset = _dataset(config)
    assert isinstance(dataset["transport_overlay_path"], str)
    assert dataset["transport_overlay_sha256"] is None
    assert isinstance(dataset["fault_overlay_path"], str)
    assert dataset["fault_overlay_sha256"] is None


def test_ablation_configs_change_only_supported_model_switches() -> None:
    expected = {
        "ptf_linear.py": ("ptf_mode", "linear"),
        "ptf_none.py": ("ptf_mode", "none"),
        "router_static.py": ("routing_mode", "static"),
        "router_uniform.py": ("routing_mode", "uniform"),
        "concat_capacity_matched.py": ("routing_mode", "concat"),
        "no_reliability.py": ("use_reliability", False),
        "no_delay_metadata.py": ("use_delay_metadata", False),
    }
    baseline = _load_config(MAIN)["model"]
    assert isinstance(baseline, dict)
    for name, (field, value) in expected.items():
        model = _load_config(CONFIG_ROOT / "ablations" / name)["model"]
        assert isinstance(model, dict)
        changed = {key for key in baseline if model.get(key) != baseline.get(key)}
        assert changed == {field}
        assert model[field] == value
        assert model["teacher"] == baseline["teacher"]

    no_distillation = _load_config(CONFIG_ROOT / "ablations" / "no_distillation.py")
    model = no_distillation["model"]
    assert model["teacher"] is None
    assert model["teacher_checkpoint"] is None
    assert model["distillation"] is None
    assert _dataset(no_distillation, "train")["include_clean_teacher"] is False


def test_unreported_values_are_labeled_and_no_digest_is_fabricated() -> None:
    config = _load_config(MAIN)
    for key in (
        "implementation_choices_model",
        "implementation_choices_dataset",
        "implementation_choices_runtime",
    ):
        choices = config[key]
        assert isinstance(choices, dict)
        assert choices["status"] == (
            "implementation choice; the paper does not report these values"
        )

    dataset_choices = config["implementation_choices_dataset"]
    assert dataset_choices["controlled_manifest_splits"] == ("train", "val")
    assert dataset_choices["max_capture_skew_ms"] == 200
    assert dataset_choices["pair_identity"] == (
        "dairc-v{vehicle_frame_id}-i{infrastructure_frame_id}"
    )

    for path in CONFIG_ROOT.rglob("*.py"):
        text = path.read_text()
        assert re.search(r"[0-9a-f]{64}", text) is None
    assert _dataset(config, "train")["transport_overlay_sha256"] is None
    assert _dataset(config, "train")["fault_overlay_sha256"] is None
