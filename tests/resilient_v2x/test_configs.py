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
VEHICLE = CONFIG_ROOT / "dair_vehicle_pretrain.py"
BASELINE_ROOT = CONFIG_ROOT / "baselines"
FFNET_BASELINE = (
    ROOT / "configs" / "ffnet" / "config_basemodel_veh_only_complemented.py"
)
FFNET_OFFICIAL = (
    ROOT
    / "configs"
    / "ffnet"
    / "config_basemodel_official_3class_complemented.py"
)

CONTROLLED_BASELINES = {
    "bevfusion.py": {
        "baseline_name": "bevfusion",
        "baseline_cfg": {
            "age_decay": 0.25,
            "hidden_channels": 256,
            "norm_groups": 32,
        },
    },
    "cobevt.py": {
        "baseline_name": "cobevt",
        "baseline_cfg": {
            "age_decay": 0.25,
            "num_heads": 8,
            "dropout": 0.0,
            "mlp_ratio": 2.0,
        },
    },
    "ffnet.py": {
        "baseline_name": "ffnet",
        "baseline_cfg": {
            "age_decay": 0.0,
            "hidden_channels": 256,
            "norm_groups": 32,
            "max_displacement_per_age": 1.0,
        },
    },
    "coformernet.py": {
        "baseline_name": "coformernet",
        "baseline_cfg": {
            "age_decay": 0.25,
            "num_heads": 8,
            "window_size": 4,
            "dropout": 0.0,
            "mlp_ratio": 2.0,
        },
    },
    "v2x_vit.py": {
        "baseline_name": "v2x_vit",
        "baseline_cfg": {
            "age_decay": 0.25,
            "num_heads": 8,
            "window_sizes": (2, 4),
            "dropout": 0.0,
            "mlp_ratio": 2.0,
        },
    },
}


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
            assert config.model.type in {
                "ControlledCooperativeBaselineNet",
                "ResilientV2XNet",
                "VehiclePointPillarsPretrainNet",
            }
            assert config.test_dataloader.dataset.type == ("ResilientTemporalDataset")
            assert config.test_dataloader.dataset.split == "val"


def test_controlled_baselines_share_protocol_without_resilient_modules() -> None:
    main = _load_config(MAIN)
    main_model = main["model"]
    assert isinstance(main_model, dict)
    found = {
        path.name for path in BASELINE_ROOT.glob("*.py") if path.name != "_base_.py"
    }
    assert found == set(CONTROLLED_BASELINES)

    shared_model_fields = {
        "grid_spec",
        "lidar_encoder",
        "camera_encoder",
        "bbox_head",
        "detection_projection",
        "data_preprocessor",
    }
    forbidden_model_fields = {
        "teacher",
        "teacher_checkpoint",
        "distillation",
        "ptf_mode",
        "routing_mode",
        "use_reliability",
        "use_delay_metadata",
        "delta_t_ms",
    }
    expected_model_fields = {
        "type",
        "baseline_name",
        "baseline_cfg",
        *shared_model_fields,
    }
    for name, expected in CONTROLLED_BASELINES.items():
        config = _load_config(BASELINE_ROOT / name)
        model = config["model"]
        assert isinstance(model, dict)
        assert set(model) == expected_model_fields
        assert model["type"] == "ControlledCooperativeBaselineNet"
        assert model["baseline_name"] == expected["baseline_name"]
        assert model["baseline_cfg"] == expected["baseline_cfg"]
        assert forbidden_model_fields.isdisjoint(model)
        for field in shared_model_fields:
            assert model[field] == main_model[field]

        experiment = config["experiment"]
        assert experiment["stage"] == "controlled_baseline"
        assert experiment["protocol"] == ("causal multimodal cooperative detection")
        assert experiment["claim_status"] == (
            "controlled adaptation; not an exact source-paper reproduction"
        )

        train = _dataset(config, "train")
        assert train["include_clean_teacher"] is False
        expected_train = copy.deepcopy(_dataset(main, "train"))
        expected_train["include_clean_teacher"] = False
        assert train == expected_train
        assert _dataset(config, "val") == _dataset(main, "val")
        assert _dataset(config, "test") == _dataset(main, "test")
        assert config["val_evaluator"] == main["val_evaluator"]
        assert config["test_evaluator"] == main["test_evaluator"]
        assert config["optim_wrapper"] == main["optim_wrapper"]
        assert config["train_cfg"] == main["train_cfg"]


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
        "detection_projection",
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
        "resolution": 0.625,
        "height": 128,
        "width": 128,
    }
    assert model["ptf_mode"] == "nonlinear"
    assert model["routing_mode"] == "dynamic"
    assert model["use_reliability"] is True
    assert model["use_delay_metadata"] is True
    assert model["delta_t_ms"] == 100

    lidar = model["lidar_encoder"]
    assert isinstance(lidar, dict)
    assert lidar["type"] == "SharedPointPillarsBEVEncoder"
    assert lidar["legacy_voxel_coordinate_order"] is True
    assert lidar["voxel_encoder"]["type"] == "PillarFeatureNet"
    expected_spatial_norm = {"type": "BN", "eps": 0.001, "momentum": 0.01}
    assert lidar["voxel_encoder"]["norm_cfg"] == {
        "type": "BN1d",
        "eps": 0.001,
        "momentum": 0.01,
    }
    assert lidar["backbone"]["norm_cfg"] == expected_spatial_norm
    assert lidar["neck"]["norm_cfg"] == expected_spatial_norm
    assert lidar["middle_encoder"] == {
        "type": "PointPillarsScatter",
        "in_channels": 64,
        "output_shape": (256, 256),
    }
    assert sum(lidar["neck"]["out_channels"]) == 384
    assert lidar["output_projection"] == {
        "type": "BEVChannelProjection",
        "in_channels": 384,
        "out_channels": 256,
        "num_groups": 32,
    }
    assert lidar["output_channels"] == 256
    assert (lidar["output_height"], lidar["output_width"]) == (128, 128)

    camera = model["camera_encoder"]
    assert isinstance(camera, dict)
    assert camera["type"] == "SharedResNetLSSBEVEncoder"
    assert camera["image_backbone"]["depth"] == 50
    assert camera["image_backbone"]["out_indices"] == (1, 2, 3)
    assert camera["image_backbone"]["frozen_stages"] == 1
    assert camera["image_backbone"]["norm_eval"] is True
    assert camera["image_backbone"]["norm_cfg"] == {
        "type": "BN",
        "requires_grad": False,
    }
    assert camera["image_backbone"]["init_cfg"] == {
        "type": "Pretrained",
        "checkpoint": ("https://download.pytorch.org/models/resnet50-0676ba61.pth"),
    }
    assert camera["image_neck"]["norm_cfg"]["type"] == "GN"
    assert camera["view_transform"]["type"] == "LSSTransform"
    assert camera["view_transform"]["out_channels"] == 256
    assert camera["view_transform"]["xbound"] == (0.0, 80.0, 0.625)
    assert camera["view_transform"]["ybound"] == (-40.0, 40.0, 0.625)
    assert (camera["output_height"], camera["output_width"]) == (128, 128)
    assert camera["view_transform_output_order"] == "xy"

    head = model["bbox_head"]
    assert isinstance(head, dict)
    assert head["type"] == "Anchor3DHead"
    assert head["num_classes"] == 1
    assert head["in_channels"] == head["feat_channels"] == 384
    assert model["detection_projection"] == {
        "type": "BEVChannelProjection",
        "in_channels": 256,
        "out_channels": 384,
        "num_groups": 32,
    }
    assert "train_cfg" in head and "test_cfg" in head
    assert head["anchor_generator"]["sizes"] == [[4.35, 1.91, 1.59]]
    assert head["train_cfg"]["assigner"][0] == {
        "type": "Max3DIoUAssigner",
        "iou_calculator": {"type": "mmdet3d.BboxOverlapsNearest3D"},
        "pos_iou_thr": 0.5,
        "neg_iou_thr": 0.35,
        "min_pos_iou": 0.35,
        "ignore_iof_thr": -1,
    }
    assert head["test_cfg"]["score_thr"] == 0.05
    assert head["test_cfg"]["max_num"] == 100

    teacher = model["teacher"]
    assert isinstance(teacher, dict)
    for key in (
        "type",
        "grid_spec",
        "lidar_encoder",
        "camera_encoder",
        "bbox_head",
        "detection_projection",
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


def test_dataset_and_runtime_are_seeded_and_runner_compatible() -> None:
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
        "point_cloud_range",
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
        assert dataset["point_cloud_range"] == [
            0.0,
            -40.0,
            -3.0,
            80.0,
            40.0,
            1.0,
        ]
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
        "deterministic": False,
        "diff_rank_seed": False,
    }
    assert config["env_cfg"]["cudnn_benchmark"] is False
    assert config["train_cfg"]["max_epochs"] == 50
    assert config["train_cfg"]["val_interval"] == 10
    assert config["default_hooks"]["checkpoint"]["save_best"] == (
        "resilient_v2x/car_bev_ap_r40_0.70"
    )
    assert config["optim_wrapper"]["optimizer"] == {
        "type": "AdamW",
        "lr": 0.0001,
        "betas": (0.95, 0.99),
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


def test_vehicle_pretrain_uses_ffnet_grid_with_transfer_compatible_shapes() -> None:
    vehicle = _load_config(VEHICLE)
    main = _load_config(MAIN)
    model = vehicle["model"]
    assert model["type"] == "VehiclePointPillarsPretrainNet"
    assert model["data_preprocessor"] == main["model"]["data_preprocessor"]
    encoder = model["lidar_encoder"]
    main_encoder = main["model"]["lidar_encoder"]
    assert encoder["voxelize_cfg"]["voxel_size"] == [0.16, 0.16, 4.0]
    assert encoder["middle_encoder"]["output_shape"] == (576, 576)
    assert (encoder["output_height"], encoder["output_width"]) == (288, 288)
    assert encoder["output_projection"] is None
    assert encoder["output_channels"] == 384
    assert main_encoder["output_projection"]["in_channels"] == 384
    assert main_encoder["output_projection"]["out_channels"] == 256
    for component in ("voxel_encoder", "backbone", "neck"):
        vehicle_component = dict(encoder[component])
        main_component = dict(main_encoder[component])
        vehicle_component.pop("voxel_size", None)
        vehicle_component.pop("point_cloud_range", None)
        main_component.pop("voxel_size", None)
        main_component.pop("point_cloud_range", None)
        assert vehicle_component == main_component
    head = model["bbox_head"]
    main_head = main["model"]["bbox_head"]
    assert head["in_channels"] == head["feat_channels"] == 384
    for key in (
        "num_classes",
        "in_channels",
        "feat_channels",
        "use_direction_classifier",
        "bbox_coder",
        "loss_cls",
        "loss_bbox",
        "loss_dir",
    ):
        assert head[key] == main_head[key]
    assert head["anchor_generator"]["type"] == ("AlignedAnchor3DRangeGenerator")
    assert head["anchor_generator"]["sizes"] == [[3.9, 1.6, 1.56]]
    assert head["anchor_generator"]["ranges"][0][2] == -2.66
    assert head["anchor_generator"]["ranges"][0][5] == -2.66
    assert head["assign_per_class"] is True
    assert head["test_cfg"]["score_thr"] == 0.2
    assert head["test_cfg"]["max_num"] == 300
    assert vehicle["val_evaluator"]["max_detections"] == 300
    assert vehicle["train_cfg"]["max_epochs"] == 80
    assert vehicle["train_cfg"]["val_interval"] == 10
    assert vehicle["param_scheduler"] == [
        {
            "type": "CosineAnnealingLR",
            "T_max": 32,
            "eta_min": 0.01,
            "by_epoch": True,
            "begin": 0,
            "end": 32,
            "convert_to_iter_based": True,
        },
        {
            "type": "CosineAnnealingLR",
            "T_max": 48,
            "eta_min": 0.0000001,
            "by_epoch": True,
            "begin": 32,
            "end": 80,
            "convert_to_iter_based": True,
        },
        {
            "type": "CosineAnnealingMomentum",
            "T_max": 32,
            "eta_min": 0.85 / 0.95,
            "by_epoch": True,
            "begin": 0,
            "end": 32,
            "convert_to_iter_based": True,
        },
        {
            "type": "CosineAnnealingMomentum",
            "T_max": 48,
            "eta_min": 1,
            "begin": 32,
            "end": 80,
            "convert_to_iter_based": True,
        },
    ]
    for split in ("train", "val", "test"):
        dataset = _dataset(vehicle, split)
        assert dataset["load_lidar"] is True
        assert dataset["load_camera"] is False
        assert dataset["point_cloud_range"] == [
            0.0,
            -46.08,
            -3.0,
            92.16,
            46.08,
            1.0,
        ]
    train = _dataset(vehicle, "train")
    assert train["include_clean_teacher"] is False
    assert train["transport_overlay_path"] is None
    assert train["fault_overlay_path"] is None
    assert vehicle["default_hooks"]["checkpoint"]["filename_tmpl"] == (
        "vehicle_epoch_{}.pth"
    )
    assert vehicle["experiment"]["stage"] == "vehicle_pretrain"
    assert vehicle["vehicle_pretrain_contract"]["shared_checkpoint_prefixes"] == (
        "lidar_encoder.",
        "bbox_head.",
    )
    assert vehicle["vehicle_pretrain_contract"]["expected_global_batch_size"] == 8
    contract = vehicle["vehicle_pretrain_contract"]
    assert contract["expected_dataset_passes"] == 80
    assert contract["expected_optimizer_steps_80e"] == 48240


def test_ffnet_baseline_packs_required_3d_box_metadata_for_validation() -> None:
    config = _load_config(FFNET_BASELINE)
    for pipeline_name in ("train_pipeline", "test_pipeline"):
        pack = config[pipeline_name][-1]
        assert pack["type"] == "Pack3DDetDAIRInputs"
        assert {"sample_id", "box_mode_3d", "box_type_3d"} <= set(
            pack["meta_keys"]
        )


def test_ffnet_baseline_preserves_official_cyclic_learning_rate() -> None:
    config = _load_config(FFNET_BASELINE)
    assert config["optim_wrapper"]["optimizer"]["lr"] == pytest.approx(1e-3)
    schedulers = config["param_scheduler"]
    assert schedulers[0]["eta_min"] == pytest.approx(1e-2)
    assert schedulers[1]["eta_min"] == pytest.approx(1e-7)
    assert schedulers[0]["begin"] == 0
    assert schedulers[0]["end"] == 16
    assert schedulers[1]["begin"] == 16
    assert schedulers[1]["end"] == 40
    contract = config["ffnet_reproduction_contract"]
    assert contract["initial_learning_rate"] == pytest.approx(1e-3)
    assert contract["peak_learning_rate"] == pytest.approx(1e-2)
    assert contract["final_learning_rate"] == pytest.approx(1e-7)


def test_ffnet_official_config_restores_three_classes_and_legacy_9d_pfn() -> None:
    config = _load_config(FFNET_OFFICIAL)
    model = config["model"]
    encoder = model["voxel_encoder"]
    head = model["bbox_head"]

    assert encoder["type"] == "FFNetLegacyPillarFeatureNet"
    assert encoder["in_channels"] == 4
    assert encoder["with_cluster_center"] is True
    assert encoder["with_voxel_center"] is True
    assert encoder["with_distance"] is False
    assert encoder["legacy"] is True
    assert model["mode"] == "fusion"
    assert model["legacy_voxel_coordinate_order"] is True
    assert model["data_preprocessor"]["infrastructure_intensity_scale"] == 1.0
    assert head["num_classes"] == 3
    assert head["dir_offset"] == 0.0
    assert head["dir_limit_offset"] == 1.0
    assert head["assign_per_class"] is False
    assert head["anchor_generator"]["sizes"] == [
        [0.6, 0.8, 1.73],
        [0.6, 1.76, 1.73],
        [1.6, 3.9, 1.56],
    ]
    assert config["test_evaluator"]["car_label_index"] == 2
    contract = config["ffnet_reproduction_contract"]
    assert contract["training_world_size"] == 1
    assert contract["global_batch_size"] == 2
    assert contract["expected_optimizer_steps"] == 192920
    assert contract["inference_mode"] == "fusion"
    assert contract["prepared_infrastructure_intensity_scale"] == 1.0
    assert contract["voxel_coordinate_order"].startswith("z-y-x")

    source = (
        ROOT
        / "transvision/models/voxel_encoders/ffnet_legacy_pillar_encoder.py"
    ).read_text(encoding="utf-8")
    compile(source, "ffnet_legacy_pillar_encoder.py", "exec")
    assert "decorated_channels += 3" in source
    assert "decorated_channels += 2" in source
    assert "features[:, :, :2]" in source

    detector_source = (
        ROOT / "transvision/models/detectors/v2x_voxelnet.py"
    ).read_text(encoding="utf-8")
    assert "coordinates[:, [2, 1, 0]]" in detector_source


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
    assert dataset_choices["max_capture_skew_ms"] == 75
    assert dataset_choices["prepared_dataset_version"] == "resilient_v2x_v2"
    assert dataset_choices["pair_identity"] == (
        "dairc-v{vehicle_frame_id}-i{infrastructure_frame_id}"
    )
    runtime_choices = config["implementation_choices_runtime"]
    assert runtime_choices["deterministic_cuda"] is False
    assert "Anchor3DHead" in runtime_choices["deterministic_cuda_reason"]

    for path in CONFIG_ROOT.rglob("*.py"):
        text = path.read_text()
        assert re.search(r"[0-9a-f]{64}", text) is None
    assert _dataset(config, "train")["transport_overlay_sha256"] is None
    assert _dataset(config, "train")["fault_overlay_sha256"] is None
