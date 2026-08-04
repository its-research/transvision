from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "tools/resilient_v2x" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluation_contract(sample_ids: list[str]) -> dict[str, object]:
    from transvision.dataset.resilient_v2x_manifest import canonical_json_bytes

    return {
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(
            canonical_json_bytes(sample_ids)
        ).hexdigest(),
    }


def _prediction_sample(
    sample_id: str, *, invalid_box: bool = False
) -> dict[str, object]:
    predicted_boxes = (
        [[1.0, 2.0, 0.0, 4.0, 2.0, 2.0]]
        if invalid_box
        else [[1.0, 2.0, 0.0, 4.0, 2.0, 2.0, 0.0]]
    )
    return {
        "sample_id": sample_id,
        "predicted_boxes_lidar_bottom_center": predicted_boxes,
        "predicted_scores": [0.9],
        "predicted_labels": [0],
        "ground_truth_boxes_lidar_bottom_center": [[1.0, 2.0, 0.0, 4.0, 2.0, 2.0, 0.0]],
        "ground_truth_labels": [0],
        "diagnostic": {
            "method": "resilient_v2x",
            "overall_supported": True,
            "branches": [
                {"agent": "ego", "modality": "lidar", "supported": True},
                {"agent": "rsu", "modality": "lidar", "supported": True},
                {"agent": "ego", "modality": "camera", "supported": True},
                {"agent": "rsu", "modality": "camera", "supported": True},
            ],
            "routing": {
                "expert_support": [True, True, True, True],
                "weights": [0.25, 0.25, 0.25, 0.25],
                "not_applicable_reason": None,
            },
        },
    }


def _write_predictions(
    path: Path,
    sample_ids: list[str],
    *,
    invalid_box: bool = False,
) -> dict[str, object]:
    from transvision.evaluation.resilient_v2x_evidence import (
        seal_document,
        write_document,
    )

    document = seal_document(
        "resilient_v2x_predictions",
        {
            "coordinate_convention": (
                "[x,y,z_bottom,length,width,height,yaw] in current ego LiDAR"
            ),
            "point_cloud_range": [0.0, -40.0, -3.0, 80.0, 40.0, 1.0],
            "iou_thresholds": [0.5, 0.7],
            "max_detections": 100,
            "sample_count": len(sample_ids),
            "samples": [
                _prediction_sample(sample_id, invalid_box=invalid_box)
                for sample_id in sample_ids
            ],
        },
    )
    write_document(path, document)
    return document


def _metrics(sample_count: int = 2) -> dict[str, object]:
    return {
        "resilient_v2x/sample_count": float(sample_count),
        "resilient_v2x/car_ground_truth_count": float(sample_count),
        "resilient_v2x/car_prediction_count": float(sample_count),
        "resilient_v2x/car_bev_ap_r40_0.50": 50.0,
        "resilient_v2x/car_bev_ap_r40_0.70": 40.0,
        "resilient_v2x/car_3d_ap_r40_0.50": 30.0,
        "resilient_v2x/car_3d_ap_r40_0.70": 20.0,
        "resilient_v2x/unsupported_sample_count": 0.0,
    }


def _write_scalars(root: Path, metrics: dict[str, object], name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": 1, **metrics}
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_upload_selection_matches_manifest_runtime_payload() -> None:
    upload = _load_script("upload_clearml_dataset.py")
    manifest, _ = upload._load_manifest(
        ROOT / "artifacts/resilient_v2x/dair/temporal_manifest_v2.json"
    )
    paths = upload._runtime_relative_paths(manifest)
    assert len(paths) == 25_646
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert sum(path.endswith(".bin") for path in paths) == 13_208
    assert sum(path.endswith(".jpg") for path in paths) == 12_438


def test_v2_overlays_bind_current_manifest() -> None:
    runner = _load_script("clearml_train.py")
    artifact_root = ROOT / "artifacts/resilient_v2x/dair_v2"
    training = json.loads(
        (artifact_root / "training_overlays.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (artifact_root / "evaluation_overlays.json").read_text(encoding="utf-8")
    )
    assert (
        training["temporal_manifest_sha256"] == runner.EXPECTED_MANIFEST_CONTENT_SHA256
    )
    assert (
        evaluation["temporal_manifest_sha256"]
        == runner.EXPECTED_MANIFEST_CONTENT_SHA256
    )


def test_validation_matrix_has_twelve_unique_configs() -> None:
    runner = _load_script("clearml_train.py")
    configs = {
        runner._condition_config(delay, condition)
        for delay in runner.DELAYS_MS
        for condition in runner.CONDITIONS
    }
    assert len(configs) == 12
    assert all(path.is_file() for path in configs)


def test_ddp_processes_enter_through_determinism_wrapper() -> None:
    runner = _load_script("clearml_train.py")
    command = runner._torchrun(4, "tools/train.py", "config.py")
    module_index = command.index("--module")
    assert command[module_index + 1 :] == [
        "tools.resilient_v2x.run_deterministic",
        "tools/train.py",
        "config.py",
    ]
    probe_env = runner._runtime_environment(runner.os.environ, "rtx5090")
    probe_env["PYTHONPATH"] = str(ROOT)
    probe = runner.subprocess.run(
        [
            runner.sys.executable,
            "-m",
            command[module_index + 1],
        ],
        cwd=ROOT,
        env=probe_env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 1
    assert "ValueError: a Python entry point is required" in probe.stderr
    assert "tools/resilient_v2x/run_deterministic.py" not in command


def test_determinism_wrapper_disables_tf32() -> None:
    import torch

    launcher = _load_script("run_deterministic.py")
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        launcher._disable_tf32()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
        torch.backends.cudnn.benchmark = original_benchmark


def test_determinism_wrapper_enables_trusted_checkpoint_loading_before_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_script("run_deterministic.py")
    events: list[tuple[str, object]] = []
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0")
    monkeypatch.setattr(launcher.sys, "argv", ["run_deterministic.py"])
    monkeypatch.setattr(
        launcher,
        "_disable_tf32",
        lambda: events.append(
            (
                "torch",
                launcher.os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
            )
        ),
    )
    monkeypatch.setattr(
        launcher.runpy,
        "run_path",
        lambda path, *, run_name: events.append(
            ("entry_point", (path, run_name, tuple(launcher.sys.argv)))
        ),
    )

    assert launcher.main(["entry.py", "--flag"]) == 0
    assert events == [
        ("torch", "1"),
        ("entry_point", ("entry.py", "__main__", ("entry.py", "--flag"))),
    ]


@pytest.mark.parametrize("value", ["1", "y", "yes", "true", "TRUE"])
def test_determinism_wrapper_rejects_forced_weights_only_policy(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    launcher = _load_script("run_deterministic.py")
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", value)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    with pytest.raises(RuntimeError, match="conflicts"):
        launcher.main(["entry.py"])
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in launcher.os.environ


def test_checkpoint_policy_selects_exact_final_epoch(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    best = tmp_path / "best_resilient_v2x.pth"
    penultimate = tmp_path / "teacher_epoch_49.pth"
    final = tmp_path / "teacher_epoch_50.pth"
    best.touch()
    penultimate.touch()
    final.touch()

    assert runner._checkpoint(tmp_path, "teacher", 50) == final.resolve()


def test_evaluation_contract_is_canonical_and_content_addressed() -> None:
    runner = _load_script("clearml_train.py")
    contract = runner._load_evaluation_contract(
        ROOT / "artifacts/resilient_v2x/dair_v2/evaluation_overlays.json"
    )

    assert contract["content_sha256"] == (
        runner.EXPECTED_EVALUATION_INDEX_CONTENT_SHA256
    )
    assert contract["sample_ids_sha256"] == (
        runner.EXPECTED_EVALUATION_SAMPLE_IDS_SHA256
    )
    assert len(contract["sample_ids"]) == runner.EXPECTED_EVALUATION_SAMPLE_COUNT


def test_prediction_evidence_accepts_sealed_expected_cohort(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    sample_ids = ["sample-a", "sample-b"]
    contract = _evaluation_contract(sample_ids)
    prediction_path = tmp_path / "predictions.json"
    expected = _write_predictions(prediction_path, sample_ids)

    assert runner._validate_prediction_document(prediction_path, contract) == expected


def test_prediction_evidence_rejects_wrong_sample_order(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    sample_ids = ["sample-a", "sample-b"]
    contract = _evaluation_contract(sample_ids)
    prediction_path = tmp_path / "predictions.json"
    _write_predictions(prediction_path, list(reversed(sample_ids)))

    with pytest.raises(ValueError, match="sample order"):
        runner._validate_prediction_document(prediction_path, contract)


def test_prediction_evidence_rejects_invalid_box_shape(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    sample_ids = ["sample-a"]
    contract = _evaluation_contract(sample_ids)
    prediction_path = tmp_path / "predictions.json"
    _write_predictions(prediction_path, sample_ids, invalid_box=True)

    with pytest.raises(ValueError, match="seven finite numbers"):
        runner._validate_prediction_document(prediction_path, contract)


def test_scalars_require_one_complete_finite_metric_row(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    expected = _metrics()
    scalars = _write_scalars(
        tmp_path,
        expected,
        "run/vis_data/scalars.json",
    )

    actual_path, actual_metrics = runner._metrics_from_scalars(tmp_path, 2)

    assert actual_path == scalars.resolve()
    assert actual_metrics == expected


def test_scalars_reject_incomplete_metric_row(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    incomplete = _metrics()
    incomplete.pop("resilient_v2x/car_3d_ap_r40_0.70")
    _write_scalars(tmp_path, incomplete, "run/scalars.json")

    with pytest.raises(ValueError, match="metric keys are incomplete"):
        runner._metrics_from_scalars(tmp_path, 2)


def test_scalars_reject_nonfinite_metric(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    nonfinite = _metrics()
    nonfinite["resilient_v2x/car_bev_ap_r40_0.50"] = float("nan")
    _write_scalars(tmp_path, nonfinite, "run/scalars.json")

    with pytest.raises(ValueError, match="finite numbers"):
        runner._metrics_from_scalars(tmp_path, 2)


def test_scalars_reject_ambiguous_multiple_files(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    _write_scalars(tmp_path, _metrics(), "run-a/scalars.json")
    _write_scalars(tmp_path, _metrics(), "run-b/scalars.json")

    with pytest.raises(ValueError, match="exactly one scalars.json"):
        runner._metrics_from_scalars(tmp_path, 2)


def test_metrics_accept_mmengine_timestamped_logger_fallback(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    expected = _metrics()
    logger_path = tmp_path / "run/20260803_120102.json"
    logger_path.parent.mkdir(parents=True)
    logger_path.write_text(json.dumps(expected), encoding="utf-8")

    actual_path, actual_metrics = runner._metrics_from_scalars(tmp_path, 2)

    assert actual_path == logger_path.resolve()
    assert actual_metrics == expected


def test_metrics_reject_ambiguous_mmengine_logger_fallback(tmp_path: Path) -> None:
    runner = _load_script("clearml_train.py")
    for timestamp in ("20260803_120102", "20260803_120103"):
        logger_path = tmp_path / "run" / f"{timestamp}.json"
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        logger_path.write_text(json.dumps(_metrics()), encoding="utf-8")

    with pytest.raises(ValueError, match="timestamped metric log"):
        runner._metrics_from_scalars(tmp_path, 2)


def test_formal_reference_quality_gate_rejects_high_iou_collapse() -> None:
    runner = _load_script("clearml_train.py")
    runner._validate_formal_reference_metrics(_metrics())

    low_bev = _metrics()
    low_bev["resilient_v2x/car_bev_ap_r40_0.70"] = 0.165
    with pytest.raises(RuntimeError, match="BEV AP@0.7 failed"):
        runner._validate_formal_reference_metrics(low_bev)

    zero_3d = _metrics()
    zero_3d["resilient_v2x/car_3d_ap_r40_0.70"] = 0.0
    with pytest.raises(RuntimeError, match="3D AP@0.7 failed"):
        runner._validate_formal_reference_metrics(zero_3d)


def test_condition_result_writer_is_canonical_and_sealed(tmp_path: Path) -> None:
    from transvision.evaluation.resilient_v2x_evidence import read_document

    runner = _load_script("clearml_train.py")
    result_path = tmp_path / "metrics.json"
    written = runner._write_sealed_document(
        result_path,
        runner.CONDITION_RESULT_DOCUMENT_TYPE,
        {"condition_id": "delay_000_full", "metrics": _metrics()},
    )

    assert (
        read_document(
            result_path,
            expected_type=runner.CONDITION_RESULT_DOCUMENT_TYPE,
        )
        == written
    )


def test_runtime_profile_cli_is_closed_and_defaults_to_legacy() -> None:
    runner = _load_script("clearml_train.py")

    assert runner._parser().parse_args(["--dataset-id", "dataset"]).runtime_profile == (
        "legacy"
    )
    assert (
        runner._parser()
        .parse_args(["--dataset-id", "dataset", "--runtime-profile", "rtx5090"])
        .runtime_profile
        == "rtx5090"
    )
    with pytest.raises(SystemExit):
        runner._parser().parse_args(
            ["--dataset-id", "dataset", "--runtime-profile", "unknown"]
        )


def test_vehicle_pretrain_stages_are_closed_and_best_checkpoint_is_selected(
    tmp_path: Path,
) -> None:
    runner = _load_script("clearml_train.py")
    assert (
        runner._parser()
        .parse_args(["--dataset-id", "dataset", "--stage", "vehicle"])
        .stage
        == "vehicle"
    )
    assert (
        runner._parser()
        .parse_args(["--dataset-id", "dataset", "--stage", "vehicle_teacher"])
        .stage
        == "vehicle_teacher"
    )
    checkpoint = (
        tmp_path
        / "best_resilient_v2x_car_bev_ap_r40_0.70_vehicle_epoch_17.pth"
    )
    checkpoint.write_bytes(b"vehicle checkpoint")
    assert runner._best_checkpoint(tmp_path, "vehicle") == checkpoint.resolve()


def test_rtx5090_runtime_keeps_local_visualization_backend_headless() -> None:
    runner = _load_script("clearml_train.py")

    assert runner._runtime_cfg_options("legacy") == ()
    assert runner._runtime_cfg_options("rtx5090") == (
        "visualizer._scope_=mmengine",
        "visualizer.type=Visualizer",
        "visualizer.vis_backends.0._scope_=mmengine",
    )
    with pytest.raises(ValueError, match="unknown runtime profile"):
        runner._runtime_cfg_options("unknown")


def test_rtx5090_runtime_environment_enforces_python_safe_path() -> None:
    runner = _load_script("clearml_train.py")
    base_env = {
        "EXISTING": "preserved",
        "PYTHONSAFEPATH": "0",
    }

    assert runner._runtime_environment(base_env, "legacy") == base_env
    assert runner._runtime_environment(base_env, "rtx5090") == {
        "EXISTING": "preserved",
        "PYTHONSAFEPATH": "1",
    }
    with pytest.raises(ValueError, match="unknown runtime profile"):
        runner._runtime_environment(base_env, "unknown")


def test_runtime_profile_tags_preserve_legacy_and_remove_a100_for_rtx() -> None:
    runner = _load_script("clearml_train.py")

    assert runner._runtime_tags("legacy") == list(runner.LEGACY_TASK_TAGS)
    rtx_tags = runner._merged_task_tags(
        ["existing", "A100", "a100-worker"],
        "rtx5090",
    )
    assert "existing" in rtx_tags
    assert all("a100" not in tag.casefold() for tag in rtx_tags)
    assert {"RTX5090", "sm120", "FP32"}.issubset(rtx_tags)


def test_rtx5090_first_run_rejects_amp_before_external_imports() -> None:
    runner = _load_script("clearml_train.py")

    with pytest.raises(ValueError, match="requires FP32"):
        runner.main(
            [
                "--dataset-id",
                "dataset",
                "--runtime-profile",
                "rtx5090",
                "--amp",
            ]
        )


def test_rtx5090_runtime_contract_helper_requires_exact_runtime() -> None:
    runner = _load_script("clearml_train.py")
    contract = {
        "python": [3, 12],
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "cuda_available": True,
        "gpu_count": 4,
        "capabilities": [[12, 0]] * 4,
        "torch_arch_list": ["sm_80", "sm_120"],
        "packages": dict(runner.RTX5090_EXPECTED_PACKAGES),
        "custom_ops": {name: True for name in runner.RTX5090_CUSTOM_OP_MODULES},
    }

    runner._validate_rtx5090_runtime_contract(contract)
    wrong_packages = dict(contract)
    wrong_packages["packages"] = {
        **contract["packages"],
        "mmcv": "2.2.0",
    }
    with pytest.raises(RuntimeError, match="package versions"):
        runner._validate_rtx5090_runtime_contract(wrong_packages)


@pytest.mark.parametrize(
    ("member_name", "member_type"),
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        ("fifo", tarfile.FIFOTYPE),
    ],
)
def test_rtx5090_bootstrap_rejects_unsafe_tar_members(
    tmp_path: Path,
    member_name: str,
    member_type: bytes,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    archive = tmp_path / "bundle.tar.gz"
    member = tarfile.TarInfo(member_name)
    member.type = member_type
    if member.isreg():
        member.size = 1
        payload = io.BytesIO(b"x")
    else:
        member.linkname = "target"
        payload = None
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.addfile(member, payload)

    with pytest.raises(ValueError, match="unsafe|links and special"):
        bootstrap._safe_extract_tar(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()


def test_rtx5090_bootstrap_extracts_valid_tar(tmp_path: Path) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    archive = tmp_path / "bundle.tar.gz"
    member = tarfile.TarInfo("nested/value.txt")
    member.size = 2
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(b"ok"))

    destination = bootstrap._safe_extract_tar(archive, tmp_path / "extract")

    assert (destination / "nested/value.txt").read_bytes() == b"ok"


def test_rtx5090_bootstrap_applies_hash_gated_metrics_compatibility(
    tmp_path: Path,
) -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    patched = (ROOT / "tools/resilient_v2x/clearml_train.py").read_text(
        encoding="utf-8"
    )
    baseline = patched
    for old, new in bootstrap.CLEARML_TRAIN_METRICS_REPLACEMENTS:
        assert baseline.count(new) == 1
        baseline = baseline.replace(new, old)
    assert hashlib.sha256(baseline.encode("utf-8")).hexdigest() == (
        bootstrap.CLEARML_TRAIN_BASELINE_SHA256
    )
    assert hashlib.sha256(patched.encode("utf-8")).hexdigest() == (
        bootstrap.CLEARML_TRAIN_METRICS_COMPAT_SHA256
    )

    source_root = tmp_path / "source"
    runner_path = source_root / "tools/resilient_v2x/clearml_train.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(baseline, encoding="utf-8")

    actual_path = bootstrap._apply_source_runner_metrics_compatibility(source_root)

    assert actual_path == runner_path.resolve()
    assert runner_path.read_text(encoding="utf-8") == patched


def test_rtx5090_embedded_smoke_scripts_compile() -> None:
    for script_name in ("clearml_5090_build.py", "clearml_5090_bootstrap.py"):
        module = _load_script(script_name)
        model_smoke = module.MODEL_SMOKE
        model_import = "import mmdet3d.models"
        scope_import = "from mmengine.registry import init_default_scope"
        transvision_call = "register_all_modules()"
        config_call = 'Config.fromfile("configs/resilient_v2x/dair_clean_teacher.py")'
        scope_call = 'init_default_scope(config.get("default_scope", "mmdet3d"))'
        assert "from mmdet3d.utils import register_all_modules" not in model_smoke
        assert model_smoke.index(scope_import) < model_smoke.index(scope_call)
        assert model_smoke.index(model_import) < model_smoke.index(transvision_call)
        assert model_smoke.index(transvision_call) < model_smoke.index(config_call)
        assert model_smoke.index(config_call) < model_smoke.index(scope_call)
        assert '"default_hooks.visualization": None' not in model_smoke
        assert '"visualizer.vis_backends": []' not in model_smoke
        assert '"visualizer._scope_": "mmengine"' in model_smoke
        assert '"visualizer.type": "Visualizer"' in model_smoke
        assert (
            '"visualizer.vis_backends.0._scope_": "mmengine"' in model_smoke
        )
        assert "type(visualizer) is not Visualizer" in model_smoke
        assert "type(backends[0]) is not LocalVisBackend" in model_smoke
        assert "config.visualizer.save_dir = save_dir" in model_smoke
        assert (
            'scalars_path = Path(save_dir) / "vis_data" / "scalars.json"'
            in model_smoke
        )
        assert 'visualizer.add_scalar("headless/smoke", 1.0, step=0)' in model_smoke
        compile(model_smoke, script_name, "exec")

    bootstrap = _load_script("clearml_5090_bootstrap.py")
    assert bootstrap.RUNTIME_NATIVE_SMOKE.count("point_cloud_range=") == 1
    bootstrap._compile_embedded_smoke_scripts()


def test_rtx5090_bootstrap_accepts_clearml_completed_status_variants() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")

    class CompletedStatus:
        value = "completed"

    artifacts = {name: object() for name in bootstrap.EXPECTED_ARTIFACTS}
    for status in ("completed", CompletedStatus(), lambda: "completed"):
        task = SimpleNamespace(status=status, artifacts=artifacts)
        assert bootstrap._require_completed_build_task(task) == artifacts
    with pytest.raises(RuntimeError, match="not completed"):
        bootstrap._require_completed_build_task(
            SimpleNamespace(status="in_progress", artifacts=artifacts)
        )


def test_rtx5090_bootstrap_runner_command_forwards_contract() -> None:
    bootstrap = _load_script("clearml_5090_bootstrap.py")
    args = SimpleNamespace(
        training_dataset_id="training-dataset",
        gpus=4,
        stage="validate",
        max_epochs=3,
        teacher_checkpoint=Path("teacher.pth"),
        student_checkpoint=Path("student.pth"),
        amp=True,
    )

    command = bootstrap._runner_command(
        args,
        python=Path("/opt/resilient-v2x-5090/bin/python"),
        runner=Path("/workspace/source/tools/resilient_v2x/clearml_train.py"),
    )

    assert command == [
        "/opt/resilient-v2x-5090/bin/python",
        "/workspace/source/tools/resilient_v2x/clearml_train.py",
        "--runtime-profile",
        "rtx5090",
        "--dataset-id",
        "training-dataset",
        "--gpus",
        "4",
        "--stage",
        "validate",
        "--max-epochs",
        "3",
        "--teacher-checkpoint",
        "teacher.pth",
        "--student-checkpoint",
        "student.pth",
        "--amp",
    ]
