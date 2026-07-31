from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest
import zstandard

from tools.resilient_v2x import train_controlled_baseline as module
from transvision.dataset.resilient_v2x_manifest import content_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sealed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = content_sha256(result)
    return result


def _overlay_descriptor(path: Path, raw: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstandard.ZstdCompressor(level=1).compress(raw)
    path.write_bytes(compressed)
    return {
        "path": path.name,
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_size": len(compressed),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "uncompressed_size": len(raw),
    }


def _protocol_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    manifest = _sealed(
        {
            "schema_version": 2,
            "artifact_type": "test_temporal_manifest",
            "split_sha256": "0" * 64,
            "splits": {"train": [], "val": []},
        }
    )
    manifest_path = tmp_path / "temporal_manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setenv("RESILIENT_V2X_MANIFEST", str(manifest_path))
    data_root = tmp_path / "prepared-data"
    data_root.mkdir()
    monkeypatch.setenv("RESILIENT_V2X_DATA_ROOT", str(data_root))
    monkeypatch.setenv("RESILIENT_V2X_SPLIT_SHA256", "0" * 64)

    overlay_dir = tmp_path / "overlays"
    transport = _overlay_descriptor(
        overlay_dir / "train_transport.jsonl.zst",
        b"sealed transport bytes",
    )
    fault = _overlay_descriptor(
        overlay_dir / "train_fault.jsonl.zst",
        b"sealed fault bytes",
    )
    index = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_training_overlays",
            "temporal_manifest_sha256": manifest["content_sha256"],
            "protocol_seed": 20250218,
            "split": "train",
            "overlays": {
                "transport": transport,
                "fault": fault,
            },
        }
    )
    index_path = overlay_dir / "training_overlays.json"
    _write_json(index_path, index)
    return manifest_path, index_path, transport, fault


def test_plan_maps_baseline_and_injects_sealed_training_inputs_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, index_path, transport, fault = _protocol_fixture(
        tmp_path,
        monkeypatch,
    )
    assert {
        "v2x_vit",
        "cobevt",
        "coformernet",
        "ffnet",
        "bevfusion",
    }.issubset(module.BASELINE_CONFIGS)
    assert all(
        path.parent == module.BASELINE_CONFIG_DIR
        for path in module.BASELINE_CONFIGS.values()
    )

    work_dir = tmp_path / "training"
    plan = module.build_training_plan(
        baseline="cobevt",
        training_index=index_path,
        work_dir=work_dir,
        seed=73,
    )
    module._verify_planned_overlay(plan, prefix="transport")
    module._verify_planned_overlay(plan, prefix="fault")
    resolved = runpy.run_path(plan["resolved_config"])

    assert resolved["model"]["type"] == "ControlledCooperativeBaselineNet"
    assert resolved["model"]["baseline_name"] == "cobevt"
    assert resolved["randomness"]["seed"] == 73
    assert resolved["resume"] is False
    assert resolved["work_dir"] == str(work_dir.resolve())
    for dataloader_name in (
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ):
        dataloader = resolved[dataloader_name]
        assert dataloader["sampler"]["seed"] == 73
        assert dataloader["dataset"]["seed"] == 73
        assert dataloader["dataset"]["manifest_path"] == str(manifest_path.resolve())
        assert dataloader["dataset"]["data_root"] == str(
            (tmp_path / "prepared-data").resolve()
        )
        assert dataloader["dataset"]["expected_split_hash"] == "0" * 64
        assert dataloader["dataset"]["include_clean_teacher"] is False

    train_dataset = resolved["train_dataloader"]["dataset"]
    assert train_dataset["transport_overlay_path"] == str(
        (index_path.parent / transport["path"]).resolve()
    )
    assert train_dataset["transport_overlay_sha256"] == transport["uncompressed_sha256"]
    assert train_dataset["fault_overlay_path"] == str(
        (index_path.parent / fault["path"]).resolve()
    )
    assert train_dataset["fault_overlay_sha256"] == fault["uncompressed_sha256"]
    for dataloader_name in ("val_dataloader", "test_dataloader"):
        dataset = resolved[dataloader_name]["dataset"]
        assert dataset["transport_overlay_path"] is None
        assert dataset["transport_overlay_sha256"] is None
        assert dataset["fault_overlay_path"] is None
        assert dataset["fault_overlay_sha256"] is None

    assert plan["seed"] == 73
    assert plan["data_root"] == str((tmp_path / "prepared-data").resolve())
    assert plan["expected_split_hash"] == "0" * 64
    assert (
        plan["manifest_content_sha256"]
        == json.loads(manifest_path.read_text())["content_sha256"]
    )
    assert (
        plan["training_index_content_sha256"]
        == json.loads(index_path.read_text())["content_sha256"]
    )
    assert (
        plan["baseline_config_sha256"]
        == hashlib.sha256(module.BASELINE_CONFIGS["cobevt"].read_bytes()).hexdigest()
    )
    assert (
        plan["resolved_config_sha256"]
        == hashlib.sha256(Path(plan["resolved_config"]).read_bytes()).hexdigest()
    )
    assert plan["content_sha256"] == content_sha256(plan)
    assert json.loads((work_dir / "training_plan.json").read_text()) == plan


def test_manifest_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    index["temporal_manifest_sha256"] = "f" * 64
    index["content_sha256"] = content_sha256(index)
    _write_json(index_path, index)

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="does not match RESILIENT_V2X_MANIFEST",
    ):
        module.build_training_plan(
            baseline="v2x_vit",
            training_index=index_path,
            work_dir=tmp_path / "training",
        )


def test_compressed_overlay_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, transport, _ = _protocol_fixture(tmp_path, monkeypatch)
    overlay = index_path.parent / str(transport["path"])
    overlay.write_bytes(b"X" * overlay.stat().st_size)

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="compressed SHA-256 mismatch",
    ):
        module.build_training_plan(
            baseline="bevfusion",
            training_index=index_path,
            work_dir=tmp_path / "training",
        )


def test_stale_checkpoint_or_log_requires_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    work_dir.mkdir()
    (work_dir / "epoch_1.pth").write_bytes(b"stale checkpoint")
    log_dir = work_dir / "20260731_120000"
    log_dir.mkdir()
    (log_dir / "train.log").write_text("stale log", encoding="utf-8")

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="contains checkpoint/log artifacts",
    ):
        module.build_training_plan(
            baseline="coformernet",
            training_index=index_path,
            work_dir=work_dir,
        )
    assert not (work_dir / "resolved_config.py").exists()
    assert not (work_dir / "training_plan.json").exists()


def test_resume_rejects_orphan_checkpoint_without_sealed_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    work_dir.mkdir()
    (work_dir / "epoch_1.pth").write_bytes(b"orphan checkpoint")

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="without an existing sealed training_plan.json",
    ):
        module.build_training_plan(
            baseline="coformernet",
            training_index=index_path,
            work_dir=work_dir,
            resume=True,
        )
    assert not (work_dir / "resolved_config.py").exists()
    assert not (work_dir / "training_plan.json").exists()


def test_dry_run_publishes_plan_without_constructing_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)

    def fail_if_executed(plan: object) -> object:
        raise AssertionError(f"dry-run constructed Runner for {plan!r}")

    monkeypatch.setattr(module, "execute_training_plan", fail_if_executed)
    work_dir = tmp_path / "dry"
    result = module.main(
        [
            "--baseline",
            "bevfusion",
            "--training-index",
            str(index_path),
            "--work-dir",
            str(work_dir),
            "--seed",
            "91",
            "--dry-run",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["baseline"] == "bevfusion"
    assert output["seed"] == 91
    assert Path(output["resolved_config"]).is_file()
    assert Path(output["plan_path"]).is_file()


def test_resume_rejects_different_existing_plan_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    module.build_training_plan(
        baseline="cobevt",
        training_index=index_path,
        work_dir=work_dir,
        seed=101,
    )
    (work_dir / "epoch_1.pth").write_bytes(b"checkpoint")
    old_plan = (work_dir / "training_plan.json").read_bytes()
    old_config = (work_dir / "resolved_config.py").read_bytes()

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="different training plan",
    ):
        module.build_training_plan(
            baseline="cobevt",
            training_index=index_path,
            work_dir=work_dir,
            seed=102,
            resume=True,
        )
    assert (work_dir / "training_plan.json").read_bytes() == old_plan
    assert (work_dir / "resolved_config.py").read_bytes() == old_config


def test_matching_resume_sets_resolved_resume_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    module.build_training_plan(
        baseline="v2x_vit",
        training_index=index_path,
        work_dir=work_dir,
        seed=17,
    )
    (work_dir / "epoch_1.pth").write_bytes(b"checkpoint")

    resumed = module.build_training_plan(
        baseline="v2x_vit",
        training_index=index_path,
        work_dir=work_dir,
        seed=17,
        resume=True,
    )
    resolved = runpy.run_path(resumed["resolved_config"])

    assert resumed["resume"] is True
    assert resolved["resume"] is True


def test_resume_rejects_effective_merged_config_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    module.build_training_plan(
        baseline="v2x_vit",
        training_index=index_path,
        work_dir=work_dir,
    )
    (work_dir / "epoch_1.pth").write_bytes(b"checkpoint")
    old_plan = (work_dir / "training_plan.json").read_bytes()
    old_config = (work_dir / "resolved_config.py").read_bytes()
    original_loader = module._load_python_config

    def changed_loader(path: Path) -> dict[str, object]:
        config = original_loader(path)
        config["randomness"]["deterministic"] = True
        return config

    monkeypatch.setattr(module, "_load_python_config", changed_loader)
    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="different training plan",
    ):
        module.build_training_plan(
            baseline="v2x_vit",
            training_index=index_path,
            work_dir=work_dir,
            resume=True,
        )
    assert (work_dir / "training_plan.json").read_bytes() == old_plan
    assert (work_dir / "resolved_config.py").read_bytes() == old_config


def test_resume_rejects_tampered_existing_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"
    module.build_training_plan(
        baseline="cobevt",
        training_index=index_path,
        work_dir=work_dir,
    )
    (work_dir / "epoch_1.pth").write_bytes(b"checkpoint")
    plan_path = work_dir / "training_plan.json"
    old_plan = plan_path.read_bytes()
    resolved_path = work_dir / "resolved_config.py"
    resolved_path.write_bytes(resolved_path.read_bytes() + b"\n# tampered\n")
    tampered_config = resolved_path.read_bytes()

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="existing resolved config changed",
    ):
        module.build_training_plan(
            baseline="cobevt",
            training_index=index_path,
            work_dir=work_dir,
            resume=True,
        )
    assert plan_path.read_bytes() == old_plan
    assert resolved_path.read_bytes() == tampered_config


def test_execute_rechecks_overlay_compressed_file_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, transport, _ = _protocol_fixture(tmp_path, monkeypatch)
    plan = module.build_training_plan(
        baseline="ffnet",
        training_index=index_path,
        work_dir=tmp_path / "training",
    )
    overlay = index_path.parent / str(transport["path"])
    overlay.write_bytes(b"X" * overlay.stat().st_size)

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="compressed SHA-256 changed after planning",
    ):
        module.execute_training_plan(plan)


def test_execute_rechecks_overlay_uncompressed_stream_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    index["overlays"]["fault"]["uncompressed_sha256"] = "f" * 64
    index["content_sha256"] = content_sha256(index)
    _write_json(index_path, index)
    plan = module.build_training_plan(
        baseline="coformernet",
        training_index=index_path,
        work_dir=tmp_path / "training",
    )

    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="uncompressed SHA-256 mismatch",
    ):
        module.execute_training_plan(plan)


def test_data_root_and_split_hash_are_required_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, _, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "training"

    monkeypatch.delenv("RESILIENT_V2X_DATA_ROOT")
    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="RESILIENT_V2X_DATA_ROOT",
    ):
        module.build_training_plan(
            baseline="ffnet",
            training_index=index_path,
            work_dir=work_dir,
        )

    monkeypatch.setenv(
        "RESILIENT_V2X_DATA_ROOT",
        str(tmp_path / "prepared-data"),
    )
    monkeypatch.setenv("RESILIENT_V2X_SPLIT_SHA256", "F" * 64)
    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="RESILIENT_V2X_SPLIT_SHA256",
    ):
        module.build_training_plan(
            baseline="ffnet",
            training_index=index_path,
            work_dir=work_dir,
        )

    monkeypatch.setenv("RESILIENT_V2X_SPLIT_SHA256", "1" * 64)
    with pytest.raises(
        module.ControlledBaselineTrainingError,
        match="does not match temporal manifest",
    ):
        module.build_training_plan(
            baseline="ffnet",
            training_index=index_path,
            work_dir=work_dir,
        )

    assert not (work_dir / "resolved_config.py").exists()
    assert not (work_dir / "training_plan.json").exists()


def test_sampler_seed_propagation_traverses_lists_and_tuples() -> None:
    tree = {
        "branches": [
            {"sampler": {"seed": 1}},
            ({"batch_sampler": {"seed": 2}},),
        ]
    }

    module._seed_sampler_tree(tree, 73)

    assert tree["branches"][0]["sampler"]["seed"] == 73
    assert tree["branches"][1][0]["batch_sampler"]["seed"] == 73
