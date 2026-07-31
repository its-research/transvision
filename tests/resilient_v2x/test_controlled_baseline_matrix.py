from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest
import zstandard

from tools.resilient_v2x import evaluate_controlled_baselines as module
from transvision.dataset.resilient_v2x_manifest import content_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sealed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = content_sha256(result)
    return result


def _overlay_entry(
    path: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    raw = b"".join(
        module.canonical_json_bytes(record) + b"\n" for record in records
    )
    compressed = zstandard.ZstdCompressor(level=1).compress(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        "path": path.name,
        "record_count": len(records),
        "uncompressed_size": len(raw),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_size": len(compressed),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _protocol_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[tuple[int, str], tuple[str, str | None]]]:
    manifest = _sealed(
        {
            "schema_version": 2,
            "artifact_type": "test_temporal_manifest",
            "split_sha256": "0" * 64,
            "splits": {"val": ["sample-a"]},
        }
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setenv("RESILIENT_V2X_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESILIENT_V2X_SPLIT_SHA256", "0" * 64)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    monkeypatch.setenv("RESILIENT_V2X_DATA_ROOT", str(data_root))

    overlay_dir = tmp_path / "overlays"
    transports = []
    faults = []
    expected: dict[tuple[int, str], tuple[str, str | None]] = {}
    for delay in module.DELAYS:
        transport_name = f"transport_{delay:03d}.jsonl.zst"
        transport_entry = _overlay_entry(
            overlay_dir / transport_name,
            [{"kind": f"transport-{delay}", "sample_id": "sample-a"}],
        )
        transport_digest = str(transport_entry["uncompressed_sha256"])
        transports.append(
            {
                "delay_ms": delay,
                "overlay": transport_entry,
            }
        )
        expected[(delay, "Full")] = (transport_digest, None)
        for condition in ("L-Fail", "C-Fail"):
            fault_name = f"fault_{delay:03d}_{condition.lower()}.jsonl.zst"
            fault_entry = _overlay_entry(
                overlay_dir / fault_name,
                [
                    {
                        "kind": f"fault-{delay}-{condition}",
                        "sample_id": "sample-a",
                    }
                ],
            )
            fault_digest = str(fault_entry["uncompressed_sha256"])
            faults.append(
                {
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": "E+R",
                    "agents": ["ego", "rsu"],
                    "duration": 1,
                    "temporal_manifest_sha256": manifest["content_sha256"],
                    "transport_overlay_sha256": transport_digest,
                    "overlay": fault_entry,
                }
            )
            expected[(delay, condition)] = (transport_digest, fault_digest)

    index = _sealed(
        {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_evaluation_overlays",
            "temporal_manifest_sha256": manifest["content_sha256"],
            "delays_ms": list(module.DELAYS),
            "conditions": list(module.CONDITIONS),
            "agent_scopes": ["E+R"],
            "requested_duration": 1,
            "sample_ids": ["sample-a"],
            "sample_ids_sha256": hashlib.sha256(
                module.canonical_json_bytes(("sample-a",))
            ).hexdigest(),
            "transport_overlays": transports,
            "fault_overlays": faults,
        }
    )
    index_path = overlay_dir / "evaluation_overlays.json"
    _write_json(index_path, index)
    checkpoint = tmp_path / "baseline.pth"
    checkpoint.write_bytes(b"controlled baseline checkpoint")
    return manifest_path, index_path, checkpoint, expected


def test_plan_resolves_all_twelve_conditions_and_replaces_only_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, index, checkpoint, expected = _protocol_fixture(
        tmp_path,
        monkeypatch,
    )
    work_dir = tmp_path / "work"
    plan = module.build_evaluation_plan(
        baseline="cobevt",
        checkpoint=checkpoint,
        overlay_index=index,
        work_dir=work_dir,
    )

    assert plan["manifest"] == str(manifest.resolve())
    assert plan["data_root"] == str((tmp_path / "data_root").resolve())
    assert plan["split_sha256"] == "0" * 64
    assert plan["content_sha256"] == content_sha256(plan)
    assert plan["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    runs = plan["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 12
    assert {(item["delay_ms"], item["condition"]) for item in runs} == {
        (delay, condition)
        for delay in module.DELAYS
        for condition in module.CONDITIONS
    }

    baseline = module._load_python_config(module.BASELINE_CONFIGS["cobevt"])
    baseline_model = baseline["model"]
    for run in runs:
        delay = run["delay_ms"]
        condition = run["condition"]
        resolved = runpy.run_path(run["resolved_config"])
        source = module._load_python_config(Path(run["condition_config"]))
        assert resolved["model"] == baseline_model
        assert resolved["default_hooks"] == source["default_hooks"]
        assert resolved["env_cfg"] == source["env_cfg"]
        assert resolved["randomness"] == source["randomness"]
        assert resolved["test_cfg"] == source["test_cfg"]
        assert resolved["val_evaluator"] == source["val_evaluator"]

        dataset = resolved["test_dataloader"]["dataset"]
        source_dataset = source["test_dataloader"]["dataset"]
        for key, value in source_dataset.items():
            if key not in {
                "data_root",
                "expected_split_hash",
                "manifest_path",
                "transport_overlay_path",
                "transport_overlay_sha256",
                "fault_overlay_path",
                "fault_overlay_sha256",
            }:
                assert dataset[key] == value
        assert dataset["data_root"] == plan["data_root"]
        assert dataset["expected_split_hash"] == plan["split_sha256"]
        transport_digest, fault_digest = expected[(delay, condition)]
        assert dataset["transport_overlay_sha256"] == transport_digest
        assert dataset["fault_overlay_sha256"] == fault_digest
        assert resolved["test_evaluator"]["prediction_output"] == run["predictions"]
        if condition == "Full":
            assert dataset["fault_overlay_path"] is None
            assert run["fault_overlay"] is None
        else:
            assert dataset["fault_overlay_path"] == run["fault_overlay"]
        assert Path(run["checkpoint_sha256_file"]).read_text().strip() == (
            plan["checkpoint_sha256"]
        )
        assert Path(run["resolved_config"]).is_file()
        assert not Path(run["predictions"]).exists()

    stored_plan = json.loads((work_dir / "evaluation_plan.json").read_text())
    assert stored_plan["runs"] == runs
    assert stored_plan["content_sha256"] == content_sha256(stored_plan)
    assert stored_plan["plan_path"] == str(
        (work_dir / "evaluation_plan.json").resolve()
    )


def test_manifest_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    index["temporal_manifest_sha256"] = "f" * 64
    index["content_sha256"] = content_sha256(index)
    _write_json(index_path, index)

    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="temporal manifest content hash does not match",
    ):
        module.build_evaluation_plan(
            baseline="v2x_vit",
            checkpoint=checkpoint,
            overlay_index=index_path,
            work_dir=tmp_path / "work",
        )


def test_data_root_and_split_identity_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("RESILIENT_V2X_DATA_ROOT")
    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="RESILIENT_V2X_DATA_ROOT",
    ):
        module.build_evaluation_plan(
            baseline="v2x_vit",
            checkpoint=checkpoint,
            overlay_index=index,
            work_dir=tmp_path / "missing-root",
        )

    monkeypatch.setenv("RESILIENT_V2X_DATA_ROOT", str(tmp_path / "data_root"))
    monkeypatch.setenv("RESILIENT_V2X_SPLIT_SHA256", "1" * 64)
    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="does not match temporal manifest",
    ):
        module.build_evaluation_plan(
            baseline="v2x_vit",
            checkpoint=checkpoint,
            overlay_index=index,
            work_dir=tmp_path / "wrong-split",
        )


def test_compressed_overlay_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    relative = index["transport_overlays"][0]["overlay"]["path"]
    overlay = index_path.parent / relative
    overlay.write_bytes(b"X" * overlay.stat().st_size)

    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="compressed SHA-256 mismatch",
    ):
        module.build_evaluation_plan(
            baseline="cobevt",
            checkpoint=checkpoint,
            overlay_index=index_path,
            work_dir=tmp_path / "work",
        )


def test_overlay_sample_set_must_match_fixed_index_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index_path, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    entry = index["transport_overlays"][0]["overlay"]
    replacement = _overlay_entry(
        index_path.parent / entry["path"],
        [{"kind": "transport-0", "sample_id": "different-sample"}],
    )
    index["transport_overlays"][0]["overlay"] = replacement
    index["content_sha256"] = content_sha256(index)
    _write_json(index_path, index)

    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="sample IDs differ from the fixed cohort",
    ):
        module.build_evaluation_plan(
            baseline="cobevt",
            checkpoint=checkpoint,
            overlay_index=index_path,
            work_dir=tmp_path / "work",
        )


def test_dry_run_outputs_selected_plan_without_importing_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, index, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)

    def fail_if_executed(plan: object) -> object:
        raise AssertionError(f"dry-run executed Runner for {plan!r}")

    monkeypatch.setattr(module, "execute_evaluation_plan", fail_if_executed)
    work_dir = tmp_path / "dry"
    result = module.main(
        [
            "--baseline",
            "bevfusion",
            "--checkpoint",
            str(checkpoint),
            "--overlay-index",
            str(index),
            "--work-dir",
            str(work_dir),
            "--delays",
            "100",
            "300",
            "--conditions",
            "Full",
            "C-Fail",
            "--dry-run",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["delays_ms"] == [100, 300]
    assert output["conditions"] == ["Full", "C-Fail"]
    assert len(output["runs"]) == 4
    assert not (work_dir / "metrics.json").exists()
    assert all(Path(item["resolved_config"]).is_file() for item in output["runs"])
    assert all(not Path(item["predictions"]).exists() for item in output["runs"])


def test_ffnet_is_available_to_the_controlled_matrix() -> None:
    assert "ffnet" in module.BASELINES
    config = module._load_python_config(module.BASELINE_CONFIGS["ffnet"])
    assert config["model"]["baseline_name"] == "ffnet"


def test_stale_predictions_or_metrics_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, index, checkpoint, _ = _protocol_fixture(tmp_path, monkeypatch)
    work_dir = tmp_path / "work"
    plan = module.build_evaluation_plan(
        baseline="coformernet",
        checkpoint=checkpoint,
        overlay_index=index,
        work_dir=work_dir,
        delays=(0,),
        conditions=("Full",),
    )
    prediction = Path(plan["runs"][0]["predictions"])
    prediction.write_text("stale", encoding="utf-8")
    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="stale predictions evidence",
    ):
        module.build_evaluation_plan(
            baseline="coformernet",
            checkpoint=checkpoint,
            overlay_index=index,
            work_dir=work_dir,
            delays=(0,),
            conditions=("Full",),
        )

    prediction.unlink()
    (work_dir / "metrics.json").write_text("stale", encoding="utf-8")
    with pytest.raises(
        module.ControlledBaselineEvaluationError,
        match="stale metrics evidence",
    ):
        module.build_evaluation_plan(
            baseline="coformernet",
            checkpoint=checkpoint,
            overlay_index=index,
            work_dir=work_dir,
            delays=(0,),
            conditions=("Full",),
        )
