from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_bootstrap():
    path = ROOT / "tools/resilient_v2x/clearml_5090_bootstrap.py"
    spec = importlib.util.spec_from_file_location("staging_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _completed_runner_work_dir(
    tmp_path: Path,
    bootstrap,
) -> tuple[Path, dict[str, bytes]]:
    work_dir = tmp_path / "runner-work"
    work_dir.mkdir()
    checkpoint_sha256 = "a" * 64
    plan_runs: list[dict[str, object]] = []
    metric_runs: list[dict[str, object]] = []
    rewritten_configs: dict[str, bytes] = {}

    for index, condition_id in enumerate(bootstrap._controlled_condition_ids()):
        condition_dir = work_dir / condition_id
        condition_dir.mkdir()
        original_config = f"# planned config {condition_id}\nvalue = {index}\n".encode()
        rewritten_config = (
            f"# MMEngine equivalent rewrite {condition_id}\nvalue={index}\n"
        ).encode()
        rewritten_configs[condition_id] = rewritten_config
        resolved_config = condition_dir / "resolved_config.py"
        resolved_config.write_bytes(rewritten_config)
        (condition_dir / "checkpoint.sha256").write_text(
            f"{checkpoint_sha256}\n",
            encoding="ascii",
        )
        prediction_raw = (
            json.dumps({"condition_id": condition_id, "samples": [index]}) + "\n"
        ).encode()
        predictions = condition_dir / "predictions.json"
        predictions.write_bytes(prediction_raw)
        delay = (0, 100, 200, 300)[index // 3]
        condition = ("Full", "L-Fail", "C-Fail")[index % 3]
        plan_runs.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "resolved_config": str(resolved_config.resolve()),
                "resolved_config_sha256": hashlib.sha256(original_config).hexdigest(),
                "predictions": str(predictions.resolve()),
                "checkpoint_sha256_file": str(
                    (condition_dir / "checkpoint.sha256").resolve()
                ),
            }
        )
        metric_runs.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "predictions": str(predictions.resolve()),
                "prediction_sha256": hashlib.sha256(prediction_raw).hexdigest(),
            }
        )

        # Real MMEngine-style work-dir pollution which must never enter evidence.
        timestamp = condition_dir / "20260810_204200"
        vis_data = timestamp / "vis_data"
        vis_data.mkdir(parents=True)
        (timestamp / "20260810_204200.log").write_text("runner log\n")
        (timestamp / "20260810_204200.json").write_text("{}\n")
        (vis_data / "config.py").write_text("# dumped config\n")
        (vis_data / "scalars.json").write_text("{}\n")
        (condition_dir / "last_checkpoint").write_text("epoch_50.pth\n")

    plan: dict[str, object] = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "work_dir": str(work_dir.resolve()),
        "plan_path": str((work_dir / "evaluation_plan.json").resolve()),
        "metrics_output": str((work_dir / "metrics.json").resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "runs": plan_runs,
    }
    plan["content_sha256"] = bootstrap._producer_content_sha256(plan)
    metrics = {
        "complete": True,
        "planned_run_count": 12,
        "runs": metric_runs,
    }
    _write_json(work_dir / "evaluation_plan.json", plan)
    _write_json(work_dir / "metrics.json", metrics)
    (work_dir / "runner-root.log").write_text("not evidence\n")
    return work_dir, rewritten_configs


def _staged_files(stage) -> set[str]:
    return {
        path.relative_to(stage.root).as_posix()
        for path in stage.root.rglob("*")
        if path.is_file()
    }


def _archive_stage(
    stage,
    destination: Path,
    *,
    extra_member: bool = False,
    nul_member_name: bool = False,
) -> None:
    victim = "delay_000_full/predictions.json"
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
    ) as archive:
        for receipt in stage.members:
            archive_name = (
                f"{receipt.relative_path}X"
                if nul_member_name and receipt.relative_path == victim
                else receipt.relative_path
            )
            archive.write(
                stage.root / receipt.relative_path,
                arcname=archive_name,
            )
        if extra_member:
            info = zipfile.ZipInfo("runner.log", (2026, 8, 10, 20, 42, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o400) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, b"pollution\n")
    if nul_member_name:
        original = f"{victim}X".encode("ascii")
        replacement = victim.encode("ascii") + b"\x00"
        raw = destination.read_bytes()
        assert raw.count(original) == 2
        destination.write_bytes(raw.replace(original, replacement))


class _Artifact:
    def __init__(self, path: Path, *, artifact_type: str = "custom") -> None:
        self.path = path
        self.type = artifact_type
        self.url = f"http://10.100.34.118:8081/artifacts/{path.name}"
        self.copy_requests: list[dict[str, object]] = []

    def get_local_copy(self, **kwargs) -> str:
        self.copy_requests.append(dict(kwargs))
        return str(self.path)


class _MappingArtifact(_Artifact):
    def __init__(
        self,
        path: Path,
        value: object,
        *,
        artifact_type: str = "dict",
        external_url: bool = False,
    ) -> None:
        super().__init__(path, artifact_type=artifact_type)
        self.value = value
        self.get_calls = 0
        if external_url:
            self.url = f"https://example.invalid/{path.name}"

    def get(self, *, force_download: bool = False) -> object:
        del force_download
        self.get_calls += 1
        return self.value


class _UploadTask:
    def __init__(
        self,
        root: Path,
        *,
        extra_zip_member: bool = False,
        nul_zip_member: bool = False,
        drift_plan: bool = False,
        drift_metrics: bool = False,
        external_metrics_url: bool = False,
        wrong_metrics_type: bool = False,
        mutate_stage_during_upload: bool = False,
    ) -> None:
        self.root = root
        self.extra_zip_member = extra_zip_member
        self.nul_zip_member = nul_zip_member
        self.drift_plan = drift_plan
        self.drift_metrics = drift_metrics
        self.external_metrics_url = external_metrics_url
        self.wrong_metrics_type = wrong_metrics_type
        self.mutate_stage_during_upload = mutate_stage_during_upload
        self.artifacts: dict[str, object] = {"run_contract": object()}
        self.uploads: list[tuple[str, object, bool]] = []
        self.flushes = 0
        self.reloads = 0

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, artifact_object, wait_on_upload))
        if name == "evaluation_plan":
            server_path = self.root / "server-evaluation-plan.json"
            raw = Path(str(artifact_object)).read_bytes()
            if self.drift_plan:
                raw += b" \n"
            server_path.write_bytes(raw)
            self.artifacts[name] = _Artifact(server_path)
        elif name == "controlled_baseline_metrics":
            server_path = self.root / "server-controlled-metrics.json"
            server_value = json.loads(json.dumps(artifact_object))
            if self.drift_metrics:
                server_value["complete"] = False
            _write_json(server_path, server_value)
            self.artifacts[name] = _MappingArtifact(
                server_path,
                artifact_object,
                artifact_type="custom" if self.wrong_metrics_type else "dict",
                external_url=self.external_metrics_url,
            )
        elif name == "controlled_baseline_evidence":
            stage_root = Path(str(artifact_object))
            stage = self.stage
            server_path = self.root / "server-controlled-evidence.zip"
            _archive_stage(
                stage,
                server_path,
                extra_member=self.extra_zip_member,
                nul_member_name=self.nul_zip_member,
            )
            self.artifacts[name] = _Artifact(server_path, artifact_type="archive")
            if self.mutate_stage_during_upload:
                condition = stage_root / "delay_000_full"
                prediction = condition / "predictions.json"
                condition.chmod(0o700)
                prediction.chmod(0o600)
                prediction.write_bytes(b"changed during upload\n")
                prediction.chmod(0o400)
                condition.chmod(0o500)
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.flushes += 1

    def reload(self) -> None:
        self.reloads += 1


def test_runner_pollution_is_excluded_and_rewritten_configs_are_resealed(
    tmp_path: Path,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, rewritten = _completed_runner_work_dir(tmp_path, bootstrap)
    stage = bootstrap._stage_controlled_baseline_evidence(
        work_dir=work_dir,
        staging_dir=tmp_path / "sealed-stage",
    )

    plan, metrics = bootstrap._verify_controlled_evidence_stage(stage)
    expected = {"evaluation_plan.json", "metrics.json"}
    for condition_id in bootstrap._controlled_condition_ids():
        expected.update(
            {
                f"{condition_id}/resolved_config.py",
                f"{condition_id}/checkpoint.sha256",
                f"{condition_id}/predictions.json",
            }
        )
    assert len(expected) == 38
    assert _staged_files(stage) == expected
    assert len(stage.members) == 38
    assert stat.S_IMODE(stage.root.stat().st_mode) == 0o500
    assert not any("20260810" in name or "vis_data" in name for name in expected)
    assert "runner-root.log" not in expected
    for run in plan["runs"]:
        condition_id = run["condition_id"]
        assert (
            run["resolved_config_sha256"]
            == hashlib.sha256(rewritten[condition_id]).hexdigest()
        )
    assert plan["content_sha256"] == bootstrap._producer_content_sha256(plan)
    assert all("resolved_config_sha256" not in run for run in metrics["runs"])
    assert "content_sha256" not in metrics
    evaluator = ROOT / "tools/resilient_v2x/evaluate_controlled_baselines.py"
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == (
        bootstrap.CONTROLLED_BASELINE_PROTOCOL_EVALUATOR_SHA256
    )


@pytest.mark.parametrize("attack", ("missing", "symlink", "fifo"))
def test_staging_rejects_missing_or_symlink_required_member(
    tmp_path: Path,
    attack: str,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    target = work_dir / "delay_000_full/predictions.json"
    target.unlink()
    if attack == "symlink":
        target.symlink_to(work_dir / "delay_000_l_fail/predictions.json")
    elif attack == "fifo":
        os.mkfifo(target)

    with pytest.raises((OSError, RuntimeError)):
        bootstrap._stage_controlled_baseline_evidence(
            work_dir=work_dir,
            staging_dir=tmp_path / "rejected-stage",
        )


@pytest.mark.parametrize("document", ("metrics", "predictions"))
def test_staging_rejects_stale_plan_or_config_hash_fields(
    tmp_path: Path,
    document: str,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    metrics_path = work_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if document == "metrics":
        metrics["runs"][0]["resolved_config_sha256"] = "b" * 64
        _write_json(metrics_path, metrics)
    else:
        prediction_path = work_dir / "delay_000_full/predictions.json"
        prediction = json.loads(prediction_path.read_text())
        prediction["evaluation_plan_content_sha256"] = "b" * 64
        _write_json(prediction_path, prediction)
        metrics["runs"][0]["prediction_sha256"] = hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
        _write_json(metrics_path, metrics)

    with pytest.raises(RuntimeError, match="stale plan/config hash fields"):
        bootstrap._stage_controlled_baseline_evidence(
            work_dir=work_dir,
            staging_dir=tmp_path / "stale-hash-stage",
        )


def test_staging_rejects_source_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    original = bootstrap._copy_fd_payload
    attacked = False

    def mutate_after_copy(source: int, destination: int):
        nonlocal attacked
        result = original(source, destination)
        if not attacked:
            attacked = True
            (work_dir / "metrics.json").write_bytes(b'{"changed":true}\n')
        return result

    monkeypatch.setattr(bootstrap, "_copy_fd_payload", mutate_after_copy)
    with pytest.raises(RuntimeError, match="changed while being staged"):
        bootstrap._stage_controlled_baseline_evidence(
            work_dir=work_dir,
            staging_dir=tmp_path / "toctou-stage",
        )


@pytest.mark.parametrize("attack", ("parent_mkdir", "stage_secure_open"))
def test_staging_setup_failure_closes_source_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    original = bootstrap._open_secure_evidence_root
    source_descriptors: list[int] = []
    calls = 0

    def observe_or_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if attack == "stage_secure_open" and calls == 2:
            raise OSError("injected secure stage open failure")
        result = original(*args, **kwargs)
        if calls == 1:
            source_descriptors.append(result[1])
        return result

    monkeypatch.setattr(bootstrap, "_open_secure_evidence_root", observe_or_fail)
    if attack == "parent_mkdir":
        blocked_parent = tmp_path / "not-a-directory"
        blocked_parent.write_text("block mkdir\n")
        staging_dir = blocked_parent / "stage"
    else:
        staging_dir = tmp_path / "stage-open-failure"

    with pytest.raises(OSError):
        bootstrap._stage_controlled_baseline_evidence(
            work_dir=work_dir,
            staging_dir=staging_dir,
        )
    assert len(source_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(source_descriptors[0])


def test_stage_parent_failure_closes_source_member_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir()
    _, source_root = bootstrap._open_secure_evidence_root(
        work_dir,
        context="test source root",
    )
    _, destination_root = bootstrap._open_secure_evidence_root(
        stage_root,
        context="test destination root",
    )
    opened_member_descriptors: list[int] = []
    original = bootstrap._open_evidence_member

    def record_source_member(*args, **kwargs):
        result = original(*args, **kwargs)
        opened_member_descriptors.extend([result[0], *result[1]])
        return result

    def fail_stage_parent(*_args, **_kwargs):
        raise OSError("injected stage parent failure")

    monkeypatch.setattr(bootstrap, "_open_evidence_member", record_source_member)
    monkeypatch.setattr(bootstrap, "_open_stage_parent", fail_stage_parent)
    try:
        with pytest.raises(OSError):
            bootstrap._stage_copy_evidence_member(
                source_root,
                destination_root,
                bootstrap.PurePosixPath(
                    "delay_000_full",
                    "predictions.json",
                ),
            )
        assert opened_member_descriptors
        for descriptor in opened_member_descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        os.close(destination_root)
        os.close(source_root)


@pytest.mark.parametrize(
    ("helper", "failed_open_index"),
    (("source_member", 0), ("source_member", 1), ("stage_parent", 0)),
)
def test_secure_open_helpers_close_new_descriptor_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper: str,
    failed_open_index: int,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    _, root_descriptor = bootstrap._open_secure_evidence_root(
        work_dir,
        context="test evidence root",
    )
    original_open = bootstrap.os.open
    original_fstat = bootstrap.os.fstat
    opened: list[int] = []
    injected = False

    def record_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_selected_fstat(descriptor: int):
        nonlocal injected
        if (
            not injected
            and len(opened) > failed_open_index
            and descriptor == opened[failed_open_index]
        ):
            injected = True
            raise OSError("injected fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(bootstrap.os, "open", record_open)
    monkeypatch.setattr(bootstrap.os, "fstat", fail_selected_fstat)
    relative = bootstrap.PurePosixPath(
        "delay_000_full",
        "predictions.json",
    )
    try:
        with pytest.raises(OSError, match="injected fstat failure"):
            if helper == "source_member":
                bootstrap._open_evidence_member(
                    root_descriptor,
                    relative,
                    context="fault-injected source member",
                )
            else:
                bootstrap._open_stage_parent(
                    root_descriptor,
                    relative,
                    context="fault-injected stage parent",
                )
        assert injected is True
        assert len(opened) > failed_open_index
        for descriptor in opened:
            with pytest.raises(OSError):
                original_fstat(descriptor)
    finally:
        os.close(root_descriptor)


def test_stage_verification_rejects_extra_member(tmp_path: Path) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    stage = bootstrap._stage_controlled_baseline_evidence(
        work_dir=work_dir,
        staging_dir=tmp_path / "sealed-stage",
    )
    stage.root.chmod(0o700)
    (stage.root / "unexpected.log").write_text("pollution\n")
    stage.root.chmod(0o500)

    with pytest.raises(RuntimeError, match="root identity|exact inventory"):
        bootstrap._verify_controlled_evidence_stage(stage)


def test_upload_uses_stage_and_readback_zip_is_exact(tmp_path: Path) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    stage = bootstrap._stage_controlled_baseline_evidence(
        work_dir=work_dir,
        staging_dir=tmp_path / "sealed-stage",
    )
    task = _UploadTask(tmp_path)
    task.stage = stage

    bootstrap._upload_controlled_baseline_artifacts(
        task,
        evidence_stage=stage,
    )

    assert task.uploads[0][1] == str(stage.root / "evaluation_plan.json")
    assert task.uploads[2][1] == str(stage.root)
    assert task.uploads[2][1] != str(work_dir)
    assert task.flushes == task.reloads == 1
    metrics_artifact = task.artifacts["controlled_baseline_metrics"]
    assert isinstance(metrics_artifact, _MappingArtifact)
    assert metrics_artifact.get_calls == 0
    assert metrics_artifact.copy_requests == [
        {
            "extract_archive": False,
            "raise_on_error": True,
            "force_download": True,
        }
    ]
    assert (tmp_path / "server-evaluation-plan.json").read_bytes() == (
        stage.root / "evaluation_plan.json"
    ).read_bytes()
    with zipfile.ZipFile(tmp_path / "server-controlled-evidence.zip") as archive:
        assert set(archive.namelist()) == {
            receipt.relative_path for receipt in stage.members
        }


@pytest.mark.parametrize(
    "attack",
    (
        "extra_zip_member",
        "nul_zip_member",
        "drift_plan",
        "drift_metrics",
        "external_metrics_url",
        "wrong_metrics_type",
    ),
)
def test_server_readback_rejects_plan_or_zip_drift(
    tmp_path: Path,
    attack: str,
) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    stage = bootstrap._stage_controlled_baseline_evidence(
        work_dir=work_dir,
        staging_dir=tmp_path / "sealed-stage",
    )
    task = _UploadTask(
        tmp_path,
        extra_zip_member=attack == "extra_zip_member",
        nul_zip_member=attack == "nul_zip_member",
        drift_plan=attack == "drift_plan",
        drift_metrics=attack == "drift_metrics",
        external_metrics_url=attack == "external_metrics_url",
        wrong_metrics_type=attack == "wrong_metrics_type",
    )
    task.stage = stage

    with pytest.raises(
        RuntimeError,
        match=(
            "exact inventory|unsafe|differ from sealed staging|"
            "differs from sealed staging|must use|not a dict"
        ),
    ):
        bootstrap._upload_controlled_baseline_artifacts(
            task,
            evidence_stage=stage,
        )


def test_server_readback_refuses_downloader_that_cannot_disable_extraction(
    tmp_path: Path,
) -> None:
    bootstrap = _load_bootstrap()
    artifact_path = tmp_path / "evidence.zip"
    artifact_path.write_bytes(b"not opened")

    class LegacyArtifact:
        type = "archive"
        url = "http://10.100.34.118:8081/artifacts/evidence.zip"

        @staticmethod
        def get_local_copy(*, raise_on_error: bool, force_download: bool) -> str:
            assert raise_on_error is True
            assert force_download is True
            raise AssertionError("unsafe automatic extraction path was called")

    with pytest.raises(RuntimeError, match="cannot disable automatic archive"):
        bootstrap._read_uploaded_artifact_file(
            LegacyArtifact(),
            context="legacy evidence",
        )


def test_upload_rejects_staging_toctou(tmp_path: Path) -> None:
    bootstrap = _load_bootstrap()
    work_dir, _ = _completed_runner_work_dir(tmp_path, bootstrap)
    stage = bootstrap._stage_controlled_baseline_evidence(
        work_dir=work_dir,
        staging_dir=tmp_path / "sealed-stage",
    )
    task = _UploadTask(tmp_path, mutate_stage_during_upload=True)
    task.stage = stage

    with pytest.raises(RuntimeError, match="root identity|member drifted"):
        bootstrap._upload_controlled_baseline_artifacts(
            task,
            evidence_stage=stage,
        )
