from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.resilient_v2x.profile import (
    _deployment_student_state_dict,
    _device_metadata,
)
from transvision.evaluation.resilient_v2x_evidence import (
    EVIDENCE_DOCUMENT_TYPE,
    EvidenceError,
    build_evidence_document,
    canonical_json_bytes,
    capture_git_state,
    digest_artifact,
    read_document,
    seal_document,
    verify_document,
    write_document,
)
from transvision.evaluation.resilient_v2x_profile import (
    GpuMemoryProfile,
    LatencyProfile,
    ParameterProfile,
    ProfilingError,
    RuntimeProfile,
    build_profile_document,
    count_parameters,
    profile_runtime,
    profile_torch_flops,
    unavailable_flops,
)


ROOT = Path(__file__).resolve().parents[2]


class _StudentTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.student = nn.Linear(3, 2, bias=False)
        self.frozen_student = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.register_buffer("student_buffer", torch.zeros(5))
        self.teacher = nn.Linear(2, 2, bias=False)
        self.teacher.register_buffer("teacher_buffer", torch.zeros(3))


class _Wrapped(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = _StudentTeacher()


def test_deployment_checkpoint_filter_removes_only_teacher_state() -> None:
    state, excluded = _deployment_student_state_dict(
        {
            "module.lidar_encoder.weight": torch.ones(1),
            "module.teacher.teacher.lidar_encoder.weight": torch.ones(1),
            "module.teacher.teacher.camera_encoder.weight": torch.ones(1),
            "module.teacher_adapter.weight": torch.ones(1),
        }
    )

    assert set(state) == {
        "lidar_encoder.weight",
        "teacher_adapter.weight",
    }
    assert excluded == 2
    with pytest.raises(RuntimeError, match="no training-only teacher"):
        _deployment_student_state_dict({"student.weight": torch.ones(1)})


def test_device_metadata_normalizes_torch_version_to_plain_string() -> None:
    class TorchVersion(str):
        pass

    fake_torch = SimpleNamespace(
        __version__=TorchVersion("2.10.0+cu128"),
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(
                name="test-gpu",
                total_memory=123,
                major=12,
                minor=0,
            ),
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
    )

    metadata = _device_metadata(fake_torch, SimpleNamespace(index=None))

    assert metadata["torch_version"] == "2.10.0+cu128"
    assert type(metadata["torch_version"]) is str


def _minimal_profile() -> dict[str, object]:
    parameters = ParameterProfile(
        parameter_count=1,
        trainable_parameter_count=1,
        parameter_bytes=4,
        buffer_count=0,
        buffer_bytes=0,
        excluded_parameter_count=0,
        excluded_parameter_bytes=0,
        excluded_buffer_count=0,
        excluded_buffer_bytes=0,
        excluded_module_paths=("teacher",),
    )
    latency = LatencyProfile(
        warmup_iterations=0,
        measured_iterations=1,
        samples_ms=(1.0,),
        mean_ms=1.0,
        median_ms=1.0,
        p90_ms=1.0,
        minimum_ms=1.0,
        maximum_ms=1.0,
        synchronized=True,
    )
    runtime = RuntimeProfile(
        latency=latency,
        gpu_memory=GpuMemoryProfile(
            status="unavailable",
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            method=None,
            reason="test backend has no GPU",
        ),
    )
    return build_profile_document(
        parameters=parameters,
        runtime=runtime,
        flops=unavailable_flops("not measured in this unit test"),
        boundary="unit-test callable",
        batch_size=1,
        device={"type": "test"},
    )


def test_parameter_profile_excludes_teacher_but_counts_frozen_student_state() -> None:
    profile = count_parameters(_Wrapped())

    assert profile.parameter_count == 10  # 3*2 trainable + 4 frozen
    assert profile.trainable_parameter_count == 6
    assert profile.parameter_bytes == 40
    assert profile.buffer_count == 5
    assert profile.buffer_bytes == 20
    assert profile.excluded_parameter_count == 4
    assert profile.excluded_parameter_bytes == 16
    assert profile.excluded_buffer_count == 3
    assert profile.excluded_buffer_bytes == 12

    with pytest.raises(ProfilingError, match="sequence of paths"):
        count_parameters(_Wrapped(), excluded_module_paths="teacher")


def test_runtime_profile_has_exact_call_boundary_and_interpolated_p90() -> None:
    runs: list[str] = []
    synchronizations: list[str] = []
    resets: list[str] = []
    clock_values = iter((0, 1_000_000, 10_000_000, 13_000_000, 20_000_000, 25_000_000))

    profile = profile_runtime(
        lambda: runs.append("run"),
        warmup_iterations=2,
        measured_iterations=3,
        synchronize=lambda: synchronizations.append("sync"),
        clock_ns=lambda: next(clock_values),
        reset_peak_memory=lambda: resets.append("reset"),
        peak_allocated_bytes=lambda: 123,
        peak_reserved_bytes=lambda: 456,
        memory_method="fake exact backend",
    )

    assert len(runs) == 5
    assert len(synchronizations) == 9
    assert resets == ["reset"]
    assert profile.latency.samples_ms == (1.0, 3.0, 5.0)
    assert profile.latency.mean_ms == pytest.approx(3.0)
    assert profile.latency.median_ms == pytest.approx(3.0)
    assert profile.latency.p90_ms == pytest.approx(4.6)
    assert profile.latency.synchronized is True
    assert profile.gpu_memory.peak_allocated_bytes == 123
    assert profile.gpu_memory.peak_reserved_bytes == 456


def test_runtime_profile_without_memory_backend_is_explicitly_unavailable() -> None:
    clock_values = iter((0, 0))
    profile = profile_runtime(
        lambda: None,
        warmup_iterations=0,
        measured_iterations=1,
        clock_ns=lambda: next(clock_values),
    )

    assert profile.latency.synchronized is False
    assert profile.gpu_memory.status == "unavailable"
    assert profile.gpu_memory.peak_allocated_bytes is None


def test_runtime_profile_rejects_partial_memory_hooks() -> None:
    with pytest.raises(ProfilingError, match="supplied together"):
        profile_runtime(
            lambda: None,
            measured_iterations=1,
            reset_peak_memory=lambda: None,
        )


def test_torch_profiler_reports_supported_linear_flops_or_a_reason() -> None:
    layer = nn.Linear(4, 3, bias=False)
    inputs = torch.ones(2, 4)

    profile = profile_torch_flops(
        lambda: layer(inputs),
        include_cuda_activity=False,
    )

    if profile.status == "measured":
        assert profile.flop_count is not None and profile.flop_count > 0
        assert profile.operator_events_with_flops > 0
    else:
        assert profile.flop_count is None
        assert profile.reason


def test_canonical_document_hash_detects_nested_metric_tampering(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"car_3d_ap_r40_0.70":42.5}', encoding="utf-8")
    metrics_artifact = digest_artifact(metrics_path, "evaluation_metrics")
    profile = _minimal_profile()
    document = build_evidence_document(
        run_id="dair-val-delay-200ms-seed-0",
        metrics={"car_3d_ap_r40_0.70": 42.5},
        complexity_profile=profile,
        artifacts=[metrics_artifact],
        conditions={"delay_ms": 200, "fault": "none"},
        measured_at_utc="2026-07-27T12:00:00Z",
    )
    output = write_document(tmp_path / "evidence.json", document)

    restored = read_document(output, expected_type=EVIDENCE_DOCUMENT_TYPE)
    assert restored == document
    provenance = restored["metric_provenance"]
    assert provenance["inference_performed_by_evidence_builder"] is False
    assert provenance["metric_recomputation_performed_by_evidence_builder"] is False
    tampered = json.loads(json.dumps(document))
    tampered["metrics"]["car_3d_ap_r40_0.70"] = 99.0
    with pytest.raises(EvidenceError, match="hash mismatch"):
        verify_document(tampered)


def test_document_hash_is_mapping_order_independent_and_reserved_fields_fail() -> None:
    first = seal_document("test_document", {"b": 2, "a": {"y": 1, "x": 0}})
    second = seal_document("test_document", {"a": {"x": 0, "y": 1}, "b": 2})

    assert first["content_sha256"] == second["content_sha256"]
    assert first["content_sha256"] == hashlib.sha256(
        canonical_json_bytes({
            "schema_version": 1,
            "document_type": "test_document",
            "a": {"x": 0, "y": 1},
            "b": 2,
        })
    ).hexdigest()
    with pytest.raises(EvidenceError, match="reserved"):
        seal_document("test_document", {"schema_version": 999})


def test_read_document_rejects_noncanonical_json_even_with_valid_hash(tmp_path: Path) -> None:
    document = _minimal_profile()
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    with pytest.raises(EvidenceError, match="not canonical"):
        read_document(path)


def test_prediction_role_must_reference_hashed_existing_predictions(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text("{}", encoding="utf-8")

    with pytest.raises(EvidenceError, match="does not identify"):
        build_evidence_document(
            run_id="test-run",
            metrics={"metric": 1.0},
            complexity_profile=_minimal_profile(),
            artifacts=[digest_artifact(artifact_path, "evaluation_metrics")],
            prediction_artifact_role="evaluation_predictions",
        )


def test_git_state_hash_captures_tracked_patch_and_untracked_content(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
        )

    git("init")
    git("config", "user.name", "Evidence Test")
    git("config", "user.email", "evidence@example.invalid")
    tracked = repository / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")

    clean = capture_git_state(repository)
    tracked.write_text("changed\n", encoding="utf-8")
    (repository / "new.txt").write_text("new evidence\n", encoding="utf-8")
    dirty = capture_git_state(repository)

    assert clean["dirty"] is False
    assert dirty["dirty"] is True
    assert dirty["working_tree_sha256"] != clean["working_tree_sha256"]
    assert dirty["tracked_patch_size_bytes"] > 0
    assert dirty["untracked_files"] == [
        {
            "path": "new.txt",
            "size_bytes": 13,
            "sha256": hashlib.sha256(b"new evidence\n").hexdigest(),
        }
    ]


def test_build_evidence_cli_uses_existing_files_without_mmengine(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.pkl"
    profile = tmp_path / "profile.json"
    output = tmp_path / "evidence.json"
    metrics.write_text('{"metrics":{"ap":12.5}}', encoding="utf-8")
    predictions.write_bytes(b"predictions-from-the-existing-evaluation-run")
    write_document(profile, _minimal_profile())

    result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/build_evidence.py",
            "--run-id",
            "existing-run",
            "--metrics",
            str(metrics),
            "--metrics-key",
            "metrics",
            "--profile",
            str(profile),
            "--predictions",
            str(predictions),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    evidence = read_document(output, expected_type=EVIDENCE_DOCUMENT_TYPE)
    assert summary["inference_performed"] is False
    assert summary["metric_recomputation_performed"] is False
    assert evidence["metrics"] == {"ap": 12.5}
    assert evidence["metric_provenance"]["prediction_artifact_role"] == (
        "evaluation_predictions"
    )
