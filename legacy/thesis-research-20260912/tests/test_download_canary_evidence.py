from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "clearml" / "download_canary_evidence.py"
SPEC = importlib.util.spec_from_file_location("download_canary_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


TASK_ID = "b" * 32
SCRIPT_SHA = "a" * 64
DIFF_SHA = "d" * 64
TASK_NAME = "rtpv2x__synth-causal-canary-v1__canary__seed3407__1234abcd"


def _event(
    *,
    source: str = "ego",
    truth_id: int = 0,
    event_time: int = 1,
    arrival_time: int = 1,
) -> dict[str, object]:
    return {
        "source": source,
        "truth_id": truth_id,
        "event_time": event_time,
        "arrival_time": arrival_time,
        "position": [1.0, 2.0],
        "velocity": [0.1, 0.2],
        "reliability": 0.8,
    }


def _metrics(*, a100: bool = True, messages_available: int = 2) -> dict[str, object]:
    naive = {
        "position_rmse": 0.6,
        "prediction_ade": 1.4,
        "prediction_fde": 1.9,
        "identity_switches": 5,
        "future_messages_consumed": 0,
        "late_messages_consumed": 1,
        "messages_available": messages_available,
    }
    reliability = {
        "position_rmse": 0.5,
        "prediction_ade": 1.1,
        "prediction_fde": 1.5,
        "identity_switches": 3,
        "future_messages_consumed": 0,
        "late_messages_consumed": 1,
        "messages_available": messages_available,
    }
    device = (
        "NVIDIA A100-SXM4-80GB, 81920, 535.129.03"
        if a100
        else "NVIDIA RTX 3060, 12288, 535.129.03"
    )
    return {
        "protocol_id": MODULE.PROTOCOL_ID,
        "scientific_claim_allowed": False,
        "diagnostic_only": True,
        "naive": naive,
        "reliability": reliability,
        "delta": {
            "position_rmse": reliability["position_rmse"] - naive["position_rmse"],
            "prediction_ade": reliability["prediction_ade"] - naive["prediction_ade"],
            "identity_switches": reliability["identity_switches"]
            - naive["identity_switches"],
        },
        "deterministic_replay": True,
        "gpu": {"available": True, "devices": [device]},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_bundle(directory: Path, *, a100: bool = True) -> tuple[dict[str, Path], str]:
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    events_path = directory / "events.jsonl"
    manifest_path = directory / "run_manifest.json"
    events = [
        _event(source="ego", truth_id=0, event_time=1, arrival_time=1),
        _event(source="rsu", truth_id=1, event_time=1, arrival_time=3),
    ]
    _write_json(metrics_path, _metrics(a100=a100, messages_available=len(events)))
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    config = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "seed": 3407,
        "steps": 72,
        "packet_loss": 0.08,
        "max_latency": 5,
        "require_gpu": True,
        "require_a100": True,
        "clearml_mode": "required",
    }
    config_sha = MODULE.canonical_sha256(config)
    manifest = {
        "schema_version": 1,
        "protocol_id": MODULE.PROTOCOL_ID,
        "scientific_claim_allowed": False,
        "diagnostic_only": True,
        "clearml_task_id": TASK_ID,
        "clearml_task_name": TASK_NAME,
        "clearml_project": MODULE.CLEARML_PROJECT,
        "clearml_required": True,
        "config": config,
        "config_sha256": config_sha,
        "script_sha256": SCRIPT_SHA,
        "standalone_diff_sha256": DIFF_SHA,
        "metrics_sha256": MODULE.file_sha256(metrics_path),
        "events_sha256": MODULE.file_sha256(events_path),
        "artifact_commit_order": ["metrics", "events", "run_manifest"],
        "python": "3.11.9 test-runtime",
        "platform": "Linux-test-x86_64",
        "pid": 123,
    }
    _write_json(manifest_path, manifest)
    return {
        "metrics": metrics_path,
        "run_manifest": manifest_path,
        "events": events_path,
    }, config_sha


class _FakeArtifact:
    def __init__(self, path: Path, *, advertise_size: bool = True) -> None:
        self.path = path
        self.size = path.stat().st_size if advertise_size else None
        self.hash = MODULE.file_sha256(path)
        self.calls: list[dict[str, object]] = []

    def get_local_copy(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return str(self.path)


class _FakeTask:
    def __init__(
        self,
        artifacts: dict[str, _FakeArtifact],
        *,
        task_id: str = TASK_ID,
        status: str = "published",
    ) -> None:
        self.id = task_id
        self.status = status
        self.artifacts = artifacts


class _NoisyFailingArtifact:
    def __init__(self, size: int, artifact_hash: str) -> None:
        self.size = size
        self.hash = artifact_hash

    def get_local_copy(self, **_: object) -> str:
        print("https://internal.invalid/?token=must-not-leak")
        print("Authorization: Bearer must-not-leak", file=sys.stderr)
        os.write(1, b"https://fd.invalid/?token=must-not-leak\n")
        os.write(2, b"Authorization: Bearer fd-must-not-leak\n")
        raise RuntimeError("must-not-leak")


class _RemoteArtifact:
    def __init__(
        self,
        source: Path,
        *,
        local_result: object = None,
        local_error: Exception | None = None,
        url: str = "https://published.invalid/files/task/metrics.json",
    ) -> None:
        self.size = source.stat().st_size
        self.hash = MODULE.file_sha256(source)
        self.url = url
        self.local_result = local_result
        self.local_error = local_error
        self.calls: list[dict[str, object]] = []

    def get_local_copy(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.local_error is not None:
            raise self.local_error
        return self.local_result


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = (
            headers if headers is not None else {"Content-Length": str(len(payload))}
        )
        self.closed = False

    def iter_content(self, chunk_size: int) -> object:
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class DownloadCanaryEvidenceTests(unittest.TestCase):
    def test_local_copy_success_never_loads_trusted_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged, _ = _build_bundle(root / "source")
            task = _FakeTask(
                {name: _FakeArtifact(path) for name, path in staged.items()}
            )
            stage = root / "stage"
            stage.mkdir()
            loader = mock.Mock(side_effect=AssertionError("fallback must not load"))
            MODULE.stage_task_artifacts(task, stage, TASK_ID, fallback_loader=loader)
            loader.assert_not_called()

    def test_failed_or_none_local_copy_uses_only_trusted_origin(self) -> None:
        for local_error in (RuntimeError("401"), None):
            with self.subTest(failure=type(local_error).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    staged, _ = _build_bundle(root / "source")
                    remote_metrics = _RemoteArtifact(
                        staged["metrics"],
                        local_error=local_error,
                        url=(
                            "https://published.invalid:9443/artifacts/task/metrics.json"
                        ),
                    )
                    artifacts = {
                        name: _FakeArtifact(path) for name, path in staged.items()
                    }
                    artifacts["metrics"] = remote_metrics
                    requests: list[tuple[str, dict[str, object]]] = []
                    response = _FakeResponse(staged["metrics"].read_bytes())

                    def request_get(url: str, **kwargs: object) -> _FakeResponse:
                        requests.append((url, kwargs))
                        return response

                    stage = root / "stage"
                    stage.mkdir()
                    MODULE.stage_task_artifacts(
                        _FakeTask(artifacts),
                        stage,
                        TASK_ID,
                        trusted_files_base="https://trusted.invalid:8081",
                        request_get=request_get,
                        auth_headers_provider=lambda: {
                            "Authorization": "Bearer test-secret"
                        },
                    )
                    self.assertEqual(
                        requests[0][0],
                        "https://trusted.invalid:8081/artifacts/task/metrics.json",
                    )
                    self.assertNotIn("published.invalid", requests[0][0])
                    self.assertEqual(
                        requests[0][1]["headers"],
                        {"Authorization": "Bearer test-secret"},
                    )
                    self.assertFalse(requests[0][1]["allow_redirects"])
                    self.assertTrue(response.closed)

    def test_trusted_url_rejects_queries_userinfo_fragments_and_traversal(self) -> None:
        bad_artifact_urls = [
            "ftp://published.invalid/file.json",
            "https://user@published.invalid/file.json",
            "https://published.invalid/file.json#fragment",
            "https://published.invalid/file.json?token=x",
            "https://published.invalid/a/../file.json",
            "https://published.invalid/a/%2e%2e/file.json",
            "https://published.invalid/a/%2Fetc/file.json",
            "https://published.invalid/a\\file.json",
        ]
        for artifact_url in bad_artifact_urls:
            with self.subTest(artifact_url=artifact_url):
                with self.assertRaises(MODULE.EvidenceValidationError):
                    MODULE.trusted_artifact_download_url(
                        artifact_url, "https://trusted.invalid:8081"
                    )
        bad_bases = [
            "ftp://trusted.invalid",
            "https://user@trusted.invalid",
            "https://trusted.invalid/base",
            "https://trusted.invalid/?token=x",
            "https://trusted.invalid/#fragment",
        ]
        for trusted_base in bad_bases:
            with self.subTest(trusted_base=trusted_base):
                with self.assertRaises(MODULE.EvidenceValidationError):
                    MODULE.trusted_artifact_download_url(
                        "https://published.invalid/file.json", trusted_base
                    )

    def test_trusted_download_rejects_redirect_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def auth() -> dict[str, str]:
                return {"Authorization": "Bearer secret"}

            cases = (
                ("redirect", _FakeResponse(b"x", status_code=302), 10),
                (
                    "oversize-header",
                    _FakeResponse(b"x", headers={"Content-Length": "11"}),
                    10,
                ),
                ("oversize-stream", _FakeResponse(b"x" * 11, headers={}), 10),
            )
            for label, response, max_bytes in cases:
                with self.subTest(label=label):
                    destination = root / f"{label}.json"
                    with self.assertRaises(MODULE.EvidenceValidationError):
                        MODULE.download_from_trusted_files_server(
                            artifact_url="https://published.invalid/file.json",
                            trusted_files_base="https://trusted.invalid:8081",
                            destination=destination,
                            max_bytes=max_bytes,
                            request_get=(
                                lambda *_args, response=response, **_kwargs: response
                            ),
                            auth_headers_provider=auth,
                        )
                    self.assertTrue(response.closed)
                    self.assertFalse(destination.exists())

    def test_trusted_fallback_still_enforces_advertised_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged, _ = _build_bundle(root / "source")
            remote_metrics = _RemoteArtifact(staged["metrics"])
            remote_metrics.hash = "0" * 64
            artifacts = {name: _FakeArtifact(path) for name, path in staged.items()}
            artifacts["metrics"] = remote_metrics
            stage = root / "stage"
            stage.mkdir()
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.stage_task_artifacts(
                    _FakeTask(artifacts),
                    stage,
                    TASK_ID,
                    trusted_files_base="https://trusted.invalid:8081",
                    request_get=lambda *_args, **_kwargs: _FakeResponse(
                        staged["metrics"].read_bytes()
                    ),
                    auth_headers_provider=lambda: {"Authorization": "Bearer secret"},
                )

    def test_valid_bundle_passes_strict_content_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged, config_sha = _build_bundle(Path(temporary))
            receipt = MODULE.validate_staged_bundle(
                staged,
                expected_task_id=TASK_ID,
                expected_protocol=MODULE.PROTOCOL_ID,
                expected_script_sha256=SCRIPT_SHA,
                expected_diff_sha256=DIFF_SHA,
                expected_config_sha256=config_sha,
                require_a100=True,
            )
        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["event_count"], 2)
        self.assertEqual(set(receipt["artifacts"]), set(MODULE.ARTIFACT_SPECS))

    def test_sensitive_keys_urls_private_ips_and_paths_are_rejected(self) -> None:
        bad_values = [
            {"token": "not-printed"},
            {"note": "Bearer not-printed"},
            {"note": "https://clearml.invalid/artifact"},
            {"note": "10.1.2.3"},
            {"note": "/internal/worker/path"},
            {"note": "C:\\internal\\worker"},
            {"note": "worker path is /home/alice/cache"},
            {"note": "worker path is C:\\Users\\alice\\cache"},
            {"note": "worker address is [fe80::1]"},
            {"note": "worker address is ::1"},
            {"note": "worker hostname is gpu-01.internal"},
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(MODULE.EvidenceValidationError):
                    MODULE.assert_no_sensitive(value)

    def test_symlink_extension_and_size_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            symlink = directory / "link.json"
            try:
                symlink.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {type(exc).__name__}")
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_regular_source(symlink, ".json", 100)

            wrong_extension = directory / "metrics.txt"
            wrong_extension.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_regular_source(wrong_extension, ".json", 100)

            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_regular_source(target, ".json", 2)

    def test_jsonl_causality_and_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            future = _event(event_time=5, arrival_time=4)
            path.write_text(json.dumps(future) + "\n", encoding="utf-8")
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_events(path)

            missing_field = _event()
            missing_field.pop("velocity")
            path.write_text(json.dumps(missing_field) + "\n", encoding="utf-8")
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_events(path)

    def test_a100_requirement_is_enforced(self) -> None:
        with self.assertRaises(MODULE.EvidenceValidationError):
            MODULE.validate_metrics(_metrics(a100=False), MODULE.PROTOCOL_ID, True)
        self.assertEqual(
            MODULE.validate_metrics(_metrics(a100=False), MODULE.PROTOCOL_ID, False),
            2,
        )

    def test_manifest_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged, config_sha = _build_bundle(Path(temporary))
            metrics = json.loads(staged["metrics"].read_text(encoding="utf-8"))
            metrics["naive"]["position_rmse"] = 9.0
            _write_json(staged["metrics"], metrics)
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_staged_bundle(
                    staged,
                    expected_task_id=TASK_ID,
                    expected_protocol=MODULE.PROTOCOL_ID,
                    expected_script_sha256=SCRIPT_SHA,
                    expected_diff_sha256=DIFF_SHA,
                    expected_config_sha256=config_sha,
                    require_a100=True,
                )

    def test_manifest_diff_and_a100_contract_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged, config_sha = _build_bundle(Path(temporary))
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_staged_bundle(
                    staged,
                    expected_task_id=TASK_ID,
                    expected_protocol=MODULE.PROTOCOL_ID,
                    expected_script_sha256=SCRIPT_SHA,
                    expected_diff_sha256="e" * 64,
                    expected_config_sha256=config_sha,
                    require_a100=True,
                )

            manifest = json.loads(staged["run_manifest"].read_text(encoding="utf-8"))
            manifest["config"]["require_a100"] = False
            config_sha = MODULE.canonical_sha256(manifest["config"])
            manifest["config_sha256"] = config_sha
            _write_json(staged["run_manifest"], manifest)
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_staged_bundle(
                    staged,
                    expected_task_id=TASK_ID,
                    expected_protocol=MODULE.PROTOCOL_ID,
                    expected_script_sha256=SCRIPT_SHA,
                    expected_diff_sha256=DIFF_SHA,
                    expected_config_sha256=config_sha,
                    require_a100=True,
                )

    def test_task_must_be_published_with_exact_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged, _ = _build_bundle(Path(temporary))
            artifacts = {name: _FakeArtifact(path) for name, path in staged.items()}
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_task(_FakeTask(artifacts, status="completed"), TASK_ID)
            artifacts["unexpected"] = _FakeArtifact(staged["metrics"])
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.validate_task(_FakeTask(artifacts), TASK_ID)

    def test_third_party_download_noise_and_errors_are_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            staged, _ = _build_bundle(source)
            artifacts = {name: _FakeArtifact(path) for name, path in staged.items()}
            artifacts["metrics"] = _NoisyFailingArtifact(
                staged["metrics"].stat().st_size,
                MODULE.file_sha256(staged["metrics"]),
            )
            task = _FakeTask(artifacts)
            stage = Path(temporary) / "stage"
            stage.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            saved_stdout = os.dup(1)
            saved_stderr = os.dup(2)
            try:
                os.dup2(stdout_write, 1)
                os.dup2(stderr_write, 2)
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    with self.assertRaises(MODULE.EvidenceValidationError):
                        MODULE.stage_task_artifacts(task, stage, TASK_ID)
            finally:
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
                for descriptor in (
                    saved_stdout,
                    saved_stderr,
                    stdout_write,
                    stderr_write,
                ):
                    os.close(descriptor)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(os.read(stdout_read, 4096), b"")
            self.assertEqual(os.read(stderr_read, 4096), b"")
            os.close(stdout_read)
            os.close(stderr_read)

    def test_missing_advertised_size_is_allowed_but_hash_remains_mandatory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged, _ = _build_bundle(root / "source")
            artifacts = {
                name: _FakeArtifact(path, advertise_size=False)
                for name, path in staged.items()
            }
            task = _FakeTask(artifacts)
            stage = root / "stage"
            stage.mkdir()
            downloaded = MODULE.stage_task_artifacts(task, stage, TASK_ID)
            self.assertEqual(set(downloaded), set(MODULE.ARTIFACT_SPECS))

            artifacts["metrics"].hash = "0" * 64
            changed_stage = root / "changed-stage"
            changed_stage.mkdir()
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.stage_task_artifacts(task, changed_stage, TASK_ID)

    def test_present_advertised_size_must_match_downloaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged, _ = _build_bundle(root / "source")
            artifacts = {name: _FakeArtifact(path) for name, path in staged.items()}
            artifacts["metrics"].size += 1
            stage = root / "stage"
            stage.mkdir()
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.stage_task_artifacts(_FakeTask(artifacts), stage, TASK_ID)

    def test_second_published_snapshot_must_keep_null_sizes_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged, config_sha = _build_bundle(root / "source")
            initial_artifacts = {
                name: _FakeArtifact(path, advertise_size=False)
                for name, path in staged.items()
            }
            refreshed_artifacts = {
                name: _FakeArtifact(path, advertise_size=False)
                for name, path in staged.items()
            }
            refreshed_artifacts["metrics"].hash = "f" * 64
            tasks = iter([_FakeTask(initial_artifacts), _FakeTask(refreshed_artifacts)])
            args = argparse.Namespace(
                task_id=TASK_ID,
                expected_protocol=MODULE.PROTOCOL_ID,
                expected_script_sha256=SCRIPT_SHA,
                expected_diff_sha256=DIFF_SHA,
                expected_config_sha256=config_sha,
                output=str(root / "validated"),
                require_a100=True,
            )
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.run(args, task_loader=lambda _: next(tasks))
            self.assertFalse((root / "validated").exists())

    def test_source_swap_between_lstat_and_open_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            replacement = root / "replacement.json"
            destination = root / "destination.json"
            source.write_text('{"original":true}\n', encoding="utf-8")
            replacement.write_text('{"changed":true}\n', encoding="utf-8")
            real_open = os.open
            swapped = False

            def swap_then_open(
                path: object, flags: int, *args: object, **kwargs: object
            ) -> int:
                nonlocal swapped
                if not swapped and Path(path) == source:
                    os.replace(replacement, source)
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(MODULE.os, "open", side_effect=swap_then_open):
                with self.assertRaises(MODULE.EvidenceValidationError):
                    MODULE.copy_regular_artifact(source, destination, ".json", 1024)
            self.assertFalse(destination.exists())

    def test_receipt_atomic_write_does_not_clobber_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "receipt.json"
            receipt.write_text("preserve-me\n", encoding="utf-8")
            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.write_json_atomic(receipt, {"validated": True})
            self.assertEqual(receipt.read_text(encoding="utf-8"), "preserve-me\n")

    def test_publish_does_not_clobber_a_racing_destination_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            stage.mkdir()
            for filename in (
                "metrics.json",
                "events.jsonl",
                "run_manifest.json",
                "receipt.json",
            ):
                (stage / filename).write_text("staged\n", encoding="utf-8")
            output = root / "output"
            real_link = os.link
            raced = False

            def race_then_link(
                source: object,
                destination: object,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal raced
                if not raced:
                    (output / str(destination)).write_text(
                        "attacker-content\n", encoding="utf-8"
                    )
                    raced = True
                real_link(source, destination, *args, **kwargs)

            with mock.patch.object(MODULE.os, "link", side_effect=race_then_link):
                with self.assertRaises(MODULE.EvidenceValidationError):
                    MODULE.publish_staged_directory(stage, output)
            self.assertFalse(output.exists())

    def test_run_atomically_publishes_receipt_and_only_safe_success_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            staged, config_sha = _build_bundle(source)
            task = _FakeTask(
                {name: _FakeArtifact(path) for name, path in staged.items()}
            )
            output = root / "validated"
            args = argparse.Namespace(
                task_id=TASK_ID,
                expected_protocol=MODULE.PROTOCOL_ID,
                expected_script_sha256=SCRIPT_SHA,
                expected_diff_sha256=DIFF_SHA,
                expected_config_sha256=config_sha,
                output=str(output),
                require_a100=True,
            )
            result = MODULE.run(args, task_loader=lambda _: task)
            receipt_path = output / "receipt.json"
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["task_status"], "published")
            self.assertEqual(set(result), {"task_id", "validated", "receipt_sha256"})
            self.assertEqual(result["receipt_sha256"], MODULE.file_sha256(receipt_path))
            self.assertFalse(any(root.glob(".canary-evidence.*")))
            for artifact in task.artifacts.values():
                self.assertEqual(
                    artifact.calls,
                    [
                        {
                            "extract_archive": False,
                            "raise_on_error": True,
                            "force_download": True,
                        }
                    ],
                )

            with self.assertRaises(MODULE.EvidenceValidationError):
                MODULE.run(args, task_loader=lambda _: task)

    def test_module_import_does_not_require_clearml(self) -> None:
        isolated_spec = importlib.util.spec_from_file_location(
            "download_canary_evidence_isolated", MODULE_PATH
        )
        assert isolated_spec is not None and isolated_spec.loader is not None
        isolated: ModuleType = importlib.util.module_from_spec(isolated_spec)
        isolated_spec.loader.exec_module(isolated)
        self.assertTrue(callable(isolated.validate_staged_bundle))

    def test_main_stdout_contains_only_three_allowlisted_fields(self) -> None:
        args = argparse.Namespace()
        result = {
            "task_id": TASK_ID,
            "validated": True,
            "receipt_sha256": "c" * 64,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            MODULE, "parse_args", return_value=args
        ), mock.patch.object(
            MODULE, "run", return_value=result
        ), contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):
            self.assertEqual(MODULE.main(), 0)
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(set(rendered), {"task_id", "validated", "receipt_sha256"})
        self.assertEqual(stderr.getvalue(), "")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            MODULE, "parse_args", return_value=args
        ), mock.patch.object(
            MODULE,
            "run",
            side_effect=MODULE.EvidenceValidationError("must-not-be-printed"),
        ), contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):
            self.assertEqual(MODULE.main(), 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "download_canary_evidence_failed\n")


if __name__ == "__main__":
    unittest.main()
