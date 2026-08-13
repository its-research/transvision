from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/collect_clearml_formal_models.py"
CONTROLLER_ID = "a" * 32
GATE_ID = "b" * 32
TEACHER_ID = "c" * 32
TEACHER_MODEL_ID = "d" * 32


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "collect_clearml_formal_models", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("seal_sha256", None)
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    result["seal_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def _content_seal(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("content_sha256", None)
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    result["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def _initialization_audit(subject: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": subject,
        "common_teacher_initialization_verified": True,
    }


def _canonical_sha(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _URLArtifact:
    def __init__(
        self,
        raw: bytes,
        *,
        url: str,
        size: int | None = None,
        sha256: str | None = None,
    ) -> None:
        self.url = url
        self.size = len(raw) if size is None else size
        self.hash = hashlib.sha256(raw).hexdigest() if sha256 is None else sha256
        self.get_calls = 0

    def get(self) -> object:
        self.get_calls += 1
        raise AssertionError("URL artifacts must use authenticated fileserver access")


class _ArtifactResponse:
    def __init__(
        self,
        *,
        url: str,
        raw: bytes,
        status_code: int = 200,
        content_length: object | None = None,
    ) -> None:
        self.url = url
        self.raw = raw
        self.status_code = status_code
        if content_length is None:
            content_length = len(raw)
        self.headers = {"Content-Length": str(content_length)}
        self.close_calls = 0

    def iter_content(self, *, chunk_size: int):
        for offset in range(0, len(self.raw), chunk_size):
            yield self.raw[offset : offset + chunk_size]

    def close(self) -> None:
        self.close_calls += 1


class _Model:
    def __init__(
        self,
        *,
        model_id: str,
        task_id: str,
        name: str,
        url: str,
        local_copy: Path,
    ) -> None:
        self.id = model_id
        self.task = task_id
        self.name = name
        self.url = url
        self.local_copy = local_copy
        self.download_calls: list[dict[str, object]] = []
        self.http_calls: list[dict[str, object]] = []
        self.on_download = None
        self.response_statuses: list[int] = []

    def get_local_copy(self, **kwargs: object) -> str:
        self.download_calls.append(dict(kwargs))
        raise AssertionError("collector must not use the ClearML model cache")


class _RemoteResponse:
    def __init__(self, *, url: str, path: Path, status_code: int) -> None:
        self.url = url
        self.path = path
        self.status_code = status_code
        self.headers = {"Content-Length": str(path.stat().st_size)}
        self.close_calls = 0

    def iter_content(self, *, chunk_size: int):
        with self.path.open("rb") as stream:
            while True:
                block = stream.read(chunk_size)
                if not block:
                    return
                yield block

    def close(self) -> None:
        self.close_calls += 1


class _Session:
    def __init__(self) -> None:
        self.token_generation = 1
        self.refresh_calls = 0

    def add_auth_headers(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer test-token-{self.token_generation}"
        return headers

    def refresh_token(self) -> None:
        self.refresh_calls += 1
        self.token_generation += 1


class _Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str = "completed",
        parameters: dict[str, object] | None = None,
        artifacts: dict[str, _Artifact] | None = None,
        model: _Model | None = None,
        reload_statuses: list[str] | None = None,
    ) -> None:
        self.id = task_id
        self.status = status
        self.parameters = dict(parameters or {})
        self.artifacts = dict(artifacts or {})
        self.model = model
        self.reload_statuses = list(reload_statuses or [])
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1
        if self.reload_statuses:
            self.status = self.reload_statuses.pop(0)

    def get_parameters(self) -> dict[str, object]:
        return dict(self.parameters)

    def get_models(self) -> dict[str, list[object]]:
        return {"input": [], "output": [] if self.model is None else [self.model]}


class _Tasks:
    def __init__(self, tasks: dict[str, _Task]) -> None:
        self.tasks = tasks
        self.get_calls: list[str] = []
        self.session = _Session()

    def get_task(self, *, task_id: str) -> _Task:
        self.get_calls.append(task_id)
        return self.tasks[task_id]

    def _get_default_session(self) -> _Session:
        return self.session


def _task_id(index: int) -> str:
    return f"{index + 1:032x}"


def _model_id(index: int) -> str:
    return f"{index + 1001:032x}"


def _formal_manifest(
    module,
    files: dict[str, Path],
    *,
    seeded: bool = False,
    spaced_urls: bool = False,
):
    entries: list[dict[str, object]] = []
    for index, subject in enumerate(module.SUBJECT_ORDER, start=1):
        path = files[subject]
        model_url = f"http://10.100.34.118:8081/models/{subject}_epoch_50.pth"
        if spaced_urls:
            model_url = (
                "http://10.100.34.118:8081/ResilientV2X/Training/"
                f"ResilientV2X post-main {index:02d} {subject} "
                f"[manual-queue-20260808].{_task_id(index)}/models/"
                f"{subject}_epoch_50.pth"
            )
        entry: dict[str, object] = {
            "index": index,
            "subject": subject,
            "kind": module.SUBJECT_KIND[subject],
            "training_task_id": _task_id(index),
            "training_predecessor_task_id": GATE_ID,
            "model_id": _model_id(index),
            "model_name": f"ResilientV2X {subject} final checkpoint",
            "model_url": model_url,
            "checkpoint_filename": "epoch_50.pth",
            "checkpoint_sha256": _sha(path),
            "checkpoint_size_bytes": path.stat().st_size,
            "common_teacher_initialization_audit_artifact": (
                "common_teacher_initialization_audit"
            ),
            "common_teacher_initialization_audit_sha256": _canonical_sha(
                _initialization_audit(subject)
            ),
        }
        if seeded:
            entry["training_seed"] = 20250218
            entry["training_overlay_protocol_seed"] = 20250218
        entries.append(entry)
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": "DAIR-CAUSAL-1337-v1",
        "sample_count": 1337,
        "delays_ms": [0, 100, 200, 300],
        "conditions": ["Full", "L-Fail", "C-Fail"],
        "run_count": 12,
        "checkpoint_policy": "epoch_50_final_only",
        "evaluation_release_semantics": (
            "formal_manifest_after_full_training_suite_completion"
        ),
        "subject_order": list(module.SUBJECT_ORDER),
        "subject_count": 26,
        "entries": entries,
    }
    if seeded:
        payload["training_seed"] = 20250218
        payload["training_overlay_protocol_seed"] = 20250218
    return _seal(payload)


def _quality_gate(teacher_file: Path) -> dict[str, object]:
    return _content_seal(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_teacher_quality_gate",
            "passed": True,
            "quality_gate_task_id": GATE_ID,
            "teacher_task_id": TEACHER_ID,
            "validation_count": 5,
            "selection_protocol": "DAIR-CLEAN-PAIR1789-v1",
            "teacher": {
                "task_id": TEACHER_ID,
                "model_id": TEACHER_MODEL_ID,
                "model_name": "ResilientV2X clean teacher",
                "model_url": (
                    "http://10.100.34.118:8081/models/"
                    "best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_20.pth"
                ),
                "checkpoint_filename": (
                    "best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_20.pth"
                ),
                "checkpoint_size_bytes": teacher_file.stat().st_size,
                "checkpoint_sha256": _sha(teacher_file),
                "selected_epoch": 20,
            },
        }
    )


def _fixture(
    tmp_path: Path, *, seeded: bool = False, spaced_urls: bool = False
):
    module = _load_module()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    files: dict[str, Path] = {}
    for subject in module.SUBJECT_ORDER:
        path = cache / f"{subject}.pth"
        path.write_bytes((subject + "-weights").encode())
        files[subject] = path
    teacher_file = cache / "teacher.pth"
    teacher_file.write_bytes(b"teacher-weights")
    manifest = _formal_manifest(
        module, files, seeded=seeded, spaced_urls=spaced_urls
    )
    teacher_reference = _quality_gate(teacher_file)["teacher"]
    assert isinstance(teacher_reference, dict)
    manifest_entries = manifest["entries"]
    assert isinstance(manifest_entries, list)
    summary_results: list[dict[str, object]] = []
    for entry in manifest_entries:
        assert isinstance(entry, dict)
        result = {
            "index": entry["index"],
            "experiment": entry["subject"],
            "task_id": entry["training_task_id"],
            "predecessor_task_id": entry["training_predecessor_task_id"],
            "model_id": entry["model_id"],
            "model_name": entry["model_name"],
            "model_url": entry["model_url"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
        }
        if seeded:
            result["training_seed"] = entry["training_seed"]
            result["training_overlay_protocol_seed"] = entry[
                "training_overlay_protocol_seed"
            ]
        summary_results.append(result)
    summary = _seal(
        {
            "schema_version": 1,
            "summary_type": "resilient_v2x_post_main_sequential_training",
            "status": "completed",
            "controller_task_id": CONTROLLER_ID,
            "gate_task_id": GATE_ID,
            "teacher": dict(teacher_reference),
            "experiment_order": list(module.SUBJECT_ORDER),
            "task_count": 26,
            "results": summary_results,
            "formal_1337_evaluation_manifest": manifest,
            "formal_1337_manifest_artifact": "formal_1337_training_manifest",
        }
    )
    controller = _Task(
        CONTROLLER_ID,
        parameters={
            "Args/gate_task_id": GATE_ID,
            "Args/template_task_id": "f" * 32,
            "Args/worker_queues": "A100,A100,V100,5090",
        },
        artifacts={
            "formal_1337_training_manifest": _Artifact(manifest),
            "post_main_training_summary": _Artifact(summary),
        },
    )
    gate_payload = _quality_gate(teacher_file)
    gate = _Task(
        GATE_ID,
        artifacts={"teacher_quality_gate": _Artifact(gate_payload)},
    )
    teacher_model = _Model(
        model_id=TEACHER_MODEL_ID,
        task_id=TEACHER_ID,
        name="ResilientV2X clean teacher",
        url=str(teacher_reference["model_url"]),
        local_copy=teacher_file,
    )
    teacher = _Task(TEACHER_ID, model=teacher_model)
    tasks = {CONTROLLER_ID: controller, GATE_ID: gate, TEACHER_ID: teacher}
    models: list[_Model] = [teacher_model]
    entries = manifest["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        subject = str(entry["subject"])
        source_revision = module.SOURCE_REVISION_BY_SUBJECT[subject]
        source = module.EXPECTED_SOURCE_BY_REVISION[source_revision]
        model = _Model(
            model_id=str(entry["model_id"]),
            task_id=str(entry["training_task_id"]),
            name=str(entry["model_name"]),
            url=str(entry["model_url"]),
            local_copy=files[subject],
        )
        seed_key = "training_seed" if seeded else "seed"
        run_contract = {
            "task_id": entry["training_task_id"],
            "experiment": subject,
            "source_dataset_id": source["dataset_id"],
            "training_dataset_id": module.EXPECTED_TRAINING_DATASET_ID,
            "gpus": 4,
            "ddp_processes": 4,
            "max_epochs": 50,
            seed_key: 20250218,
            "amp": False,
            "precision": "FP32",
            "val_interval": 10,
            "condition_evaluation": False,
            "source_archive": {
                "name": source["archive_name"],
                "size_bytes": source["archive_size_bytes"],
                "sha256": source["archive_sha256"],
            },
        }
        final_contract = {
            "model_id": entry["model_id"],
            "name": entry["model_name"],
            "url": entry["model_url"],
            "filename": "epoch_50.pth",
            "size_bytes": entry["checkpoint_size_bytes"],
            "sha256": entry["checkpoint_sha256"],
        }
        task = _Task(
            str(entry["training_task_id"]),
            parameters={
                "Args/source_dataset_id": source["dataset_id"],
                "Args/source_archive_name": source["archive_name"],
                "Args/source_archive_bytes": str(source["archive_size_bytes"]),
                "Args/source_archive_sha256": source["archive_sha256"],
            },
            artifacts={
                "run_contract": _Artifact(run_contract),
                "final_checkpoint_contract": _Artifact(final_contract),
                "common_teacher_initialization_audit": _Artifact(
                    _initialization_audit(subject)
                ),
            },
            model=model,
        )
        tasks[task.id] = task
        models.append(model)
    models_by_url = {
        module._fileserver_url(
            model.url,
            expected_filename=model.url.rsplit("/", 1)[-1],
            context="test model URL",
        ): model
        for model in models
    }

    def open_remote(
        url: str,
        *,
        headers: dict[str, str],
        stream: bool,
        allow_redirects: bool,
        timeout: float,
    ):
        model = models_by_url[url]
        model.http_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "stream": stream,
                "allow_redirects": allow_redirects,
                "timeout": timeout,
            }
        )
        if model.on_download is not None:
            model.on_download()
        status = model.response_statuses.pop(0) if model.response_statuses else 200
        return _RemoteResponse(
            url=url, path=model.local_copy, status_code=status
        )

    module._HTTP_GET = open_remote
    module._available_bytes = lambda _path: 10**18
    return module, _Tasks(tasks), manifest, models


def test_url_artifact_rewrites_legacy_host_and_uses_authenticated_no_redirect_get(
) -> None:
    module = _load_module()
    raw = b'{"document_type":"sealed-evidence","schema_version":1}'
    legacy_url = (
        "http://10.100.34.118:8081/ResilientV2X/Training/"
        "controller/artifacts/formal_manifest.json"
    )
    current_url = legacy_url.replace("10.100.34.118", "10.100.35.118")
    artifact = _URLArtifact(raw, url=legacy_url)
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})
    response = _ArtifactResponse(url=current_url, raw=raw)
    calls: list[dict[str, object]] = []

    def open_remote(url: str, **kwargs: object) -> _ArtifactResponse:
        captured = dict(kwargs)
        captured["headers"] = dict(captured["headers"])
        calls.append({"url": url, **captured})
        return response

    module._HTTP_GET = open_remote
    refresh_calls: list[bool] = []
    payload = module._artifact_payload(
        task,
        "evidence",
        auth_header_provider=lambda: {"Authorization": "Bearer artifact-token"},
        auth_refresher=lambda: refresh_calls.append(True),
    )

    assert payload == {"document_type": "sealed-evidence", "schema_version": 1}
    assert artifact.get_calls == 0
    assert refresh_calls == []
    assert calls == [
        {
            "url": current_url,
            "headers": {
                "Authorization": "Bearer artifact-token",
                "Accept-Encoding": "identity",
            },
            "stream": True,
            "allow_redirects": False,
            "timeout": module.HTTP_DOWNLOAD_TIMEOUT_SECONDS,
        }
    ]
    assert response.close_calls == 1


def test_url_artifact_refuses_unauthenticated_artifact_get_fallback() -> None:
    module = _load_module()
    raw = b'{"schema_version":1}'
    artifact = _URLArtifact(
        raw,
        url="http://10.100.34.118:8081/artifacts/evidence.json",
    )
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})

    with pytest.raises(RuntimeError, match="requires authenticated fileserver access"):
        module._artifact_payload(task, "evidence")
    assert artifact.get_calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://10.100.36.118:8081/artifacts/evidence.json",
        "http://10.100.34.118:8081/artifacts/evidence.json?download=1",
        "http://10.100.34.118:8081/artifacts/%2e%2e/evidence.json",
        "http://10.100.34.118:8081/artifacts/%2Fetc/evidence.json",
    ],
)
def test_url_artifact_rejects_authority_or_path_drift_before_auth_request(
    url: str,
) -> None:
    module = _load_module()
    raw = b'{"schema_version":1}'
    artifact = _URLArtifact(raw, url=url)
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})

    def unexpected_request(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unsafe artifact URL reached the HTTP client")

    module._HTTP_GET = unexpected_request
    with pytest.raises(RuntimeError, match="expected durable fileserver URL"):
        module._artifact_payload(
            task,
            "evidence",
            auth_header_provider=lambda: {"Authorization": "Bearer token"},
            auth_refresher=lambda: None,
        )
    assert artifact.get_calls == 0


@pytest.mark.parametrize(
    ("sealed_size_delta", "sealed_sha256", "expected_error"),
    [
        (1, None, "Content-Length mismatch"),
        (0, "0" * 64, "SHA-256 mismatch"),
    ],
)
def test_url_artifact_fails_closed_on_sealed_size_or_sha_mismatch(
    sealed_size_delta: int,
    sealed_sha256: str | None,
    expected_error: str,
) -> None:
    module = _load_module()
    raw = b'{"schema_version":1}'
    current_url = "http://10.100.35.118:8081/artifacts/evidence.json"
    artifact = _URLArtifact(
        raw,
        url=current_url,
        size=len(raw) + sealed_size_delta,
        sha256=sealed_sha256,
    )
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})
    response = _ArtifactResponse(url=current_url, raw=raw)
    module._HTTP_GET = lambda *_args, **_kwargs: response

    with pytest.raises(RuntimeError, match=expected_error):
        module._artifact_payload(
            task,
            "evidence",
            auth_header_provider=lambda: {"Authorization": "Bearer token"},
            auth_refresher=lambda: None,
        )
    assert response.close_calls == 1
    assert artifact.get_calls == 0


def test_url_artifact_401_refreshes_once_and_retries_with_new_bearer() -> None:
    module = _load_module()
    raw = b'{"schema_version":1}'
    current_url = "http://10.100.35.118:8081/artifacts/evidence.json"
    artifact = _URLArtifact(raw, url=current_url)
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})
    responses = [
        _ArtifactResponse(url=current_url, raw=raw, status_code=401),
        _ArtifactResponse(url=current_url, raw=raw),
    ]
    calls: list[dict[str, object]] = []
    generation = {"value": 1}

    def open_remote(_url: str, **kwargs: object) -> _ArtifactResponse:
        captured = dict(kwargs)
        captured["headers"] = dict(captured["headers"])
        calls.append(captured)
        return responses[len(calls) - 1]

    def refresh() -> None:
        generation["value"] += 1

    module._HTTP_GET = open_remote
    payload = module._artifact_payload(
        task,
        "evidence",
        auth_header_provider=lambda: {
            "Authorization": f"Bearer token-{generation['value']}"
        },
        auth_refresher=refresh,
    )

    assert payload == {"schema_version": 1}
    assert [call["headers"]["Authorization"] for call in calls] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert all(call["allow_redirects"] is False for call in calls)
    assert [response.close_calls for response in responses] == [1, 1]


def test_url_artifact_rejects_redirect_response_without_following_it() -> None:
    module = _load_module()
    raw = b'{"schema_version":1}'
    current_url = "http://10.100.35.118:8081/artifacts/evidence.json"
    artifact = _URLArtifact(raw, url=current_url)
    task = _Task(CONTROLLER_ID, artifacts={"evidence": artifact})
    response = _ArtifactResponse(url=current_url, raw=raw, status_code=302)
    calls: list[dict[str, object]] = []

    def open_remote(_url: str, **kwargs: object) -> _ArtifactResponse:
        calls.append(dict(kwargs))
        return response

    module._HTTP_GET = open_remote
    with pytest.raises(RuntimeError, match="HTTP 302"):
        module._artifact_payload(
            task,
            "evidence",
            auth_header_provider=lambda: {"Authorization": "Bearer token"},
            auth_refresher=lambda: None,
        )
    assert calls[0]["allow_redirects"] is False
    assert len(calls) == 1
    assert response.close_calls == 1


def test_validates_legacy_and_explicit_seed_manifests(tmp_path: Path) -> None:
    module, _, legacy, _ = _fixture(tmp_path / "legacy")
    result = module.validate_formal_manifest(legacy)
    assert result["seed_evidence"] == "legacy_source_c_run_contract"
    assert result["training_seed"] == 20250218

    module, _, seeded, _ = _fixture(tmp_path / "seeded", seeded=True)
    result = module.validate_formal_manifest(seeded)
    assert result["seed_evidence"] == "explicit_manifest"
    assert result["training_seed"] == 20250218


def test_rejects_partial_seed_metadata(tmp_path: Path) -> None:
    module, _, manifest, _ = _fixture(tmp_path)
    manifest["training_seed"] = 20250218
    manifest = _seal(manifest)
    with pytest.raises(RuntimeError, match="partial seed"):
        module.validate_formal_manifest(manifest)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda manifest: manifest.update(sample_count=1336), "sample_count"),
        (
            lambda manifest: manifest["entries"].__setitem__(1, manifest["entries"][0]),
            "entry order|not unique",
        ),
        (
            lambda manifest: manifest["entries"][0].update(
                model_url=(
                    "http://evil.example:8081/models/support_residual_epoch_50.pth"
                )
            ),
            "fileserver URL",
        ),
        (
            lambda manifest: manifest["entries"][0].update(
                model_url=(
                    "http://10.100.34.118:8081/models/"
                    "support_residual_epoch_50.pth?download=1"
                )
            ),
            "fileserver URL",
        ),
    ],
)
def test_manifest_drift_is_rejected(tmp_path: Path, mutation, message: str) -> None:
    module, _, manifest, _ = _fixture(tmp_path)
    mutation(manifest)
    manifest = _seal(manifest)
    with pytest.raises(RuntimeError, match=message):
        module.validate_formal_manifest(manifest)


def test_manifest_seal_is_required(tmp_path: Path) -> None:
    module, _, manifest, _ = _fixture(tmp_path)
    manifest["sample_count"] = 1
    with pytest.raises(RuntimeError, match="seal_sha256 mismatch"):
        module.validate_formal_manifest(manifest)


def test_real_clearml_unencoded_space_and_bracket_urls_are_streamed(
    tmp_path: Path,
) -> None:
    module, tasks, _, models = _fixture(
        tmp_path / "fixture", spaced_urls=True
    )
    registry = module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=tmp_path / "models",
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    methods = registry["methods"]
    assert isinstance(methods, list)
    for model, record in zip(models[1:], methods, strict=True):
        assert " " in model.url and "[" in model.url and "]" in model.url
        assert record["remote_url"] == model.url
        request_url = str(model.http_calls[0]["url"])
        assert " " not in request_url
        assert "[" not in request_url and "]" not in request_url
        assert "%20" in request_url
        assert "%5B" in request_url and "%5D" in request_url
        assert not model.download_calls


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/safe%2Fescape/models/ffnet_epoch_50.pth",
        "/safe%5Cescape/models/ffnet_epoch_50.pth",
        "/safe%00escape/models/ffnet_epoch_50.pth",
        "/safe%252Fescape/models/ffnet_epoch_50.pth",
        "/safe\\escape/models/ffnet_epoch_50.pth",
        "/safe\x00escape/models/ffnet_epoch_50.pth",
    ],
)
def test_fileserver_url_rejects_encoded_separators_and_double_encoding(
    tmp_path: Path, unsafe_path: str
) -> None:
    del tmp_path
    module = _load_module()
    with pytest.raises(RuntimeError, match="durable fileserver URL"):
        module._fileserver_url(
            f"http://10.100.34.118:8081{unsafe_path}",
            expected_filename="ffnet_epoch_50.pth",
            context="malicious model URL",
        )


def test_collects_teacher_and_all_methods_atomically(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    registry_path = tmp_path / "registry.json"
    registry = module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=output,
        registry_path=registry_path,
        wait=False,
    )
    assert registry["status"] == "complete"
    assert registry["method_count"] == 26
    assert registry["teacher_model_count"] == 1
    assert registry["epoch50_method_model_count"] == 26
    assert registry["model_count"] == 27
    assert len(registry["methods"]) == 26
    revision = str(registry["revision"])
    revision_dir = output / "revisions" / revision
    assert (revision_dir / "manifest.json").is_file()
    assert (revision_dir / "COMPLETE").is_file()
    assert (output / "COMPLETE").is_file()
    assert registry_path.is_file()
    sums = (revision_dir / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 27
    assert len(list(output.rglob("*.pth"))) == 27
    assert all(len(model.http_calls) == 1 for model in models)
    assert all(not model.download_calls for model in models)


def test_second_collection_is_idempotent_and_does_not_redownload(
    tmp_path: Path,
) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    kwargs = {
        "task_class": tasks,
        "controller_task_id": CONTROLLER_ID,
        "quality_gate_task_id": GATE_ID,
        "output_dir": tmp_path / "models",
        "registry_path": tmp_path / "registry.json",
        "wait": False,
    }
    first = module.collect_formal_models(**kwargs)
    for model in models:
        model.http_calls.clear()
    second = module.collect_formal_models(**kwargs)
    assert second == first
    assert all(not model.http_calls for model in models)
    assert all(not model.download_calls for model in models)


def test_source_c_controller_shape_without_data_parameters_is_supported(
    tmp_path: Path,
) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    parameters = tasks.tasks[CONTROLLER_ID].parameters
    assert "Args/source_dataset_id" not in parameters
    assert "Args/source_archive_sha256" not in parameters
    assert "Args/training_dataset_id" not in parameters
    registry = module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=tmp_path / "models",
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    assert registry["source_revisions"] == (
        module.SOURCE_REVISION_CERTIFICATE_BY_TREE
    )
    assert registry["source_revision_by_subject"] == module.SOURCE_TREE_BY_SUBJECT
    assert registry["source_revision_counts"] == module.SOURCE_REVISION_COUNTS
    assert registry["training_dataset_id"] == module.EXPECTED_TRAINING_DATASET_ID


def test_collection_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    module = _load_module()
    root = module._safe_output_root(tmp_path / "models")
    with module._collection_lock(root):
        with pytest.raises(RuntimeError, match="already locked"):
            with module._collection_lock(root):
                pass


def test_download_failure_never_publishes_complete_collection(
    tmp_path: Path,
) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")

    def fail_download() -> None:
        raise RuntimeError("injected download failure")

    models[5].on_download = fail_download
    output = tmp_path / "models"
    registry_path = tmp_path / "registry.json"
    with pytest.raises(RuntimeError, match="authenticated model request failed"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=registry_path,
            wait=False,
        )
    assert not (output / "COMPLETE").exists()
    assert not registry_path.exists()
    assert not list(output.rglob("*.pth"))


def test_registry_write_failure_leaves_root_uncommitted(
    tmp_path: Path, monkeypatch
) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    registry_path = tmp_path / "registry.json"
    original = module._atomic_write_text

    def fail_registry(destination: Path, content: str) -> None:
        if destination == registry_path:
            raise RuntimeError("injected registry failure")
        original(destination, content)

    monkeypatch.setattr(module, "_atomic_write_text", fail_registry)
    with pytest.raises(RuntimeError, match="injected registry failure"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=registry_path,
            wait=False,
        )
    assert not (output / "COMPLETE").exists()
    revisions = list((output / "revisions").iterdir())
    assert len(revisions) == 1
    assert (revisions[0] / "COMPLETE").is_file()


def test_remote_manifest_drift_before_publish_fails_closed(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    controller = tasks.tasks[CONTROLLER_ID]

    def drift_manifest() -> None:
        payload = controller.artifacts["formal_1337_training_manifest"].value
        assert isinstance(payload, dict)
        changed = dict(payload)
        changed["sample_count"] = 1336
        controller.artifacts["formal_1337_training_manifest"].value = _seal(changed)

    models[0].on_download = drift_manifest
    output = tmp_path / "models"
    with pytest.raises(RuntimeError, match="manifest changed during collection"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert not (output / "COMPLETE").exists()
    assert not list((output / "revisions").iterdir())


def test_unexpected_checkpoint_in_output_is_rejected(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    output.mkdir()
    (output / "unexpected.pth").write_bytes(b"not-formal")
    with pytest.raises(RuntimeError, match="unexpected .pth"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert all(not model.http_calls for model in models)
    assert all(not model.download_calls for model in models)


def test_insufficient_space_fails_before_any_http_download(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    module._available_bytes = lambda _path: 0
    output = tmp_path / "models"
    with pytest.raises(RuntimeError, match="insufficient free space"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert all(not model.http_calls for model in models)
    assert all(not model.download_calls for model in models)
    assert not (output / "COMPLETE").exists()


def test_space_preflight_uses_the_actual_staging_filesystem(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    checked: list[Path] = []

    def available(path: Path) -> int:
        checked.append(path.resolve(strict=True))
        return 10**18

    module._available_bytes = available
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=output,
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    assert checked == [(tmp_path / ".models.staging").resolve(strict=True)]


def test_existing_corrupt_model_fails_closed(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=output,
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    target = next((output / "revisions").rglob("*.pth"))
    target.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="final identity verification"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_run_contract_seed_is_independently_audited(tmp_path: Path) -> None:
    module, tasks, manifest, _ = _fixture(tmp_path / "fixture")
    first = manifest["entries"][0]
    task = tasks.tasks[str(first["training_task_id"])]
    contract = task.artifacts["run_contract"].value
    assert isinstance(contract, dict)
    contract["seed"] = 1
    with pytest.raises(RuntimeError, match="run contract seed mismatch"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_run_contract_source_archive_is_independently_audited(
    tmp_path: Path,
) -> None:
    module, tasks, manifest, _ = _fixture(tmp_path / "fixture")
    first = manifest["entries"][0]
    task = tasks.tasks[str(first["training_task_id"])]
    contract = task.artifacts["run_contract"].value
    assert isinstance(contract, dict)
    source_archive = contract["source_archive"]
    assert isinstance(source_archive, dict)
    source_archive["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source archive identity mismatch"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_initialization_audit_content_is_independently_audited(
    tmp_path: Path,
) -> None:
    module, tasks, manifest, _ = _fixture(tmp_path / "fixture")
    first = manifest["entries"][0]
    task = tasks.tasks[str(first["training_task_id"])]
    audit = task.artifacts["common_teacher_initialization_audit"].value
    assert isinstance(audit, dict)
    audit["common_teacher_initialization_verified"] = False
    with pytest.raises(RuntimeError, match="initialization audit SHA-256 mismatch"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_summary_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    controller = tasks.tasks[CONTROLLER_ID]
    summary = controller.artifacts["post_main_training_summary"].value
    assert isinstance(summary, dict)
    summary["task_count"] = 25
    controller.artifacts["post_main_training_summary"].value = _seal(summary)
    with pytest.raises(RuntimeError, match="task_count"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_no_wait_rejects_incomplete_dependencies(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    tasks.tasks[CONTROLLER_ID].status = "in_progress"
    with pytest.raises(RuntimeError, match="must both be completed"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


def test_wait_supports_dependency_transitions(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    gate = tasks.tasks[GATE_ID]
    controller = tasks.tasks[CONTROLLER_ID]
    gate.status = "in_progress"
    gate.reload_statuses = ["completed"]
    controller.status = "queued"
    controller.reload_statuses = ["in_progress", "completed"]
    sleeps: list[float] = []
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=tmp_path / "models",
        registry_path=tmp_path / "registry.json",
        wait=True,
        poll_seconds=2.0,
        timeout_hours=1.0,
        monotonic_clock=lambda: 0.0,
        sleeper=sleeps.append,
    )
    assert sleeps == [2.0]


def test_output_root_symlink_is_rejected(tmp_path: Path) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "models"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=link,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )


@pytest.mark.parametrize(
    "registry_location",
    [
        "output_equal",
        "output_descendant",
        "output_ancestor",
        "revision_checkpoint",
        "staging_descendant",
        "symlink_into_output",
        "symlink_ancestor_into_output",
    ],
)
def test_registry_path_must_be_strictly_isolated_from_collection_trees(
    tmp_path: Path, registry_location: str
) -> None:
    module, tasks, manifest, models = _fixture(tmp_path / "fixture")
    output = tmp_path / "collection" / "models"
    if registry_location == "output_equal":
        registry_path = output
    elif registry_location == "output_descendant":
        registry_path = output / "registry.json"
    elif registry_location == "output_ancestor":
        registry_path = output.parent
    elif registry_location == "revision_checkpoint":
        registry_path = (
            output
            / "revisions"
            / str(manifest["seal_sha256"])
            / "methods"
            / "support_residual"
            / _task_id(1)
            / "support_residual_epoch_50.pth"
        )
    elif registry_location == "staging_descendant":
        registry_path = output.parent / ".models.staging" / "registry.json"
    else:
        output.mkdir(parents=True)
        link = tmp_path / "registry-link"
        link.symlink_to(output, target_is_directory=True)
        registry_path = (
            link / "nested" / "registry.json"
            if registry_location == "symlink_ancestor_into_output"
            else link / "registry.json"
        )
    with pytest.raises(RuntimeError, match="isolated|symlink"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=registry_path,
            wait=False,
        )
    assert not (output / "COMPLETE").exists()
    assert not (output / "nested").exists()
    assert all(not model.http_calls for model in models)


def test_stale_root_completion_is_quarantined_before_download_failure(
    tmp_path: Path,
) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    output.mkdir()
    marker = output / "COMPLETE"
    marker.write_text("stale marker\n", encoding="utf-8")

    def fail_download() -> None:
        raise RuntimeError("injected download failure")

    models[0].on_download = fail_download
    with pytest.raises(RuntimeError, match="authenticated model request failed"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert not marker.exists()
    quarantined = list(output.glob(".COMPLETE.stale.*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "stale marker\n"


def test_valid_root_completion_survives_failed_idempotent_revalidation(
    tmp_path: Path, monkeypatch
) -> None:
    module, tasks, _, _ = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    registry_path = tmp_path / "registry.json"
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=output,
        registry_path=registry_path,
        wait=False,
    )
    marker = output / "COMPLETE"
    original = marker.read_bytes()

    def fail_revalidation(**_kwargs: object) -> None:
        raise RuntimeError("injected remote revalidation failure")

    monkeypatch.setattr(module, "_revalidate_captured_sources", fail_revalidation)
    with pytest.raises(RuntimeError, match="injected remote revalidation failure"):
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=registry_path,
            wait=False,
        )
    assert marker.read_bytes() == original
    assert not list(output.glob(".COMPLETE.stale.*"))


def test_clearml_cache_api_is_never_called(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=tmp_path / "models",
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    assert all(model.http_calls for model in models)
    assert all(not model.download_calls for model in models)


def test_bearer_header_is_used_but_never_persisted(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    output = tmp_path / "models"
    registry_path = tmp_path / "registry.json"
    registry = module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=output,
        registry_path=registry_path,
        wait=False,
    )
    for model in models:
        call = model.http_calls[0]
        headers = call["headers"]
        assert isinstance(headers, dict)
        assert headers["Authorization"] == "Bearer test-token-1"
        assert headers["Accept-Encoding"] == "identity"
        assert call["allow_redirects"] is False
        assert call["stream"] is True
    persisted = json.dumps(registry, sort_keys=True)
    persisted += registry_path.read_text(encoding="utf-8")
    for path in output.rglob("*"):
        if path.is_file() and path.suffix != ".pth":
            persisted += path.read_text(encoding="utf-8")
    assert "test-token" not in persisted
    assert "Authorization" not in persisted


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_authenticated_download_session_ignores_environment_proxies(
    monkeypatch, scheme: str
) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(key, "http://proxy.invalid:3128")
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    module = _load_module()
    assert module._HTTP_SESSION.trust_env is False
    assert module._HTTP_GET.__self__ is module._HTTP_SESSION
    settings = module._HTTP_SESSION.merge_environment_settings(
        f"{scheme}://10.100.34.118:8081/models/model.pth",
        {},
        None,
        None,
        None,
    )
    assert not settings["proxies"]


def test_response_and_request_exception_do_not_retain_bearer() -> None:
    module = _load_module()
    response = requests.Response()
    response.status_code = 200
    response.url = "http://10.100.34.118:8081/models/model.pth"
    response.request = requests.Request(
        "GET",
        response.url,
        headers={"Authorization": "Bearer response-secret"},
    ).prepare()
    module._HTTP_GET = lambda *_args, **_kwargs: response
    observed = module._authenticated_model_response(
        url=response.url,
        auth_header_provider=lambda: {"Authorization": "Bearer live-token"},
        auth_refresher=lambda: None,
    )
    assert observed is response
    assert "Authorization" not in response.request.headers

    error = requests.RequestException("injected request failure")
    error.request = requests.Request(
        "GET",
        response.url,
        headers={"Authorization": "Bearer exception-secret"},
    ).prepare()

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise error

    module._HTTP_GET = fail_request
    with pytest.raises(RuntimeError, match="authenticated model request failed"):
        module._authenticated_model_response(
            url=response.url,
            auth_header_provider=lambda: {"Authorization": "Bearer live-token"},
            auth_refresher=lambda: None,
        )
    assert "Authorization" not in error.request.headers


def test_first_401_refreshes_once_and_retries_with_new_header(
    tmp_path: Path,
) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    models[0].response_statuses = [401, 200]
    module.collect_formal_models(
        task_class=tasks,
        controller_task_id=CONTROLLER_ID,
        quality_gate_task_id=GATE_ID,
        output_dir=tmp_path / "models",
        registry_path=tmp_path / "registry.json",
        wait=False,
    )
    assert tasks.session.refresh_calls == 1
    assert len(models[0].http_calls) == 2
    assert models[0].http_calls[0]["headers"]["Authorization"] == (
        "Bearer test-token-1"
    )
    assert models[0].http_calls[1]["headers"]["Authorization"] == (
        "Bearer test-token-2"
    )


def test_second_401_fails_without_a_third_attempt(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    models[0].response_statuses = [401, 401, 200]
    output = tmp_path / "models"
    with pytest.raises(RuntimeError, match="after one refresh") as error:
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=output,
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert "test-token" not in str(error.value)
    assert tasks.session.refresh_calls == 1
    assert len(models[0].http_calls) == 2
    assert models[0].response_statuses == [200]
    assert not (output / "COMPLETE").exists()


def test_non_401_http_failure_is_not_retried(tmp_path: Path) -> None:
    module, tasks, _, models = _fixture(tmp_path / "fixture")
    models[0].response_statuses = [403, 200]
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        module.collect_formal_models(
            task_class=tasks,
            controller_task_id=CONTROLLER_ID,
            quality_gate_task_id=GATE_ID,
            output_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
            wait=False,
        )
    assert "test-token" not in str(error.value)
    assert tasks.session.refresh_calls == 0
    assert len(models[0].http_calls) == 1
    assert models[0].response_statuses == [200]


def test_cli_defaults_to_project_dedicated_model_directory() -> None:
    module = _load_module()
    args = module._parser().parse_args(
        [
            "--training-controller-task-id",
            CONTROLLER_ID,
            "--teacher-quality-gate-task-id",
            GATE_ID,
        ]
    )
    assert args.output_dir == (
        ROOT / "artifacts/trained_models/formal-v5-mixed-5c984ad4-ad511d88"
    )
    assert args.registry_path == ROOT / "docs/resilient_v2x/model-registry.json"
