from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_source_d_evidence.py"
BUILDER_PATH = ROOT / "tools/resilient_v2x/formal_source_d_seed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_formal_source_d_evidence", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_seal(payload: dict[str, object]) -> None:
    observed = payload["seal_sha256"]
    unhashed = dict(payload)
    unhashed.pop("seal_sha256")
    canonical = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert observed == _digest_text(canonical)


def _reseal(payload: dict[str, object]) -> None:
    payload.pop("seal_sha256", None)
    payload["seal_sha256"] = _digest_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _rehash_equivalence(payload: dict[str, object]) -> None:
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = _digest_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _SourceTask:
    def __init__(
        self,
        module,
        source: str,
        *,
        task_id: str | None = None,
        status: str = "completed",
        statuses: list[str] | None = None,
        repository: object = "",
        working_dir: object = ".",
        entry_point: object | None = None,
        script_diff: object | None = None,
        parent: object | None = None,
        reload_ok: bool | None = None,
    ) -> None:
        self.id = task_id or module.SOURCE_C_TASK_ID
        self.status = status
        self.statuses = list(statuses or [])
        self.parent = module.SOURCE_C_PARENT_TASK_ID if parent is None else parent
        self.reload_ok = reload_ok
        self.reload_calls = 0
        self._reload_skip_flag = True
        self.data = SimpleNamespace(
            id=self.id,
            parent=self.parent,
            script=SimpleNamespace(
                repository=repository,
                working_dir=working_dir,
                entry_point=(
                    module.SOURCE_C_ENTRY_POINT if entry_point is None else entry_point
                ),
                diff=source if script_diff is None else script_diff,
            ),
        )

    def _reload(self) -> object:
        assert self._reload_skip_flag is False
        self.reload_calls += 1
        if self.statuses:
            self.status = self.statuses.pop(0)
        return self.data if self.reload_ok is None else self.reload_ok


class _OutputTask:
    def __init__(
        self,
        module,
        *,
        task_id: str = "f" * 32,
        parent: str | None = None,
        upload_failure: str | None = None,
        flush_failure: str | None = None,
        readback_drift: str | None = None,
        readback_missing: str | None = None,
        reload_ok: bool | None = None,
        repository: object = "",
        working_dir: object = ".",
        entry_point: object | None = None,
        script_diff: object | None = None,
    ) -> None:
        self.id = task_id
        self.parent = module.SOURCE_C_TASK_ID if parent is None else parent
        self.data = SimpleNamespace(
            id=self.id,
            parent=self.parent,
            script=SimpleNamespace(
                repository=repository,
                working_dir=working_dir,
                entry_point=(
                    module.PRODUCER_ENTRY_POINT if entry_point is None else entry_point
                ),
                diff=(
                    MODULE_PATH.read_bytes().decode("utf-8")
                    if script_diff is None
                    else script_diff
                ),
            ),
        )
        self.artifacts: dict[str, _Artifact] = {}
        self.upload_failure = upload_failure
        self.flush_failure = flush_failure
        self.readback_drift = readback_drift
        self.readback_missing = readback_missing
        self.reload_ok = reload_ok
        self.uploads: list[str] = []
        self.flushes: list[str] = []
        self.reload_calls = 0
        self._last_upload: str | None = None
        self._reload_skip_flag = True

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        assert wait_on_upload is True
        self.uploads.append(name)
        self._last_upload = name
        if name == self.upload_failure:
            return False
        value = copy.deepcopy(artifact_object)
        if name == self.readback_drift:
            assert isinstance(value, dict)
            value["drift"] = True
        if name != self.readback_missing:
            self.artifacts[name] = _Artifact(value)
        return True

    def flush(self, *, wait_for_uploads: bool) -> bool | None:
        assert wait_for_uploads is True
        assert self._last_upload is not None
        self.flushes.append(self._last_upload)
        if self._last_upload == self.flush_failure:
            return False
        return None

    def _reload(self) -> object:
        assert self._reload_skip_flag is False
        self.reload_calls += 1
        return self.data if self.reload_ok is None else self.reload_ok


class _Tasks:
    def __init__(self, source: _SourceTask, output: _OutputTask | None) -> None:
        self.source = source
        self.output = output
        self.requested: list[str] = []

    def get_task(self, *, task_id: str) -> _SourceTask:
        self.requested.append(task_id)
        return self.source

    def current_task(self) -> _OutputTask | None:
        return self.output


class _FakeBuilder:
    def __init__(
        self,
        module,
        source_d: str,
        *,
        mutation: str | None = None,
    ) -> None:
        self.module = module
        self.source_d = source_d
        self.mutation = mutation
        self._STAGING_ANCHOR_NAMES = module.STAGING_ANCHOR_NAMES
        if mutation == "builder_staging_anchors":
            self._STAGING_ANCHOR_NAMES = module.STAGING_ANCHOR_NAMES[:-1]

    def _artifact(self) -> dict[str, object]:
        module = self.module
        artifact: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "resilient_v2x_formal_source_d_seed_diff",
            "transformation_id": module.TRANSFORMATION_ID,
            "source_c": {"sha256": module.SOURCE_C_SHA256},
            "source_d": {"sha256": module.SOURCE_D_SHA256},
            "seed_contract": {
                "default_training_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
            },
            "diff": [
                {
                    "index": index,
                    "name": name,
                    "expected_count": 1,
                    "observed_count": 1,
                }
                for index, name in enumerate(
                    (
                        *(
                            f"seed_replacement_{seed_index}"
                            for seed_index in range(
                                1,
                                module.DECLARED_REPLACEMENT_COUNT
                                - len(module.STAGING_ANCHOR_NAMES)
                                + 1,
                            )
                        ),
                        *module.STAGING_ANCHOR_NAMES,
                    ),
                    start=1,
                )
            ],
            "equivalence": {
                "only_declared_anchor_replacements": True,
                "declared_replacement_count": module.DECLARED_REPLACEMENT_COUNT,
                "unchanged_segment_count": module.UNCHANGED_SEGMENT_COUNT,
                "source_c_replay_sha256": module.SOURCE_C_SHA256,
                "source_d_replay_sha256": module.SOURCE_D_SHA256,
                "source_d_compiles": True,
            },
        }
        if self.mutation == "replacement_count":
            artifact["diff"] = artifact["diff"][:-1]  # type: ignore[index]
        elif self.mutation == "unchanged_count":
            artifact["equivalence"]["unchanged_segment_count"] = 24  # type: ignore[index]
        elif self.mutation == "float_unchanged_count":
            artifact["equivalence"]["unchanged_segment_count"] = 23.0  # type: ignore[index]
        elif self.mutation == "float_expected_count":
            artifact["diff"][0]["expected_count"] = 1.0  # type: ignore[index]
        elif self.mutation == "float_overlay_seed":
            artifact["seed_contract"]["training_overlay_protocol_seed"] = 20_250_218.0  # type: ignore[index]
        elif self.mutation == "staging_anchor":
            artifact["diff"][-1]["name"] = "drifted_staging_anchor"  # type: ignore[index]
        elif self.mutation == "source_c_replay":
            artifact["equivalence"]["source_c_replay_sha256"] = "0" * 64  # type: ignore[index]
        elif self.mutation == "overlay_seed":
            artifact["seed_contract"]["training_overlay_protocol_seed"] = 1  # type: ignore[index]
        elif self.mutation == "transformation":
            artifact["transformation_id"] = "drifted"
        artifact["artifact_sha256"] = (
            "0" * 64
            if self.mutation == "equivalence_seal"
            else hashlib.sha256(
                json.dumps(
                    artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
        )
        return artifact

    def build_source_d(self, _source_c: str) -> SimpleNamespace:
        source_d = self.source_d
        source_d_sha256 = self.module.SOURCE_D_SHA256
        if self.mutation == "source_d_bytes":
            source_d += "# drift\n"
        elif self.mutation == "source_d_declared_sha":
            source_d_sha256 = "0" * 64
        return SimpleNamespace(
            source_d_text=source_d,
            source_d_sha256=source_d_sha256,
            artifact=self._artifact(),
        )

    def verify_source_d(
        self,
        source_c: str,
        _source_d: str,
        _artifact: object,
    ) -> SimpleNamespace:
        if self.mutation == "verify_result":
            return SimpleNamespace(
                source_d_text="different",
                source_d_sha256="0" * 64,
                artifact={},
            )
        return self.build_source_d(source_c)


def _args(**kwargs: object) -> SimpleNamespace:
    values = {"poll_seconds": 1.0, "timeout_hours": 1.0}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _world(module, monkeypatch, *, output: _OutputTask | None = None):
    source_c = "#!/usr/bin/env python3\n# sealed source C fixture\n"
    source_d = source_c + (
        "def _validate_rtx5090_runtime_contract_multi_gpu(contract):\n"
        "    return contract\n"
        "_portable_runtime_validator = "
        "_validate_rtx5090_runtime_contract_multi_gpu\n"
        "# explicit training seed\n"
    )
    source_d += "".join(
        f"# {fragment}\n" * expected_count
        for fragment, expected_count in module.STAGING_REQUIRED_FRAGMENTS
    )
    monkeypatch.setattr(module, "SOURCE_C_SHA256", _digest_text(source_c))
    monkeypatch.setattr(module, "SOURCE_D_SHA256", _digest_text(source_d))
    builder = _FakeBuilder(module, source_d)
    monkeypatch.setattr(
        module,
        "EQUIVALENCE_ARTIFACT_SHA256",
        builder._artifact()["artifact_sha256"],
    )
    source = _SourceTask(module, source_c)
    output = output or _OutputTask(module)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    return source_c, source_d, source, output, tasks, builder


def test_fixed_live_identities_and_builder_bytes_are_pinned() -> None:
    module = _load_module()
    assert hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest() == (
        "e94d5c1702bcd50d1ac79d8b2ad6a314f8986a49d5ac45f07a3d5e62f06384df"
    )
    assert module.SOURCE_C_TASK_ID == "95e72da24d464ab08d117dedabd6652e"
    assert module.SOURCE_C_PARENT_TASK_ID == ("6525107e60ae4104a2800731d74ecd4e")
    assert module.PRODUCER_ENTRY_POINT == ("clearml_formal_source_d_evidence.py")
    assert module.SOURCE_C_SHA256 == (
        "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
    )
    assert module.SOURCE_D_SHA256 == (
        "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
    )
    assert module.EQUIVALENCE_ARTIFACT_SHA256 == (
        "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
    )
    assert hashlib.sha256(BUILDER_PATH.read_bytes()).hexdigest() == (
        module.BUILDER_SOURCE_SHA256
    )
    assert module.TRANSFORMATION_ID == "source-c-to-source-d-explicit-seed-evidence-v2"
    assert module.DECLARED_REPLACEMENT_COUNT == 22
    assert module.UNCHANGED_SEGMENT_COUNT == 23
    assert module.TRAINING_OVERLAY_PROTOCOL_SEED == 20_250_218


def test_standalone_source_embeds_exact_builder_and_needs_no_checkout() -> None:
    module = _load_module()
    standalone = module.generate_standalone_source()
    assert _digest_text(standalone) == (
        "d5b759f38d39a9f349ab6e716c07fda53eb3e1635687ce4a077e0405920ffeec"
    )
    compile(standalone, "<standalone>", "exec")
    tree = ast.parse(standalone)
    encoded = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "_EMBEDDED_BUILDER_SOURCE_B64":
            encoded = ast.literal_eval(node.value)
            break
    assert isinstance(encoded, str) and encoded
    assert base64.b64decode(encoded, validate=True) == BUILDER_PATH.read_bytes()
    namespace = {
        "__name__": "standalone_source_d_evidence_fixture",
        "__file__": "/definitely/missing/clearml_formal_source_d_evidence.py",
    }
    exec(compile(standalone, "<standalone>", "exec"), namespace)
    builder = namespace["_load_builder_module"]()
    assert builder.EXPECTED_SOURCE_C_SHA256 == module.SOURCE_C_SHA256


def test_run_publishes_ordered_sealed_evidence(monkeypatch) -> None:
    module = _load_module()
    source_c, source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    receipt = module.run(_args(), task_class=tasks)
    assert tasks.requested == [module.SOURCE_C_TASK_ID]
    assert output.uploads == list(module.PUBLICATION_ORDER)
    assert output.flushes == list(module.PUBLICATION_ORDER)
    assert output.reload_calls == 12
    assert output._reload_skip_flag is True
    snapshot = output.artifacts[module.SOURCE_C_SNAPSHOT_ARTIFACT].get()
    source_d_artifact = output.artifacts[module.SOURCE_D_SCRIPT_ARTIFACT].get()
    equivalence = output.artifacts[module.EQUIVALENCE_ARTIFACT].get()
    assert snapshot["script_diff"] == source_c
    assert source_d_artifact["script"] == source_d
    assert equivalence["artifact_sha256"] == module.EQUIVALENCE_ARTIFACT_SHA256
    assert receipt["complete"] is True
    assert receipt["publication_order"] == list(module.PUBLICATION_ORDER)
    assert receipt["provenance"]["source_c_task_id"] == module.SOURCE_C_TASK_ID
    assert receipt["provenance"]["output_parent_task_id"] == (module.SOURCE_C_TASK_ID)
    assert receipt["provenance"]["builder_source_sha256"] == (
        module.BUILDER_SOURCE_SHA256
    )
    assert receipt["provenance"]["source_c_task_parent"] == (
        module.SOURCE_C_PARENT_TASK_ID
    )
    assert receipt["provenance"]["producer_entry_point"] == (
        module.PRODUCER_ENTRY_POINT
    )
    assert (
        receipt["provenance"]["producer_script_sha256"]
        == hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    )
    transformation = receipt["transformation"]
    assert transformation["declared_replacement_count"] == 22
    assert transformation["unchanged_segment_count"] == 23
    assert transformation["training_overlay_protocol_seed"] == 20_250_218
    assert transformation["portable_runner_load_marker"] == (
        module.PORTABLE_RUNNER_LOAD_MARKER
    )
    assert transformation["portable_runner_load_marker_count"] == 2
    assert transformation["legacy_runner_load_target_anchor_count"] == 0
    _verify_seal(snapshot)
    _verify_seal(source_d_artifact)
    _verify_seal(receipt)


def test_main_initializes_and_binds_the_current_clearml_task(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    _source_c, _source_d, source, output, _tasks, _builder = _world(module, monkeypatch)

    class _TaskApi:
        init_calls: list[dict[str, object]] = []

        @classmethod
        def init(cls, **kwargs: object) -> _OutputTask:
            cls.init_calls.append(dict(kwargs))
            return output

        @classmethod
        def get_task(cls, *, task_id: str) -> _SourceTask:
            assert task_id == module.SOURCE_C_TASK_ID
            return source

    monkeypatch.setattr(module, "Task", _TaskApi)
    assert module.main(["--poll-seconds", "1", "--timeout-hours", "1"]) == 0
    assert _TaskApi.init_calls == [
        {
            "project_name": module.DEFAULT_PROJECT,
            "task_name": "ResilientV2X formal Source-D transformation evidence",
            "reuse_last_task_id": False,
            "output_uri": module.FILES_SERVER_URI,
        }
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["complete"] is True


@pytest.mark.parametrize("snapshot", (None, False, 0, True, 1, ""))
def test_raw_server_reload_rejects_primitive_snapshots(
    monkeypatch, snapshot: object
) -> None:
    module = _load_module()

    class _BadSource(_SourceTask):
        def _reload(self) -> object:
            assert self._reload_skip_flag is False
            self.reload_calls += 1
            return snapshot

    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _BadSource(module, source_c)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="no snapshot"):
        module.run(_args(), task_class=tasks)
    assert source._reload_skip_flag is True
    assert output.uploads == []


@pytest.mark.parametrize("target", ("source", "output"))
def test_raw_server_reload_rejects_wrong_snapshot_task_id(
    monkeypatch, target: str
) -> None:
    module = _load_module()

    class _WrongSnapshotSource(_SourceTask):
        def _reload(self) -> object:
            snapshot = copy.copy(super()._reload())
            snapshot.id = "a" * 32
            return snapshot

    class _WrongSnapshotOutput(_OutputTask):
        def _reload(self) -> object:
            snapshot = copy.copy(super()._reload())
            snapshot.id = "b" * 32
            return snapshot

    output = _WrongSnapshotOutput(module) if target == "output" else None
    source_c, _source_d, source, output, tasks, builder = _world(
        module, monkeypatch, output=output
    )
    if target == "source":
        source = _WrongSnapshotSource(module, source_c)
        tasks = _Tasks(source, output)
        monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="snapshot identity mismatch"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_public_reload_path_is_not_trusted(monkeypatch) -> None:
    module = _load_module()

    class _PublicReloadMustNotRun(_SourceTask):
        def reload(self) -> None:
            raise AssertionError("public reload swallows ClearML backend errors")

    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _PublicReloadMustNotRun(module, source_c)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    receipt = module.run(_args(), task_class=tasks)
    assert receipt["complete"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("repository", "https://example.invalid/repo.git", "standalone"),
        ("working_dir", "tools", "working directory"),
        ("entry_point", "other.py", "entry point"),
        ("script_diff", b"not text", "raw text"),
        ("script_diff", "# changed\n", "SHA-256"),
    ),
)
def test_source_script_metadata_and_sha_drift_fail_closed(
    monkeypatch, field: str, value: object, message: str
) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    values = {field: value}
    source = _SourceTask(module, source_c, **values)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_source_identity_mismatch_fails_before_publication(monkeypatch) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(module, source_c, task_id="a" * 32)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="identity mismatch"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_integer_output_task_id_is_rejected_before_publication(monkeypatch) -> None:
    module = _load_module()
    output = _OutputTask(module, task_id=int("1" * 32))
    _source_c, _source_d, source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="lowercase 32-hex"):
        module.run(_args(), task_class=tasks)
    assert source.reload_calls == 0
    assert output.uploads == []


@pytest.mark.parametrize("parent", ("", "1" * 32))
def test_wrong_source_parent_fails_before_wait_or_publication(
    monkeypatch, parent: str
) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(module, source_c, parent=parent)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="parent mismatch"):
        module.run(_args(), task_class=tasks)
    assert source.reload_calls == 0
    assert output.uploads == []


@pytest.mark.parametrize("parent", (False, 0))
def test_nonstring_source_parent_is_not_treated_as_root(
    monkeypatch, parent: object
) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(module, source_c, parent=parent)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="task parent must"):
        module.run(_args(), task_class=tasks)
    assert source.reload_calls == 0
    assert output.uploads == []


@pytest.mark.parametrize("status", ("failed", "stopped", "published", "mystery"))
def test_noncompleted_source_status_fails_closed(monkeypatch, status: str) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(module, source_c, status=status)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_source_waits_then_completes(monkeypatch) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(
        module,
        source_c,
        status="queued",
        statuses=["in_progress", "completed"],
    )
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    sleeps: list[float] = []
    module.run(_args(), task_class=tasks, sleeper=sleeps.append)
    assert sleeps == [1.0]
    assert source.reload_calls == 14
    assert source._reload_skip_flag is True


def test_final_source_status_drift_fails_before_publication(monkeypatch) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(
        module,
        source_c,
        status="completed",
        statuses=["completed", "failed"],
    )
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="final status drifted"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_final_source_identity_drift_fails_before_publication(monkeypatch) -> None:
    module = _load_module()

    class _DriftingSource(_SourceTask):
        def _reload(self) -> object:
            result = super()._reload()
            if self.reload_calls == 2:
                self.id = "a" * 32
            return result

    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _DriftingSource(module, source_c)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="final identity drifted"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_final_source_parent_drift_fails_before_publication(monkeypatch) -> None:
    module = _load_module()

    class _DriftingSource(_SourceTask):
        def _reload(self) -> object:
            result = super()._reload()
            if self.reload_calls == 2:
                self.parent = "1" * 32
                self.data.parent = self.parent
            return result

    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _DriftingSource(module, source_c)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="final parent drifted"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize("drifted_parent", (False, 0))
def test_final_source_parent_type_drift_fails_before_publication(
    monkeypatch, drifted_parent: object
) -> None:
    module = _load_module()

    class _DriftingSource(_SourceTask):
        def _reload(self) -> object:
            result = super()._reload()
            if self.reload_calls == 2:
                self.parent = drifted_parent
                self.data.parent = drifted_parent
            return result

    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _DriftingSource(module, source_c)
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="task parent must"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_final_output_parent_drift_fails_before_publication(monkeypatch) -> None:
    module = _load_module()

    class _DriftingOutput(_OutputTask):
        def _reload(self) -> object:
            result = super()._reload()
            if self.reload_calls == 1:
                self.parent = "a" * 32
                self.data.parent = self.parent
            return result

    output = _DriftingOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="output parent drifted"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize("parent", (False, 0))
def test_nonstring_output_parent_is_rejected_before_publication(
    monkeypatch, parent: object
) -> None:
    module = _load_module()
    output = _OutputTask(module, parent=parent)
    _source_c, _source_d, source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="task parent must"):
        module.run(_args(), task_class=tasks)
    assert source.reload_calls == 0
    assert output.uploads == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("repository", "https://example.invalid/repo.git", "standalone"),
        ("working_dir", "tools", "working directory"),
        ("entry_point", "other.py", "entry point"),
        ("script_diff", b"not text", "raw text"),
        ("script_diff", "", "raw text"),
        ("script_diff", "# attacker-controlled producer\n", "script bytes"),
    ),
)
def test_initial_output_producer_metadata_or_bytes_drift_fails_closed(
    monkeypatch, field: str, value: object, message: str
) -> None:
    module = _load_module()
    output = _OutputTask(module, **{field: value})
    _source_c, _source_d, source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert source.reload_calls == 0
    assert output.uploads == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("repository", "https://example.invalid/repo.git", "standalone"),
        ("working_dir", "tools", "working directory"),
        ("entry_point", "other.py", "entry point"),
        ("diff", "# attacker-controlled producer\n", "script bytes"),
    ),
)
def test_final_output_producer_metadata_or_bytes_drift_fails_closed(
    monkeypatch, field: str, value: object, message: str
) -> None:
    module = _load_module()

    class _DriftingOutput(_OutputTask):
        def _reload(self) -> object:
            result = super()._reload()
            if self.reload_calls == 1:
                setattr(self.data.script, field, value)
            return result

    output = _DriftingOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_source_wait_timeout_has_no_artifacts(monkeypatch) -> None:
    module = _load_module()
    source_c, _source_d, _source, output, _tasks, builder = _world(module, monkeypatch)
    source = _SourceTask(module, source_c, status="in_progress")
    tasks = _Tasks(source, output)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    values = iter((0.0, 3601.0))
    with pytest.raises(TimeoutError, match="Source-C"):
        module.run(
            _args(),
            task_class=tasks,
            monotonic_clock=lambda: next(values),
            sleeper=lambda _seconds: None,
        )
    assert output.uploads == []


@pytest.mark.parametrize(
    ("task_id", "parent", "alias_object", "message"),
    (
        ("f" * 32, "a" * 32, False, "parent mismatch"),
        ("95e72da24d464ab08d117dedabd6652e", None, False, "aliases"),
        ("not-an-id", None, False, "32-hex"),
        ("95e72da24d464ab08d117dedabd6652e", None, True, "aliases"),
    ),
)
def test_output_parent_identity_and_alias_are_rejected(
    monkeypatch,
    task_id: str,
    parent: str | None,
    alias_object: bool,
    message: str,
) -> None:
    module = _load_module()
    source_c, _source_d, source, _output, _tasks, builder = _world(module, monkeypatch)
    if alias_object:
        source.id = task_id
        output = source
    else:
        output = _OutputTask(module, task_id=task_id, parent=parent)
    tasks = _Tasks(source, output)  # type: ignore[arg-type]
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)


def test_missing_current_output_task_is_rejected(monkeypatch) -> None:
    module = _load_module()
    _source_c, _source_d, source, _output, _tasks, builder = _world(module, monkeypatch)
    tasks = _Tasks(source, None)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError, match="current output"):
        module.run(_args(), task_class=tasks)


def test_builder_byte_drift_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    drifted = tmp_path / "formal_source_d_seed.py"
    drifted.write_bytes(BUILDER_PATH.read_bytes() + b"\n# drift\n")
    with pytest.raises(module.SourceDEvidenceError, match="builder byte SHA-256"):
        module._builder_source_bytes(drifted)


@pytest.mark.parametrize(
    "mutation",
    (
        "source_d_bytes",
        "source_d_declared_sha",
        "verify_result",
        "equivalence_seal",
        "replacement_count",
        "unchanged_count",
        "overlay_seed",
        "transformation",
    ),
)
def test_builder_output_drift_never_publishes(monkeypatch, mutation: str) -> None:
    module = _load_module()
    _source_c, source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    builder = _FakeBuilder(module, source_d, mutation=mutation)
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)
    with pytest.raises(module.SourceDEvidenceError):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("builder_staging_anchors", "builder staging anchors"),
        ("staging_anchor", "staging replacement evidence"),
        ("source_c_replay", "equivalence evidence"),
        ("float_expected_count", "replacement evidence"),
        ("float_unchanged_count", "equivalence evidence"),
        ("float_overlay_seed", "overlay seed contract"),
    ),
)
def test_resealed_builder_anchor_replay_and_float_drift_never_publishes(
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    module = _load_module()
    _source_c, source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    builder = _FakeBuilder(module, source_d, mutation=mutation)
    monkeypatch.setattr(
        module,
        "EQUIVALENCE_ARTIFACT_SHA256",
        builder._artifact()["artifact_sha256"],
    )
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)

    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize("mutation", ("portable_marker", "legacy_target"))
def test_builder_cannot_repin_nonportable_source_d(monkeypatch, mutation: str) -> None:
    module = _load_module()
    _source_c, source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    if mutation == "portable_marker":
        source_d = source_d.replace(
            module.PORTABLE_RUNNER_LOAD_MARKER,
            "_drifted_runtime_contract_multi_gpu",
            1,
        )
        message = "portable runner-load marker count"
    else:
        source_d += module.LEGACY_RUNNER_LOAD_TARGET_ANCHOR
        message = "legacy runner-load target anchor"
    monkeypatch.setattr(module, "SOURCE_D_SHA256", _digest_text(source_d))
    builder = _FakeBuilder(module, source_d)
    monkeypatch.setattr(
        module,
        "EQUIVALENCE_ARTIFACT_SHA256",
        builder._artifact()["artifact_sha256"],
    )
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)

    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize(
    ("original", "mutation"),
    (
        (
            "if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (",
            "if len(ordered) != 38.0 or sum(item.size_bytes for item in ordered) > (",
        ),
        ("changed during readback", "changed after readback"),
        ("st_nlink != 1", "st_nlink < 1"),
        (
            "or entries_on_disk != 38\n",
            "or entries_on_disk != 38.0\n",
        ),
        (
            'reloader = getattr(task, "_reload", None)',
            'reloader = getattr(task, "reload", None)',
        ),
        (
            '("controlled_baseline_evidence", str(evidence_stage.root))',
            '("controlled_baseline_evidence", str(work_dir))',
        ),
    ),
)
def test_builder_cannot_repin_staging_security_drift(
    monkeypatch,
    original: str,
    mutation: str,
) -> None:
    module = _load_module()
    _source_c, source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    assert original in source_d
    source_d = source_d.replace(original, mutation, 1)
    monkeypatch.setattr(module, "SOURCE_D_SHA256", _digest_text(source_d))
    builder = _FakeBuilder(module, source_d)
    monkeypatch.setattr(
        module,
        "EQUIVALENCE_ARTIFACT_SHA256",
        builder._artifact()["artifact_sha256"],
    )
    monkeypatch.setattr(module, "_load_builder_module", lambda _path=None: builder)

    with pytest.raises(
        module.SourceDEvidenceError,
        match="staging invariant count|unsafe evidence upload",
    ):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


@pytest.mark.parametrize("failure_index", range(4))
def test_upload_failure_stops_before_complete_receipt(
    monkeypatch, failure_index: int
) -> None:
    module = _load_module()
    output = _OutputTask(module, upload_failure=module.PUBLICATION_ORDER[failure_index])
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="failed to upload"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER[: failure_index + 1])
    assert module.RECEIPT_ARTIFACT not in output.artifacts


@pytest.mark.parametrize("failure_index", range(4))
def test_flush_false_stops_before_later_publication(
    monkeypatch, failure_index: int
) -> None:
    module = _load_module()
    output = _OutputTask(module, flush_failure=module.PUBLICATION_ORDER[failure_index])
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="failed to flush"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER[: failure_index + 1])
    if failure_index < 3:
        assert module.RECEIPT_ARTIFACT not in output.artifacts


def test_flush_integer_zero_stops_before_later_publication(monkeypatch) -> None:
    module = _load_module()

    class _ZeroFlushOutput(_OutputTask):
        def flush(self, *, wait_for_uploads: bool) -> int:
            super().flush(wait_for_uploads=wait_for_uploads)
            return 0

    output = _ZeroFlushOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="failed to flush"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == [module.SOURCE_C_SNAPSHOT_ARTIFACT]
    assert module.RECEIPT_ARTIFACT not in output.artifacts


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("source_parent", "publication parent drifted"),
        ("output_parent", "output parent drifted"),
        ("producer", "script bytes"),
    ),
)
def test_publication_midflight_binding_drift_stops_after_current_upload(
    monkeypatch, drift: str, message: str
) -> None:
    module = _load_module()

    class _MidflightDriftOutput(_OutputTask):
        source: _SourceTask

        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            uploaded = super().upload_artifact(
                name,
                artifact_object=artifact_object,
                wait_on_upload=wait_on_upload,
            )
            if len(self.uploads) == 1:
                if drift == "source_parent":
                    self.source.parent = "1" * 32
                    self.source.data.parent = self.source.parent
                elif drift == "output_parent":
                    self.parent = "2" * 32
                    self.data.parent = self.parent
                else:
                    self.data.script.diff = "# producer drift during publication\n"
            return uploaded

    output = _MidflightDriftOutput(module)
    _source_c, _source_d, source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    output.source = source
    with pytest.raises(module.SourceDEvidenceError, match=message):
        module.run(_args(), task_class=tasks)
    assert output.uploads == [module.SOURCE_C_SNAPSHOT_ARTIFACT]
    assert module.RECEIPT_ARTIFACT not in output.artifacts


@pytest.mark.parametrize("mode", ("readback_drift", "readback_missing"))
def test_readback_failure_stops_before_receipt(monkeypatch, mode: str) -> None:
    module = _load_module()
    target = module.SOURCE_D_SCRIPT_ARTIFACT
    output = _OutputTask(module, **{mode: target})
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="artifact|seal"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER[:2])
    assert module.RECEIPT_ARTIFACT not in output.artifacts


def test_existing_artifact_drift_fails_before_new_upload(monkeypatch) -> None:
    module = _load_module()
    _source_c, _source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    output.artifacts[module.SOURCE_C_SNAPSHOT_ARTIFACT] = _Artifact({"drift": True})
    with pytest.raises(module.SourceDEvidenceError, match="drifted|seal"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_retained_upload_alias_cannot_mutate_frozen_nested_expected(
    monkeypatch,
) -> None:
    module = _load_module()

    class _RetainingOutput(_OutputTask):
        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            assert wait_on_upload is True
            self.uploads.append(name)
            self._last_upload = name
            self.artifacts[name] = _Artifact(artifact_object)
            if name == module.SOURCE_D_SCRIPT_ARTIFACT:
                snapshot = self.artifacts[module.SOURCE_C_SNAPSHOT_ARTIFACT].value
                assert isinstance(snapshot, dict)
                script = snapshot["script"]
                assert isinstance(script, dict)
                script["sha256"] = "0" * 64
            return True

    output = _RetainingOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="seal|canonical"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER[:3])
    assert module.RECEIPT_ARTIFACT not in output.artifacts


def test_boolean_schema_version_cannot_alias_integer_one(monkeypatch) -> None:
    module = _load_module()

    class _BooleanSchemaOutput(_OutputTask):
        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            assert wait_on_upload is True
            self.uploads.append(name)
            self._last_upload = name
            value = copy.deepcopy(artifact_object)
            assert isinstance(value, dict)
            if name == module.SOURCE_C_SNAPSHOT_ARTIFACT:
                assert value["schema_version"] == 1
                value["schema_version"] = True
                _reseal(value)
            self.artifacts[name] = _Artifact(value)
            return True

    output = _BooleanSchemaOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="canonical"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == [module.SOURCE_C_SNAPSHOT_ARTIFACT]
    assert module.RECEIPT_ARTIFACT not in output.artifacts


@pytest.mark.parametrize(
    "mutation", ("nested_seed", "nested_seed_rehashed", "artifact_hash")
)
def test_equivalence_readback_hash_is_recomputed_live(
    monkeypatch, mutation: str
) -> None:
    module = _load_module()

    class _MutatingEquivalenceOutput(_OutputTask):
        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            assert wait_on_upload is True
            self.uploads.append(name)
            self._last_upload = name
            value = copy.deepcopy(artifact_object)
            assert isinstance(value, dict)
            if name == module.EQUIVALENCE_ARTIFACT:
                if mutation.startswith("nested_seed"):
                    seed_contract = value["seed_contract"]
                    assert isinstance(seed_contract, dict)
                    seed_contract["training_overlay_protocol_seed"] = 1
                if mutation == "nested_seed_rehashed":
                    _rehash_equivalence(value)
                elif mutation == "artifact_hash":
                    value["artifact_sha256"] = "0" * 64
            self.artifacts[name] = _Artifact(value)
            return True

    output = _MutatingEquivalenceOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="artifact SHA-256|canonical"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER[:3])
    assert module.RECEIPT_ARTIFACT not in output.artifacts


def test_receipt_cross_artifact_mutation_fails_even_when_resealed(
    monkeypatch,
) -> None:
    module = _load_module()

    class _MutatingReceiptOutput(_OutputTask):
        def upload_artifact(
            self,
            name: str,
            *,
            artifact_object: object,
            wait_on_upload: bool,
        ) -> bool:
            assert wait_on_upload is True
            self.uploads.append(name)
            self._last_upload = name
            value = copy.deepcopy(artifact_object)
            assert isinstance(value, dict)
            if name == module.RECEIPT_ARTIFACT:
                artifact_hashes = value["artifact_hashes"]
                assert isinstance(artifact_hashes, dict)
                artifact_hashes[module.EQUIVALENCE_ARTIFACT] = "0" * 64
                _reseal(value)
            self.artifacts[name] = _Artifact(value)
            return True

    output = _MutatingReceiptOutput(module)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="artifact hashes"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == list(module.PUBLICATION_ORDER)


def test_out_of_order_existing_artifact_is_rejected(monkeypatch) -> None:
    module = _load_module()
    _source_c, _source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    output.artifacts[module.SOURCE_D_SCRIPT_ARTIFACT] = _Artifact({})
    with pytest.raises(module.SourceDEvidenceError, match="publication order"):
        module.run(_args(), task_class=tasks)
    assert output.uploads == []


def test_exact_existing_artifacts_are_idempotent(monkeypatch) -> None:
    module = _load_module()
    _source_c, _source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    first = module.run(_args(), task_class=tasks)
    uploads = list(output.uploads)
    flushes = list(output.flushes)
    second = module.run(_args(), task_class=tasks)
    assert second == first
    assert output.uploads == uploads
    assert output.flushes == flushes


def test_exact_partial_prefix_can_resume_after_failed_flush(monkeypatch) -> None:
    module = _load_module()
    output = _OutputTask(module, flush_failure=module.SOURCE_D_SCRIPT_ARTIFACT)
    _source_c, _source_d, _source, output, tasks, _builder = _world(
        module, monkeypatch, output=output
    )
    with pytest.raises(module.SourceDEvidenceError, match="failed to flush"):
        module.run(_args(), task_class=tasks)
    assert list(output.artifacts) == list(module.PUBLICATION_ORDER[:2])
    output.flush_failure = None
    receipt = module.run(_args(), task_class=tasks)
    assert receipt["complete"] is True
    assert output.uploads == list(module.PUBLICATION_ORDER)


@pytest.mark.parametrize(
    ("poll_seconds", "timeout_hours"),
    ((0, 1), (1, 0), (float("nan"), 1), (1, float("inf")), (True, 1)),
)
def test_invalid_wait_configuration_is_rejected(
    monkeypatch, poll_seconds: object, timeout_hours: object
) -> None:
    module = _load_module()
    _source_c, _source_d, _source, output, tasks, _builder = _world(module, monkeypatch)
    with pytest.raises(ValueError, match="finite and positive"):
        module.run(
            _args(poll_seconds=poll_seconds, timeout_hours=timeout_hours),
            task_class=tasks,
        )
    assert output.uploads == []
