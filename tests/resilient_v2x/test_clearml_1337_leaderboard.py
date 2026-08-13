from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_1337_leaderboard.py"
CONTROLLER_PATH = ROOT / "tools/resilient_v2x/clearml_5090_training_controller.py"
WATCHER_PATH = ROOT / "tools/resilient_v2x/clearml_1337_dependency_watcher.py"
CONTROLLER_ID = "e" * 32
WATCHER_ID = "f" * 32


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "clearml_1337_leaderboard",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


class _Task:
    def __init__(
        self,
        task_id: str,
        *,
        status: str = "completed",
        artifacts: dict[str, _Artifact] | None = None,
        parameters: dict[str, object] | None = None,
        input_model_id: str | None = None,
        reload_statuses: list[str] | None = None,
    ) -> None:
        self.id = task_id
        self.status = status
        self.artifacts = artifacts or {}
        self.parameters = parameters or {}
        self.input_model_id = input_model_id
        self.reload_statuses = list(reload_statuses or [])
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1
        if self.reload_statuses:
            self.status = self.reload_statuses.pop(0)

    def get_parameters(self, cast: bool = False) -> dict[str, object]:
        assert cast is False
        return dict(self.parameters)

    def get_models(self) -> dict[str, list[object]]:
        inputs = (
            []
            if self.input_model_id is None
            else [SimpleNamespace(id=self.input_model_id)]
        )
        return {"input": inputs, "output": []}


class _OutputTask(_Task):
    def __init__(self) -> None:
        super().__init__("c" * 32)
        self.uploads: list[tuple[str, object, bool]] = []
        self.tags: list[str] = []
        self.flush_calls = 0

    def set_tags(self, tags: list[str]) -> None:
        self.tags = list(tags)

    def upload_artifact(
        self,
        name: str,
        *,
        artifact_object: object,
        wait_on_upload: bool,
    ) -> bool:
        self.uploads.append((name, artifact_object, wait_on_upload))
        self.artifacts[name] = _Artifact(artifact_object)
        return True

    def flush(self, *, wait_for_uploads: bool) -> None:
        assert wait_for_uploads is True
        self.flush_calls += 1


class _TaskClass:
    tasks: dict[str, _Task] = {}
    current: _Task | None = None

    @classmethod
    def get_task(cls, *, task_id: str) -> _Task:
        return cls.tasks[task_id]

    @classmethod
    def current_task(cls) -> _Task | None:
        return cls.current


def _training_manifest(module, *, seeded: bool = True) -> dict[str, object]:
    entries = []
    for index, subject in enumerate(module.FORMAL_SUBJECT_ORDER, start=1):
        entry = {
            "index": index,
            "subject": subject,
            "kind": module.SUBJECT_KIND[subject],
            "training_task_id": f"{index:032x}",
            "training_predecessor_task_id": f"{index + 500:032x}",
            "model_id": f"{index + 100:032x}",
            "model_name": f"ResilientV2X {subject} final checkpoint",
            "model_url": (f"http://10.100.34.118:8081/models/{subject}_epoch_50.pth"),
            "checkpoint_filename": "epoch_50.pth",
            "checkpoint_sha256": f"{index:064x}",
            "checkpoint_size_bytes": 1000 + index,
            "common_teacher_initialization_audit_artifact": (
                "common_teacher_initialization_audit"
            ),
            "common_teacher_initialization_audit_sha256": "a" * 64,
        }
        if seeded:
            entry.update(
                {
                    "training_seed": module.DEFAULT_TRAINING_SEED,
                    "training_overlay_protocol_seed": (
                        module.TRAINING_OVERLAY_PROTOCOL_SEED
                    ),
                }
            )
        entries.append(entry)
    payload = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": module.PROTOCOL_ID,
        "sample_count": module.SAMPLE_COUNT,
        "delays_ms": list(module.DELAYS_MS),
        "conditions": list(module.CONDITIONS),
        "run_count": 12,
        "checkpoint_policy": module.CHECKPOINT_POLICY,
        "evaluation_release_semantics": module.RELEASE_SEMANTICS,
        "subject_order": list(module.FORMAL_SUBJECT_ORDER),
        "subject_count": 26,
        "entries": entries,
    }
    if seeded:
        payload.update(
            {
                "training_seed": module.DEFAULT_TRAINING_SEED,
                "training_overlay_protocol_seed": (
                    module.TRAINING_OVERLAY_PROTOCOL_SEED
                ),
            }
        )
    return module._seal(payload)


def _evaluation_plan(module) -> dict[str, object]:
    queues = ("GPU4-A100", "GPU4-A100", "GPU4-V100", "GPU4-5090")
    entries = [
        {
            "subject": subject,
            "evaluation_task_id": f"{index + 1000:032x}",
            "queue": queues[(index - 1) % len(queues)],
        }
        for index, subject in enumerate(module.FORMAL_SUBJECT_ORDER, start=1)
    ]
    source_equivalence = module._seal(
        {
            "schema_version": 1,
            "contract": "fixture-source-equivalence",
            "source_revisions": module.SOURCE_REVISION_CERTIFICATE_BY_TREE,
        }
    )
    module.SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = source_equivalence[
        "seal_sha256"
    ]
    source_subject_map = module._seal(
        {
            "schema_version": 1,
            "contract": "fixture-source-subject-map",
            "source_revision_by_subject": {
                subject: module.SOURCE_TREE_BY_REVISION[
                    module.SOURCE_REVISION_BY_SUBJECT[subject]
                ]
                for subject in module.FORMAL_SUBJECT_ORDER
            },
        }
    )
    module.SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = source_subject_map[
        "seal_sha256"
    ]
    return module._seal(
        {
            "schema_version": 2,
            "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
            "training_controller_task_id": CONTROLLER_ID,
            "training_provenance_task_id": "e" * 32,
            "training_provenance_seal_sha256": "f" * 64,
            "source_revision_equivalence": source_equivalence,
            "source_revision_equivalence_seal_sha256": source_equivalence[
                "seal_sha256"
            ],
            "source_revision_subject_map": source_subject_map,
            "source_revision_subject_map_seal_sha256": source_subject_map[
                "seal_sha256"
            ],
            "evaluation_source_revision_tree_sha256": (
                module.OLD_SOURCE_TREE_SHA256
            ),
            "evaluation_source_revision": module.SOURCE_REVISION_CERTIFICATE_BY_TREE[
                module.OLD_SOURCE_TREE_SHA256
            ],
            "evaluation_template_task_id": "d" * 32,
            "protocol_id": module.PROTOCOL_ID,
            "sample_count": module.SAMPLE_COUNT,
            "delays_ms": list(module.DELAYS_MS),
            "conditions": list(module.CONDITIONS),
            "run_count": 12,
            "subject_order": list(module.FORMAL_SUBJECT_ORDER),
            "entries": entries,
        }
    )


def _subject_base(module, subject: str) -> float:
    if subject == module.PRIMARY_SUBJECT:
        return 85.0
    if subject == "support_residual":
        return 90.0
    if subject in module.BASELINE_SUBJECTS:
        return 50.0 + module.BASELINE_SUBJECTS.index(subject)
    return 75.0


def _metrics(module, subject: str, checkpoint_sha256: str) -> dict[str, object]:
    metric_offsets = {
        "resilient_v2x/car_bev_ap_r40_0.50": 5.0,
        "resilient_v2x/car_bev_ap_r40_0.70": 0.0,
        "resilient_v2x/car_3d_ap_r40_0.50": -3.0,
        "resilient_v2x/car_3d_ap_r40_0.70": -10.0,
    }
    condition_penalties = {"Full": 0.0, "L-Fail": 10.0, "C-Fail": 5.0}
    base = _subject_base(module, subject)
    runs = []
    for delay in module.DELAYS_MS:
        for condition in module.CONDITIONS:
            metrics = {
                key: base + offset - delay / 100.0 - condition_penalties[condition]
                for key, offset in metric_offsets.items()
            }
            metrics.update(module.COUNT_METRICS)
            runs.append(
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "metrics": metrics,
                    "predictions": f"/evidence/{subject}/{delay}/{condition}.json",
                    "prediction_sha256": "b" * 64,
                    "prediction_content_sha256": "c" * 64,
                    "sample_count": module.SAMPLE_COUNT,
                    "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
                    "ground_truth_count": module.GROUND_TRUTH_COUNT,
                    "unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
                }
            )
    return {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": 12,
        "baseline": subject,
        "protocol_id": module.PROTOCOL_ID,
        "checkpoint": f"/checkpoints/{subject}_epoch_50.pth",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": module.MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": module.OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": module.SAMPLE_IDS_SHA256,
        "expected_sample_count": module.SAMPLE_COUNT,
        "expected_ground_truth_count": module.GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": module.UNSUPPORTED_SAMPLE_COUNT,
        "runs": runs,
    }


def _world(
    module,
    *,
    seeded: bool = True,
    controller_status: str = "completed",
    watcher_status: str = "completed",
    controller_reload_statuses: list[str] | None = None,
    watcher_reload_statuses: list[str] | None = None,
):
    training_manifest = _training_manifest(module, seeded=seeded)
    evaluation_plan = _evaluation_plan(module)
    controller = _Task(
        CONTROLLER_ID,
        status=controller_status,
        artifacts={module.TRAINING_MANIFEST_ARTIFACT: _Artifact(training_manifest)},
        reload_statuses=controller_reload_statuses,
    )
    watcher = _Task(
        WATCHER_ID,
        status=watcher_status,
        artifacts={module.EVALUATION_PLAN_ARTIFACT: _Artifact(evaluation_plan)},
        reload_statuses=watcher_reload_statuses,
    )
    tasks: dict[str, _Task] = {
        CONTROLLER_ID: controller,
        WATCHER_ID: watcher,
    }
    for training, evaluation in zip(
        training_manifest["entries"],
        evaluation_plan["entries"],
        strict=True,
    ):
        subject = training["subject"]
        tasks[evaluation["evaluation_task_id"]] = _Task(
            evaluation["evaluation_task_id"],
            parameters={
                "Args/stage": "baseline_validate",
                "Args/source_dataset_id": module.OLD_SOURCE_DATASET_ID,
                "Args/source_archive_name": module.OLD_SOURCE_ARCHIVE_NAME,
                "Args/source_archive_bytes": str(module.OLD_SOURCE_ARCHIVE_BYTES),
                "Args/source_archive_sha256": module.OLD_SOURCE_ARCHIVE_SHA256,
                "Args/controlled_baseline": subject,
                "Args/controlled_baseline_task_id": training["training_task_id"],
                "Args/controlled_baseline_model_id": training["model_id"],
                "Args/controlled_baseline_checkpoint_sha256": training[
                    "checkpoint_sha256"
                ],
                "Args/predecessor_task_id": training["training_task_id"],
                "Args/gpus": "4",
                "Args/max_epochs": "50",
                "Args/amp": "False",
            },
            input_model_id=training["model_id"],
            artifacts={
                module.METRICS_ARTIFACT: _Artifact(
                    _metrics(module, subject, training["checkpoint_sha256"])
                )
            },
        )
    _TaskClass.tasks = tasks
    output = _OutputTask()
    _TaskClass.current = output
    return training_manifest, evaluation_plan, controller, watcher, output


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        watcher_task_id=WATCHER_ID,
        training_controller_task_id=CONTROLLER_ID,
        poll_seconds=1.0,
        timeout_hours=1.0,
    )


def test_registry_has_exact_26_subjects_and_14_external_baselines() -> None:
    module = _load_module()
    controller = _load_path("leaderboard_controller_contract", CONTROLLER_PATH)
    watcher = _load_path("leaderboard_watcher_contract", WATCHER_PATH)

    assert len(module.FORMAL_SUBJECT_ORDER) == 26
    assert len(set(module.FORMAL_SUBJECT_ORDER)) == 26
    assert len(module.BASELINE_SUBJECTS) == 14
    assert set(module.BASELINE_SUBJECTS) < set(module.FORMAL_SUBJECT_ORDER)
    assert module.PRIMARY_SUBJECT not in module.BASELINE_SUBJECTS
    assert not set(module.BASELINE_SUBJECTS) & module.ABLATION_SUBJECTS
    assert not set(module.BASELINE_SUBJECTS) & module.IMPROVEMENT_SUBJECTS
    assert module.FORMAL_SUBJECT_ORDER == controller.EXPERIMENT_ORDER
    assert module.FORMAL_SUBJECT_ORDER == watcher.FORMAL_SUBJECT_ORDER
    assert module.BASELINE_SUBJECTS == tuple(
        spec.name for spec in controller.EXPERIMENT_SPECS if spec.kind == "baseline"
    )


def test_run_builds_sealed_leaderboard_for_all_26_subjects() -> None:
    module = _load_module()
    _, _, _, _, output = _world(module)

    leaderboard = module.run(
        _args(),
        task_class=_TaskClass,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=lambda _: None,
    )

    assert leaderboard["subject_count"] == 26
    assert leaderboard["baseline_count"] == 14
    assert leaderboard["schema_version"] == 3
    assert leaderboard["training_seed"] == module.DEFAULT_TRAINING_SEED
    assert leaderboard["training_overlay_protocol_seed"] == (
        module.TRAINING_OVERLAY_PROTOCOL_SEED
    )
    assert leaderboard["subject_order"] == list(module.FORMAL_SUBJECT_ORDER)
    assert len(leaderboard["results"]) == 26
    assert {result["subject"] for result in leaderboard["results"]} == set(
        module.FORMAL_SUBJECT_ORDER
    )
    assert {
        result["source_revision_tree_sha256"]
        for result in leaderboard["results"]
    } == {module.OLD_SOURCE_TREE_SHA256, module.NEW_SOURCE_TREE_SHA256}
    assert module._seal(leaderboard) == leaderboard
    assert output.uploads == [(module.LEADERBOARD_ARTIFACT, leaderboard, True)]
    assert output.flush_calls == 1

    primary = next(
        result
        for result in leaderboard["results"]
        if result["subject"] == module.PRIMARY_SUBJECT
    )
    summary = primary["metrics"][module.LEADERSHIP_METRIC]
    assert summary["clean_0ms"] == 85.0
    assert summary["mean_12"] == pytest.approx(78.5)
    assert summary["worst_12"] == {
        "value": 72.0,
        "delay_ms": 300,
        "condition": "L-Fail",
    }
    assert summary["full_mean"] == pytest.approx(83.5)
    assert summary["l_fail_mean"] == pytest.approx(73.5)
    assert summary["c_fail_mean"] == pytest.approx(78.5)
    assert summary["full_300"] == 82.0
    assert summary["pdr"] == pytest.approx(100.0 * 3.0 / 85.0)

    leadership = leaderboard["leadership"]
    assert leadership["comparison_pool"] == list(module.BASELINE_SUBJECTS)
    assert "support_residual" not in leadership["comparison_pool"]
    assert leadership["aggregate_dimensions"]["clean_0ms"] == {
        "resilient_v2x_value": 85.0,
        "best_baseline_value": 63.0,
        "best_baseline_subject": "how2comm",
        "best_baseline_subjects": ["how2comm"],
        "margin": 22.0,
        "leads": True,
    }
    assert leadership["aggregate_dimensions"]["mean_12"]["margin"] == pytest.approx(
        22.0
    )
    assert leadership["aggregate_dimensions"]["worst_12"]["margin"] == pytest.approx(
        22.0
    )
    assert leadership["leads_all_aggregate"] is True
    assert len(leadership["condition_comparisons"]) == 12
    assert {item["margin"] for item in leadership["condition_comparisons"]} == {22.0}
    assert leadership["conditions_won"] == 12
    assert leadership["conditions_won_fraction"] == "12/12"
    assert leadership["leads_all_conditions"] is True


def test_legacy_manifest_remains_supported_without_inventing_seed_evidence() -> None:
    module = _load_module()
    _, _, _, _, output = _world(module, seeded=False)

    leaderboard = module.run(
        _args(),
        task_class=_TaskClass,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=lambda _: None,
    )

    assert leaderboard["schema_version"] == 3
    assert "training_seed" not in leaderboard
    assert "training_overlay_protocol_seed" not in leaderboard


def test_run_waits_for_controller_and_watcher_completion() -> None:
    module = _load_module()
    _, _, controller, watcher, output = _world(
        module,
        controller_status="created",
        watcher_status="created",
        controller_reload_statuses=["in_progress", "completed"],
        watcher_reload_statuses=["in_progress", "completed"],
    )
    sleeps: list[float] = []

    module.run(
        _args(),
        task_class=_TaskClass,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert sleeps == [1.0]
    assert controller.reload_calls >= 2
    assert watcher.reload_calls >= 2


def test_metrics_artifact_accepts_downloaded_json_path(tmp_path: Path) -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][0]["evaluation_task_id"]
    evaluation_task = _TaskClass.tasks[evaluation_task_id]
    metrics = evaluation_task.artifacts[module.METRICS_ARTIFACT].value
    metrics_path = tmp_path / "controlled_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    evaluation_task.artifacts[module.METRICS_ARTIFACT] = _Artifact(metrics_path)

    leaderboard = module.run(
        _args(),
        task_class=_TaskClass,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=lambda _: None,
    )

    assert leaderboard["subject_count"] == 26
    assert leaderboard["leadership"]["conditions_won"] == 12


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (float("nan"), "finite and within"),
        (101.0, "finite and within"),
        (None, "must be numeric"),
    ],
)
def test_ap_gate_rejects_nonfinite_out_of_range_or_missing(
    replacement: object,
    message: str,
) -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][10]["evaluation_task_id"]
    artifact = _TaskClass.tasks[evaluation_task_id].artifacts[module.METRICS_ARTIFACT]
    artifact.value["runs"][0]["metrics"][module.LEADERSHIP_METRIC] = replacement

    with pytest.raises(ValueError, match=message):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


@pytest.mark.parametrize(
    ("parameter", "replacement"),
    [
        ("Args/controlled_baseline_task_id", "1" * 32),
        ("Args/controlled_baseline_model_id", "2" * 32),
        ("Args/controlled_baseline_checkpoint_sha256", "3" * 64),
    ],
)
def test_join_rejects_training_task_model_or_checkpoint_drift(
    parameter: str,
    replacement: str,
) -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][5]["evaluation_task_id"]
    _TaskClass.tasks[evaluation_task_id].parameters[parameter] = replacement

    with pytest.raises(RuntimeError, match=parameter):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_metrics_gate_rejects_count_or_condition_drift() -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][0]["evaluation_task_id"]
    metrics = (
        _TaskClass.tasks[evaluation_task_id].artifacts[module.METRICS_ARTIFACT].value
    )
    metrics["runs"][0]["sample_count"] = module.SAMPLE_COUNT - 1

    with pytest.raises(ValueError, match="sample_count mismatch"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_metrics_checkpoint_and_condition_matrix_are_pinned() -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][0]["evaluation_task_id"]
    metrics = (
        _TaskClass.tasks[evaluation_task_id].artifacts[module.METRICS_ARTIFACT].value
    )
    metrics["checkpoint_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )

    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][0]["evaluation_task_id"]
    metrics = (
        _TaskClass.tasks[evaluation_task_id].artifacts[module.METRICS_ARTIFACT].value
    )
    metrics["runs"][1]["condition"] = "Full"
    with pytest.raises(ValueError, match="order mismatch"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_completed_evaluation_must_register_exact_training_input_model() -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    evaluation_task_id = evaluation_plan["entries"][3]["evaluation_task_id"]
    _TaskClass.tasks[evaluation_task_id].input_model_id = "9" * 32

    with pytest.raises(RuntimeError, match="input model mismatch"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_leadership_reports_negative_margins_without_failing_evidence() -> None:
    module = _load_module()
    _, evaluation_plan, _, _, output = _world(module)
    primary_index = module.FORMAL_SUBJECT_ORDER.index(module.PRIMARY_SUBJECT)
    evaluation_task_id = evaluation_plan["entries"][primary_index]["evaluation_task_id"]
    metrics = (
        _TaskClass.tasks[evaluation_task_id].artifacts[module.METRICS_ARTIFACT].value
    )
    for run in metrics["runs"]:
        run["metrics"][module.LEADERSHIP_METRIC] = 20.0

    leaderboard = module.run(
        _args(),
        task_class=_TaskClass,
        output_task=output,
        monotonic=lambda: 0.0,
        sleeper=lambda _: None,
    )

    dimensions = leaderboard["leadership"]["aggregate_dimensions"]
    assert {item["leads"] for item in dimensions.values()} == {False}
    assert all(item["margin"] < 0.0 for item in dimensions.values())
    assert leaderboard["leadership"]["leads_all_aggregate"] is False
    assert leaderboard["leadership"]["conditions_won"] == 0
    assert leaderboard["leadership"]["conditions_won_fraction"] == "0/12"
    assert leaderboard["leadership"]["leads_all_conditions"] is False


def test_plan_must_keep_all_26_subjects_and_valid_seal() -> None:
    module = _load_module()
    _, evaluation_plan, _, watcher, output = _world(module)
    evaluation_plan["entries"].pop()
    watcher.artifacts[module.EVALUATION_PLAN_ARTIFACT] = _Artifact(
        module._seal(evaluation_plan)
    )

    with pytest.raises(ValueError, match="entry count"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )


def test_existing_identical_artifact_is_idempotent_but_drift_fails() -> None:
    module = _load_module()
    training_manifest, evaluation_plan, _, _, output = _world(module)
    payload = module.build_leaderboard(
        task_class=_TaskClass,
        controller_task_id=CONTROLLER_ID,
        watcher_task_id=WATCHER_ID,
        training_manifest=training_manifest,
        evaluation_plan=evaluation_plan,
    )
    output.artifacts[module.LEADERBOARD_ARTIFACT] = _Artifact(payload)

    assert (
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )
        == payload
    )
    assert output.uploads == []

    drifted = dict(payload)
    drifted["subject_count"] = 25
    output.artifacts[module.LEADERBOARD_ARTIFACT] = _Artifact(drifted)
    with pytest.raises(RuntimeError, match="drifted"):
        module.run(
            _args(),
            task_class=_TaskClass,
            output_task=output,
            monotonic=lambda: 0.0,
            sleeper=lambda _: None,
        )
