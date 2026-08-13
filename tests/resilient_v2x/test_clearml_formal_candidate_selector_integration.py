from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = ROOT / "tools/resilient_v2x/clearml_formal_candidate_selector.py"
AUDIT_PATH = ROOT / "tools/resilient_v2x/clearml_formal_comparability_audit.py"
AUDIT_TEST_PATH = (
    ROOT / "tests/resilient_v2x/test_clearml_formal_comparability_audit.py"
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_audit_payload_is_accepted_by_selector(monkeypatch) -> None:
    audit = _load(AUDIT_PATH, "selector_integration_audit")
    audit_test = _load(AUDIT_TEST_PATH, "selector_integration_audit_test")
    world = audit_test._world(audit)
    payload = audit_test._run(audit, world)

    selector = _load(SELECTOR_PATH, "selector_integration_selector")
    pins = {
        "DEFAULT_AUDIT_TASK_ID": world.output.id,
        "DEFAULT_LEADERBOARD_TASK_ID": audit_test.LEADERBOARD_ID,
        "TRAINING_CONTROLLER_TASK_ID": audit_test.CONTROLLER_ID,
        "SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID": (
            audit.SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        ),
        "TRAINING_PROVENANCE_TASK_ID": audit_test.PROVENANCE_ID,
        "WATCHER_TASK_ID": audit_test.WATCHER_ID,
        "LEGACY_PARENT_TASK_ID": audit_test.LEGACY_PARENT_ID,
        "NO_DISTILLATION_PARENT_TASK_ID": audit_test.NO_DIST_PARENT_ID,
        "RECOVERY_PARENT_TASK_ID": audit_test.RECOVERY_PARENT_ID,
        "LEGACY_TRAINING_SCRIPT_SHA256": audit.LEGACY_TRAINING_SCRIPT_SHA256,
        "CANONICAL_TRAINING_SCRIPT_SHA256": audit.CANONICAL_TRAINING_SCRIPT_SHA256,
        "EVALUATION_SCRIPT_SHA256": audit.EVALUATION_SCRIPT_SHA256,
        "SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256": (
            audit.SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        ),
        "SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256": (
            audit.SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        ),
    }
    for name, value in pins.items():
        monkeypatch.setattr(selector, name, value)

    (
        results,
        leaderboard_seal,
        manifest_seal,
        plan_seal,
        source_binding,
        seeded,
    ) = selector._validate_leaderboard(world.leaderboard)
    records, chain = selector._validate_audit(
        payload,
        audit_task_id=world.output.id,
        leaderboard_task_id=audit_test.LEADERBOARD_ID,
        leaderboard_seal=leaderboard_seal,
        leaderboard_manifest_seal=manifest_seal,
        leaderboard_plan_seal=plan_seal,
        leaderboard_results=results,
        leaderboard_source_binding=source_binding,
        leaderboard_seeded=seeded,
    )

    assert list(records) == list(selector.SUBJECT_ORDER)
    assert chain["training_provenance_task_id"] == audit_test.PROVENANCE_ID
    assert (
        chain["training_script_equivalence"] == payload["training_script_equivalence"]
    )
    training_by_subject = {
        record["subject"]: record for record in payload["training_tasks"]
    }
    assert training_by_subject["support_residual"]["script_sha256"] == (
        audit.LEGACY_TRAINING_SCRIPT_SHA256
    )
    assert training_by_subject["ffnet"]["script_sha256"] == (
        audit.CANONICAL_TRAINING_SCRIPT_SHA256
    )
    assert records["support_residual"]["script_sha256"] == (
        audit.EVALUATION_SCRIPT_SHA256
    )
