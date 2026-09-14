from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "clearml" / "publish_task.py"
SPEC = importlib.util.spec_from_file_location("publish_task", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


TASK_ID = "b824e7114467492aa71f87c0dad0d555"
TASK_NAME = "rtpv2x__synth-causal-canary-v1__canary-secure__seed3407"


class _Status:
    def __init__(self, value: str) -> None:
        self.value = value


class _Artifact:
    def __init__(self, artifact_hash: str, size: int | None) -> None:
        self.hash = artifact_hash
        self.size = size


def _artifacts(*, size: int | None = None) -> dict[str, _Artifact]:
    return {
        "events": _Artifact("7" * 64, size),
        "metrics": _Artifact("e" * 64, size),
        "run_manifest": _Artifact("1" * 64, size),
    }


class _Task:
    def __init__(
        self,
        *,
        status: str,
        artifacts: dict[str, _Artifact] | None = None,
        task_id: str = TASK_ID,
        project: str = MODULE.CLEARML_PROJECT,
        name: str = TASK_NAME,
    ) -> None:
        self.id = task_id
        self.status = _Status(status)
        self.artifacts = _artifacts() if artifacts is None else artifacts
        self.name = name
        self.project = project
        self.publish_calls = 0

    def get_project_name(self) -> str:
        return self.project

    def publish(self) -> None:
        self.publish_calls += 1


class PublishTaskTests(unittest.TestCase):
    def test_valid_task_is_published_and_refetched_with_stable_snapshot(self) -> None:
        before = _Task(status="completed")
        after = _Task(status="published")
        loader = mock.Mock(side_effect=[before, after])

        status, snapshot_sha = MODULE.publish_task(TASK_ID, loader=loader)

        self.assertEqual(status, "published")
        self.assertRegex(snapshot_sha, r"^[0-9a-f]{64}$")
        self.assertEqual(before.publish_calls, 1)
        self.assertEqual(
            loader.call_args_list, [mock.call(TASK_ID), mock.call(TASK_ID)]
        )

    def test_missing_sizes_are_valid_and_covered_by_stability_check(self) -> None:
        before = _Task(status="completed", artifacts=_artifacts(size=None))
        after = _Task(status="published", artifacts=_artifacts(size=None))
        status, _ = MODULE.publish_task(
            TASK_ID, loader=mock.Mock(side_effect=[before, after])
        )
        self.assertEqual(status, "published")

        changed = _Task(status="published", artifacts=_artifacts(size=1))
        with self.assertRaisesRegex(
            MODULE.PublishGateError, "ARTIFACT_SNAPSHOT_CHANGED"
        ):
            MODULE.publish_task(
                TASK_ID,
                loader=mock.Mock(side_effect=[_Task(status="completed"), changed]),
            )

    def test_task_id_is_exact_lowercase_hex(self) -> None:
        invalid = ["B" * 32, "a" * 31, "a" * 33, "g" * 32, "../" + "a" * 32]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(MODULE.PublishGateError, "INVALID_TASK_ID"):
                    MODULE.validate_task_id(value)

    def test_metadata_gate_rejects_every_required_mismatch(self) -> None:
        cases = [
            (_Task(status="completed", task_id="a" * 32), "TASK_ID_MISMATCH"),
            (_Task(status="completed", project="Other"), "PROJECT_MISMATCH"),
            (_Task(status="completed", name="other__task"), "TASK_NAME_MISMATCH"),
            (_Task(status="failed"), "TASK_STATUS_MISMATCH"),
            (
                _Task(
                    status="completed", artifacts={"metrics": _Artifact("e" * 64, None)}
                ),
                "INVALID_ARTIFACT_SET",
            ),
        ]
        for task, code in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(MODULE.PublishGateError, code):
                    MODULE.inspect_task(task, TASK_ID, "completed")

    def test_invalid_artifact_size_and_hash_fail_closed(self) -> None:
        bad_rows = [
            (_Artifact("e" * 64, 0), "INVALID_ARTIFACT_SIZE"),
            (_Artifact("e" * 64, True), "INVALID_ARTIFACT_SIZE"),
            (_Artifact("E" * 64, None), "INVALID_ARTIFACT_HASH"),
            (_Artifact("e" * 63, None), "INVALID_ARTIFACT_HASH"),
        ]
        for artifact, code in bad_rows:
            artifacts = _artifacts()
            artifacts["metrics"] = artifact
            with self.subTest(code=code):
                with self.assertRaisesRegex(MODULE.PublishGateError, code):
                    MODULE.inspect_task(
                        _Task(status="completed", artifacts=artifacts),
                        TASK_ID,
                        "completed",
                    )

    def test_publish_exception_and_post_publish_changes_use_fixed_codes(self) -> None:
        before = _Task(status="completed")
        before.publish = mock.Mock(side_effect=RuntimeError("secret URL"))
        with self.assertRaisesRegex(MODULE.PublishGateError, "TASK_PUBLISH_FAILED"):
            MODULE.publish_task(TASK_ID, loader=mock.Mock(return_value=before))

        before = _Task(status="completed")
        changed_artifacts = _artifacts()
        changed_artifacts["metrics"] = _Artifact("a" * 64, None)
        after = _Task(status="published", artifacts=changed_artifacts)
        with self.assertRaisesRegex(
            MODULE.PublishGateError, "ARTIFACT_SNAPSHOT_CHANGED"
        ):
            MODULE.publish_task(TASK_ID, loader=mock.Mock(side_effect=[before, after]))

    def test_main_success_and_failure_outputs_are_strict(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            MODULE,
            "publish_task",
            return_value=("published", "f" * 64),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = MODULE.main([TASK_ID])
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            f"task_id={TASK_ID}\nstatus=published\nartifact_snapshot_sha256={'f' * 64}\n",
        )
        self.assertEqual(stderr.getvalue(), "")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = MODULE.main(["https://secret.invalid/?token=x"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "INVALID_TASK_ID\n")

    def test_fd_level_third_party_output_is_suppressed(self) -> None:
        script = f"""
import importlib.util
import os
spec = importlib.util.spec_from_file_location('publish_task_fd_test', {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def noisy():
    os.write(1, b'https://secret.invalid/?token=x\\n')
    os.write(2, b'Authorization: Bearer secret\\n')
module._quiet_call('NOISY_FAILED', noisy)
print('ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
