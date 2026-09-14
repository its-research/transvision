#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import inspect_task  # noqa: E402


TASK_ID = "b5633ba6208744aeac309c50a0e195e7"
OTHER_TASK_ID = "a" * 32
SERVER_HASH_A = "a" * 64
SERVER_HASH_B = "b" * 64
INTERNAL_URL = "http://10.0.0.1:8080/private"


class FakeStatus:
    value = "completed"


class FakeArtifact:
    def __init__(self, server_hash: object, size: object) -> None:
        self.hash = server_hash
        self.content_size = size
        self.url = INTERNAL_URL
        self.type = "custom"
        self.mode = "output"


class FakeScript:
    def __init__(self, diff: object) -> None:
        self.diff = diff
        self.repository = INTERNAL_URL
        self.working_dir = "/home/private/worktree"
        self.requirements = {"pip": ["private-package"]}


class FakeData:
    def __init__(self, diff: object) -> None:
        self.script = FakeScript(diff)
        self.status_reason = "secret failure reason"
        self.container = {"image": "private.registry/image"}
        self.execution = {"docker_cmd": "secret"}


class FakeTask:
    def __init__(
        self,
        *,
        task_id: object = TASK_ID,
        name: object = "rtpv2x__causal-canary_v1",
        project: object = "Thesis/RTP-V2X",
        status: object = FakeStatus(),
        last_iteration: object = 17,
        diff: object = "print('token=secret')\n# " + INTERNAL_URL,
        artifacts: object | None = None,
    ) -> None:
        self.id = task_id
        self.name = name
        self.status = status
        self.data = FakeData(diff)
        self.artifacts = artifacts if artifacts is not None else {
            "metrics": FakeArtifact(SERVER_HASH_A.upper(), 4096),
            "events": FakeArtifact(SERVER_HASH_B, None),
        }
        self._project = project
        self._last_iteration = last_iteration

    def get_project_name(self) -> object:
        return self._project

    def get_last_iteration(self) -> object:
        return self._last_iteration

    def get_status_message(self) -> object:
        raise AssertionError("status message must not be read")

    def get_reported_console_output(self, **_: object) -> object:
        raise AssertionError("console must not be read")

    def get_base_docker(self) -> object:
        raise AssertionError("container metadata must not be read")


def recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(recursive_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(recursive_keys(item) for item in value)) if value else set()
    return set()


class InspectTaskTest(unittest.TestCase):
    def test_summary_is_an_exact_safe_field_whitelist(self) -> None:
        task = FakeTask()
        diff = task.data.script.diff
        self.assertIsInstance(diff, str)
        with mock.patch.object(inspect_task, "_load_task", return_value=task):
            summary = inspect_task._task_summary(TASK_ID)

        expected = {
            "artifacts": [
                {"name": "events", "server_hash": SERVER_HASH_B, "size": None},
                {"name": "metrics", "server_hash": SERVER_HASH_A, "size": 4096},
            ],
            "id": TASK_ID,
            "last_iteration": 17,
            "name": "rtpv2x__causal-canary_v1",
            "project": "Thesis/RTP-V2X",
            "source": {
                "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                "diff_size": len(diff.encode("utf-8")),
            },
            "status": "completed",
        }
        self.assertEqual(summary, expected)
        self.assertEqual(set(summary), inspect_task.TASK_FIELDS)
        self.assertEqual(set(summary["source"]), inspect_task.SOURCE_FIELDS)
        self.assertTrue(
            all(set(artifact) == inspect_task.ARTIFACT_FIELDS for artifact in summary["artifacts"])
        )

        rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(INTERNAL_URL, rendered)
        self.assertNotIn("secret", rendered)
        forbidden = {
            "console",
            "container",
            "docker",
            "docker_cmd",
            "execution",
            "repository",
            "requirements",
            "status_message",
            "status_reason",
            "url",
            "working_dir",
        }
        self.assertTrue(recursive_keys(summary).isdisjoint(forbidden))

    def test_unsafe_values_fail_closed_without_echoing_them(self) -> None:
        unsafe_tasks = (
            FakeTask(name=INTERNAL_URL),
            FakeTask(project="Thesis\nprivate"),
            FakeTask(status="running"),
            FakeTask(last_iteration=-1),
            FakeTask(artifacts={"../metrics": FakeArtifact(SERVER_HASH_A, 1)}),
            FakeTask(artifacts={"metrics": FakeArtifact(INTERNAL_URL, 1)}),
            FakeTask(artifacts={"metrics": FakeArtifact(SERVER_HASH_A, -1)}),
            FakeTask(diff={"repository": INTERNAL_URL}),
            FakeTask(task_id=OTHER_TASK_ID),
        )
        for task in unsafe_tasks:
            with self.subTest(task=task):
                with mock.patch.object(inspect_task, "_load_task", return_value=task):
                    with self.assertRaises(inspect_task.InspectionError) as raised:
                        inspect_task._task_summary(TASK_ID)
                self.assertNotIn(INTERNAL_URL, str(raised.exception))

        loader = mock.Mock()
        with mock.patch.object(inspect_task, "_load_task", loader):
            with self.assertRaises(inspect_task.InspectionError):
                inspect_task._task_summary("not-a-task-id")
        loader.assert_not_called()

    def test_console_option_is_rejected_before_task_lookup(self) -> None:
        loader = mock.Mock()
        stderr = io.StringIO()
        with mock.patch.object(inspect_task, "_load_task", loader):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    inspect_task.parse_args([TASK_ID, "--console-lines", "10"])
        self.assertEqual(raised.exception.code, 2)
        loader.assert_not_called()

    def test_output_uses_atomic_replace_and_matches_stdout(self) -> None:
        task = FakeTask()
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "task.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(inspect_task, "_load_task", return_value=task):
                with mock.patch.object(
                    inspect_task.os, "replace", wraps=real_replace
                ) as replace:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        result = inspect_task.main([TASK_ID, "--output", str(output)])

            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(output.read_text(encoding="utf-8"), stdout.getvalue())
            replace.assert_called_once()
            self.assertEqual(list(output.parent.glob("*.tmp")), [])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), inspect_task.TASK_FIELDS)


if __name__ == "__main__":
    unittest.main()
