from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "preflight.py"
SPEC = importlib.util.spec_from_file_location("clearml_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class FakeEndpoint:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def get_all(self, **_kwargs: object) -> list[object]:
        return self.values


class FakeTaskAPI:
    duplicates: list[object] = []

    @classmethod
    def get_tasks(cls, **_kwargs: object) -> list[object]:
        return cls.duplicates


def client(*, entries: list[object], workers: list[object]) -> SimpleNamespace:
    queue = {"id": "internal-queue-id", "entries": entries}
    return SimpleNamespace(
        queues=FakeEndpoint([queue]),
        workers=FakeEndpoint(workers),
    )


class ClearMLPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTaskAPI.duplicates = []

    def inspect(
        self, *, entries: list[object], workers: list[object]
    ) -> dict[str, object]:
        return preflight.inspect_preflight(
            client=client(entries=entries, workers=workers),
            task_api=FakeTaskAPI,
            queue_name="GPU4-A100",
            project_name="Thesis/RTP-V2X",
            task_name="safe-name",
            require_empty=True,
        )

    def test_summary_contains_counts_but_no_topology_or_task_details(self) -> None:
        workers = [
            {
                "id": "private-host-A100:gpu0",
                "queues": [{"id": "internal-queue-id"}],
                "task": {"id": "private-task-id", "name": "private-task-name"},
                "last_report_time": "private-time",
            },
            {
                "id": "private-host-A100:gpu1",
                "queues": [{"id": "internal-queue-id"}],
                "task": "",
            },
        ]
        report = self.inspect(entries=[], workers=workers)
        encoded = json.dumps(report, sort_keys=True)
        self.assertEqual(report["workers"], {"reporting": 2, "idle": 1, "busy": 1})
        for forbidden in (
            "private-host",
            "private-task",
            "private-time",
            "internal-queue-id",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_unknown_nonempty_task_shape_is_fail_closed_as_busy(self) -> None:
        worker = {
            "queues": [{"id": "internal-queue-id"}],
            "task": {"unexpected": "shape"},
        }
        report = self.inspect(entries=[], workers=[worker])
        self.assertEqual(report["workers"], {"reporting": 1, "idle": 0, "busy": 1})

    def test_queue_entries_and_duplicate_tasks_fail_without_identifiers(self) -> None:
        worker = {"queues": [{"id": "internal-queue-id"}], "task": ""}
        with self.assertRaisesRegex(preflight.PreflightError, "not empty: 1 entries"):
            self.inspect(entries=[{"task": "secret-entry-id"}], workers=[worker])

        FakeTaskAPI.duplicates = [{"id": "secret-duplicate-id"}]
        with self.assertRaisesRegex(preflight.PreflightError, "exists: 1") as raised:
            self.inspect(entries=[], workers=[worker])
        self.assertNotIn("secret-duplicate-id", str(raised.exception))

    def test_main_loads_clearml_lazily_and_prints_only_safe_json(self) -> None:
        fake_task = type(
            "Task", (), {"get_tasks": classmethod(lambda cls, **kwargs: [])}
        )
        fake_client = client(
            entries=[],
            workers=[{"queues": [{"id": "internal-queue-id"}], "task": ""}],
        )
        clearml_module = SimpleNamespace(Task=fake_task)
        client_module = SimpleNamespace(APIClient=lambda: fake_client)
        argv = [
            "preflight.py",
            "--queue",
            "GPU4-A100",
            "--project",
            "Thesis/RTP-V2X",
            "--task-name",
            "safe-name",
            "--require-empty",
        ]
        output = io.StringIO()
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "clearml": clearml_module,
                    "clearml.backend_api.session.client": client_module,
                },
            ),
            mock.patch("sys.argv", argv),
            redirect_stdout(output),
        ):
            preflight.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["workers"]["idle"], 1)
        self.assertNotIn("internal-queue-id", output.getvalue())


if __name__ == "__main__":
    unittest.main()
