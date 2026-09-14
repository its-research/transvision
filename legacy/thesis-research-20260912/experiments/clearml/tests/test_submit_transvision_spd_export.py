from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "submit_transvision_spd_export.py"
SPEC = importlib.util.spec_from_file_location("transvision_export_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class ResolvedFixture:
    def __init__(self, test: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "thesis"
        self.source = self.root / "transvision"
        self.contracts = self.root / "external-contracts"
        self.dataset = self.root / "dataset-mount"
        self.checkpoints = self.root / "checkpoint-mount"
        for directory in (
            self.repository,
            self.source,
            self.contracts,
            self.dataset,
            self.checkpoints,
        ):
            directory.mkdir(parents=True)

        self._copy_repository_contracts()
        self.thesis_commit = self._commit(self.repository)

        self.detector_config = (
            self.source / "configs/resilient_v2x/baselines/coformernet.py"
        )
        self.detector_config.parent.mkdir(parents=True)
        self.detector_config.write_text("model = dict(type='fixture')\n", encoding="utf-8")
        self.runtime_source = self.source / "transvision_runtime.py"
        self.runtime_source.write_text(
            "def export_detections():\n    raise RuntimeError('fixture only')\n",
            encoding="utf-8",
        )
        self.source_commit = self._commit(self.source)
        self._git(
            self.source,
            "remote",
            "add",
            "origin",
            preflight.TRANSVISION_REPOSITORY_URL,
        )

        self.checkpoint = self.checkpoints / "coformernet-final.pth"
        self.checkpoint.write_bytes(b"frozen-checkpoint-fixture\n")
        self._write_control("data-access.json", b'{"authorized":true}\n')
        self._write_control("release-identity.json", b'{"release":"fixture"}\n')
        self._write_control("split.json", b'{"split":"validation"}\n')
        for subject in sorted(preflight.RIGHTS_SUBJECTS):
            self._write_control(
                f"rights-{subject}.json",
                json.dumps({"subject": subject, "authorized": True}).encode() + b"\n",
            )
        self._write_control("environment.json", b'{"environment":"fixture"}\n')
        self._write_source_seal()
        self.document = self._resolved_document()
        self.config_path = self.contracts / "resolved-export.json"
        self.write_config()

    def _copy_repository_contracts(self) -> None:
        for relative in (
            preflight.CONTRACT_RELATIVE_PATH,
            preflight.FIXED_DETECTION_RELATIVE_PATH,
        ):
            target = self.repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preflight.REPOSITORY_ROOT / relative, target)

    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _commit(self, root: Path) -> str:
        subprocess.run(
            ["git", "init", "--quiet", str(root)],
            check=True,
            capture_output=True,
        )
        self._git(root, "add", ".")
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Export Contract Test",
                "-c",
                "user.email=export-contract@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
            capture_output=True,
        )
        return self._git(root, "rev-parse", "HEAD")

    def _write_control(self, name: str, payload: bytes) -> None:
        (self.contracts / name).write_bytes(payload)

    def _write_source_seal(self) -> None:
        relative_paths = (
            "configs/resilient_v2x/baselines/coformernet.py",
            "transvision_runtime.py",
        )
        payload = {
            "schema_version": 1,
            "seal_kind": "source_tree",
            "git_commit": self.source_commit,
            "entries": [
                {
                    "path": relative,
                    "size": (self.source / relative).stat().st_size,
                    "sha256": sha256_file(self.source / relative),
                    "git_mode": self._git(
                        self.source, "ls-tree", "HEAD", relative
                    ).split()[0],
                    "git_blob": self._git(
                        self.source, "rev-parse", f"HEAD:{relative}"
                    ),
                }
                for relative in relative_paths
            ],
        }
        payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        seal = {**payload, "manifest_sha256": payload_sha256}
        self.source_tree_sha256 = payload_sha256
        (self.contracts / "source-tree.json").write_bytes(canonical_json_bytes(seal))

    def _control_binding(self, name: str) -> tuple[str, str]:
        return name, sha256_file(self.contracts / name)

    def _resolved_document(self) -> dict[str, object]:
        document = json.loads(
            (
                preflight.REPOSITORY_ROOT / preflight.TEMPLATE_RELATIVE_PATH
            ).read_text(encoding="utf-8")
        )
        document.update(
            {
                "config_kind": "transvision_spd_detection_export_resolved",
                "execution_enabled": False,
                "task_creation_allowed": False,
                "execution_status": (
                    "resolved_for_local_preflight_only_terminally_blocked"
                ),
            }
        )
        data_access = document["data_access"]
        assert isinstance(data_access, dict)
        for path_field, hash_field, name in (
            ("receipt_path", "receipt_sha256", "data-access.json"),
            (
                "release_identity_manifest_path",
                "release_identity_manifest_sha256",
                "release-identity.json",
            ),
            ("split_manifest_path", "split_manifest_sha256", "split.json"),
        ):
            data_access[path_field], data_access[hash_field] = self._control_binding(name)

        rights = document["rights"]
        assert isinstance(rights, list)
        for entry in rights:
            assert isinstance(entry, dict)
            name = f"rights-{entry['subject']}.json"
            entry.update(
                {
                    "authorization_basis": "private_authorization",
                    "execute_allowed": True,
                    "modify_allowed": entry["subject"] != "detector_checkpoint",
                    "redistribute_allowed": False,
                    "receipt_path": name,
                    "receipt_sha256": sha256_file(self.contracts / name),
                }
            )

        code = document["code"]
        assert isinstance(code, dict)
        code.update(
            {
                "thesis_commit": self.thesis_commit,
                "transvision_root": str(self.source),
                "transvision_commit": self.source_commit,
                "source_tree_seal_path": "source-tree.json",
                "source_tree_seal_sha256": sha256_file(
                    self.contracts / "source-tree.json"
                ),
                "source_tree_sha256": self.source_tree_sha256,
            }
        )

        detector = document["detector"]
        assert isinstance(detector, dict)
        detector.update(
            {
                "model_id": "coformernet_controlled_adaptation",
                "config_path": "configs/resilient_v2x/baselines/coformernet.py",
                "config_sha256": sha256_file(self.detector_config),
                "checkpoint_path": self.checkpoint.name,
                "checkpoint_sha256": sha256_file(self.checkpoint),
                "environment_manifest_path": "environment.json",
                "environment_manifest_sha256": sha256_file(
                    self.contracts / "environment.json"
                ),
                "backend_status": "unresolved",
                "runtime_status": "unresolved",
            }
        )

        digest = "sha256:" + "d" * 64
        container = document["container"]
        assert isinstance(container, dict)
        container.update(
            {
                "image_reference": "registry.example/rtpv2x/transvision@" + digest,
                "final_oci_digest": digest,
            }
        )
        mounts = document["mounts"]
        assert isinstance(mounts, dict)
        mounts.update(
            {
                "dataset_root": str(self.dataset),
                "checkpoint_root": str(self.checkpoints),
            }
        )
        return document

    def write_config(self) -> str:
        self.config_path.write_text(
            json.dumps(self.document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return sha256_file(self.config_path)

    @staticmethod
    def observation() -> dict[str, object]:
        return {
            "queue_name": "GPU4-A100",
            "queue_accepting_tasks": True,
            "online_worker_count": 1,
            "visible_gpu_count": 1,
            "device_name": "NVIDIA A100 fixture",
            "active_same_name_task_count": 0,
        }

    def run(
        self,
        *,
        environment: dict[str, str] | None = None,
        observation: dict[str, object] | None = None,
        read_only_probe: object | None = None,
        expected_config_sha256: str | None = None,
    ) -> None:
        preflight.preflight_resolved_config(
            self.config_path,
            expected_config_sha256=(
                expected_config_sha256 or sha256_file(self.config_path)
            ),
            contracts_root=self.contracts,
            observation=observation or self.observation(),
            environment=(
                {preflight.ACKNOWLEDGEMENT_VARIABLE: "1"}
                if environment is None
                else environment
            ),
            repository_root=self.repository,
            read_only_probe=read_only_probe or (lambda _path: True),
        )


class TransVisionExportPreflightTests(unittest.TestCase):
    def assert_blocked(self, reason: str, function: object) -> None:
        assert callable(function)
        with self.assertRaises(preflight.ExportPreflightError) as raised:
            function()
        self.assertEqual(raised.exception.code, reason)

    def test_repository_template_is_disabled_and_diagnostic_only(self) -> None:
        report = preflight.validate_repository_template()
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(report["task_creation_allowed"])
        self.assertFalse(report["scientific_claim_allowed"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["formal_fixed_detection_input_allowed"])
        self.assertFalse(report["h1_support_allowed"])
        self.assertFalse(report["score_is_reliability"])
        self.assertFalse(report["authenticated_live_preflight_implemented"])

    def test_check_template_cli_has_no_submission_side_effect(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(preflight.main(["check-template"]), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["network_access_performed"])
        self.assertFalse(report["submission_performed"])
        self.assertEqual(report["preflight"], "template_valid")

    def test_module_contains_no_clearml_import_or_task_creation_call(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from clearml import", source)
        self.assertNotIn("import clearml", source)
        self.assertNotIn("Task.create", source)
        self.assertNotIn("Task.enqueue", source)

    def test_repository_template_cannot_be_used_as_resolved_config(self) -> None:
        fixture = ResolvedFixture(self)
        self.assert_blocked(
            "external_resolved_config_required",
            lambda: preflight.preflight_resolved_config(
                preflight.REPOSITORY_ROOT / preflight.TEMPLATE_RELATIVE_PATH,
                expected_config_sha256=sha256_file(
                    preflight.REPOSITORY_ROOT / preflight.TEMPLATE_RELATIVE_PATH
                ),
                contracts_root=preflight.REPOSITORY_ROOT,
                observation=fixture.observation(),
                environment={preflight.ACKNOWLEDGEMENT_VARIABLE: "1"},
                read_only_probe=lambda _path: True,
            ),
        )

    def test_all_resolved_local_gates_end_at_backend_runtime_blocker(self) -> None:
        fixture = ResolvedFixture(self)
        self.assertFalse(fixture.document["execution_enabled"])
        self.assertFalse(fixture.document["task_creation_allowed"])
        self.assert_blocked(
            "production_backend_and_runtime_unresolved",
            fixture.run,
        )

    def test_preflight_reads_but_never_sets_explicit_data_access_ack(self) -> None:
        fixture = ResolvedFixture(self)
        empty_environment: dict[str, str] = {}
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assert_blocked(
                "explicit_data_access_acknowledgement_required",
                lambda: fixture.run(environment=empty_environment),
            )
            self.assertNotIn(preflight.ACKNOWLEDGEMENT_VARIABLE, os.environ)

    def test_unresolved_or_insufficient_rights_fail_closed(self) -> None:
        fixture = ResolvedFixture(self)
        rights = fixture.document["rights"]
        assert isinstance(rights, list)
        source = next(
            entry
            for entry in rights
            if isinstance(entry, dict) and entry["subject"] == "transvision_source"
        )
        assert isinstance(source, dict)
        source["authorization_basis"] = "unverified"
        fixture.write_config()
        self.assert_blocked("rights_authorization_unresolved", fixture.run)

        source["authorization_basis"] = "private_authorization"
        source["modify_allowed"] = False
        fixture.write_config()
        self.assert_blocked("rights_modify_not_authorized", fixture.run)

    def test_dirty_or_wrong_git_identity_fails_closed(self) -> None:
        fixture = ResolvedFixture(self)
        (fixture.repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        self.assert_blocked(
            "thesis_commit_or_worktree_mismatch",
            fixture.run,
        )

    def test_transvision_remote_identity_is_verified_without_network(self) -> None:
        fixture = ResolvedFixture(self)
        fixture._git(
            fixture.source,
            "remote",
            "set-url",
            "origin",
            "https://example.invalid/not-transvision.git",
        )
        self.assert_blocked("transvision_remote_identity_mismatch", fixture.run)

    def test_resolved_config_and_asset_hashes_are_mandatory(self) -> None:
        fixture = ResolvedFixture(self)
        stale = sha256_file(fixture.config_path)
        fixture.document["execution_status"] = "changed"
        fixture.write_config()
        self.assert_blocked(
            "resolved_config_hash_mismatch",
            lambda: fixture.run(expected_config_sha256=stale),
        )

        fixture = ResolvedFixture(self)
        fixture.checkpoint.write_bytes(b"tampered\n")
        self.assert_blocked("detector_checkpoint_hash_mismatch", fixture.run)

    def test_resolved_config_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        fixture = ResolvedFixture(self)
        payload = fixture.config_path.read_text(encoding="utf-8")
        duplicate = payload.replace(
            '  "execution_enabled": false,',
            '  "execution_enabled": false,\n  "execution_enabled": false,',
            1,
        )
        fixture.config_path.write_text(duplicate, encoding="utf-8")
        self.assert_blocked("resolved_config_invalid", fixture.run)

        fixture = ResolvedFixture(self)
        nonfinite = fixture.config_path.read_text(encoding="utf-8").replace(
            '  "schema_version": 1,',
            '  "schema_version": NaN,',
            1,
        )
        fixture.config_path.write_text(nonfinite, encoding="utf-8")
        self.assert_blocked("resolved_config_invalid", fixture.run)

    def test_read_only_mounts_are_independent_mandatory_gates(self) -> None:
        fixture = ResolvedFixture(self)
        dataset = fixture.dataset.resolve()

        def dataset_writable(path: Path) -> bool:
            return path.resolve() != dataset

        self.assert_blocked(
            "dataset_mount_not_read_only",
            lambda: fixture.run(read_only_probe=dataset_writable),
        )

        checkpoint = fixture.checkpoints.resolve()

        def checkpoint_writable(path: Path) -> bool:
            return path.resolve() != checkpoint

        self.assert_blocked(
            "checkpoint_mount_not_read_only",
            lambda: fixture.run(read_only_probe=checkpoint_writable),
        )

    def test_final_oci_digest_and_a100_observation_are_mandatory(self) -> None:
        fixture = ResolvedFixture(self)
        container = fixture.document["container"]
        assert isinstance(container, dict)
        container["final_oci_digest"] = None
        fixture.write_config()
        self.assert_blocked("final_oci_digest_invalid", fixture.run)

        fixture = ResolvedFixture(self)
        observation = fixture.observation()
        observation["device_name"] = "NVIDIA H100 fixture"
        self.assert_blocked(
            "a100_device_required",
            lambda: fixture.run(observation=observation),
        )

        observation = fixture.observation()
        observation["queue_accepting_tasks"] = False
        self.assert_blocked(
            "a100_queue_not_accepting_tasks",
            lambda: fixture.run(observation=observation),
        )

    def test_source_seal_binds_commit_file_bytes_and_payload_hash(self) -> None:
        fixture = ResolvedFixture(self)
        seal_path = fixture.contracts / "source-tree.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["git_commit"] = "f" * 40
        seal_path.write_bytes(canonical_json_bytes(seal))
        code = fixture.document["code"]
        assert isinstance(code, dict)
        code["source_tree_seal_sha256"] = sha256_file(seal_path)
        fixture.write_config()
        self.assert_blocked("source_tree_seal_invalid", fixture.run)

    def test_source_seal_must_cover_the_complete_tracked_tree(self) -> None:
        fixture = ResolvedFixture(self)
        seal_path = fixture.contracts / "source-tree.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["entries"] = seal["entries"][:-1]
        payload = dict(seal)
        del payload["manifest_sha256"]
        payload_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        seal["manifest_sha256"] = payload_sha256
        seal_path.write_bytes(canonical_json_bytes(seal))
        code = fixture.document["code"]
        assert isinstance(code, dict)
        code["source_tree_seal_sha256"] = sha256_file(seal_path)
        code["source_tree_sha256"] = payload_sha256
        fixture.write_config()
        self.assert_blocked("source_tree_seal_invalid", fixture.run)

    def test_repository_fsmonitor_configuration_cannot_execute(self) -> None:
        fixture = ResolvedFixture(self)
        marker = fixture.root / "fsmonitor-was-executed"
        hook = fixture.root / "malicious-fsmonitor.sh"
        hook.write_text(
            "#!/bin/sh\nprintf invoked > '" + str(marker) + "'\nexit 0\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)
        for repository in (fixture.repository, fixture.source):
            fixture._git(repository, "config", "core.fsmonitor", str(hook))
        self.assert_blocked(
            "production_backend_and_runtime_unresolved",
            fixture.run,
        )
        self.assertFalse(marker.exists())

    def test_repository_clean_filter_configuration_cannot_execute(self) -> None:
        fixture = ResolvedFixture(self)
        marker = fixture.root / "clean-filter-was-executed"
        filter_program = fixture.root / "malicious-clean-filter.sh"
        filter_program.write_text(
            "#!/bin/sh\nprintf invoked > '" + str(marker) + "'\ncat\n",
            encoding="utf-8",
        )
        filter_program.chmod(0o755)
        attributes = fixture.repository / ".gitattributes"
        probe = fixture.repository / "filter-probe.txt"
        attributes.write_text("filter-probe.txt filter=evil\n", encoding="utf-8")
        probe.write_text("unchanged bytes\n", encoding="utf-8")
        fixture._git(
            fixture.repository,
            "config",
            "filter.evil.clean",
            str(filter_program),
        )
        fixture._git(fixture.repository, "config", "filter.evil.required", "true")
        fixture._git(fixture.repository, "add", ".gitattributes", "filter-probe.txt")
        subprocess.run(
            [
                "git",
                "-C",
                str(fixture.repository),
                "-c",
                "user.name=Export Contract Test",
                "-c",
                "user.email=export-contract@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "add filter fixture",
            ],
            check=True,
            capture_output=True,
        )
        marker.unlink(missing_ok=True)
        code = fixture.document["code"]
        assert isinstance(code, dict)
        code["thesis_commit"] = fixture._git(
            fixture.repository, "rev-parse", "HEAD"
        )
        fixture.write_config()
        self.assert_blocked(
            "production_backend_and_runtime_unresolved",
            fixture.run,
        )
        self.assertFalse(marker.exists())

    def test_git_replace_ref_cannot_substitute_the_expected_commit_tree(self) -> None:
        fixture = ResolvedFixture(self)
        original_commit = fixture.thesis_commit
        replacement_file = fixture.repository / "replacement-only.txt"
        replacement_file.write_text("replacement tree\n", encoding="utf-8")
        fixture._git(fixture.repository, "add", "replacement-only.txt")
        subprocess.run(
            [
                "git",
                "-C",
                str(fixture.repository),
                "-c",
                "user.name=Export Contract Test",
                "-c",
                "user.email=export-contract@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "replacement tree",
            ],
            check=True,
            capture_output=True,
        )
        replacement_commit = fixture._git(
            fixture.repository, "rev-parse", "HEAD"
        )
        fixture._git(fixture.repository, "reset", "--hard", original_commit)
        fixture._git(
            fixture.repository,
            "replace",
            original_commit,
            replacement_commit,
        )
        fixture._git(fixture.repository, "reset", "--hard", original_commit)
        self.assertEqual(
            fixture._git(fixture.repository, "rev-parse", "HEAD"),
            original_commit,
        )
        self.assertTrue(replacement_file.exists())
        self.assert_blocked(
            "thesis_commit_or_worktree_mismatch",
            fixture.run,
        )


if __name__ == "__main__":
    unittest.main()
