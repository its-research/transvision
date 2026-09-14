#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import freeze_v2xseq_inputs  # noqa: E402
import v2xseq_real_canary  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class RealCanaryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.dataset = self.root / "dataset"
        self.repository.mkdir()
        self.dataset.mkdir()
        self.clearml_environment = mock.patch.dict(
            os.environ, {"CLEARML_TASK_ID": "a" * 32}, clear=False
        )
        self.clearml_environment.start()
        self.addCleanup(self.clearml_environment.stop)
        write_json(
            self.dataset / "data_info.json",
            [
                {
                    "vehicle_frame": "000001",
                    "infrastructure_frame": "009001",
                    "vehicle_sequence": "train-seq",
                    "infrastructure_sequence": "train-seq",
                },
                {
                    "vehicle_frame": "000002",
                    "infrastructure_frame": "009002",
                    "vehicle_sequence": "validation-seq",
                    "infrastructure_sequence": "validation-seq",
                },
            ],
        )
        write_json(
            self.dataset / "v1.0-trainval" / "sample.json",
            [
                {
                    "token": "000001",
                    "timestamp": 1_000_000,
                    "scene_token": "train-seq",
                    "prev": "",
                    "next": "",
                },
                {
                    "token": "000002",
                    "timestamp": 2_000_000,
                    "scene_token": "validation-seq",
                    "prev": "",
                    "next": "",
                },
            ],
        )
        write_json(
            self.dataset / "v1.0-trainval" / "scene.json",
            [
                {
                    "token": "train-seq",
                    "nbr_samples": 1,
                    "first_sample_token": "000001",
                    "last_sample_token": "000001",
                },
                {
                    "token": "validation-seq",
                    "nbr_samples": 1,
                    "first_sample_token": "000002",
                    "last_sample_token": "000002",
                },
            ],
        )
        write_json(
            self.dataset / "vehicle-side" / "data_info.json",
            [
                {
                    "frame_id": "000001",
                    "sequence_id": "train-seq",
                    "pointcloud_timestamp": "1000000",
                },
                {
                    "frame_id": "000002",
                    "sequence_id": "validation-seq",
                    "pointcloud_timestamp": "2000000",
                },
            ],
        )
        write_json(
            self.dataset / "label" / "000001.json",
            [
                {
                    "track_id": "t1",
                    "type": "Car",
                    "veh_pointcloud_timestamp": "1000000",
                    "3d_location": {"x": 1, "y": 2, "z": 0},
                }
            ],
        )
        write_json(
            self.dataset / "label" / "000002.json",
            [
                {
                    "track_id": "t2",
                    "type": "Car",
                    "veh_pointcloud_timestamp": "2000000",
                    "3d_location": {"x": 3, "y": 4, "z": 0},
                }
            ],
        )
        train_vehicle = self.dataset / "vehicle-side" / "velodyne" / "000001.pcd"
        train_infrastructure = (
            self.dataset / "infrastructure-side" / "velodyne" / "009001.pcd"
        )
        validation_vehicle = self.dataset / "vehicle-side" / "velodyne" / "000002.pcd"
        validation_infrastructure = (
            self.dataset / "infrastructure-side" / "velodyne" / "009002.pcd"
        )
        train_vehicle.parent.mkdir(parents=True)
        train_infrastructure.parent.mkdir(parents=True)
        train_vehicle.write_bytes(b"train-vehicle-points")
        train_infrastructure.write_bytes(b"train-infrastructure-points")
        validation_vehicle.write_bytes(b"validation-vehicle-points")
        validation_infrastructure.write_bytes(b"validation-infrastructure-points")

        write_json(
            self.root / "official-split.json",
            {
                "train": ["train-seq"],
                "val": ["validation-seq"],
                "test": [],
            },
        )
        split_args = argparse.Namespace(
            dataset_root=str(self.dataset),
            source=str(self.root / "official-split.json"),
            protocol_id="V2XSEQ-TRACK-v1",
            require_complete=True,
            cooperative_info="data_info.json",
            source_key=None,
        )
        self.split = freeze_v2xseq_inputs.freeze_split(split_args)
        self.split_path = self.root / "frozen-split.json"
        write_json(self.split_path, self.split)

        self.identity_args = argparse.Namespace(
            dataset_root=str(self.dataset),
            sequence=["train-seq", "validation-seq"],
            release_id="official-test-fixture",
            acknowledge_license=True,
            max_frames=1,
            cooperative_info="data_info.json",
            vehicle_info="vehicle-side/data_info.json",
            label_template="label/{vehicle_frame}.json",
            asset_template=[
                "vehicle:lidar:vehicle-side/velodyne/{vehicle_frame}.pcd",
                "infrastructure:lidar:infrastructure-side/velodyne/{infrastructure_frame}.pcd",
            ],
        )
        self.identity = freeze_v2xseq_inputs.freeze_identity(self.identity_args)
        self.identity_path = self.root / "identity.json"
        write_json(self.identity_path, self.identity)

        self.protocol_path = self.repository / "protocol.json"
        write_json(
            self.protocol_path,
            {
                "schema_version": 1,
                "protocol_id": "V2XSEQ-TRACK-v1",
                "status": "frozen",
                "dataset": "V2X-Seq-SPD",
                "causal_rule": "arrival_time <= decision_time",
                "time_model": {
                    "native_timestamp_unit": "microseconds",
                    "sampling_interval_ns": 100_000_000,
                },
                "coordinate_frame": "vehicle lidar frame at decision time",
                "classes": ["vehicle"],
                "fault_grid": {"fixed_latency_ms": [0]},
                "tracking_metrics": ["AMOTA", "AMOTP"],
                "unresolved": [],
            },
        )
        self.adapter_path = self.repository / "adapter.py"
        self.adapter_path.write_text(
            """
class Adapter:
    def __init__(self, context):
        self.context = context
        self.evaluated = False
    def runtime_environment(self):
        return {"framework": "fixture", "framework_version": "1", "device_name": "NVIDIA A100 fixture"}
    def forward(self, frame, training):
        if training:
            assert "labels" in frame
            assert frame["sequence_id"] == "train-seq"
            frame["labels"][0]["track_id"] = "training-side-mutation"
        else:
            assert "labels" not in frame
            assert frame["sequence_id"] == "validation-seq"
        return {"training": training, "frame_id": frame["frame_id"]}
    def backward(self, output):
        return {"loss": 1.0, "backward_completed": True}
    def optimizer_step(self): pass
    def evaluate(self, predictions, targets):
        assert all("labels" in target for target in targets)
        assert all("assets" not in target for target in targets)
        assert all(target["sequence_id"] == "validation-seq" for target in targets)
        predictions[0]["prediction"]["frame_id"] = "evaluation-side-mutation"
        targets[0]["labels"][0]["track_id"] = "evaluation-side-mutation"
        self.evaluated = True
        return {"AMOTA": 0.0, "AMOTP": 1.0}
    def save_checkpoint(self, path):
        assert not self.evaluated
        with open(path, "wb") as stream: stream.write(b"checkpoint")
def build_adapter(context): return Adapter(context)
""".lstrip(),
            encoding="utf-8",
        )
        self.config_path = self.repository / "config.json"
        self.config = {
            "schema_version": 1,
            "canary_type": "real_v2xseq",
            "scientific_claim_allowed": False,
            "protocol": "V2XSEQ-TRACK-v1",
            "stage": "canary",
            "template_config": {"path": "config.json", "sha256": "f" * 64},
            "seed": 3407,
            "clearml": {
                "project": "Thesis/RTP-V2X",
                "queue": "GPU4-A100",
                "require_task_id": True,
            },
            "data": {
                "dataset": "V2X-Seq-SPD",
                "read_only_mount_required": True,
                "protocol_path": "protocol.json",
                "identity_manifest": str(self.identity_path),
                "identity_manifest_sha256": v2xseq_real_canary.sha256_file(
                    self.identity_path
                ),
                "frozen_split": str(self.split_path),
                "frozen_split_sha256": v2xseq_real_canary.sha256_file(
                    self.split_path
                ),
                "training": {
                    "partition": "train",
                    "sequence_id": "train-seq",
                    "max_frames": 1,
                },
                "evaluation": {
                    "partition": "validation",
                    "sequence_id": "validation-seq",
                    "max_frames": 1,
                },
                "required_assets": [
                    {"agent": "vehicle", "modality": "lidar"},
                    {"agent": "infrastructure", "modality": "lidar"},
                ],
            },
            "model": {
                "name": "fixture-centerpoint",
                "implementation_status": "ready",
                "initialization": "from_scratch",
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "source_tree_seal": {
                    "manifest_path": None,
                    "manifest_file_sha256": None,
                },
                "installed_environment_seal": {
                    "manifest_path": None,
                    "manifest_file_sha256": None,
                },
            },
            "adapter": {
                "path": "adapter.py",
                "sha256": v2xseq_real_canary.sha256_file(self.adapter_path),
                "factory": "build_adapter",
                "settings": {},
                "package_root": ".",
                "execution_enabled": True,
                "execution_status": "enabled",
                "package_seal": {
                    "manifest_path": None,
                    "manifest_file_sha256": None,
                },
            },
            "runtime": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "amp": False,
                "require_gpu_name_substring": "A100",
            },
            "evidence": {
                "save_predictions": True,
                "save_metrics": True,
                "save_checkpoint": True,
                "sha256_manifest": True,
                "registry_write_allowed": False,
            },
        }
        write_json(self.config_path, self.config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_args(self, output_name: str = "output") -> argparse.Namespace:
        return argparse.Namespace(
            repository_root=str(self.repository),
            dataset_root=str(self.dataset),
            config="config.json",
            contracts_root=None,
            output=str(self.root / output_name),
            clearml_task_id="a" * 32,
            allow_dirty_diagnostic=False,
        )

    @staticmethod
    def source_evidence() -> dict[str, dict[str, str]]:
        return {
            "adapter": {"git_commit": "a" * 40, "sha256": "b" * 64},
            "provenance": {"git_commit": "a" * 40, "sha256": "f" * 64},
            "protocol": {"git_commit": "a" * 40, "sha256": "d" * 64},
            "runner": {"git_commit": "a" * 40, "sha256": "e" * 64},
        }

    @staticmethod
    def execution_provenance() -> dict[str, object]:
        return {
            "resolved_config_sha256": "1" * 64,
            "template_config_sha256": "2" * 64,
            "source_tree": {"git_commit": "a" * 40},
            "adapter_package": {"git_commit": "a" * 40},
            "installed_environment": {
                "file_sha256": "3" * 64,
                "manifest_sha256": "4" * 64,
            },
        }

    def run_patches(self):
        return (
            mock.patch.object(v2xseq_real_canary, "ensure_read_only_dataset"),
            mock.patch.object(
                v2xseq_real_canary,
                "git_identity",
                return_value=("a" * 40, False),
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "probe_gpu",
                return_value={"query": "fixture", "rows": ["NVIDIA A100 fixture"]},
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "enforce_production_execution_gate",
                return_value=self.execution_provenance(),
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "verify_canary_head_sources",
                return_value=self.source_evidence(),
            ),
        )

    def refresh_adapter_hash(self) -> None:
        self.config["adapter"]["sha256"] = v2xseq_real_canary.sha256_file(  # type: ignore[index]
            self.adapter_path
        )
        write_json(self.config_path, self.config)

    def test_identity_detects_raw_sensor_mutation(self) -> None:
        v2xseq_real_canary.validate_identity(self.dataset, self.identity)
        path = self.dataset / "vehicle-side" / "velodyne" / "000001.pcd"
        path.write_bytes(b"mutated")
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "mismatch"):
            v2xseq_real_canary.validate_identity(self.dataset, self.identity)

    def test_split_rejects_overlap(self) -> None:
        broken = json.loads(json.dumps(self.split))
        broken["partitions"]["test"] = ["train-seq"]
        broken["partition_sha256"] = v2xseq_real_canary.sha256_bytes(
            v2xseq_real_canary.canonical_bytes(broken["partitions"])
        )
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "overlap"):
            v2xseq_real_canary.validate_split(broken, protocol_id="V2XSEQ-TRACK-v1")

    def test_identity_rejects_unknown_entry_fields(self) -> None:
        broken = json.loads(json.dumps(self.identity))
        raw_sensor = next(
            entry for entry in broken["entries"] if entry["role"] == "raw_sensor"
        )
        raw_sensor["ground_truth"] = [{"track_id": "leak"}]
        broken["content_sha256"] = v2xseq_real_canary.sha256_bytes(
            v2xseq_real_canary.canonical_bytes(broken["entries"])
        )
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "strict"):
            v2xseq_real_canary.validate_identity(self.dataset, broken)

    def test_identity_rejects_nonofficial_sensor_layout(self) -> None:
        broken = json.loads(json.dumps(self.identity))
        raw_sensor = next(
            entry
            for entry in broken["entries"]
            if entry["role"] == "raw_sensor" and entry["agent"] == "vehicle"
        )
        invalid_path = self.dataset / "vehicle-side" / "velodyne" / "000099.txt"
        invalid_path.write_bytes(b"label-shaped-bytes")
        raw_sensor["path"] = "vehicle-side/velodyne/000099.txt"
        raw_sensor["size"] = invalid_path.stat().st_size
        raw_sensor["sha256"] = v2xseq_real_canary.sha256_file(invalid_path)
        broken["entries"].sort(key=lambda entry: entry["path"])
        broken["content_sha256"] = v2xseq_real_canary.sha256_bytes(
            v2xseq_real_canary.canonical_bytes(broken["entries"])
        )
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "official SPD"):
            v2xseq_real_canary.validate_identity(self.dataset, broken)

    def test_sequence_rejects_duplicate_sensor_key(self) -> None:
        broken = json.loads(json.dumps(self.identity))
        duplicate_path = self.dataset / "vehicle-side" / "velodyne" / "000099.pcd"
        duplicate_path.write_bytes(b"duplicate-vehicle-points")
        broken["entries"].append(
            {
                "path": "vehicle-side/velodyne/000099.pcd",
                "size": duplicate_path.stat().st_size,
                "sha256": v2xseq_real_canary.sha256_file(duplicate_path),
                "role": "raw_sensor",
                "sequence": "train-seq",
                "vehicle_frame": "000001",
                "infrastructure_frame": "009001",
                "agent": "vehicle",
                "modality": "lidar",
            }
        )
        broken["entries"].sort(key=lambda entry: entry["path"])
        broken["content_sha256"] = v2xseq_real_canary.sha256_bytes(
            v2xseq_real_canary.canonical_bytes(broken["entries"])
        )
        index = v2xseq_real_canary.validate_identity(self.dataset, broken)
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "duplicate"):
            v2xseq_real_canary.load_sequence(
                self.dataset,
                index,
                sequence_id="train-seq",
                max_frames=1,
                required_assets=[
                    ("vehicle", "lidar"),
                    ("infrastructure", "lidar"),
                ],
            )

    def test_sequence_never_consumes_label_from_another_sequence(self) -> None:
        broken = json.loads(json.dumps(self.identity))
        train_label = next(
            entry
            for entry in broken["entries"]
            if entry["role"] == "label" and entry["vehicle_frame"] == "000001"
        )
        train_label["sequence"] = "validation-seq"
        broken["content_sha256"] = v2xseq_real_canary.sha256_bytes(
            v2xseq_real_canary.canonical_bytes(broken["entries"])
        )
        index = v2xseq_real_canary.validate_identity(self.dataset, broken)
        with self.assertRaisesRegex(
            v2xseq_real_canary.ContractError, "exactly one label"
        ):
            v2xseq_real_canary.load_sequence(
                self.dataset,
                index,
                sequence_id="train-seq",
                max_frames=1,
                required_assets=[
                    ("vehicle", "lidar"),
                    ("infrastructure", "lidar"),
                ],
            )

    def test_identity_freeze_requires_observation_side_timestamp(self) -> None:
        vehicle_info_path = self.dataset / "vehicle-side" / "data_info.json"
        rows = json.loads(vehicle_info_path.read_text(encoding="utf-8"))
        rows[0].pop("pointcloud_timestamp")
        write_json(vehicle_info_path, rows)
        with self.assertRaisesRegex(
            v2xseq_real_canary.ContractError, "pointcloud_timestamp"
        ):
            freeze_v2xseq_inputs.freeze_identity(self.identity_args)

    def test_pending_protocol_is_fail_closed(self) -> None:
        protocol = load = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        load["status"] = "pending_data_readback"
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "must be frozen"):
            v2xseq_real_canary.validate_protocol(protocol)

    def test_json_loader_rejects_duplicate_keys_and_non_finite_values(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"protocol": "a", "protocol": "b"}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            v2xseq_real_canary.ContractError, "cannot read JSON"
        ):
            v2xseq_real_canary.load_json(duplicate)

        non_finite = self.root / "non-finite.json"
        non_finite.write_text('{"metric": NaN}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            v2xseq_real_canary.ContractError, "cannot read JSON"
        ):
            v2xseq_real_canary.load_json(non_finite)

    def test_dataset_path_rejects_intermediate_symlink(self) -> None:
        alias = self.dataset / "linked-labels"
        alias.symlink_to(self.dataset / "label", target_is_directory=True)
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "symlink"):
            v2xseq_real_canary.safe_dataset_path(
                alias.parent, "linked-labels/000001.json"
            )

    def test_repository_file_rejects_intermediate_symlink(self) -> None:
        real_directory = self.repository / "real"
        real_directory.mkdir()
        (real_directory / "file.json").write_text("{}\n", encoding="utf-8")
        (self.repository / "alias").symlink_to(real_directory, target_is_directory=True)
        with self.assertRaisesRegex(v2xseq_real_canary.ContractError, "symlink"):
            v2xseq_real_canary.resolve_repository_file(
                self.repository, "alias/file.json", "fixture"
            )

    def test_clearml_task_id_must_come_from_agent_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                v2xseq_real_canary.ContractError, "agent environment"
            ):
                v2xseq_real_canary.validate_clearml_task_id("a" * 32)

    def test_gpu_preflight_rejects_non_a100_device(self) -> None:
        result = mock.Mock(stdout="NVIDIA H100, 580.65, 81559\n")
        with (
            mock.patch.object(
                v2xseq_real_canary.subprocess, "run", return_value=result
            ),
            self.assertRaisesRegex(v2xseq_real_canary.ContractError, "A100"),
        ):
            v2xseq_real_canary.probe_gpu("A100")

    def test_dataset_mount_must_be_read_only(self) -> None:
        with (
            mock.patch.object(v2xseq_real_canary.os, "access", return_value=True),
            self.assertRaisesRegex(v2xseq_real_canary.ContractError, "read-only"),
        ):
            v2xseq_real_canary.ensure_read_only_dataset(self.dataset)

    def test_contracts_mount_must_be_read_only(self) -> None:
        with (
            mock.patch.object(v2xseq_real_canary.os, "access", return_value=True),
            self.assertRaisesRegex(v2xseq_real_canary.ContractError, "read-only"),
        ):
            v2xseq_real_canary.ensure_read_only_contracts_root(self.repository)

    def test_production_execution_gate_requires_external_resolved_config(self) -> None:
        with self.assertRaisesRegex(
            v2xseq_real_canary.ContractError,
            "external resolved config",
        ):
            v2xseq_real_canary.enforce_production_execution_gate(
                self.config["model"], self.config["adapter"]
            )

    def test_production_execution_gate_rejects_dirty_head(self) -> None:
        contracts = self.root / "dirty-contracts"
        contracts.mkdir()
        resolved_config = contracts / "resolved.json"
        write_json(resolved_config, {"resolved": True})
        with (
            mock.patch.object(
                v2xseq_real_canary, "ensure_read_only_contracts_root"
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "git_identity",
                return_value=("c" * 40, True),
            ),
            self.assertRaisesRegex(v2xseq_real_canary.ContractError, "clean Git"),
        ):
            v2xseq_real_canary.enforce_production_execution_gate(
                self.config["model"],
                self.config["adapter"],
                repository_root=self.repository,
                contracts_root=contracts,
                resolved_config_path=resolved_config,
                resolved_config_sha256=v2xseq_real_canary.sha256_file(
                    resolved_config
                ),
            )

    def test_production_execution_gate_accepts_complete_external_seals(self) -> None:
        contracts = self.root / "contracts"
        contracts.mkdir()
        seal_paths: dict[str, Path] = {}
        for name in ("source.json", "package.json", "environment.json"):
            path = contracts / name
            write_json(path, {})
            seal_paths[name] = path
        resolved_config = contracts / "resolved.json"
        write_json(resolved_config, {"resolved": True})

        package_root = self.repository / "adapter-package"
        package_root.mkdir()
        package_adapter = package_root / "adapter.py"
        package_lock = package_root / "requirements.lock"
        package_adapter.write_text("# sealed adapter\n", encoding="utf-8")
        package_lock.write_text("torch==2.8.0\n", encoding="utf-8")
        template = self.repository / v2xseq_real_canary.TEMPLATE_CONFIG_PATH
        write_json(
            template,
            {
                "schema_version": 1,
                "canary_type": "real_v2xseq",
                "stage": "canary",
                "scientific_claim_allowed": False,
                "adapter": {
                    "execution_enabled": False,
                    "execution_status": "disabled_pending_inputs",
                },
                "evidence": {"registry_write_allowed": False},
            },
        )

        source_paths = {
            v2xseq_real_canary.TEMPLATE_CONFIG_PATH,
            "runner.py",
        }
        package_paths = {
            "adapter-package/adapter.py",
            "adapter-package/requirements.lock",
        }
        source_document = {
            "git_commit": "c" * 40,
            "manifest_sha256": "1" * 64,
            "entries": [{"path": path} for path in sorted(source_paths)],
        }
        package_document = {
            "git_commit": "c" * 40,
            "manifest_sha256": "2" * 64,
            "entries": [{"path": path} for path in sorted(package_paths)],
        }
        environment_document = {"manifest_sha256": "3" * 64}
        model = {
            "implementation_status": "ready",
            "source_tree_seal": {
                "manifest_path": "source.json",
                "manifest_file_sha256": v2xseq_real_canary.sha256_file(
                    seal_paths["source.json"]
                ),
            },
            "installed_environment_seal": {
                "manifest_path": "environment.json",
                "manifest_file_sha256": v2xseq_real_canary.sha256_file(
                    seal_paths["environment.json"]
                ),
            },
        }
        adapter = {
            "execution_enabled": True,
            "execution_status": "enabled",
            "package_root": "adapter-package",
            "package_seal": {
                "manifest_path": "package.json",
                "manifest_file_sha256": v2xseq_real_canary.sha256_file(
                    seal_paths["package.json"]
                ),
            },
        }
        with (
            mock.patch.object(
                v2xseq_real_canary,
                "git_identity",
                return_value=("c" * 40, False),
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "validate_git_seal",
                side_effect=[source_document, package_document],
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "validate_installed_environment_manifest",
                return_value=environment_document,
            ),
            mock.patch.object(
                v2xseq_real_canary,
                "ensure_read_only_contracts_root",
            ),
        ):
            evidence = v2xseq_real_canary.enforce_production_execution_gate(
                model,
                adapter,
                repository_root=self.repository,
                contracts_root=contracts,
                resolved_config_path=resolved_config,
                resolved_config_sha256=v2xseq_real_canary.sha256_file(
                    resolved_config
                ),
                template_config={
                    "path": v2xseq_real_canary.TEMPLATE_CONFIG_PATH,
                    "sha256": v2xseq_real_canary.sha256_file(template),
                },
                required_source_paths=source_paths,
                required_package_paths=package_paths,
            )
        self.assertEqual(evidence["source_tree"]["git_commit"], "c" * 40)
        self.assertEqual(evidence["adapter_package"]["git_commit"], "c" * 40)
        self.assertEqual(
            evidence["installed_environment"]["manifest_sha256"], "3" * 64
        )

    def test_repository_spd_template_remains_disabled_and_diagnostic_only(self) -> None:
        repository_root = MODULE_ROOT.parents[1]
        template = json.loads(
            (
                repository_root
                / v2xseq_real_canary.TEMPLATE_CONFIG_PATH
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(template["scientific_claim_allowed"])
        self.assertFalse(template["adapter"]["execution_enabled"])
        self.assertNotEqual(template["adapter"]["execution_status"], "enabled")
        self.assertNotEqual(template["model"]["implementation_status"], "ready")
        self.assertFalse(template["evidence"]["registry_write_allowed"])

    def test_adapter_bytes_are_pinned_by_resolved_config(self) -> None:
        self.adapter_path.write_text(
            self.adapter_path.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
        read_only, git, gpu, gate, source = self.run_patches()
        with (
            read_only,
            git,
            gpu,
            gate,
            source,
            self.assertRaisesRegex(
                v2xseq_real_canary.ContractError, "adapter SHA-256"
            ),
        ):
            v2xseq_real_canary.run_canary(self.run_args("adapter-hash-output"))

    def test_evaluator_cannot_rewrite_pre_target_checkpoint(self) -> None:
        self.adapter_path.write_text(
            """
class Adapter:
    def __init__(self, context): self.checkpoint = None
    def runtime_environment(self):
        return {"framework": "fixture", "framework_version": "1", "device_name": "NVIDIA A100 fixture"}
    def forward(self, frame, training):
        return {"training": training, "frame_id": frame["frame_id"]}
    def backward(self, output):
        return {"loss": 1.0, "backward_completed": True}
    def optimizer_step(self): pass
    def save_checkpoint(self, path):
        self.checkpoint = path
        with open(path, "wb") as stream: stream.write(b"checkpoint")
    def evaluate(self, predictions, targets):
        with open(self.checkpoint, "wb") as stream: stream.write(b"target-tainted")
        return {"AMOTA": 0.0, "AMOTP": 1.0}
def build_adapter(context): return Adapter(context)
""".lstrip(),
            encoding="utf-8",
        )
        self.refresh_adapter_hash()
        read_only, git, gpu, gate, source = self.run_patches()
        with (
            read_only,
            git,
            gpu,
            gate,
            source,
            self.assertRaisesRegex(
                v2xseq_real_canary.ContractError,
                "modified frozen pre-target artifact",
            ),
        ):
            v2xseq_real_canary.run_canary(self.run_args("tainted-output"))

    def test_checkpoint_save_cannot_rewrite_frozen_predictions(self) -> None:
        self.adapter_path.write_text(
            """
from pathlib import Path
class Adapter:
    def __init__(self, context): pass
    def runtime_environment(self):
        return {"framework": "fixture", "framework_version": "1", "device_name": "NVIDIA A100 fixture"}
    def forward(self, frame, training):
        return {"training": training, "frame_id": frame["frame_id"]}
    def backward(self, output):
        return {"loss": 1.0, "backward_completed": True}
    def optimizer_step(self): pass
    def save_checkpoint(self, path):
        Path(path).write_bytes(b"checkpoint")
        Path(path).with_name("predictions.jsonl").write_bytes(b"tampered")
    def evaluate(self, predictions, targets):
        return {"AMOTA": 0.0, "AMOTP": 1.0}
def build_adapter(context): return Adapter(context)
""".lstrip(),
            encoding="utf-8",
        )
        self.refresh_adapter_hash()
        read_only, git, gpu, gate, source = self.run_patches()
        with (
            read_only,
            git,
            gpu,
            gate,
            source,
            self.assertRaisesRegex(
                v2xseq_real_canary.ContractError,
                "modified frozen predictions while saving checkpoint",
            ),
        ):
            v2xseq_real_canary.run_canary(
                self.run_args("checkpoint-tainted-output")
            )

    def test_control_snapshot_hash_precedes_later_path_replacement(self) -> None:
        original_validate_protocol = v2xseq_real_canary.validate_protocol

        def replace_config_after_snapshot(document: object) -> dict[str, object]:
            replacement = json.loads(self.config_path.read_text(encoding="utf-8"))
            replacement["replacement_marker"] = True
            write_json(self.config_path, replacement)
            return original_validate_protocol(document)

        read_only, git, gpu, gate, source = self.run_patches()
        with (
            read_only,
            git,
            gpu,
            gate,
            source,
            mock.patch.object(
                v2xseq_real_canary,
                "validate_protocol",
                side_effect=replace_config_after_snapshot,
            ),
            self.assertRaisesRegex(
                v2xseq_real_canary.ContractError, "control file changed"
            ),
        ):
            v2xseq_real_canary.run_canary(
                self.run_args("control-swapped-output")
            )

    def test_real_canary_calls_all_phases_and_exports_hashed_evidence(self) -> None:
        read_only, git, gpu, gate, source = self.run_patches()
        with (
            read_only,
            git,
            gpu,
            gate,
            source,
        ):
            manifest = v2xseq_real_canary.run_canary(self.run_args())
        output = self.root / "output"
        self.assertEqual(manifest["claim_scope"], "diagnostic_only")
        self.assertFalse(manifest["scientific_claim_allowed"])
        self.assertEqual(manifest["clearml_task_id"], "a" * 32)
        self.assertFalse(manifest["registry_write_allowed"])
        self.assertEqual(
            manifest["execution_provenance"], self.execution_provenance()
        )
        self.assertEqual(manifest["training_sequence_id"], "train-seq")
        self.assertEqual(manifest["evaluation_sequence_id"], "validation-seq")
        self.assertEqual(manifest["training_frame_ids"], ["000001"])
        self.assertEqual(manifest["evaluation_frame_ids"], ["000002"])
        self.assertEqual(
            set(manifest["source_head_evidence"]),
            {"adapter", "provenance", "protocol", "runner"},
        )
        self.assertEqual(
            set(manifest["artifacts"]), set(v2xseq_real_canary.REQUIRED_ARTIFACTS)
        )
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(v2xseq_real_canary.sha256_file(output / name), digest)
        events = (output / "stdout.log").read_text(encoding="utf-8")
        for phase in (
            "forward_complete",
            "backward_complete",
            "optimizer_step_complete",
            "evaluation_complete",
            "checkpoint_complete",
        ):
            self.assertIn(phase, events)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["metrics"], {"AMOTA": 0.0, "AMOTP": 1.0})
        environment = json.loads(
            (output / "environment.json").read_text(encoding="utf-8")
        )
        self.assertIn("NVIDIA A100 fixture", environment["gpu"]["rows"][0])
        self.assertEqual(environment["adapter"]["framework"], "fixture")
        predictions = (output / "predictions.jsonl").read_text(encoding="utf-8")
        self.assertNotIn('"labels"', predictions)
        self.assertNotIn("evaluation-side-mutation", predictions)
        self.assertEqual((output / "checkpoint.bin").read_bytes(), b"checkpoint")
        self.assertEqual(
            manifest["pre_target_artifacts"]["predictions.jsonl"],
            v2xseq_real_canary.sha256_file(output / "predictions.jsonl"),
        )
        self.assertEqual(
            manifest["pre_target_artifacts"]["checkpoint.bin"],
            v2xseq_real_canary.sha256_file(output / "checkpoint.bin"),
        )
        expected_training_target = {
            "sequence_id": "train-seq",
            "frame_id": "000001",
            "timestamp": 1_000_000,
            "labels": [
                {
                    "track_id": "t1",
                    "type": "Car",
                    "veh_pointcloud_timestamp": "1000000",
                    "3d_location": {"x": 1, "y": 2, "z": 0},
                }
            ],
        }
        self.assertEqual(
            manifest["payload_evidence"]["training_target_sha256"],
            v2xseq_real_canary.sha256_bytes(
                v2xseq_real_canary.canonical_bytes(expected_training_target)
            ),
        )
        self.assertEqual(
            set(manifest["payload_evidence"]),
            {
                "training_observation_sha256",
                "training_target_sha256",
                "evaluation_observation_sha256",
                "evaluation_target_sha256",
            },
        )
        self.assertIn("run_manifest.json", (output / "SHA256SUMS").read_text())


if __name__ == "__main__":
    unittest.main()
