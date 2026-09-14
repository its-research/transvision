#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import freeze_v2xseq_tfd_inputs  # noqa: E402
import v2xseq_tfd_canary  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_row(
    timestamp: int,
    actor_id: str,
    tag: str,
    *,
    cooperative: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "city": "fixture-city",
        "timestamp": timestamp,
        "id": actor_id,
        "type": "VEHICLE",
        "sub_type": "Car",
        "tag": tag,
        "x": float(timestamp) + (1.0 if actor_id == "target" else 0.0),
        "y": 2.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 1.5,
        "theta": 0.0,
        "v_x": 1.0,
        "v_y": 0.0,
        "intersect_id": "001",
    }
    if cooperative:
        row.update(
            {
                "vic_tag": "car",
                "from_side": "vehicle view",
                "car_side_id": actor_id,
                "road_side_id": actor_id,
            }
        )
    return row


def traffic_light_row(timestamp: int) -> dict[str, object]:
    return {
        "city": "fixture-city",
        "timestamp": timestamp,
        "x": 0.0,
        "y": 0.0,
        "direction": "north",
        "lane_id": "lane-1",
        "color_1": "green",
        "remain_1": 10,
        "color_2": "red",
        "remain_2": 0,
        "color_3": "yellow",
        "remain_3": 0,
        "intersect_id": "001",
    }


class TFDForecastCanaryContractTest(unittest.TestCase):
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
        for scene in ("train-scene", "validation-scene", "test-scene"):
            cooperative_rows: list[dict[str, object]] = []
            vehicle_rows: list[dict[str, object]] = []
            infrastructure_rows: list[dict[str, object]] = []
            for timestamp in range(4):
                cooperative_rows.append(
                    trajectory_row(timestamp, "av", "AV", cooperative=True)
                )
                cooperative_rows.append(
                    trajectory_row(
                        timestamp, "target", "TARGET_AGENT", cooperative=True
                    )
                )
                vehicle_rows.append(
                    trajectory_row(
                        timestamp, "target", "TARGET_AGENT", cooperative=False
                    )
                )
                infrastructure_rows.append(
                    trajectory_row(
                        timestamp, "road-target", "OTHERS", cooperative=False
                    )
                )
            write_csv(
                self.dataset
                / "cooperative-vehicle-infrastructure"
                / "cooperative-trajectories"
                / f"{scene}.csv",
                cooperative_rows,
            )
            write_csv(
                self.dataset
                / "cooperative-vehicle-infrastructure"
                / "vehicle-trajectories"
                / f"{scene}.csv",
                vehicle_rows,
            )
            write_csv(
                self.dataset
                / "cooperative-vehicle-infrastructure"
                / "infrastructure-trajectories"
                / f"{scene}.csv",
                infrastructure_rows,
            )
            write_csv(
                self.dataset
                / "cooperative-vehicle-infrastructure"
                / "traffic-light"
                / f"{scene}.csv",
                [traffic_light_row(timestamp) for timestamp in range(4)],
            )
        write_json(
            self.dataset / "maps" / "hdmap001.json",
            {"LANE": {"lane-1": {"centerline": [[0, 0], [1, 0]]}}},
        )

        self.split_source = self.root / "official-split.json"
        write_json(
            self.split_source,
            {
                "train": ["train-scene"],
                "val": ["validation-scene"],
                "test": ["test-scene"],
            },
        )
        split_args = argparse.Namespace(
            dataset_root=str(self.dataset),
            source=str(self.split_source),
            protocol_id="V2XSEQ-FORECAST-v1",
            source_key=None,
            require_complete=True,
            cooperative_directory="cooperative-vehicle-infrastructure/cooperative-trajectories",
        )
        self.split = freeze_v2xseq_tfd_inputs.freeze_split(split_args)
        self.split_path = self.root / "frozen-split.json"
        write_json(self.split_path, self.split)

        identity_args = argparse.Namespace(
            dataset_root=str(self.dataset),
            scene=["train-scene", "validation-scene"],
            release_id="official-test-fixture",
            acknowledge_license=True,
            cooperative_template="cooperative-vehicle-infrastructure/cooperative-trajectories/{scene_id}.csv",
            vehicle_template="cooperative-vehicle-infrastructure/vehicle-trajectories/{scene_id}.csv",
            infrastructure_template="cooperative-vehicle-infrastructure/infrastructure-trajectories/{scene_id}.csv",
            traffic_light_template="cooperative-vehicle-infrastructure/traffic-light/{scene_id}.csv",
            map_template="maps/hdmap{intersection_id}.json",
        )
        self.identity = freeze_v2xseq_tfd_inputs.freeze_identity(identity_args)
        self.identity_path = self.root / "identity.json"
        write_json(self.identity_path, self.identity)

        self.reference_implementation_path = (
            self.repository / "experiments" / "rtp_v2x" / "evaluator_conformance.py"
        )
        self.reference_implementation_path.parent.mkdir(parents=True)
        self.reference_implementation_path.write_text(
            "# fixture implementation\n", encoding="utf-8"
        )
        self.reference_evaluator_path = self.repository / "evaluator-conformance.json"
        write_json(
            self.reference_evaluator_path,
            {
                "schema_version": 1,
                "contract_id": "RTPV2X-EVALUATOR-CONFORMANCE-v1",
                "status": "reference_conformance_only",
                "scientific_claim_allowed": False,
                "official_evaluator_equivalence": False,
                "implementation": "experiments/rtp_v2x/evaluator_conformance.py",
                "forecasting": {
                    "metrics": {
                        name: "fixture definition"
                        for name in (
                            "minADE",
                            "minFDE",
                            "MR",
                            "NLL",
                            "Brier",
                            "ECE",
                            "AURC",
                        )
                    }
                },
                "numeric_policy": {"conformance_absolute_tolerance": 1e-12},
            },
        )
        self.protocol_path = self.repository / "protocol.json"
        write_json(
            self.protocol_path,
            {
                "schema_version": 1,
                "protocol_id": "V2XSEQ-FORECAST-v1",
                "status": "frozen",
                "dataset": "V2X-Seq-TFD",
                "task": "vehicle-infrastructure cooperative multimodal trajectory forecasting",
                "coordinate_frame": "world map frame",
                "target_tag": "TARGET_AGENT",
                "causal_rule": "all history observations must have arrival_time <= decision_time",
                "arrival_model": "arrival_time_ns = event_time_ns",
                "time_model": {
                    "native_timestamp_unit": "deciseconds",
                    "native_timestamp_scale_to_ns": 100000000,
                    "sampling_interval_ns": 100000000,
                },
                "history_steps": 2,
                "future_steps": 2,
                "history_duration_ns": 200000000,
                "forecast_duration_ns": 200000000,
                "modes": 2,
                "spatial_dimensions": 2,
                "metrics": ["minADE", "minFDE", "MR", "NLL", "ECE", "Brier", "AURC"],
                "miss_rate_threshold_m": 2.0,
                "calibration_definition": {
                    "nll_sigma_m": 1.0,
                    "ece_event": "mode reaches final displacement threshold",
                    "ece_bins": 10,
                    "brier_event": "mode reaches final displacement threshold",
                    "aurc_coverage_grid": "all_prefixes",
                },
                "reference_evaluator_contract": {
                    "contract_id": "RTPV2X-EVALUATOR-CONFORMANCE-v1",
                    "path": "evaluator-conformance.json",
                    "sha256": v2xseq_tfd_canary.sha256_file(
                        self.reference_evaluator_path
                    ),
                },
                "fault_grid": {"fixed_latency_ms": [0]},
                "unresolved": [],
            },
        )

        self.upstream_path = self.repository / "upstream.json"
        self.upstream_revision = "b" * 40
        write_json(
            self.upstream_path,
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "v2x_graph",
                        "repository": "https://github.com/AIR-THU/V2X-Graph",
                        "revision": self.upstream_revision,
                        "license": "Apache-2.0",
                        "license_file_observed": True,
                        "redistribution_allowed_by_this_manifest": True,
                        "capabilities": {"v2x_seq_tfd_forecasting": True},
                    }
                ],
            },
        )
        self.lock_path = self.repository / "requirements.lock"
        self.lock_path.write_text("torch==2.8.0\n", encoding="utf-8")
        self.isolation_path = self.repository / "isolation.json"
        self.isolation_path.write_bytes(
            (
                MODULE_ROOT.parents[0]
                / "clearml"
                / "protocols"
                / "history-only-subprocess-v1.json"
            ).read_bytes()
        )
        self.adapter_path = self.repository / "adapter.py"
        self.adapter_path.write_text(
            """
class Adapter:
    def __init__(self, context):
        context["protocol"]["calibration_definition"]["nll_sigma_m"] = 100.0
        self.context = context
    def runtime_environment(self):
        return {"framework": "fixture", "framework_version": "1", "device_name": "NVIDIA A100 fixture"}
    def forward(self, sample, training):
        if training:
            assert "ground_truth" in sample
            return {"loss_tensor": "fixture"}
        assert "ground_truth" not in sample
        assert all(row["arrival_time_ns"] <= sample["history"]["decision_time_ns"] for row in sample["history"]["cooperative_trajectories"])
        return {
            "scene_id": sample["scene_id"],
            "target_id": sample["target_id"],
            "modes": [
                {"mode_id": "m0", "probability": 0.6, "trajectory": [[3.0, 2.0], [4.0, 2.0]]},
                {"mode_id": "m1", "probability": 0.4, "trajectory": [[3.0, 2.1], [4.0, 2.1]]},
            ],
        }
    def backward(self, output):
        return {"loss": 1.0, "gradient_norm": 0.5, "backward_completed": True}
    def optimizer_step(self):
        return {"parameter_update_norm": 0.1, "optimizer_step_completed": True}
    def evaluate(self, predictions, targets):
        assert all("ground_truth" in target for target in targets)
        predictions[0]["modes"][0]["trajectory"][0][0] = 999.0
        targets[0]["ground_truth"]["positions"][0][0] = 999.0
        return {"minADE": 0.0, "minFDE": 0.0, "MR": 0.0, "NLL": 3.679742140862615, "ECE": 0.4, "Brier": 0.16000000000000003, "AURC": 0.0}
    def save_checkpoint(self, path):
        with open(path, "wb") as stream: stream.write(b"checkpoint")
def build_adapter(context): return Adapter(context)
""".lstrip(),
            encoding="utf-8",
        )
        self.config_path = self.repository / "config.json"
        self.config = {
            "schema_version": 1,
            "canary_type": "real_v2xseq_tfd_forecast",
            "scientific_claim_allowed": False,
            "protocol": "V2XSEQ-FORECAST-v1",
            "stage": "canary",
            "template_config": {
                "path": "config.json",
                "sha256": "f" * 64,
            },
            "seed": 3407,
            "clearml": {
                "project": "Thesis/RTP-V2X",
                "queue": "GPU4-A100",
                "require_task_id": True,
            },
            "data": {
                "dataset": "V2X-Seq-TFD",
                "read_only_mount_required": True,
                "cooperative_directory": "cooperative-vehicle-infrastructure/cooperative-trajectories",
                "protocol_path": "protocol.json",
                "identity_manifest": str(self.identity_path),
                "identity_manifest_sha256": v2xseq_tfd_canary.sha256_file(
                    self.identity_path
                ),
                "frozen_split": str(self.split_path),
                "frozen_split_sha256": v2xseq_tfd_canary.sha256_file(self.split_path),
                "training_scene_id": "train-scene",
                "evaluation_scene_ids": ["validation-scene"],
            },
            "model": {
                "name": "fixture-v2x-graph",
                "implementation_status": "ready",
                "initialization": "from_scratch",
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "upstream_manifest_path": "upstream.json",
                "upstream_id": "v2x_graph",
                "upstream_revision": self.upstream_revision,
                "dependency_lock_path": "requirements.lock",
                "dependency_lock_sha256": v2xseq_tfd_canary.sha256_file(self.lock_path),
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
                "sha256": v2xseq_tfd_canary.sha256_file(self.adapter_path),
                "factory": "build_adapter",
                "package_root": "adapter-package",
                "execution_enabled": True,
                "execution_status": "enabled",
                "package_seal": {
                    "manifest_path": None,
                    "manifest_file_sha256": None,
                },
                "isolation": {
                    "mode": "history_only_subprocess",
                    "manifest_path": "isolation.json",
                    "manifest_file_sha256": v2xseq_tfd_canary.sha256_file(
                        self.isolation_path
                    ),
                },
            },
            "runtime": {
                "epochs": 1,
                "batch_size": 1,
                "subprocess_timeout_seconds": 10.0,
                "require_gpu_name_substring": "A100",
            },
            "evidence": {
                "save_multimodal_predictions": True,
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
            output=str(self.root / output_name),
            clearml_task_id="a" * 32,
            allow_dirty_diagnostic=False,
            contracts_root=None,
        )

    def run_patches(self):
        return (
            mock.patch.object(v2xseq_tfd_canary, "ensure_read_only_dataset"),
            mock.patch.object(
                v2xseq_tfd_canary,
                "git_identity",
                return_value=("c" * 40, False),
            ),
            mock.patch.object(
                v2xseq_tfd_canary,
                "probe_gpu",
                return_value={"query": "fixture", "rows": ["NVIDIA A100 fixture"]},
            ),
        )

    def test_identity_detects_trajectory_mutation(self) -> None:
        v2xseq_tfd_canary.validate_identity(self.dataset, self.identity)
        path = (
            self.dataset
            / "cooperative-vehicle-infrastructure"
            / "cooperative-trajectories"
            / "train-scene.csv"
        )
        path.write_text(
            path.read_text(encoding="utf-8") + "mutated\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "mismatch"):
            v2xseq_tfd_canary.validate_identity(self.dataset, self.identity)

    def test_split_requires_three_nonempty_disjoint_partitions(self) -> None:
        broken = json.loads(json.dumps(self.split))
        broken["partitions"]["test"] = ["train-scene"]
        broken["partition_sha256"] = v2xseq_tfd_canary.sha256_bytes(
            v2xseq_tfd_canary.canonical_bytes(broken["partitions"])
        )
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "overlap"):
            v2xseq_tfd_canary.validate_split(broken, protocol_id="V2XSEQ-FORECAST-v1")
        incomplete = json.loads(json.dumps(self.split))
        incomplete["scene_universe_sha256"] = v2xseq_tfd_canary.sha256_bytes(
            v2xseq_tfd_canary.canonical_bytes(
                ["extra-scene", "test-scene", "train-scene", "validation-scene"]
            )
        )
        with self.assertRaisesRegex(
            v2xseq_tfd_canary.ContractError, "complete scene universe"
        ):
            v2xseq_tfd_canary.validate_split(
                incomplete, protocol_id="V2XSEQ-FORECAST-v1"
            )

    def test_pending_or_incomplete_protocol_is_fail_closed(self) -> None:
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        protocol["status"] = "pending_data_readback"
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "must be frozen"):
            v2xseq_tfd_canary.validate_protocol(protocol)
        protocol["status"] = "frozen"
        protocol["calibration_definition"]["ece_bins"] = None
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "ece_bins"):
            v2xseq_tfd_canary.validate_protocol(protocol)

    def test_prediction_rejects_invalid_distribution(self) -> None:
        protocol = v2xseq_tfd_canary.validate_protocol(
            json.loads(self.protocol_path.read_text(encoding="utf-8"))
        )
        sample = v2xseq_tfd_canary.load_forecast_sample(
            self.dataset,
            v2xseq_tfd_canary.validate_identity(self.dataset, self.identity),
            scene_id="validation-scene",
            protocol=protocol,
        )
        invalid = {
            "scene_id": "validation-scene",
            "target_id": "target",
            "modes": [
                {
                    "mode_id": "m0",
                    "probability": 0.6,
                    "trajectory": [[0.0, 0.0], [1.0, 0.0]],
                },
                {
                    "mode_id": "m1",
                    "probability": 0.6,
                    "trajectory": [[0.0, 0.0], [1.0, 0.0]],
                },
            ],
        }
        with self.assertRaises(v2xseq_tfd_canary.ContractError):
            v2xseq_tfd_canary.validate_prediction(
                invalid, sample=sample, protocol=protocol
            )

    def test_history_is_limited_to_the_exact_frozen_window(self) -> None:
        rows = [
            {
                key: str(value)
                for key, value in trajectory_row(
                    timestamp,
                    "target",
                    "TARGET_AGENT",
                    cooperative=False,
                ).items()
            }
            for timestamp in range(4)
        ]
        history = v2xseq_tfd_canary.normalized_history_rows(
            rows,
            scale_to_ns=1,
            history_timestamps_ns={1, 2},
            label="fixture trajectory",
        )
        self.assertEqual([row["event_time_ns"] for row in history], [1, 2])

    def test_unlicensed_upstream_is_rejected(self) -> None:
        manifest = json.loads(self.upstream_path.read_text(encoding="utf-8"))
        manifest["sources"][0]["license_file_observed"] = False
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "license"):
            v2xseq_tfd_canary.validate_upstream(
                manifest,
                source_id="v2x_graph",
                expected_revision=self.upstream_revision,
            )

    def test_clearml_task_id_must_come_from_agent_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                v2xseq_tfd_canary.ContractError, "agent environment"
            ):
                v2xseq_tfd_canary.validate_clearml_task_id("a" * 32)

    def test_mounted_scene_universe_must_match_frozen_split(self) -> None:
        source = (
            self.dataset
            / "cooperative-vehicle-infrastructure"
            / "cooperative-trajectories"
            / "train-scene.csv"
        )
        extra = source.with_name("unexpected-scene.csv")
        extra.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "scene universe"):
            v2xseq_tfd_canary.validate_scene_universe(
                self.dataset,
                "cooperative-vehicle-infrastructure/cooperative-trajectories",
                self.split["scene_universe_sha256"],
            )

    def test_metrics_must_match_exact_names_and_ranges(self) -> None:
        names = ["minADE", "minFDE", "MR", "NLL", "ECE", "Brier", "AURC"]
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "exactly"):
            v2xseq_tfd_canary.validate_metrics({"minADE": 0.0}, names)
        values = {name: 0.0 for name in names}
        values["MR"] = 1.1
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, r"\[0, 1\]"):
            v2xseq_tfd_canary.validate_metrics(values, names)

    def test_phase_reports_reject_unfrozen_extra_fields(self) -> None:
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "exactly"):
            v2xseq_tfd_canary.validate_backward_report(
                {
                    "loss": 1.0,
                    "gradient_norm": 0.5,
                    "backward_completed": True,
                    "diagnostic_detail": "must-not-be-persisted",
                }
            )
        with self.assertRaisesRegex(v2xseq_tfd_canary.ContractError, "exactly"):
            v2xseq_tfd_canary.validate_optimizer_report(
                {
                    "parameter_update_norm": 0.1,
                    "optimizer_step_completed": True,
                    "diagnostic_detail": "must-not-be-persisted",
                }
            )

    def test_production_execution_gate_requires_external_resolved_config(self) -> None:
        with self.assertRaisesRegex(
            v2xseq_tfd_canary.ContractError,
            "external resolved config",
        ):
            v2xseq_tfd_canary.enforce_production_execution_gate(
                self.config["model"], self.config["adapter"]
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
        template = self.repository / "template.json"
        write_json(template, {"execution_enabled": False})

        source_paths = {
            "template.json",
            "isolation.json",
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
            "source_tree_seal": {
                "manifest_path": "source.json",
                "manifest_file_sha256": v2xseq_tfd_canary.sha256_file(
                    seal_paths["source.json"]
                ),
            },
            "installed_environment_seal": {
                "manifest_path": "environment.json",
                "manifest_file_sha256": v2xseq_tfd_canary.sha256_file(
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
                "manifest_file_sha256": v2xseq_tfd_canary.sha256_file(
                    seal_paths["package.json"]
                ),
            },
            "isolation": {
                "mode": "history_only_subprocess",
                "manifest_path": "isolation.json",
                "manifest_file_sha256": v2xseq_tfd_canary.sha256_file(
                    self.isolation_path
                ),
            },
        }
        with (
            mock.patch.object(
                v2xseq_tfd_canary,
                "git_identity",
                return_value=("c" * 40, False),
            ),
            mock.patch.object(
                v2xseq_tfd_canary,
                "validate_git_seal",
                side_effect=[source_document, package_document],
            ),
            mock.patch.object(
                v2xseq_tfd_canary,
                "validate_installed_environment_manifest",
                return_value=environment_document,
            ),
            mock.patch.object(
                v2xseq_tfd_canary,
                "ensure_read_only_contracts_root",
            ),
        ):
            evidence = v2xseq_tfd_canary.enforce_production_execution_gate(
                model,
                adapter,
                repository_root=self.repository,
                contracts_root=contracts,
                resolved_config_path=resolved_config,
                resolved_config_sha256=v2xseq_tfd_canary.sha256_file(resolved_config),
                template_config={
                    "path": "template.json",
                    "sha256": v2xseq_tfd_canary.sha256_file(template),
                },
                required_source_paths=source_paths,
                required_package_paths=package_paths,
            )
        self.assertEqual(evidence["source_tree"]["git_commit"], "c" * 40)
        self.assertEqual(
            evidence["isolation_contract_sha256"],
            v2xseq_tfd_canary.sha256_file(self.isolation_path),
        )

    def test_adapter_bytes_are_pinned_by_config_hash(self) -> None:
        self.adapter_path.write_text(
            self.adapter_path.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
        read_only, git, gpu = self.run_patches()
        with read_only, git, gpu:
            with self.assertRaisesRegex(
                v2xseq_tfd_canary.ContractError, "adapter SHA-256"
            ):
                v2xseq_tfd_canary.run_canary(self.run_args("adapter-hash-output"))

    def test_config_cannot_enable_scientific_claims(self) -> None:
        self.config["scientific_claim_allowed"] = True
        write_json(self.config_path, self.config)
        read_only, git, gpu = self.run_patches()
        with read_only, git, gpu:
            with self.assertRaisesRegex(
                v2xseq_tfd_canary.ContractError, "scientific_claim_allowed"
            ):
                v2xseq_tfd_canary.run_canary(self.run_args())

    def test_real_tfd_canary_calls_all_phases_and_exports_hashed_evidence(self) -> None:
        read_only, git, gpu = self.run_patches()
        diagnostic_gate_bypass = mock.patch.object(
            v2xseq_tfd_canary,
            "enforce_production_execution_gate",
            return_value={
                "resolved_config_sha256": "a" * 64,
                "template_config_sha256": "b" * 64,
            },
        )
        with read_only, git, gpu, diagnostic_gate_bypass:
            manifest = v2xseq_tfd_canary.run_canary(self.run_args())
        output = self.root / "output"
        self.assertEqual(manifest["claim_scope"], "diagnostic_only")
        self.assertFalse(manifest["scientific_claim_allowed"])
        self.assertEqual(manifest["training_scene_id"], "train-scene")
        self.assertEqual(manifest["evaluation_scene_ids"], ["validation-scene"])
        self.assertEqual(
            set(manifest["artifacts"]), set(v2xseq_tfd_canary.REQUIRED_ARTIFACTS)
        )
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(v2xseq_tfd_canary.sha256_file(output / name), digest)
        events = (output / "stdout.log").read_text(encoding="utf-8")
        for phase in (
            "training_forward_complete",
            "backward_complete",
            "optimizer_step_complete",
            "evaluation_forward_complete",
            "evaluation_complete",
            "checkpoint_complete",
        ):
            self.assertIn(phase, events)
        predictions = (output / "predictions.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("ground_truth", predictions)
        row = json.loads(predictions)
        self.assertEqual(len(row["prediction"]["modes"]), 2)
        self.assertEqual(
            row["prediction"]["modes"][0]["trajectory"][0][0],
            3.0,
        )
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(
            set(metrics["metrics"]),
            {"minADE", "minFDE", "MR", "NLL", "ECE", "Brier", "AURC"},
        )
        sums = (output / "SHA256SUMS").read_text(encoding="ascii")
        self.assertIn("run_manifest.json", sums)
        self.assertIn("predictions.jsonl", sums)


class RepositoryTFDReferenceContractTest(unittest.TestCase):
    def test_forecast_protocol_pins_current_reference_evaluator_bytes(self) -> None:
        repository = MODULE_ROOT.parents[1]
        protocol = json.loads(
            (
                repository
                / "experiments"
                / "clearml"
                / "protocols"
                / "v2xseq-forecast-v1.json"
            ).read_text(encoding="utf-8")
        )
        path, document = v2xseq_tfd_canary.validate_reference_evaluator_contract(
            repository, protocol
        )
        self.assertTrue(path.is_file())
        self.assertFalse(document["official_evaluator_equivalence"])
        self.assertFalse(document["scientific_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
