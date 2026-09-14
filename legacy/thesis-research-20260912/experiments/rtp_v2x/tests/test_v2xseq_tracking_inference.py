from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.adapters.transvision_spd import schema as detection_schema  # noqa: E402
from experiments.rtp_v2x import v2xseq_tracking_inference as tracking  # noqa: E402


def digest(character: str) -> str:
    return character * 64


def source() -> dict[str, Any]:
    return {
        "repository_url": detection_schema.REPOSITORY_URL,
        "implementation_revision": "1" * 40,
        "implementation_tree_sha256": digest("2"),
        "model_id": "coformernet_controlled_adaptation",
        "config_path": detection_schema.CONFIG_PATH_BY_MODEL[
            "coformernet_controlled_adaptation"
        ],
        "config_sha256": digest("3"),
        "checkpoint_role": "canonical_final",
        "checkpoint_sha256": digest("4"),
        "teacher_checkpoint_sha256": None,
        "inference_adapter_sha256": digest("5"),
        "environment_manifest_sha256": digest("6"),
    }


def detection(
    index: int,
    *,
    x: float,
    y: float = 0.0,
    score: float = 0.9,
) -> dict[str, Any]:
    return {
        "detection_id": f"det-{index:06d}",
        "source_label_id": 0,
        "source_class_name": "Car",
        "class_name": "vehicle",
        "box_3d": [x, y, -1.5, 4.2, 1.9, 1.6, 0.0],
        "score": score,
    }


def record(
    *,
    frame_index: int,
    detections: list[dict[str, Any]],
    sequence_id: str = "0001",
) -> dict[str, Any]:
    decision_time = 1_000 + frame_index * 100
    return {
        "schema_version": detection_schema.SCHEMA_VERSION,
        "contract_id": detection_schema.CONTRACT_ID,
        "record_kind": detection_schema.RECORD_KIND,
        "scientific_claim_allowed": False,
        "dataset": {
            "name": detection_schema.DATASET_NAME,
            "release_id": "fixture-release-v1",
            "split": "validation",
            "dataset_manifest_sha256": digest("7"),
            "split_manifest_sha256": digest("8"),
        },
        "frame": {
            "sequence_id": sequence_id,
            "vehicle_frame_id": f"{frame_index:06d}",
            "infrastructure_frame_id": f"1{frame_index:05d}",
            "vehicle_capture_time_ns": decision_time,
            "infrastructure_capture_time_ns": decision_time - 10,
            "decision_time_ns": decision_time,
        },
        "coordinate_system": detection_schema.expected_coordinate_system(),
        "source": source(),
        "policy": detection_schema.expected_policy(),
        "detections": detections,
    }


def input_bytes(records: list[dict[str, Any]]) -> bytes:
    return detection_schema.canonical_jsonl_bytes(records)


def run(records: list[dict[str, Any]]) -> tracking.TrackingArtifact:
    payload = input_bytes(records)
    return tracking.run_tracking_inference(
        payload,
        expected_input_document_sha256=hashlib.sha256(payload).hexdigest(),
    )


class TrackingInferenceContractTest(unittest.TestCase):
    def test_links_nearby_consecutive_detections_without_prediction(self) -> None:
        artifact = run(
            [
                record(frame_index=1, detections=[detection(0, x=10.0)]),
                record(frame_index=2, detections=[detection(0, x=10.5)]),
            ]
        )
        first, second = artifact.records
        self.assertEqual(first["tracks"][0]["lifecycle"], "new")
        self.assertIsNone(first["tracks"][0]["predecessor"])
        self.assertEqual(second["tracks"][0]["lifecycle"], "matched")
        self.assertEqual(
            first["tracks"][0]["track_id"], second["tracks"][0]["track_id"]
        )
        self.assertEqual(
            second["tracks"][0]["predecessor"],
            {
                "vehicle_frame_id": "000001",
                "detection_id": "det-000000",
                "decision_time_ns": 1_100,
            },
        )
        self.assertEqual(second["tracks"][0]["box_3d"][0], 10.5)

    def test_gate_and_empty_frame_retire_previous_state(self) -> None:
        artifact = run(
            [
                record(frame_index=1, detections=[detection(0, x=1.0)]),
                record(frame_index=2, detections=[detection(0, x=20.0)]),
                record(frame_index=3, detections=[]),
                record(frame_index=4, detections=[detection(0, x=20.2)]),
            ]
        )
        ids = [
            row["tracks"][0]["track_id"] for row in artifact.records if row["tracks"]
        ]
        self.assertEqual(
            ids,
            ["diag-track-00000000", "diag-track-00000001", "diag-track-00000002"],
        )
        self.assertEqual(artifact.records[2]["tracks"], [])
        self.assertEqual(artifact.records[3]["tracks"][0]["lifecycle"], "new")

    def test_stable_orders_make_equal_cost_replay_deterministic(self) -> None:
        records = [
            record(
                frame_index=1,
                detections=[
                    detection(0, x=10.0, y=-1.0, score=0.9),
                    detection(1, x=10.0, y=1.0, score=0.8),
                ],
            ),
            record(
                frame_index=2,
                detections=[
                    detection(0, x=11.0, y=0.0, score=0.9),
                    detection(1, x=9.0, y=0.0, score=0.8),
                ],
            ),
        ]
        first = run(records)
        second = run(copy.deepcopy(records))
        self.assertEqual(first.jsonl_bytes, second.jsonl_bytes)
        self.assertEqual(first.output_document_sha256, second.output_document_sha256)

    def test_output_is_non_claiming_and_bound_to_every_input_and_code_digest(
        self,
    ) -> None:
        payload = input_bytes(
            [record(frame_index=1, detections=[detection(0, x=10.0)])]
        )
        artifact = tracking.run_tracking_inference(
            payload,
            expected_input_document_sha256=hashlib.sha256(payload).hexdigest(),
        )
        output = artifact.records[0]
        self.assertTrue(output["diagnostic_only"])
        self.assertFalse(output["scientific_claim_allowed"])
        self.assertEqual(output["contract_id"], tracking.CONTRACT_ID)
        boundary = output["capability_boundary"]
        self.assertEqual(boundary["purpose"], "diagnostic_baseline_plumbing_only")
        self.assertFalse(boundary["per_agent_source_available"])
        self.assertFalse(boundary["arrival_time_available"])
        self.assertFalse(boundary["spatial_reliability_available"])
        self.assertFalse(boundary["runs_rtp_v2x"])
        self.assertFalse(boundary["runs_h1_reliability_association"])
        self.assertEqual(boundary["fixed_v1_class_mapping"], "Car_to_vehicle_only")
        self.assertTrue(boundary["class_mapping_unverified_for_scientific_protocol"])
        self.assertFalse(boundary["production_fixed_detection_input_allowed"])
        self.assertFalse(boundary["runtime_source_hash_is_execution_proof"])
        self.assertEqual(
            boundary["production_contract_upgrade_requirement"],
            "data_readback_then_separately_versioned_v2",
        )
        self.assertFalse(boundary["scientific_evidence_eligible"])
        self.assertEqual(
            boundary["detection_score_role"],
            "passthrough_only_not_spatial_reliability_or_association_term",
        )
        self.assertEqual(output["association_policy"]["capability_boundary"], boundary)
        self.assertEqual(
            output["input_provenance"]["fixed_detection_document_sha256"],
            hashlib.sha256(payload).hexdigest(),
        )
        self.assertRegex(
            output["input_provenance"]["fixed_detection_contract_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            output["input_provenance"]["detection_schema_implementation_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            output["input_provenance"]["fixed_detection_record_sha256"],
            detection_schema.record_sha256(detection_schema.loads_jsonl(payload)[0]),
        )
        provenance = output["tracker_provenance"]
        self.assertEqual(
            provenance["association_contract_sha256"],
            tracking.ASSOCIATION_CONTRACT_SHA256,
        )
        for key in (
            "consumer_implementation_sha256",
            "association_implementation_sha256",
            "association_contract_sha256",
        ):
            self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")
        self.assertEqual(
            artifact.output_document_sha256,
            hashlib.sha256(artifact.jsonl_bytes).hexdigest(),
        )

    def test_detection_score_is_passthrough_not_an_association_term(self) -> None:
        def scenario(score_a: float, score_b: float) -> list[dict[str, Any]]:
            return [
                record(
                    frame_index=1,
                    detections=[
                        detection(0, x=10.0, y=-2.0, score=0.9),
                        detection(1, x=10.0, y=2.0, score=0.8),
                    ],
                ),
                record(
                    frame_index=2,
                    detections=[
                        detection(0, x=10.2, y=-2.0, score=score_a),
                        detection(1, x=10.2, y=2.0, score=score_b),
                    ],
                ),
            ]

        high_scores = run(scenario(0.99, 0.98)).records[1]["tracks"]
        low_scores = run(scenario(0.06, 0.05)).records[1]["tracks"]
        high_assignment = [
            (row["detection_id"], row["track_id"], row["lifecycle"])
            for row in high_scores
        ]
        low_assignment = [
            (row["detection_id"], row["track_id"], row["lifecycle"])
            for row in low_scores
        ]
        self.assertEqual(high_assignment, low_assignment)
        self.assertNotEqual(
            [row["score"] for row in high_scores],
            [row["score"] for row in low_scores],
        )

    def test_output_jsonl_has_frozen_float_spelling_and_final_lf(self) -> None:
        artifact = run([record(frame_index=1, detections=[detection(0, x=10.0)])])
        self.assertTrue(artifact.jsonl_bytes.endswith(b"\n"))
        self.assertIn(b"1.0000000000000000e+01", artifact.jsonl_bytes)
        parsed = json.loads(artifact.jsonl_bytes.decode("utf-8"))
        self.assertEqual(parsed["tracks"][0]["box_3d"][0], 10.0)

    def test_wrong_or_malformed_expected_digest_fails_before_consumption(self) -> None:
        payload = input_bytes([record(frame_index=1, detections=[])])
        for invalid, pattern in (
            ("a" * 63, "64 lowercase"),
            ("0" * 64, "does not match"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(tracking.TrackingInferenceError, pattern):
                    tracking.run_tracking_inference(
                        payload, expected_input_document_sha256=invalid
                    )

    def test_noncanonical_or_gt_bearing_input_is_rejected_by_fixed_schema(self) -> None:
        valid = record(frame_index=1, detections=[detection(0, x=10.0)])
        noncanonical = (json.dumps(valid) + "\n").encode("utf-8")
        invalid_gt = copy.deepcopy(valid)
        invalid_gt["detections"][0]["ground_truth"] = True
        gt_payload = (
            json.dumps(invalid_gt, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for payload in (noncanonical, gt_payload):
            with self.subTest(payload=payload[:80]):
                with self.assertRaisesRegex(
                    tracking.TrackingInferenceError, "frozen canonical contract"
                ):
                    tracking.run_tracking_inference(
                        payload,
                        expected_input_document_sha256=hashlib.sha256(
                            payload
                        ).hexdigest(),
                    )

    def test_duplicate_or_reversed_decision_time_fails_closed(self) -> None:
        first = record(frame_index=1, detections=[])
        second = record(frame_index=2, detections=[])
        second["frame"]["decision_time_ns"] = first["frame"]["decision_time_ns"]
        second["frame"]["vehicle_capture_time_ns"] = first["frame"][
            "vehicle_capture_time_ns"
        ]
        second["frame"]["infrastructure_capture_time_ns"] = first["frame"][
            "infrastructure_capture_time_ns"
        ]
        with self.assertRaisesRegex(
            detection_schema.FixedDetectionContractError, "strictly increasing"
        ):
            input_bytes([first, second])

    def test_each_sequence_has_an_independent_track_identity_space(self) -> None:
        artifact = run(
            [
                record(
                    frame_index=1,
                    sequence_id="0001",
                    detections=[detection(0, x=1.0)],
                ),
                record(
                    frame_index=1,
                    sequence_id="0002",
                    detections=[detection(0, x=2.0)],
                ),
            ]
        )
        self.assertEqual(
            [row["tracks"][0]["track_id"] for row in artifact.records],
            ["diag-track-00000000", "diag-track-00000000"],
        )

    def test_module_has_no_training_clearml_or_model_runtime_imports(self) -> None:
        module_path = Path(tracking.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint(
                {"clearml", "torch", "tensorflow", "mmdet", "mmdet3d", "transvision"}
            )
        )


class TrackingInferenceCliTest(unittest.TestCase):
    def test_cli_writes_new_file_and_refuses_overwrite(self) -> None:
        payload = input_bytes([record(frame_index=1, detections=[])])
        expected = hashlib.sha256(payload).hexdigest()
        script = Path(tracking.__file__)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "detections.jsonl"
            output_path = root / "tracks.jsonl"
            source_path.write_bytes(payload)
            command = [
                sys.executable,
                str(script),
                "--input",
                str(source_path),
                "--output",
                str(output_path),
                "--expected-input-sha256",
                expected,
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = json.loads(first.stdout)
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(summary["scientific_claim_allowed"])
            original = output_path.read_bytes()
            second = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(output_path.read_bytes(), original)

    def test_cli_rejects_symbolic_link_input(self) -> None:
        payload = input_bytes([record(frame_index=1, detections=[])])
        expected = hashlib.sha256(payload).hexdigest()
        script = Path(tracking.__file__)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "detections.jsonl"
            linked_path = root / "linked-detections.jsonl"
            output_path = root / "tracks.jsonl"
            source_path.write_bytes(payload)
            linked_path.symlink_to(source_path)
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(linked_path),
                    "--output",
                    str(output_path),
                    "--expected-input-sha256",
                    expected,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be a symbolic link", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
