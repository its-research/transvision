#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from v2xseq_label_audit import (  # noqa: E402
    AuditError,
    EXPECTED_V1_TABLES,
    audit_dataset,
    canonical_sha256,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_row(
    *, frame: str, infrastructure_frame: str, token: str, vehicle_timestamp: str
) -> dict[str, object]:
    return {
        "2d_box": {"xmax": 1.0, "xmin": 0.0, "ymax": 1.0, "ymin": 0.0},
        "3d_dimensions": {"h": 1.0, "l": 1.0, "w": 1.0},
        "3d_location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "alpha": 0.0,
        "from_side": "veh",
        "inf_frame_id": infrastructure_frame,
        "inf_pointcloud_timestamp": str(int(vehicle_timestamp) + 10),
        "inf_token": "-1",
        "inf_track_id": "-1",
        "occluded_state": 0,
        "rotation": 0.0,
        "token": token,
        "track_id": "0007",
        "truncated_state": 0,
        "type": "Car",
        "veh_frame_id": frame,
        "veh_pointcloud_timestamp": vehicle_timestamp,
        "veh_token": token,
        "veh_track_id": "0007",
    }


def build_fixture(root: Path) -> None:
    frames = ("000001", "000002")
    data_info = [
        {
            "infrastructure_frame": "009001",
            "infrastructure_sequence": "s0",
            "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0},
            "vehicle_frame": frames[0],
            "vehicle_sequence": "s0",
        },
        {
            "infrastructure_frame": "009002",
            "infrastructure_sequence": "s0",
            "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0},
            "vehicle_frame": frames[1],
            "vehicle_sequence": "s0",
        },
    ]
    write_json(root / "data_info.json", data_info)
    write_json(
        root / "label" / f"{frames[0]}.json",
        [
            label_row(
                frame=frames[0],
                infrastructure_frame="009001",
                token="ann-a",
                vehicle_timestamp="990",
            )
        ],
    )
    write_json(
        root / "label" / f"{frames[1]}.json",
        [
            label_row(
                frame=frames[1],
                infrastructure_frame="009002",
                token="ann-b",
                vehicle_timestamp="1490",
            )
        ],
    )

    tables: dict[str, object] = {table: [] for table in EXPECTED_V1_TABLES}
    tables["scene"] = [
        {
            "description": "",
            "first_sample_token": frames[0],
            "last_sample_token": frames[1],
            "log_token": "log0",
            "name": "s0",
            "nbr_samples": 2,
            "token": "s0",
        }
    ]
    tables["sample"] = [
        {
            "next": frames[1],
            "prev": "",
            "scene_token": "s0",
            "timestamp": 1000.0,
            "token": frames[0],
        },
        {
            "next": "",
            "prev": frames[0],
            "scene_token": "s0",
            "timestamp": 1500.0,
            "token": frames[1],
        },
    ]
    tables["sample_data"] = [
        {
            "filename": "vehicle-side/velodyne/000001.bin",
            "sample_token": frames[0],
            "timestamp": 990.0,
            "token": frames[0],
        },
        {
            "filename": "vehicle-side/velodyne/000002.bin",
            "sample_token": frames[1],
            "timestamp": 1490.0,
            "token": frames[1],
        },
    ]
    tables["instance"] = [
        {
            "category_token": "car",
            "first_annotation_token": "ann-a",
            "last_annotation_token": "ann-b",
            "nbr_annotations": 2,
            "token": "instance-a",
        }
    ]
    tables["sample_annotation"] = [
        {
            "instance_token": "instance-a",
            "next": "ann-b",
            "prev": "",
            "sample_token": frames[0],
            "token": "ann-a",
        },
        {
            "instance_token": "instance-a",
            "next": "",
            "prev": "ann-a",
            "sample_token": frames[1],
            "token": "ann-b",
        },
    ]
    for table, rows in tables.items():
        write_json(root / "v1.0-trainval" / f"{table}.json", rows)

    data_files = sorted(
        path for path in root.rglob("*.json") if path.name != "mirror-manifest.json"
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "status": "downloaded",
        }
        for path in data_files
    ]
    manifest = {
        "entries": entries,
        "file_count": len(entries),
        "maps_included": False,
        "official_source": False,
        "prefix": "SPD/train_val_encode/V2X-Seq-SPD-New/cooperative/",
        "purpose": "protocol and association-pipeline audit only",
        "revision": "fixture-revision",
        "scientific_claim_allowed": False,
        "source": "https://example.invalid/non-official-mirror",
        "total_bytes": sum(entry["size"] for entry in entries),
    }
    write_json(root / "mirror-manifest.json", manifest)


class V2XSeqLabelAuditTest(unittest.TestCase):
    def test_report_is_deterministic_and_preserves_claim_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fixture(root)

            first = audit_dataset(root)
            second = audit_dataset(root)

            self.assertEqual(first, second)
            digest = first["report_sha256"]
            body = copy.deepcopy(first)
            del body["report_sha256"]
            self.assertEqual(digest, canonical_sha256(body))
            self.assertFalse(first["boundary"]["official_source"])
            self.assertFalse(first["boundary"]["scientific_claim_allowed"])
            self.assertFalse(first["boundary"]["scientific_validation_performed"])
            self.assertNotIn(str(root), json.dumps(first, sort_keys=True))

            self.assertEqual(first["file_counts"]["label_files"], 2)
            self.assertEqual(first["file_counts"]["raw_sensor_payload_files"], 0)
            self.assertEqual(first["file_counts"]["referenced_sensor_payload_paths"], 2)
            self.assertEqual(first["sequence_findings"]["scene_count"], 1)
            self.assertEqual(first["track_id_findings"]["sequence_scoped_track_key_count"], 1)
            self.assertEqual(
                first["track_id_findings"]["sequence_track_to_multiple_instance_count"], 0
            )
            self.assertIsNone(first["timestamp_findings"]["timestamp_unit_from_subset"])
            self.assertEqual(
                first["timestamp_findings"]["within_scene_consecutive_sample_delta_raw"][
                    "median"
                ],
                500,
            )
            unresolved = {row["field"] for row in first["unresolved_protocol_fields"]}
            self.assertIn("official_train_validation_test_split", unresolved)
            self.assertIn("message_arrival_and_network_fault_ground_truth", unresolved)

    def test_file_tampering_fails_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fixture(root)
            label_path = root / "label" / "000001.json"
            payload = label_path.read_text(encoding="utf-8")
            self.assertIn('"Car"', payload)
            label_path.write_text(payload.replace('"Car"', '"Van"'), encoding="utf-8")

            with self.assertRaisesRegex(AuditError, "manifest hash mismatch"):
                audit_dataset(root)

    def test_manifest_cannot_enable_scientific_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_fixture(root)
            manifest_path = root / "mirror-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scientific_claim_allowed"] = True
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(AuditError, "scientific_claim_allowed=false"):
                audit_dataset(root)


if __name__ == "__main__":
    unittest.main()
