#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import export_upstream  # noqa: E402
import parity  # noqa: E402


def tensor(dtype: str, shape: list[int], values: list[object]) -> dict[str, object]:
    return parity.tensor_descriptor(dtype, shape, values)


def withheld(dtype: str, shape: list[int], token: str) -> dict[str, object]:
    return {
        "present": True,
        "dtype": dtype,
        "shape": shape,
        "canonical_values_sha256": parity.sha256_json({"token": token}),
    }


def trajectory_row(
    timestamp: int,
    actor_id: str,
    tag: str,
    y: float,
    *,
    cooperative: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "city": "fixture-city",
        "timestamp": str(timestamp),
        "id": actor_id,
        "type": "VEHICLE",
        "sub_type": "Car",
        "tag": tag,
        "x": str(float(timestamp)),
        "y": str(y),
        "z": "0.0",
        "length": "4.0",
        "width": "2.0",
        "height": "1.5",
        "theta": "0.0",
        "v_x": "1.0",
        "v_y": "0.0",
        "intersect_id": "001",
        "event_time_ns": timestamp,
        "arrival_time_ns": timestamp,
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
        "timestamp": str(timestamp),
        "x": "0.0",
        "y": "0.0",
        "direction": "north",
        "lane_id": "lane-1",
        "color_1": "green",
        "remain_1": "10",
        "color_2": "red",
        "remain_2": "0",
        "color_3": "yellow",
        "remain_3": "0",
        "intersect_id": "001",
        "event_time_ns": timestamp,
        "arrival_time_ns": timestamp,
    }


def float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


class Fixture:
    def __init__(self) -> None:
        self.scene_id = "1"
        positions: list[float] = []
        deltas: list[float] = []
        for y in (0.0, 1.0):
            for step in range(parity.HISTORY_STEPS):
                positions.extend([float(step - 49), y])
                deltas.extend([0.0 if step == 0 else 1.0, 0.0])
        edge_values = [0, 1, 1, 0]
        projection: dict[str, object] = {
            "x": tensor("torch.float32", [2, 50, 2], deltas),
            "positions": tensor("torch.float32", [2, 50, 2], positions),
            "edge_index": tensor("torch.int64", [2, 2], edge_values),
            "v2x_edge_index": tensor("torch.int64", [2, 2], [1, 0, 0, 1]),
            "v2x_pesudo_mask": tensor("torch.bool", [2], [True, True]),
            "v2x_mask": tensor("torch.bool", [2], [True, True]),
            "v2x_aa_mask": tensor("torch.bool", [2], [True, True]),
            "v2x_ins_mask": tensor("torch.bool", [2], [True, True]),
            "v2x_type_mask": tensor("torch.bool", [2], [True, True]),
            "interact_ego_mask": tensor("torch.bool", [2], [False, False]),
            "interact_road_mask": tensor("torch.bool", [2], [False, False]),
            "actor_type": tensor("torch.uint8", [2], [3, 3]),
            "num_nodes": parity.scalar_descriptor("int", 2),
            "num_car_actors": parity.scalar_descriptor("int", 1),
            "num_road_actors": parity.scalar_descriptor("int", 1),
            "ego_mask": tensor("torch.bool", [2], [True, False]),
            "road_mask": tensor("torch.bool", [2], [False, True]),
            "padding_mask": tensor("torch.bool", [2, 50], [False] * 100),
            "bos_mask": tensor(
                "torch.bool",
                [2, 50],
                [True] + [False] * 49 + [True] + [False] * 49,
            ),
            "rotate_angles": tensor("torch.float32", [2], [0.0, 0.0]),
            "lane_vectors": tensor("torch.float32", [1, 2], [1.0, 0.0]),
            "is_intersections": tensor("torch.uint8", [1], [0]),
            "turn_directions": tensor("torch.uint8", [1], [0]),
            "traffic_controls": tensor("torch.uint8", [1], [0]),
            "lane_actor_index": tensor("torch.int64", [2, 2], [0, 0, 0, 1]),
            "lane_actor_vectors": tensor(
                "torch.float32", [2, 2], [0.0, 0.0, 0.0, -1.0]
            ),
            "seq_id": parity.scalar_descriptor("int", 1),
            "av_index": parity.scalar_descriptor("int", 0),
            "agent_index": parity.scalar_descriptor("int", 1),
            "city": parity.scalar_descriptor("string", "fixture-city"),
            "origin": tensor("torch.float32", [1, 2], [49.0, 0.0]),
            "theta": tensor("torch.float32", [], [0.0]),
            "last_positions": tensor("torch.float64", [2, 2], [49.0, 0.0, 49.0, 1.0]),
        }
        export_without_hash = {
            "schema_version": 1,
            "contract_id": parity.CONTRACT_ID,
            "export_kind": parity.EXPORT_KIND,
            "claim_scope": "diagnostic_preprocessing_evidence_only",
            "scientific_claim_allowed": False,
            "upstream": {
                "repository": parity.UPSTREAM_REPOSITORY,
                "revision": parity.UPSTREAM_COMMIT,
                "source_files": dict(parity.UPSTREAM_SOURCE_SHA256),
            },
            "runtime": {
                "python": "3.8.0-fixture",
                "torch": "1.8.0-fixture",
                "torch_geometric": "1.7.2-fixture",
            },
            "scene": {
                "scene_id": self.scene_id,
                "split": "val",
                "processed_file": {
                    "basename": "1.pt",
                    "size": 1,
                    "sha256": "1" * 64,
                },
                "inputs": {
                    name: {
                        "basename": f"{self.scene_id}.csv",
                        "size": 1,
                        "sha256": character * 64,
                    }
                    for name, character in (
                        ("ego_raw", "2"),
                        ("road_raw", "3"),
                        ("association_raw", "4"),
                    )
                },
                "history_steps": 50,
                "future_steps": 50,
                "native_history_timestamps": [str(index) for index in range(50)],
                "actors": [
                    {"index": 0, "actor_id": "10", "side": "ego", "tags": ["AV"]},
                    {
                        "index": 1,
                        "actor_id": "20",
                        "side": "road",
                        "tags": ["TARGET_AGENT"],
                    },
                ],
                "association_rows": [
                    {
                        "timestamp": "0",
                        "id": "999",
                        "tag": "OTHERS",
                        "ego_side_id": "10",
                        "coop_side_id": "20",
                    }
                ],
            },
            "temporal_data": {
                "field_inventory": sorted(set(projection) | {"y"}),
                "history_projection": projection,
                "withheld_future_fields": {
                    "positions_future": withheld(
                        "torch.float32", [2, 50, 2], "positions"
                    ),
                    "y": withheld("torch.float32", [2, 50, 2], "y"),
                },
            },
            "exporter": {
                "basename": "export_upstream.py",
                "sha256": parity.sha256_file(PACKAGE_ROOT / "export_upstream.py"),
                "contract_sha256": parity.sha256_file(PACKAGE_ROOT / "parity.py"),
            },
        }
        self.export = {
            **export_without_hash,
            "content_sha256": parity.sha256_json(export_without_hash),
        }

        history = {
            "decision_time_ns": 49,
            "timestamps_ns": list(range(50)),
            "cooperative_trajectories": [
                trajectory_row(
                    step,
                    "20",
                    "TARGET_AGENT",
                    1.0,
                    cooperative=True,
                )
                for step in range(50)
            ],
            "vehicle_trajectories": [
                trajectory_row(step, "10", "AV", 0.0) for step in range(50)
            ],
            "infrastructure_trajectories": [
                trajectory_row(step, "20", "TARGET_AGENT", 1.0) for step in range(50)
            ],
            "traffic_lights": [traffic_light_row(step) for step in range(50)],
            "hd_map": {"LANE": {"lane-1": {"centerline": [[0, 0], [1, 0]]}}},
            "coordinate_frame": "world map frame",
        }
        self.sample = {
            "scene_id": self.scene_id,
            "target_id": "20",
            "history": history,
            "input_sha256": parity.sha256_json(history),
        }
        self.bindings = {
            "schema_version": 1,
            "contract_id": parity.CONTRACT_ID,
            "status": "candidate_unverified",
            "scene_id": self.scene_id,
            "upstream_export_content_sha256": self.export["content_sha256"],
            "history_input_sha256": self.sample["input_sha256"],
            "canonical_coordinate_frame": "world map frame",
            "coordinate_rule": parity.COORDINATE_RULE,
            "absolute_tolerance": 1e-5,
            "timestamp_bindings": [
                {
                    "upstream_index": step,
                    "native_timestamp": str(step),
                    "event_time_ns": step,
                }
                for step in range(50)
            ],
            "actor_bindings": [
                {
                    "upstream_index": 0,
                    "upstream_actor_id": "10",
                    "canonical_stream": "vehicle_trajectories",
                    "canonical_actor_id": "10",
                },
                {
                    "upstream_index": 1,
                    "upstream_actor_id": "20",
                    "canonical_stream": "infrastructure_trajectories",
                    "canonical_actor_id": "20",
                },
            ],
        }

    @staticmethod
    def rehash_descriptor(descriptor: dict[str, object]) -> None:
        core = {
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "values": descriptor["values"],
        }
        descriptor["sha256"] = parity.sha256_json(core)

    def rehash_export(self) -> None:
        without = {
            key: value for key, value in self.export.items() if key != "content_sha256"
        }
        self.export["content_sha256"] = parity.sha256_json(without)
        self.bindings["upstream_export_content_sha256"] = self.export["content_sha256"]

    def apply_large_coordinate_projection(self) -> None:
        base_x = 100000.123456
        base_y = 200000.654321
        heading = 0.345678901234
        for stream in (
            "cooperative_trajectories",
            "vehicle_trajectories",
            "infrastructure_trajectories",
        ):
            for row in self.sample["history"][stream]:
                step = int(row["event_time_ns"])
                row["x"] = repr(base_x + step)
                row["y"] = repr(base_y + (1.0 if str(row["id"]) == "20" else 0.0))
                row["theta"] = repr(heading)

        raw_origin_x = base_x + 49.0
        raw_origin_y = base_y
        cosine = math.cos(heading)
        sine = math.sin(heading)
        positions: list[float] = []
        deltas: list[float] = []
        for side_y in (0.0, 1.0):
            previous: tuple[float, float] | None = None
            for step in range(50):
                delta_x = base_x + step - raw_origin_x
                raw_position = (
                    delta_x * cosine + side_y * sine,
                    -delta_x * sine + side_y * cosine,
                )
                positions.extend([float32(raw_position[0]), float32(raw_position[1])])
                if previous is None:
                    raw_delta = (0.0, 0.0)
                else:
                    raw_delta = (
                        raw_position[0] - previous[0],
                        raw_position[1] - previous[1],
                    )
                deltas.extend([float32(raw_delta[0]), float32(raw_delta[1])])
                previous = raw_position

        projection = self.export["temporal_data"]["history_projection"]
        projection["positions"] = tensor("torch.float32", [2, 50, 2], positions)
        projection["x"] = tensor("torch.float32", [2, 50, 2], deltas)
        projection["origin"] = tensor(
            "torch.float32",
            [1, 2],
            [float32(raw_origin_x), float32(raw_origin_y)],
        )
        projection["theta"] = tensor("torch.float32", [], [float32(heading)])
        projection["rotate_angles"] = tensor(
            "torch.float32", [2], [float32(heading), float32(heading)]
        )
        projection["last_positions"] = tensor(
            "torch.float64",
            [2, 2],
            [
                float32(raw_origin_x),
                float32(raw_origin_y),
                float32(raw_origin_x),
                float32(raw_origin_y + 1.0),
            ],
        )
        self.sample["input_sha256"] = parity.sha256_json(self.sample["history"])
        self.bindings["history_input_sha256"] = self.sample["input_sha256"]
        self.rehash_export()


class ParityHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def test_candidate_checks_pass_but_never_verify_parity(self) -> None:
        report = parity.compare_candidate(
            self.fixture.export,
            self.fixture.sample,
            self.fixture.bindings,
        )
        self.assertEqual(report["comparison_status"], "candidate_checks_passed")
        self.assertFalse(report["parity_verified"])
        self.assertFalse(report["scientific_claim_allowed"])
        self.assertIn(
            "lane and map feature replay with the pinned DAIR map API",
            report["unresolved_categories"],
        )

    def test_verified_binding_status_is_rejected(self) -> None:
        self.fixture.bindings["status"] = "verified"
        with self.assertRaisesRegex(parity.ParityContractError, "candidate_unverified"):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_upstream_revision_drift_is_rejected(self) -> None:
        self.fixture.export["upstream"]["revision"] = "a" * 40
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "revision drift"):
            parity.validate_export_document(self.fixture.export)

    def test_upstream_source_byte_drift_is_rejected(self) -> None:
        self.fixture.export["upstream"]["source_files"]["utils.py"] = "a" * 64
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "source hash drift"):
            parity.validate_export_document(self.fixture.export)

    def test_descriptor_tampering_is_rejected(self) -> None:
        self.fixture.export["temporal_data"]["history_projection"]["x"]["values"][0] = (
            7.0
        )
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "descriptor SHA-256"):
            parity.validate_export_document(self.fixture.export)

    def test_coordinate_mismatch_is_rejected(self) -> None:
        positions = self.fixture.export["temporal_data"]["history_projection"][
            "positions"
        ]
        positions["values"][0] = 99.0
        self.fixture.rehash_descriptor(positions)
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, r"positions\[0,0,0\]"):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_joint_origin_and_position_shift_is_rejected_against_bound_av(self) -> None:
        projection = self.fixture.export["temporal_data"]["history_projection"]
        origin = projection["origin"]
        origin["values"][0] += 100.0
        self.fixture.rehash_descriptor(origin)
        positions = projection["positions"]
        for index in range(0, len(positions["values"]), 2):
            positions["values"][index] -= 100.0
        self.fixture.rehash_descriptor(positions)
        self.fixture.rehash_export()
        with self.assertRaisesRegex(
            parity.ParityContractError, "after float32 rounding"
        ):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_actor_order_mismatch_is_rejected(self) -> None:
        self.fixture.bindings["actor_bindings"][0]["upstream_actor_id"] = "20"
        with self.assertRaisesRegex(
            parity.ParityContractError, "actor binding differs"
        ):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_missing_timestamp_binding_is_rejected(self) -> None:
        self.fixture.bindings["timestamp_bindings"].pop()
        with self.assertRaisesRegex(parity.ParityContractError, "exactly 50"):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_association_edge_mismatch_is_rejected(self) -> None:
        edges = self.fixture.export["temporal_data"]["history_projection"][
            "v2x_edge_index"
        ]
        edges["values"] = [0, 0, 1, 1]
        self.fixture.rehash_descriptor(edges)
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "association witness"):
            parity.validate_export_document(self.fixture.export)

    def test_future_ground_truth_in_history_is_rejected(self) -> None:
        self.fixture.sample["history"]["ground_truth"] = [[1.0, 2.0]]
        self.fixture.sample["input_sha256"] = parity.sha256_json(
            self.fixture.sample["history"]
        )
        self.fixture.bindings["history_input_sha256"] = self.fixture.sample[
            "input_sha256"
        ]
        with self.assertRaisesRegex(
            parity.ParityContractError, "fields differ|future-bearing"
        ):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_disguised_future_key_in_history_row_is_rejected(self) -> None:
        row = self.fixture.sample["history"]["vehicle_trajectories"][0]
        row["future_positions"] = [[999.0, 999.0]]
        self.fixture.sample["input_sha256"] = parity.sha256_json(
            self.fixture.sample["history"]
        )
        self.fixture.bindings["history_input_sha256"] = self.fixture.sample[
            "input_sha256"
        ]
        with self.assertRaisesRegex(parity.ParityContractError, "future-bearing"):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_every_history_stream_rejects_unknown_row_fields(self) -> None:
        for stream in sorted(parity.HISTORY_ROW_FIELDS):
            with self.subTest(stream=stream):
                fixture = Fixture()
                fixture.sample["history"][stream][0]["unexpected_metadata"] = "x"
                fixture.sample["input_sha256"] = parity.sha256_json(
                    fixture.sample["history"]
                )
                fixture.bindings["history_input_sha256"] = fixture.sample[
                    "input_sha256"
                ]
                with self.assertRaisesRegex(
                    parity.ParityContractError, "fields differ"
                ):
                    parity.compare_candidate(
                        fixture.export, fixture.sample, fixture.bindings
                    )

    def test_history_row_missing_frozen_field_is_rejected(self) -> None:
        del self.fixture.sample["history"]["vehicle_trajectories"][0]["v_x"]
        self.fixture.sample["input_sha256"] = parity.sha256_json(
            self.fixture.sample["history"]
        )
        self.fixture.bindings["history_input_sha256"] = self.fixture.sample[
            "input_sha256"
        ]
        with self.assertRaisesRegex(parity.ParityContractError, "fields differ"):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_large_coordinates_and_heading_follow_upstream_rounding(self) -> None:
        self.fixture.apply_large_coordinate_projection()
        raw_origin = float(
            self.fixture.sample["history"]["vehicle_trajectories"][-1]["x"]
        )
        exported_origin = self.fixture.export["temporal_data"]["history_projection"][
            "origin"
        ]["values"][0]
        self.assertGreater(abs(raw_origin - exported_origin), 1e-4)
        report = parity.compare_candidate(
            self.fixture.export, self.fixture.sample, self.fixture.bindings
        )
        self.assertEqual(report["comparison_status"], "candidate_checks_passed")
        self.assertFalse(report["parity_verified"])

    def test_unrounded_large_coordinate_origin_is_rejected(self) -> None:
        self.fixture.apply_large_coordinate_projection()
        projection = self.fixture.export["temporal_data"]["history_projection"]
        raw_origin = float(
            self.fixture.sample["history"]["vehicle_trajectories"][-1]["x"]
        )
        projection["origin"]["values"][0] = raw_origin
        self.fixture.rehash_descriptor(projection["origin"])
        self.fixture.rehash_export()
        with self.assertRaisesRegex(
            parity.ParityContractError, "after float32 rounding"
        ):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_validation_export_requires_withheld_y_shape_and_presence(self) -> None:
        withheld_y = self.fixture.export["temporal_data"]["withheld_future_fields"]["y"]
        withheld_y.update(
            {
                "present": False,
                "dtype": None,
                "shape": None,
                "canonical_values_sha256": None,
            }
        )
        self.fixture.export["temporal_data"]["field_inventory"].remove("y")
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "upstream split"):
            parity.validate_export_document(self.fixture.export)

    def test_pinned_broken_test_preprocessing_path_is_rejected(self) -> None:
        self.fixture.export["scene"]["split"] = "test"
        self.fixture.rehash_export()
        with self.assertRaisesRegex(parity.ParityContractError, "dereferences null y"):
            parity.validate_export_document(self.fixture.export)

    def test_history_hash_mismatch_is_rejected(self) -> None:
        self.fixture.sample["history"]["decision_time_ns"] = 48
        with self.assertRaisesRegex(
            parity.ParityContractError, "decision time|SHA-256"
        ):
            parity.compare_candidate(
                self.fixture.export, self.fixture.sample, self.fixture.bindings
            )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(parity.ParityContractError, "duplicate JSON key"):
            parity.loads_strict('{"scene_id":"1","scene_id":"2"}')

    def test_nonfinite_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(parity.ParityContractError, "non-finite"):
            parity.loads_strict('{"value":NaN}')

    def test_exporter_import_does_not_import_torch(self) -> None:
        self.assertNotIn("torch", export_upstream.__dict__)
        self.assertNotIn("torch_geometric", export_upstream.__dict__)

    def test_scene_witness_reconstructs_pinned_first_seen_order(self) -> None:
        ego_rows = []
        road_rows = []
        for step in range(100):
            ego_rows.extend(
                [
                    {
                        "timestamp": str(step),
                        "id": "11",
                        "type": "VEHICLE",
                        "tag": "OTHERS",
                        "x": "0",
                        "y": "0",
                        "theta": "0",
                    },
                    {
                        "timestamp": str(step),
                        "id": "10",
                        "type": "VEHICLE",
                        "tag": "AV",
                        "x": "0",
                        "y": "0",
                        "theta": "0",
                    },
                ]
            )
            road_rows.append(
                {
                    "timestamp": str(step),
                    "id": "20",
                    "type": "VEHICLE",
                    "tag": "TARGET_AGENT",
                    "x": "0",
                    "y": "0",
                    "theta": "0",
                }
            )
        association_rows = [
            {
                "timestamp": f"{step}.0",
                "id": "999.0",
                "tag": "OTHERS",
                "ego_side_id": "10.0",
                "coop_side_id": "20.0",
            }
            for step in range(50)
        ]
        times, actors, associations = export_upstream.build_scene_witness(
            ego_rows, road_rows, association_rows
        )
        self.assertEqual(times[:2], ["0", "1"])
        self.assertEqual([actor["actor_id"] for actor in actors], ["11", "10", "20"])
        self.assertEqual(associations[0]["ego_side_id"], "10")
        self.assertEqual(associations[0]["coop_side_id"], "20")

    def test_scene_witness_rejects_side_identity_overlap(self) -> None:
        ego_rows = [
            {
                "timestamp": str(step),
                "id": "10",
                "type": "VEHICLE",
                "tag": "AV" if step == 0 else "OTHERS",
                "x": "0",
                "y": "0",
                "theta": "0",
            }
            for step in range(100)
        ]
        road_rows = [
            {
                "timestamp": str(step),
                "id": "10",
                "type": "VEHICLE",
                "tag": "TARGET_AGENT" if step == 0 else "OTHERS",
                "x": "0",
                "y": "0",
                "theta": "0",
            }
            for step in range(100)
        ]
        with self.assertRaisesRegex(parity.ParityContractError, "overlap"):
            export_upstream.build_scene_witness(
                ego_rows,
                road_rows,
                [
                    {
                        "timestamp": "0",
                        "id": "1",
                        "tag": "OTHERS",
                        "ego_side_id": "10",
                        "coop_side_id": "10",
                    }
                ],
            )

    def test_cli_files_reject_symlink_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text(json.dumps({"ok": True}), encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(parity.ParityContractError, "non-symlink"):
                parity.load_json_file(link, "fixture evidence")


if __name__ == "__main__":
    unittest.main()
