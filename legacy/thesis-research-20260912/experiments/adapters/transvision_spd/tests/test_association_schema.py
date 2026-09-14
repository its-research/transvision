from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.adapters.transvision_spd import association_schema  # noqa: E402
from experiments.adapters.transvision_spd import schema as fixed_schema  # noqa: E402


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "association-observations-v1.json"
)
FIXED_PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "fixed-detections-v1.json"
)


def digest(character: str) -> str:
    return character * 64


def fixed_source() -> dict[str, Any]:
    return {
        "repository_url": fixed_schema.REPOSITORY_URL,
        "implementation_revision": "1" * 40,
        "implementation_tree_sha256": digest("2"),
        "model_id": "coformernet_controlled_adaptation",
        "config_path": fixed_schema.CONFIG_PATH_BY_MODEL[
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
    *, detection_id: str = "det-000000", score: float = 0.9
) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "source_label_id": 0,
        "source_class_name": "Car",
        "class_name": "vehicle",
        "box_3d": [10.0, -2.0, -1.5, 4.35, 1.91, 1.59, 0.25],
        "score": score,
    }


def fixed_record(
    *,
    detections: list[dict[str, Any]] | None = None,
    vehicle_frame_id: str = "000001",
    infrastructure_frame_id: str = "100001",
    vehicle_capture_time_ns: int = 1_000,
    infrastructure_capture_time_ns: int = 900,
    decision_time_ns: int = 1_000,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": fixed_schema.CONTRACT_ID,
        "record_kind": fixed_schema.RECORD_KIND,
        "scientific_claim_allowed": False,
        "dataset": {
            "name": fixed_schema.DATASET_NAME,
            "release_id": "fixture-release-v1",
            "split": "validation",
            "dataset_manifest_sha256": digest("7"),
            "split_manifest_sha256": digest("8"),
        },
        "frame": {
            "sequence_id": "0001",
            "vehicle_frame_id": vehicle_frame_id,
            "infrastructure_frame_id": infrastructure_frame_id,
            "vehicle_capture_time_ns": vehicle_capture_time_ns,
            "infrastructure_capture_time_ns": infrastructure_capture_time_ns,
            "decision_time_ns": decision_time_ns,
        },
        "coordinate_system": fixed_schema.expected_coordinate_system(),
        "source": fixed_source(),
        "policy": fixed_schema.expected_policy(),
        "detections": [] if detections is None else detections,
    }


def second_fixed_record(
    *, detections: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return fixed_record(
        detections=detections,
        vehicle_frame_id="000002",
        infrastructure_frame_id="100002",
        vehicle_capture_time_ns=1_100,
        infrastructure_capture_time_ns=1_000,
        decision_time_ns=1_100,
    )


def agent_registry() -> list[dict[str, str]]:
    return [
        {
            "agent_id": "infrastructure-1",
            "agent_role": "infrastructure",
            "agent_manifest_sha256": digest("9"),
        },
        {
            "agent_id": "vehicle-1",
            "agent_role": "vehicle",
            "agent_manifest_sha256": digest("a"),
        },
    ]


def observation(
    *,
    agent_id: str = "infrastructure-1",
    event_time_ns: int = 900,
    complete_arrival_time_ns: int = 950,
    spatial_reliability: float = 0.75,
    box_3d: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "event_time_ns": event_time_ns,
        "complete_arrival_time_ns": complete_arrival_time_ns,
        "box_3d": (
            [10.0, -2.0, -1.5, 4.35, 1.91, 1.59, 0.25]
            if box_3d is None
            else box_3d
        ),
        "source_observation_record_sha256": digest("b"),
        "reliability_map_record_sha256": digest("c"),
        "spatial_reliability": spatial_reliability,
    }


def sidecar_record(
    fixed: dict[str, Any],
    fixed_records: list[dict[str, Any]],
    *,
    observations_by_detection: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    normalized_fixed_records = fixed_schema.validate_document(fixed_records)
    normalized_fixed = fixed_schema.validate_record(fixed)
    if observations_by_detection is None:
        observations_by_detection = [
            [
                observation(
                    event_time_ns=min(
                        normalized_fixed["frame"]["infrastructure_capture_time_ns"],
                        normalized_fixed["frame"]["decision_time_ns"],
                    ),
                    complete_arrival_time_ns=normalized_fixed["frame"][
                        "decision_time_ns"
                    ],
                )
            ]
            for _ in normalized_fixed["detections"]
        ]
    candidates = [
        {
            "detection_id": item["detection_id"],
            "detection_score_passthrough": item["score"],
            "source_observations": observations_by_detection[index],
        }
        for index, item in enumerate(normalized_fixed["detections"])
    ]
    return {
        "schema_version": 1,
        "contract_id": association_schema.CONTRACT_ID,
        "record_kind": association_schema.RECORD_KIND,
        "scientific_claim_allowed": False,
        "runs_h1": False,
        "production_input_allowed": False,
        "fixed_detection_binding": {
            "fixed_detection_contract_id": fixed_schema.CONTRACT_ID,
            "fixed_detection_contract_sha256": hashlib.sha256(
                FIXED_PROTOCOL_PATH.read_bytes()
            ).hexdigest(),
            "fixed_detection_document_sha256": fixed_schema.document_sha256(
                normalized_fixed_records
            ),
            "fixed_detection_record_sha256": fixed_schema.record_sha256(
                normalized_fixed
            ),
        },
        "dataset": copy.deepcopy(normalized_fixed["dataset"]),
        "frame": copy.deepcopy(normalized_fixed["frame"]),
        "source": {
            "fixed_detection_source_sha256": association_schema._fixed_source_sha256(
                normalized_fixed["source"]
            ),
            "observation_generator_sha256": digest("d"),
        },
        "agent_registry": agent_registry(),
        "reliability_map_binding": {
            "reliability_map_contract_id": "diagnostic-map-v1",
            "reliability_map_contract_sha256": digest("e"),
            "reliability_map_document_sha256": digest("f"),
        },
        "semantics": association_schema.expected_semantics(),
        "candidates": candidates,
    }


class AssociationObservationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_identity_and_diagnostic_flags_are_frozen(self) -> None:
        self.assertEqual(self.protocol["schema_version"], 1)
        self.assertEqual(self.protocol["contract_id"], association_schema.CONTRACT_ID)
        self.assertEqual(self.protocol["status"], "diagnostic_local_schema")
        self.assertFalse(self.protocol["scientific_claim_allowed"])
        self.assertFalse(self.protocol["runs_h1"])
        self.assertFalse(self.protocol["production_input_allowed"])

    def test_production_gate_requires_readback_and_reviewed_v2(self) -> None:
        gate = self.protocol["review_gate"]
        requirements = " ".join(gate["required_before_production_input"])
        self.assertIn("authorized V2X-Seq SPD data read-back", requirements)
        self.assertIn("reviewed v2", requirements)
        self.assertTrue(gate["v1_may_not_be_promoted_by_changing_flags"])

    def test_protocol_and_validator_share_semantics_and_limits(self) -> None:
        fixed = self.protocol["fixed_semantics"]
        expected = association_schema.expected_semantics()
        self.assertEqual(
            fixed["spatial_reliability_aggregation"],
            expected["spatial_reliability_aggregation"],
        )
        self.assertEqual(
            fixed["spatial_reliability_calibration"],
            expected["spatial_reliability_calibration"],
        )
        self.assertEqual(
            fixed["spatial_reliability_value_role"],
            expected["spatial_reliability_value_role"],
        )
        self.assertEqual(
            fixed["detection_score_role"], expected["detection_score_role"]
        )
        limits = self.protocol["resource_limits"]
        self.assertEqual(limits["maximum_document_bytes"], association_schema.MAX_JSONL_BYTES)
        self.assertEqual(limits["maximum_record_bytes"], association_schema.MAX_RECORD_BYTES)
        self.assertEqual(limits["maximum_document_records"], association_schema.MAX_DOCUMENT_RECORDS)
        self.assertEqual(limits["maximum_agents"], association_schema.MAX_AGENT_COUNT)
        self.assertEqual(
            limits["maximum_candidates_per_frame"],
            association_schema.MAX_CANDIDATES_PER_FRAME,
        )
        self.assertEqual(
            limits["maximum_nesting_depth"],
            association_schema.MAX_NESTING_DEPTH,
        )
        self.assertEqual(
            limits["maximum_source_observations_per_candidate"],
            association_schema.MAX_SOURCE_OBSERVATIONS_PER_CANDIDATE,
        )

    def test_protocol_and_validator_share_exact_object_fields_and_box_bounds(self) -> None:
        record_schema = self.protocol["record_schema"]
        self.assertEqual(
            set(record_schema["required_top_level_fields"]),
            association_schema.TOP_LEVEL_FIELDS,
        )
        for protocol_key, implementation_fields in (
            ("fixed_detection_binding", association_schema.FIXED_BINDING_FIELDS),
            ("dataset", association_schema.DATASET_FIELDS),
            ("frame", association_schema.FRAME_FIELDS),
            ("source", association_schema.SOURCE_FIELDS),
            ("reliability_map_binding", association_schema.RELIABILITY_MAP_BINDING_FIELDS),
            ("semantics", association_schema.SEMANTICS_FIELDS),
            ("candidate", association_schema.CANDIDATE_FIELDS),
            ("source_observation", association_schema.OBSERVATION_FIELDS),
        ):
            with self.subTest(protocol_key=protocol_key):
                self.assertEqual(
                    set(record_schema[protocol_key]["fields"]),
                    implementation_fields,
                )
        self.assertEqual(
            set(record_schema["agent_registry"]["entry_fields"]),
            association_schema.AGENT_FIELDS,
        )
        observation_schema = record_schema["source_observation"]
        policy = fixed_schema.expected_policy()
        self.assertEqual(
            observation_schema["box_3d_bottom_center_bounds_m"],
            {
                "x": policy["roi"]["x_m"],
                "y": policy["roi"]["y_m"],
                "z": policy["roi"]["z_m"],
            },
        )
        self.assertEqual(
            observation_schema["box_3d_dimension_bounds_m"],
            {
                "minimum_inclusive": fixed_schema.MIN_BOX_DIMENSION_M,
                "maximum_inclusive": fixed_schema.MAX_BOX_DIMENSION_M,
            },
        )
        self.assertEqual(
            observation_schema["box_3d_yaw_bounds_rad"],
            {
                "minimum_inclusive": -math.pi,
                "maximum_exclusive": math.pi,
            },
        )

    def test_forbidden_fields_and_hash_join_are_explicit(self) -> None:
        forbidden = set(
            self.protocol["record_schema"]["forbidden_field_names_at_any_depth"]
        )
        self.assertEqual(forbidden, association_schema.FORBIDDEN_FIELD_NAMES)
        join = self.protocol["fixed_detection_join"]
        self.assertIn("exactly one", join["cardinality"])
        self.assertEqual(join["empty_frame_representation"], [])
        self.assertIn("never reliability", join["score_binding"])

    def test_validator_has_no_runtime_ml_clearml_or_dataset_import(self) -> None:
        source_text = Path(association_schema.__file__).read_text(encoding="utf-8")
        for statement in (
            "import numpy",
            "import torch",
            "import clearml",
            "from clearml",
            "import transvision",
            "from transvision",
        ):
            with self.subTest(statement=statement):
                self.assertNotIn(statement, source_text)


class AssociationObservationRecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixed_records = [fixed_record(detections=[detection()])]
        self.value = sidecar_record(self.fixed_records[0], self.fixed_records)

    def assert_invalid(self, value: Any, pattern: str) -> None:
        with self.assertRaisesRegex(
            association_schema.AssociationObservationContractError, pattern
        ):
            association_schema.validate_record(value)

    def test_valid_record_is_detached_and_canonical(self) -> None:
        normalized = association_schema.validate_record(self.value)
        self.assertIsNot(normalized, self.value)
        self.assertEqual(
            normalized["candidates"][0]["source_observations"][0][
                "spatial_reliability"
            ],
            0.75,
        )
        canonical = association_schema.canonical_record_bytes(self.value)
        self.assertTrue(canonical.endswith(b"\n"))
        self.assertIn(b'"scientific_claim_allowed":false', canonical)
        self.assertIn(b'"runs_h1":false', canonical)
        self.assertIn(b'"production_input_allowed":false', canonical)

    def test_all_claim_and_production_flags_must_be_false(self) -> None:
        for field in (
            "scientific_claim_allowed",
            "runs_h1",
            "production_input_allowed",
        ):
            value = copy.deepcopy(self.value)
            value[field] = True
            with self.subTest(field=field):
                self.assert_invalid(value, f"{field} must be false")

    def test_unknown_fields_are_rejected_at_every_object_level(self) -> None:
        mutations = []
        top = copy.deepcopy(self.value)
        top["unexpected"] = True
        mutations.append(top)
        binding = copy.deepcopy(self.value)
        binding["fixed_detection_binding"]["unexpected"] = True
        mutations.append(binding)
        candidate = copy.deepcopy(self.value)
        candidate["candidates"][0]["unexpected"] = True
        mutations.append(candidate)
        observation_value = copy.deepcopy(self.value)
        observation_value["candidates"][0]["source_observations"][0][
            "unexpected"
        ] = True
        mutations.append(observation_value)
        for value in mutations:
            with self.subTest(keys=value.keys()):
                self.assert_invalid(value, "unknown")

    def test_gt_label_track_evaluator_and_metric_fields_are_forbidden(self) -> None:
        for field in (
            "ground_truth",
            "labels",
            "track_id",
            "evaluator",
            "metrics",
        ):
            value = copy.deepcopy(self.value)
            value["candidates"][0]["source_observations"][0][field] = "forbidden"
            with self.subTest(field=field):
                self.assert_invalid(value, "forbidden field")

    def test_agent_registry_requires_unique_stable_sorted_ids(self) -> None:
        duplicate = copy.deepcopy(self.value)
        duplicate["agent_registry"][1]["agent_id"] = "infrastructure-1"
        self.assert_invalid(duplicate, "duplicate stable agent_id")
        reversed_registry = copy.deepcopy(self.value)
        reversed_registry["agent_registry"].reverse()
        self.assert_invalid(reversed_registry, "ordered by agent_id")
        empty = copy.deepcopy(self.value)
        empty["agent_registry"] = []
        self.assert_invalid(empty, "at least one")
        oversized = copy.deepcopy(self.value)
        oversized["agent_registry"] = [
            {
                "agent_id": f"agent-{index:02d}",
                "agent_role": "remote",
                "agent_manifest_sha256": digest("a"),
            }
            for index in range(association_schema.MAX_AGENT_COUNT + 1)
        ]
        self.assert_invalid(oversized, "agent-count")

    def test_observations_require_registered_unique_sorted_agent_ids(self) -> None:
        unknown = copy.deepcopy(self.value)
        unknown["candidates"][0]["source_observations"][0]["agent_id"] = "unknown"
        self.assert_invalid(unknown, "absent from")
        duplicate = copy.deepcopy(self.value)
        duplicate["candidates"][0]["source_observations"].append(
            copy.deepcopy(duplicate["candidates"][0]["source_observations"][0])
        )
        self.assert_invalid(duplicate, "repeats agent_id")
        reversed_observations = copy.deepcopy(self.value)
        reversed_observations["candidates"][0]["source_observations"] = [
            observation(
                agent_id="vehicle-1",
                event_time_ns=900,
                complete_arrival_time_ns=950,
            ),
            observation(),
        ]
        self.assert_invalid(reversed_observations, "ordered by agent_id")
        empty = copy.deepcopy(self.value)
        empty["candidates"][0]["source_observations"] = []
        self.assert_invalid(empty, "non-empty")

    def test_causal_timestamp_order_is_strictly_enforced(self) -> None:
        for event_time, arrival_time, pattern in (
            (951, 950, "event_time_ns"),
            (900, 1_001, "decision_time_ns"),
            (-1, 900, "64-bit"),
            (900, 2**63, "64-bit"),
        ):
            value = copy.deepcopy(self.value)
            item = value["candidates"][0]["source_observations"][0]
            item["event_time_ns"] = event_time
            item["complete_arrival_time_ns"] = arrival_time
            with self.subTest(event_time=event_time, arrival_time=arrival_time):
                self.assert_invalid(value, pattern)

    def test_reliability_is_finite_unit_interval_not_score_fallback(self) -> None:
        for invalid in (-0.001, 1.001, math.nan, math.inf, True):
            value = copy.deepcopy(self.value)
            value["candidates"][0]["source_observations"][0][
                "spatial_reliability"
            ] = invalid
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, r"finite|\[0, 1\]")
        missing = copy.deepcopy(self.value)
        item = missing["candidates"][0]["source_observations"][0]
        del item["spatial_reliability"]
        item["score"] = self.value["candidates"][0]["detection_score_passthrough"]
        self.assert_invalid(missing, "missing|unknown")

    def test_box_shape_finiteness_roi_dimensions_and_yaw_are_bounded(self) -> None:
        invalid_boxes = (
            [1.0] * 6,
            [math.nan, 0, 0, 1, 1, 1, 0],
            [-0.01, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 101, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, math.pi],
        )
        for invalid in invalid_boxes:
            value = copy.deepcopy(self.value)
            value["candidates"][0]["source_observations"][0]["box_3d"] = invalid
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, "shape|finite|ROI|numeric bound|yaw")

    def test_candidate_ids_are_canonical_unique_ordinals(self) -> None:
        value = copy.deepcopy(self.value)
        value["candidates"][0]["detection_id"] = "track-1"
        self.assert_invalid(value, "canonical ordinal")
        oversized = copy.deepcopy(self.value)
        oversized["candidates"] = [
            {
                "detection_id": f"det-{index:06d}",
                "detection_score_passthrough": 0.9,
                "source_observations": [observation()],
            }
            for index in range(association_schema.MAX_CANDIDATES_PER_FRAME + 1)
        ]
        self.assert_invalid(oversized, "candidate limit")

    def test_every_hash_is_lowercase_sha256(self) -> None:
        mutations = (
            ("fixed_detection_binding", "fixed_detection_contract_sha256"),
            ("fixed_detection_binding", "fixed_detection_document_sha256"),
            ("fixed_detection_binding", "fixed_detection_record_sha256"),
            ("source", "fixed_detection_source_sha256"),
            ("source", "observation_generator_sha256"),
            ("reliability_map_binding", "reliability_map_contract_sha256"),
            ("reliability_map_binding", "reliability_map_document_sha256"),
        )
        for parent, field in mutations:
            value = copy.deepcopy(self.value)
            value[parent][field] = "A" * 64
            with self.subTest(parent=parent, field=field):
                self.assert_invalid(value, "64 lowercase")
        for field in (
            "source_observation_record_sha256",
            "reliability_map_record_sha256",
        ):
            value = copy.deepcopy(self.value)
            value["candidates"][0]["source_observations"][0][field] = "a" * 63
            with self.subTest(field=field):
                self.assert_invalid(value, "64 lowercase")

    def test_semantics_are_exact_and_score_is_passthrough_only(self) -> None:
        value = copy.deepcopy(self.value)
        value["semantics"]["spatial_reliability_calibration"] = "calibrated"
        self.assert_invalid(value, "diagnostic v1 semantics")
        value = copy.deepcopy(self.value)
        value["semantics"]["detection_score_role"] = "reliability"
        self.assert_invalid(value, "diagnostic v1 semantics")

    def test_schema_version_is_integer_and_not_boolean_or_float(self) -> None:
        for invalid in (True, 1.0, "1", 2):
            value = copy.deepcopy(self.value)
            value["schema_version"] = invalid
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, "schema_version")

    def test_signed_zero_and_numeric_spelling_are_canonical(self) -> None:
        negative = copy.deepcopy(self.value)
        positive = copy.deepcopy(self.value)
        negative["candidates"][0]["source_observations"][0]["box_3d"][1] = -0.0
        positive["candidates"][0]["source_observations"][0]["box_3d"][1] = 0.0
        self.assertEqual(
            association_schema.canonical_record_bytes(negative),
            association_schema.canonical_record_bytes(positive),
        )
        canonical = association_schema.canonical_record_bytes(self.value)
        self.assertIn(b"7.5000000000000000e-01", canonical)

    def test_record_byte_limit_is_enforced(self) -> None:
        with mock.patch.object(association_schema, "MAX_RECORD_BYTES", 1):
            with self.assertRaisesRegex(
                association_schema.AssociationObservationContractError,
                "record exceeds",
            ):
                association_schema.canonical_record_bytes(self.value)


class AssociationObservationDocumentTest(unittest.TestCase):
    def make_documents(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        fixed_records = [
            fixed_record(detections=[detection()]),
            second_fixed_record(detections=[detection(score=0.8)]),
        ]
        sidecars = [
            sidecar_record(fixed_records[0], fixed_records),
            sidecar_record(fixed_records[1], fixed_records),
        ]
        return fixed_records, sidecars

    def assert_document_invalid(
        self,
        sidecars: Any,
        fixed_records: Any,
        pattern: str,
    ) -> None:
        with self.assertRaisesRegex(
            association_schema.AssociationObservationContractError, pattern
        ):
            association_schema.validate_document(sidecars, fixed_records)

    def test_valid_join_round_trips_and_hashes_stably(self) -> None:
        fixed_records, sidecars = self.make_documents()
        normalized = association_schema.validate_document(sidecars, fixed_records)
        self.assertEqual(len(normalized), 2)
        payload = association_schema.canonical_jsonl_bytes(sidecars, fixed_records)
        fixed_payload = fixed_schema.canonical_jsonl_bytes(fixed_records)
        self.assertEqual(
            association_schema.loads_jsonl(
                payload, fixed_detection_data=fixed_payload
            ),
            normalized,
        )
        self.assertEqual(
            association_schema.document_sha256(sidecars, fixed_records),
            hashlib.sha256(payload).hexdigest(),
        )

    def test_record_only_validation_does_not_accept_a_false_join(self) -> None:
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["fixed_detection_binding"]["fixed_detection_record_sha256"] = digest(
            "0"
        )
        association_schema.validate_record(sidecars[0])
        self.assert_document_invalid(sidecars, fixed_records, "record SHA-256")

    def test_sidecar_cardinality_must_equal_fixed_document(self) -> None:
        fixed_records, sidecars = self.make_documents()
        self.assert_document_invalid(sidecars[:1], fixed_records, "exactly one")
        self.assert_document_invalid(sidecars, fixed_records[:1], "exactly one")

    def test_contract_document_and_record_digests_are_verified(self) -> None:
        for field, pattern in (
            ("fixed_detection_contract_sha256", "contract SHA-256"),
            ("fixed_detection_document_sha256", "document SHA-256"),
            ("fixed_detection_record_sha256", "record SHA-256"),
        ):
            fixed_records, sidecars = self.make_documents()
            sidecars[0]["fixed_detection_binding"][field] = digest("0")
            with self.subTest(field=field):
                self.assert_document_invalid(sidecars, fixed_records, pattern)

    def test_exact_dataset_frame_and_detector_source_are_bound(self) -> None:
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["frame"]["vehicle_frame_id"] = "other"
        self.assert_document_invalid(sidecars, fixed_records, "frame identity")
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["dataset"]["release_id"] = "other-release"
        self.assert_document_invalid(sidecars, fixed_records, "dataset")
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["source"]["fixed_detection_source_sha256"] = digest("0")
        self.assert_document_invalid(sidecars, fixed_records, "source SHA-256")

    def test_each_detection_is_referenced_once_with_exact_score(self) -> None:
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["candidates"] = []
        self.assert_document_invalid(sidecars, fixed_records, "every fixed detection")
        fixed_records, sidecars = self.make_documents()
        sidecars[0]["candidates"][0]["detection_score_passthrough"] = 0.7
        self.assert_document_invalid(sidecars, fixed_records, "exact passthrough")

    def test_empty_fixed_detection_frame_requires_explicit_empty_candidates(self) -> None:
        fixed_records = [fixed_record()]
        sidecars = [sidecar_record(fixed_records[0], fixed_records)]
        normalized = association_schema.validate_document(sidecars, fixed_records)
        self.assertEqual(normalized[0]["candidates"], [])
        sidecars[0]["candidates"] = [
            {
                "detection_id": "det-000000",
                "detection_score_passthrough": 0.9,
                "source_observations": [observation()],
            }
        ]
        self.assert_document_invalid(sidecars, fixed_records, "every fixed detection")

    def test_source_agent_registry_and_reliability_map_are_document_stable(self) -> None:
        mutations = (
            ("source", "observation_generator_sha256", digest("0"), "source identities"),
            ("agent_registry", None, None, "stable agent registry"),
            (
                "reliability_map_binding",
                "reliability_map_document_sha256",
                digest("0"),
                "reliability-map",
            ),
        )
        for parent, field, invalid, pattern in mutations:
            fixed_records, sidecars = self.make_documents()
            if parent == "agent_registry":
                sidecars[1][parent][0]["agent_role"] = "changed"
            else:
                assert field is not None and invalid is not None
                sidecars[1][parent][field] = invalid
            with self.subTest(parent=parent):
                self.assert_document_invalid(sidecars, fixed_records, pattern)

    def test_invalid_or_reordered_fixed_document_is_rejected(self) -> None:
        fixed_records, sidecars = self.make_documents()
        fixed_records.reverse()
        self.assert_document_invalid(sidecars, fixed_records, "invalid")

    def test_parser_rejects_duplicate_keys_noncanonical_framing_and_spelling(self) -> None:
        fixed_records, sidecars = self.make_documents()
        payload = association_schema.canonical_jsonl_bytes(sidecars, fixed_records)
        fixed_payload = fixed_schema.canonical_jsonl_bytes(fixed_records)
        duplicate = payload.replace(
            b'"schema_version":1',
            b'"schema_version":1,"schema_version":1',
            1,
        )
        with self.assertRaisesRegex(
            association_schema.AssociationObservationContractError, "duplicate JSON key"
        ):
            association_schema.loads_jsonl(
                duplicate, fixed_detection_data=fixed_payload
            )
        invalid_payloads = (
            payload[:-1],
            payload.replace(b"\n", b"\r\n"),
            b"\xef\xbb\xbf" + payload,
            payload + b"\n",
            json.dumps(sidecars[0], ensure_ascii=False).encode("utf-8") + b"\n",
        )
        for invalid in invalid_payloads:
            with self.subTest(invalid=invalid[:20]):
                with self.assertRaises(
                    association_schema.AssociationObservationContractError
                ):
                    association_schema.loads_jsonl(
                        invalid, fixed_detection_data=fixed_payload
                    )

    def test_parser_requires_canonical_fixed_detection_document(self) -> None:
        fixed_records, sidecars = self.make_documents()
        payload = association_schema.canonical_jsonl_bytes(sidecars, fixed_records)
        noncanonical_fixed = (
            json.dumps(fixed_records[0], ensure_ascii=False).encode("utf-8") + b"\n"
        )
        with self.assertRaisesRegex(
            association_schema.AssociationObservationContractError,
            "fixed-detection JSONL binding",
        ):
            association_schema.loads_jsonl(
                payload, fixed_detection_data=noncanonical_fixed
            )

    def test_document_byte_limit_is_enforced(self) -> None:
        fixed_records, sidecars = self.make_documents()
        one_line = len(association_schema.canonical_record_bytes(sidecars[0]))
        with mock.patch.object(association_schema, "MAX_JSONL_BYTES", one_line):
            with self.assertRaisesRegex(
                association_schema.AssociationObservationContractError,
                "document exceeds",
            ):
                association_schema.canonical_jsonl_bytes(sidecars, fixed_records)
            with self.assertRaisesRegex(
                association_schema.AssociationObservationContractError,
                "document exceeds",
            ):
                association_schema.validate_document(sidecars, fixed_records)


if __name__ == "__main__":
    unittest.main()
