from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from unittest import mock
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PACKAGE_ROOT))

import schema  # noqa: E402


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "fixed-detections-v1.json"
)


def digest(character: str) -> str:
    return character * 64


def detection(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "detection_id": "det-000000",
        "source_label_id": 0,
        "source_class_name": "Car",
        "class_name": "vehicle",
        "box_3d": [10.0, -2.0, -1.5, 4.35, 1.91, 1.59, 0.25],
        "score": 0.9,
    }
    value.update(overrides)
    return value


def source(*, model_id: str = "coformernet_controlled_adaptation") -> dict[str, Any]:
    return {
        "repository_url": schema.REPOSITORY_URL,
        "implementation_revision": "1" * 40,
        "implementation_tree_sha256": digest("2"),
        "model_id": model_id,
        "config_path": (
            "configs/resilient_v2x/dair_resilient_v2x.py"
            if model_id == "resilient_v2x"
            else "configs/resilient_v2x/baselines/coformernet.py"
        ),
        "config_sha256": digest("3"),
        "checkpoint_role": "canonical_final",
        "checkpoint_sha256": digest("4"),
        "teacher_checkpoint_sha256": (
            digest("5") if model_id == "resilient_v2x" else None
        ),
        "inference_adapter_sha256": digest("6"),
        "environment_manifest_sha256": digest("7"),
    }


def record(
    *,
    detections: list[dict[str, Any]] | None = None,
    sequence_id: str = "0001",
    vehicle_frame_id: str = "000001",
    infrastructure_frame_id: str = "100001",
    vehicle_capture_time_ns: int = 1_000,
    infrastructure_capture_time_ns: int = 900,
    decision_time_ns: int = 1_000,
    model_id: str = "coformernet_controlled_adaptation",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": schema.CONTRACT_ID,
        "record_kind": schema.RECORD_KIND,
        "scientific_claim_allowed": False,
        "dataset": {
            "name": schema.DATASET_NAME,
            "release_id": "fixture-release-v1",
            "split": "validation",
            "dataset_manifest_sha256": digest("8"),
            "split_manifest_sha256": digest("9"),
        },
        "frame": {
            "sequence_id": sequence_id,
            "vehicle_frame_id": vehicle_frame_id,
            "infrastructure_frame_id": infrastructure_frame_id,
            "vehicle_capture_time_ns": vehicle_capture_time_ns,
            "infrastructure_capture_time_ns": infrastructure_capture_time_ns,
            "decision_time_ns": decision_time_ns,
        },
        "coordinate_system": schema.expected_coordinate_system(),
        "source": source(model_id=model_id),
        "policy": schema.expected_policy(),
        "detections": [] if detections is None else detections,
    }


def second_record(**overrides: Any) -> dict[str, Any]:
    settings = {
        "vehicle_frame_id": "000002",
        "infrastructure_frame_id": "100002",
        "vehicle_capture_time_ns": 1_100,
        "infrastructure_capture_time_ns": 1_000,
        "decision_time_ns": 1_100,
    }
    settings.update(overrides)
    return record(**settings)


class FixedProtocolFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_identity_and_non_claim_scope_are_frozen(self) -> None:
        self.assertEqual(self.protocol["schema_version"], schema.SCHEMA_VERSION)
        self.assertEqual(self.protocol["contract_id"], schema.CONTRACT_ID)
        self.assertEqual(self.protocol["status"], "frozen_local_schema")
        self.assertFalse(self.protocol["scientific_claim_allowed"])
        self.assertFalse(self.protocol["scope"]["backend_included"])
        self.assertFalse(self.protocol["scope"]["upstream_code_included"])

    def test_protocol_and_validator_share_exact_fixed_semantics(self) -> None:
        fixed = self.protocol["fixed_semantics"]
        self.assertEqual(
            fixed["coordinate_system"], schema.expected_coordinate_system()
        )
        self.assertEqual(fixed["policy"], schema.expected_policy())
        self.assertEqual(
            self.protocol["record_schema"]["source"]["config_path_by_model"],
            schema.CONFIG_PATH_BY_MODEL,
        )
        self.assertEqual(
            set(self.protocol["record_schema"]["forbidden_field_names_at_any_depth"]),
            schema.FORBIDDEN_FIELD_NAMES,
        )

    def test_protocol_freezes_empty_frames_and_local_detection_ids(self) -> None:
        detection_schema = self.protocol["record_schema"]["detection"]
        self.assertEqual(detection_schema["empty_frame_representation"], [])
        self.assertIn(
            "never a trajectory identity", detection_schema["detection_id_scope"]
        )

    def test_protocol_freezes_runtime_limits_and_numeric_spelling(self) -> None:
        self.assertEqual(
            self.protocol["resource_limits"],
            {
                "maximum_document_bytes": schema.MAX_JSONL_BYTES,
                "maximum_record_bytes": schema.MAX_RECORD_BYTES,
                "maximum_document_records": schema.MAX_DOCUMENT_RECORDS,
                "maximum_nesting_depth": schema.MAX_NESTING_DEPTH,
                "minimum_box_dimension_m": schema.MIN_BOX_DIMENSION_M,
                "maximum_box_dimension_m": schema.MAX_BOX_DIMENSION_M,
            },
        )
        canonical = self.protocol["artifact_format"]["canonical_json"]
        self.assertIn(".16e", canonical["float_encoding"])
        self.assertIn("at least two", canonical["float_encoding"])
        self.assertIn("-0.0", canonical["signed_zero"])

    def test_validator_source_does_not_import_transvision(self) -> None:
        source_text = Path(schema.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import transvision", source_text)
        self.assertNotIn("from transvision", source_text)


class FixedDetectionRecordTest(unittest.TestCase):
    def assert_invalid(self, value: Any, pattern: str) -> None:
        with self.assertRaisesRegex(schema.FixedDetectionContractError, pattern):
            schema.validate_record(value)

    def test_legal_empty_frame_round_trips_as_empty_array(self) -> None:
        value = record()
        normalized = schema.validate_record(value)
        self.assertEqual(normalized["detections"], [])
        payload = schema.canonical_jsonl_bytes([value])
        self.assertTrue(payload.endswith(b"\n"))
        self.assertIn(b'"detections":[]', payload)
        self.assertEqual(schema.loads_jsonl(payload), (normalized,))
        self.assertRegex(schema.document_sha256([value]), r"^[0-9a-f]{64}$")

    def test_valid_detection_and_score_threshold_boundary_are_accepted(self) -> None:
        value = record(detections=[detection(score=0.05)])
        observed = schema.validate_record(value)["detections"][0]
        self.assertEqual(observed["box_3d"], detection()["box_3d"])
        self.assertEqual(observed["score"], 0.05)

    def test_canonical_serialization_is_key_order_independent(self) -> None:
        value = record(detections=[detection()])
        reordered = dict(reversed(list(value.items())))
        self.assertEqual(
            schema.canonical_record_bytes(value),
            schema.canonical_record_bytes(reordered),
        )
        self.assertEqual(schema.record_sha256(value), schema.record_sha256(reordered))

    def test_unknown_fields_fail_at_top_nested_and_detection_levels(self) -> None:
        cases = []
        top = record()
        top["unexpected"] = True
        cases.append(top)
        nested = record()
        nested["frame"]["unexpected"] = True
        cases.append(nested)
        inside_detection = record(detections=[detection(unexpected=True)])
        cases.append(inside_detection)
        for value in cases:
            with self.subTest(value=value):
                self.assert_invalid(value, "unknown")

    def test_gt_track_and_evaluator_fields_are_explicitly_forbidden(self) -> None:
        for field_name in (
            "ground_truth",
            "labels",
            "track_id",
            "evaluator",
            "metrics",
        ):
            value = record(detections=[detection()])
            value["detections"][0][field_name] = "forbidden"
            with self.subTest(field_name=field_name):
                self.assert_invalid(value, "forbidden field")

    def test_detection_id_is_a_frame_local_ordinal_not_an_identity(self) -> None:
        value = record(
            detections=[
                detection(detection_id="track-42"),
            ]
        )
        self.assert_invalid(value, "frame-local ordinal")

    def test_coordinate_frame_axes_and_z_reference_are_exact(self) -> None:
        mutations = (
            ("frame", "world"),
            ("z_reference", "gravity_center"),
            ("box_3d_order", "x,y,z,w,l,h,yaw"),
        )
        for key, invalid in mutations:
            value = record()
            value["coordinate_system"][key] = invalid
            with self.subTest(key=key):
                self.assert_invalid(value, "coordinate")
        value = record()
        value["coordinate_system"]["axes"]["y"] = "right"
        self.assert_invalid(value, "coordinate")

    def test_box_shape_must_be_a_seven_element_json_array(self) -> None:
        for invalid in (
            [1.0] * 6,
            [1.0] * 8,
            tuple(detection()["box_3d"]),
            "not-an-array",
        ):
            value = record(detections=[detection(box_3d=invalid)])
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, r"shape \[7\]")

    def test_nan_infinity_and_boolean_numbers_are_rejected(self) -> None:
        cases = (
            record(detections=[detection(box_3d=[math.nan, 0, 0, 1, 1, 1, 0])]),
            record(detections=[detection(score=math.inf)]),
            record(detections=[detection(score=True)]),
            record(vehicle_capture_time_ns=True),
        )
        for value in cases:
            with self.subTest(value=value):
                self.assert_invalid(value, "finite|64-bit")

    def test_strict_json_parser_rejects_nan_and_infinity_tokens(self) -> None:
        for token in ("NaN", "Infinity", "-Infinity"):
            value = record(detections=[detection()])
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
            payload = payload.replace('"score":0.9', f'"score":{token}') + "\n"
            with self.subTest(token=token):
                with self.assertRaisesRegex(
                    schema.FixedDetectionContractError, "non-finite"
                ):
                    schema.loads_jsonl(payload)

    def test_all_sha_fields_require_lowercase_sha256(self) -> None:
        mutations = (
            ("dataset", "dataset_manifest_sha256"),
            ("dataset", "split_manifest_sha256"),
            ("source", "implementation_tree_sha256"),
            ("source", "config_sha256"),
            ("source", "checkpoint_sha256"),
            ("source", "inference_adapter_sha256"),
            ("source", "environment_manifest_sha256"),
        )
        for parent, field in mutations:
            for invalid in ("a" * 63, "A" * 64, "g" * 64):
                value = record()
                value[parent][field] = invalid
                with self.subTest(parent=parent, field=field, invalid=invalid[:1]):
                    self.assert_invalid(value, "64 lowercase hexadecimal")

    def test_roi_dimensions_yaw_and_score_bounds_are_enforced(self) -> None:
        invalid_boxes = (
            [-0.01, 0, 0, 1, 1, 1, 0],
            [0, 40.01, 0, 1, 1, 1, 0],
            [0, 0, 1.01, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, -1, 1, 0],
            [0, 0, 0, 5e-324, 1, 1, 0],
            [0, 0, 0, 100.01, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, math.pi],
        )
        for box in invalid_boxes:
            value = record(detections=[detection(box_3d=box)])
            with self.subTest(box=box):
                self.assert_invalid(value, "ROI|positive|numeric bound|yaw")
        for invalid_score in (0.049999, 1.000001):
            value = record(detections=[detection(score=invalid_score)])
            with self.subTest(score=invalid_score):
                self.assert_invalid(value, "score interval")

    def test_class_roi_score_and_nms_policy_cannot_drift(self) -> None:
        mutations = (
            ("class_mapping", "source_class_name", "Vehicle"),
            ("roi", "x_m", [0.0, 100.0]),
            ("score", "threshold", 0.1),
            ("nms", "type", "circle_nms"),
            ("nms", "iou_threshold", 0.1),
            ("nms", "nms_across_levels", True),
        )
        for section, key, invalid in mutations:
            value = record()
            value["policy"][section][key] = invalid
            with self.subTest(section=section, key=key):
                self.assert_invalid(value, "policy")

    def test_teacher_checkpoint_rule_depends_on_model_identity(self) -> None:
        coformer = record()
        coformer["source"]["teacher_checkpoint_sha256"] = digest("a")
        self.assert_invalid(coformer, "null teacher")
        resilient = record(model_id="resilient_v2x")
        normalized = schema.validate_record(resilient)
        self.assertEqual(normalized["source"]["teacher_checkpoint_sha256"], digest("5"))
        resilient["source"]["teacher_checkpoint_sha256"] = None
        self.assert_invalid(resilient, "64 lowercase hexadecimal")

    def test_revision_repository_and_config_path_are_strict(self) -> None:
        invalid_mutations = (
            ("implementation_revision", "f" * 39, "40 lowercase"),
            ("implementation_revision", "F" * 40, "40 lowercase"),
            ("repository_url", "http://example.invalid/repo", "HTTPS URL"),
            ("repository_url", "https://u:p@example.invalid/repo", "HTTPS URL"),
            ("repository_url", "https://[malformed/repo", "valid HTTPS URL"),
            (
                "repository_url",
                "https://example.invalid/transvision.git",
                "must be",
            ),
            ("config_path", "../config.py", "safe relative"),
            ("config_path", "/absolute/config.py", "safe relative"),
            ("config_path", "config.json", "safe relative"),
        )
        for field, invalid, message in invalid_mutations:
            value = record()
            value["source"][field] = invalid
            with self.subTest(field=field, invalid=invalid):
                self.assert_invalid(value, message)
        mismatched = record()
        mismatched["source"]["config_path"] = (
            "configs/resilient_v2x/dair_resilient_v2x.py"
        )
        self.assert_invalid(mismatched, "does not match")

    def test_schema_version_is_exact_integer_not_boolean_or_float(self) -> None:
        for invalid in (True, 1.0, "1", 2):
            value = record()
            value["schema_version"] = invalid
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, "schema_version must be integer 1")

    def test_causal_timestamp_rule_and_integer_types_are_enforced(self) -> None:
        value = record(decision_time_ns=899)
        self.assert_invalid(value, "causal time rule")
        for invalid in (-1, 2**63, 1.5, "1000"):
            value = record()
            value["frame"]["decision_time_ns"] = invalid
            with self.subTest(invalid=invalid):
                self.assert_invalid(value, "64-bit integer")

    def test_detection_count_is_bounded_by_frozen_nms_policy(self) -> None:
        detections = [
            detection(detection_id=f"det-{index:06d}") for index in range(101)
        ]
        self.assert_invalid(record(detections=detections), "per-frame maximum")

    def test_huge_integer_and_signed_zero_are_canonicalized_safely(self) -> None:
        huge = record(detections=[detection(box_3d=[0, 0, 0, 10**1000, 1, 1, 0])])
        self.assert_invalid(huge, "numeric bound")
        negative_zero = record(
            detections=[detection(box_3d=[-0.0, -0.0, -0.0, 1, 1, 1, -0.0])]
        )
        positive_zero = record(
            detections=[detection(box_3d=[0.0, 0.0, 0.0, 1, 1, 1, 0.0])]
        )
        self.assertEqual(
            schema.canonical_record_bytes(negative_zero),
            schema.canonical_record_bytes(positive_zero),
        )

    def test_detection_array_order_is_frozen(self) -> None:
        value = record(
            detections=[
                detection(detection_id="det-000000", score=0.8),
                detection(
                    detection_id="det-000001",
                    score=0.9,
                    box_3d=[11.0, -2.0, -1.5, 4.35, 1.91, 1.59, 0.25],
                ),
            ]
        )
        self.assert_invalid(value, "descending score")

    def test_canonical_generation_enforces_record_and_document_byte_limits(
        self,
    ) -> None:
        value = record()
        with mock.patch.object(schema, "MAX_RECORD_BYTES", 1):
            with self.assertRaisesRegex(
                schema.FixedDetectionContractError, "record exceeds"
            ):
                schema.canonical_record_bytes(value)
        line_size = len(schema.canonical_record_bytes(value))
        with mock.patch.object(schema, "MAX_JSONL_BYTES", line_size - 1):
            with self.assertRaisesRegex(
                schema.FixedDetectionContractError, "document exceeds"
            ):
                schema.canonical_jsonl_bytes([value])

    def test_document_limit_fails_before_validating_every_record(self) -> None:
        values = [record() for _ in range(100)]
        with (
            mock.patch.object(schema, "MAX_JSONL_BYTES", 1),
            mock.patch.object(
                schema, "validate_record", wraps=schema.validate_record
            ) as validator,
            self.assertRaisesRegex(
                schema.FixedDetectionContractError, "document exceeds"
            ),
        ):
            schema.canonical_jsonl_bytes(values)
        self.assertEqual(validator.call_count, 0)

    def test_numeric_spelling_is_explicit_and_signed_zero_is_normalized(self) -> None:
        value = record(
            detections=[
                detection(
                    box_3d=[0.000001, -0.0, 0.0, 1.0, 1.0, 1.0, -0.0],
                    score=0.9,
                )
            ]
        )
        canonical = schema.canonical_record_bytes(value)
        self.assertIn(b"9.9999999999999995e-07", canonical)
        self.assertIn(b"0.0000000000000000e+00", canonical)
        self.assertNotIn(b"-0.0000000000000000e+00", canonical)
        for spelling in (b"1e-06", b"0.000001"):
            noncanonical = canonical.replace(b"9.9999999999999995e-07", spelling, 1)
            with self.subTest(spelling=spelling):
                with self.assertRaisesRegex(
                    schema.FixedDetectionContractError,
                    "not canonically serialized",
                ):
                    schema.loads_jsonl(noncanonical)

    def test_document_digest_does_not_materialize_canonical_jsonl(self) -> None:
        with mock.patch.object(
            schema,
            "canonical_jsonl_bytes",
            side_effect=AssertionError("must not materialize"),
        ):
            self.assertRegex(schema.document_sha256([record()]), r"^[0-9a-f]{64}$")

    def test_parser_rejects_record_count_before_parsing_any_object(self) -> None:
        payload = schema.canonical_record_bytes(record()) * 2
        with (
            mock.patch.object(schema, "MAX_DOCUMENT_RECORDS", 1),
            mock.patch.object(
                schema, "_loads_json_object", wraps=schema._loads_json_object
            ) as parser,
            self.assertRaisesRegex(
                schema.FixedDetectionContractError, "record-count limit"
            ),
        ):
            schema.loads_jsonl(payload)
        self.assertEqual(parser.call_count, 0)

    def test_float_encoder_handles_subnormal_and_maximum_exponents(self) -> None:
        self.assertEqual(
            schema._canonical_float_text(5e-324),
            "4.9406564584124654e-324",
        )
        self.assertEqual(
            schema._canonical_float_text(sys.float_info.max),
            "1.7976931348623157e+308",
        )


class FixedDetectionDocumentTest(unittest.TestCase):
    def assert_document_invalid(self, values: Any, pattern: str) -> None:
        with self.assertRaisesRegex(schema.FixedDetectionContractError, pattern):
            schema.validate_document(values)

    def test_document_accepts_ordered_frames_and_reuses_detection_ids(self) -> None:
        first = record(detections=[detection(detection_id="det-000000")])
        second = second_record(
            detections=[detection(detection_id="det-000000", score=0.8)]
        )
        observed = schema.validate_document([first, second])
        self.assertEqual(len(observed), 2)
        self.assertEqual(
            observed[0]["detections"][0]["detection_id"],
            observed[1]["detections"][0]["detection_id"],
        )

    def test_duplicate_frame_identity_is_rejected(self) -> None:
        duplicate = copy.deepcopy(record())
        duplicate["frame"]["decision_time_ns"] = 1_100
        duplicate["frame"]["vehicle_capture_time_ns"] = 1_100
        self.assert_document_invalid([record(), duplicate], "duplicate frame identity")

    def test_source_dataset_and_split_mixing_are_rejected(self) -> None:
        mixed_source = second_record()
        mixed_source["source"]["checkpoint_sha256"] = digest("a")
        self.assert_document_invalid([record(), mixed_source], "source identities")
        mixed_release = second_record()
        mixed_release["dataset"]["release_id"] = "other-release"
        self.assert_document_invalid(
            [record(), mixed_release], "dataset release or split"
        )
        mixed_split = second_record()
        mixed_split["dataset"]["split"] = "test"
        self.assert_document_invalid(
            [record(), mixed_split], "dataset release or split"
        )

    def test_document_order_and_strict_sequence_time_are_enforced(self) -> None:
        self.assert_document_invalid(
            [second_record(), record()], "strictly increasing|canonical document order"
        )
        same_time = second_record(
            vehicle_capture_time_ns=1_000,
            infrastructure_capture_time_ns=900,
            decision_time_ns=1_000,
        )
        self.assert_document_invalid([record(), same_time], "strictly increasing")

    def test_document_requires_a_nonempty_sequence(self) -> None:
        for invalid in ([], (), "not-records", b"not-records", {}):
            with self.subTest(invalid=invalid):
                self.assert_document_invalid(invalid, "sequence|at least one")

    def test_document_digest_is_stable_and_content_sensitive(self) -> None:
        first = record(detections=[detection()])
        second = second_record(detections=[detection(score=0.8)])
        original = schema.document_sha256([first, second])
        again = schema.document_sha256([dict(reversed(list(first.items()))), second])
        changed_second = copy.deepcopy(second)
        changed_second["detections"][0]["score"] = 0.7
        changed = schema.document_sha256([first, changed_second])
        self.assertEqual(original, again)
        self.assertNotEqual(original, changed)

    def test_parser_rejects_duplicate_json_keys(self) -> None:
        payload = schema.canonical_record_bytes(record()).decode("utf-8")
        payload = payload.replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
            1,
        )
        with self.assertRaisesRegex(
            schema.FixedDetectionContractError, "duplicate JSON key"
        ):
            schema.loads_jsonl(payload)

    def test_parser_rejects_noncanonical_document_framing(self) -> None:
        canonical = schema.canonical_record_bytes(record())
        invalid_payloads = (
            canonical[:-1],
            canonical.replace(b"\n", b"\r\n"),
            canonical + b"\n",
            b"\xef\xbb\xbf" + canonical,
            b"[]\n",
            b"\n",
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(schema.FixedDetectionContractError):
                    schema.loads_jsonl(payload)

    def test_parser_rejects_valid_but_noncanonical_json_spelling(self) -> None:
        noncanonical = json.dumps(record(), ensure_ascii=False) + "\n"
        with self.assertRaisesRegex(
            schema.FixedDetectionContractError, "not canonically serialized"
        ):
            schema.loads_jsonl(noncanonical)


if __name__ == "__main__":
    unittest.main()
