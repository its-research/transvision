from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "rtp_v2x"))

PROTOCOL_PATH = (
    ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "communication-measurement-v1.json"
)

import communication_meter as meter  # noqa: E402


class StepClock:
    def __init__(self, step_ns: int = 1_000_000) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        result = self.value
        self.value += self.step_ns
        return result


def diagnostic_report(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "frames": [
            meter.Frame("frame-001", {"query": [1.0, 2.0]}),
            meter.Frame("frame-002", {"query": [3.0]}),
        ],
        "encoder": meter.canonical_json_encoder,
        "transport": lambda value: value,
        "decoder": meter.canonical_json_decoder,
        "verifier": lambda expected, observed: expected == observed,
        "warmup_repetitions": 1,
        "repetitions": 2,
        "clock_ns": StepClock(),
    }
    arguments.update(overrides)
    return meter.measure_frames(**arguments)  # type: ignore[arg-type]


class CommunicationMeterTests(unittest.TestCase):
    def test_machine_readable_protocol_matches_implementation(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["schema_version"], meter.SCHEMA_VERSION)
        self.assertEqual(protocol["protocol_id"], meter.PROTOCOL_ID)
        self.assertEqual(protocol["wire_scope"], meter.WIRE_SCOPE)
        self.assertEqual(protocol["real_pipeline_scope"], meter.REAL_PIPELINE_SCOPE)
        self.assertFalse(protocol["scientific_claim_allowed"])
        self.assertEqual(
            set(protocol["future_candidate_real_context"]), meter.REAL_CONTEXT_FIELDS
        )

    def test_measures_actual_wire_bytes_and_frozen_aggregates(self) -> None:
        report = diagnostic_report()
        self.assertEqual(report["protocol_id"], meter.PROTOCOL_ID)
        self.assertFalse(report["scientific_claim_allowed"])
        self.assertEqual(report["measurement"]["sample_count"], 4)  # type: ignore[index]
        samples = report["samples"]  # type: ignore[assignment]
        expected_sizes = {
            "frame-001": len(b'{"query":[1.0,2.0]}'),
            "frame-002": len(b'{"query":[3.0]}'),
        }
        for sample in samples:  # type: ignore[union-attr]
            self.assertEqual(sample["wire_bytes"], expected_sizes[sample["frame_id"]])
            self.assertEqual(sample["encode_ns"], 1_000_000)
            self.assertEqual(sample["decode_ns"], 1_000_000)
            self.assertEqual(sample["consumer_ns"], 1_000_000)
            self.assertEqual(sample["e2e_ns"], 7_000_000)
        metrics = report["metrics"]  # type: ignore[assignment]
        self.assertEqual(metrics["communication_encode_ms"], 1.0)  # type: ignore[index]
        self.assertEqual(metrics["communication_decode_ms"], 1.0)  # type: ignore[index]
        self.assertEqual(metrics["communication_e2e_p95_ms"], 7.0)  # type: ignore[index]
        self.assertEqual(
            metrics["communication_bytes_per_frame"],  # type: ignore[index]
            sum(expected_sizes.values()) / 2,
        )

    def test_warmup_is_executed_but_not_reported(self) -> None:
        calls = 0

        def encode(value: object) -> bytes:
            nonlocal calls
            calls += 1
            return meter.canonical_json_encoder(value)

        report = diagnostic_report(encoder=encode, warmup_repetitions=3, repetitions=4)
        self.assertEqual(calls, 2 * (3 + 4))
        self.assertEqual(report["measurement"]["sample_count"], 8)  # type: ignore[index]

    def test_duplicate_or_missing_frame_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(meter.MeasurementError, "duplicate frame_id"):
            diagnostic_report(
                frames=[meter.Frame("same", {}), meter.Frame("same", {})]
            )
        with self.assertRaisesRegex(meter.MeasurementError, "frame_id"):
            diagnostic_report(frames=[meter.Frame("", {})])

    def test_nonfinite_and_unsupported_payloads_are_rejected(self) -> None:
        for payload in ({"x": math.nan}, {"x": math.inf}, {"x": object()}):
            with self.subTest(payload=payload):
                with self.assertRaises(meter.MeasurementError):
                    diagnostic_report(frames=[meter.Frame("frame", payload)])

    def test_wire_mutation_and_empty_message_fail_closed(self) -> None:
        with self.assertRaisesRegex(meter.MeasurementError, "wire integrity"):
            diagnostic_report(transport=lambda value: value + b"x")
        with self.assertRaisesRegex(meter.MeasurementError, "empty wire"):
            diagnostic_report(encoder=lambda value: b"")

    def test_backward_or_invalid_clock_is_rejected(self) -> None:
        values = iter([10, 20, 19, 30, 40, 50, 60, 70])
        with self.assertRaisesRegex(meter.MeasurementError, "clock moved backwards"):
            diagnostic_report(clock_ns=lambda: next(values))
        with self.assertRaisesRegex(meter.MeasurementError, "non-negative integer"):
            diagnostic_report(clock_ns=lambda: -1)

    def test_cross_stage_clock_rollback_is_rejected(self) -> None:
        timestamps = iter([100, 200, 300, 50, 100, 150, 200, 400])
        with self.assertRaisesRegex(meter.MeasurementError, "across measurement stages"):
            diagnostic_report(
                frames=[meter.Frame("frame", {"x": 1})],
                warmup_repetitions=0,
                repetitions=1,
                clock_ns=lambda: next(timestamps),
            )

    def test_encoder_cannot_mutate_payload_during_warmup(self) -> None:
        payload = {"x": [1, 2, 3]}

        def mutating_encoder(value: object) -> bytes:
            value["x"].pop()  # type: ignore[index,union-attr]
            return meter.canonical_json_encoder(value)

        with self.assertRaisesRegex(meter.MeasurementError, "mutated payload"):
            diagnostic_report(
                frames=[meter.Frame("frame", payload)],
                encoder=mutating_encoder,
                warmup_repetitions=1,
                repetitions=1,
            )

    def test_report_binds_payload_and_wire_hashes(self) -> None:
        report = diagnostic_report(warmup_repetitions=0, repetitions=1)
        self.assertEqual(len(report["frame_manifest"]), 2)  # type: ignore[arg-type]
        for entry in report["frame_manifest"]:  # type: ignore[union-attr]
            self.assertRegex(entry["payload_sha256"], r"^[0-9a-f]{64}$")
        for sample in report["samples"]:  # type: ignore[union-attr]
            self.assertRegex(sample["wire_sha256"], r"^[0-9a-f]{64}$")

    def test_candidate_real_is_disabled_until_audited_runner_exists(self) -> None:
        sha = "a" * 64
        context = {
            "code_commit": "b" * 40,
            "config_sha256": sha,
            "dataset_identity_sha256": sha,
            "pipeline_scope": meter.REAL_PIPELINE_SCOPE,
            "split_sha256": sha,
            "transport_id": "clearml-worker-channel-v1",
            "wire_scope": meter.WIRE_SCOPE,
        }
        with self.assertRaisesRegex(meter.MeasurementError, "is disabled"):
            diagnostic_report(
                evidence_tier="candidate_real",
                context=context,
                consumer=lambda value: value,
                warmup_repetitions=0,
            )

        bad = dict(context)
        bad["transport_id"] = "loopback"
        with self.assertRaisesRegex(meter.MeasurementError, "cannot be loopback"):
            diagnostic_report(
                evidence_tier="candidate_real",
                context=bad,
                consumer=lambda value: value,
                warmup_repetitions=0,
            )

    def test_evidence_is_self_hashed_and_never_overwritten(self) -> None:
        report = diagnostic_report(warmup_repetitions=0, repetitions=1)
        expected = hashlib.sha256(meter.canonical_bytes(report)).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "communication.json"
            observed = meter.write_evidence(output, report)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(observed, expected)
            self.assertEqual(document["report_sha256"], expected)
            with self.assertRaisesRegex(meter.MeasurementError, "already exists"):
                meter.write_evidence(output, report)

    def test_evidence_recomputes_metrics_and_rejects_claim_escalation(self) -> None:
        report = diagnostic_report(warmup_repetitions=0, repetitions=1)
        report["metrics"]["communication_e2e_p95_ms"] = 0.0  # type: ignore[index]
        with self.assertRaisesRegex(meter.MeasurementError, "does not match raw samples"):
            meter.evidence_document(report)

        report = diagnostic_report(warmup_repetitions=0, repetitions=1)
        report["scientific_claim_allowed"] = True
        with self.assertRaisesRegex(meter.MeasurementError, "cannot grant"):
            meter.evidence_document(report)

    def test_evidence_rejects_duplicate_sample_pair(self) -> None:
        report = diagnostic_report(warmup_repetitions=0, repetitions=2)
        report["samples"][1] = dict(report["samples"][0])  # type: ignore[index]
        with self.assertRaisesRegex(meter.MeasurementError, "duplicate measured"):
            meter.evidence_document(report)


if __name__ == "__main__":
    unittest.main()
