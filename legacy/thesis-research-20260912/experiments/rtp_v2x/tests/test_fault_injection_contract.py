#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = ROOT / "experiments" / "rtp_v2x"
sys.path.insert(0, str(MODULE_ROOT))

import fault_injection_contract as contract  # noqa: E402


CONTRACT_PATH = (
    ROOT / "experiments" / "clearml" / "protocols" / "fault-injection-v1.json"
)
TRACK_PROTOCOL = (
    ROOT / "experiments" / "clearml" / "protocols" / "v2xseq-track-v1.json"
)
FORECAST_PROTOCOL = (
    ROOT / "experiments" / "clearml" / "protocols" / "v2xseq-forecast-v1.json"
)
EXPECTED_FILE_SHA256 = (
    "db2f6d30fb57c011fee75d6dcf7fe99c1aff0ba5fe0160fecafb2e44b3a1a4bb"
)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("fixture root must be an object")
    return value


class FaultInjectionContractTest(unittest.TestCase):
    def test_production_contract_is_exact_and_frozen(self) -> None:
        document = contract.load_contract(CONTRACT_PATH)
        self.assertEqual(document["contract_id"], contract.CONTRACT_ID)
        self.assertEqual(document["status"], "frozen")
        self.assertEqual(document["fixed_base_seeds"], [3407, 4909, 6203])
        self.assertEqual(
            [row["id"] for row in document["conditions"]],
            list(contract.CONDITION_IDS),
        )
        self.assertEqual(contract.sha256_file(CONTRACT_PATH), EXPECTED_FILE_SHA256)
        self.assertEqual(
            contract.sha256_bytes(contract.canonical_json_bytes(document)),
            contract.EXPECTED_CANONICAL_SHA256,
        )

    def test_six_conditions_fix_required_parameters_and_application_points(self) -> None:
        document = contract.load_contract(CONTRACT_PATH)
        conditions = {row["id"]: row for row in document["conditions"]}

        self.assertEqual(
            conditions["latency_300ms"]["parameters"], {"delay_ns": 300_000_000}
        )
        self.assertEqual(
            conditions["independent_packet_loss_0p3"]["parameters"],
            {"drop_probability": 0.3},
        )
        self.assertEqual(
            conditions["gilbert_elliott_burst_loss"]["parameters"],
            {
                "initial_state": "good",
                "p_good_to_bad": 0.02,
                "p_bad_to_good": 0.2,
                "drop_probability_good": 0.02,
                "drop_probability_bad": 0.8,
                "drop_draw_precedes_transition_draw": True,
                "state_reset_scope": ["run_id", "scope_id", "source_id"],
                "emission_order": ["event_time_ns", "source_id", "message_id"],
            },
        )
        pose = conditions["pose_noise_0p5m_0p5deg"]
        self.assertEqual(pose["parameters"]["translation_x_sigma_m"], 0.5)
        self.assertEqual(pose["parameters"]["translation_y_sigma_m"], 0.5)
        self.assertEqual(pose["parameters"]["translation_z_m"], 0.0)
        self.assertEqual(pose["parameters"]["yaw_sigma_deg"], 0.5)
        self.assertEqual(
            pose["application"]["point"],
            "after_decode_before_transform_to_decision_frame",
        )
        retention = conditions["observation_retention_0p5"]
        self.assertEqual(retention["parameters"], {"retain_probability": 0.5})
        self.assertEqual(
            set(retention["application"]), {"tracking", "forecasting"}
        )

    def test_order_duplication_and_clock_assumptions_are_closed(self) -> None:
        document = contract.load_contract(CONTRACT_PATH)
        time = document["time_semantics"]
        self.assertIs(time["out_of_order_injection"], False)
        self.assertEqual(time["duplicate_message_rate"], 0.0)
        self.assertEqual(time["clock_offset_ns"], 0)
        self.assertEqual(
            time["consumer_order"],
            ["arrival_time_ns", "event_time_ns", "source_id", "message_id"],
        )
        self.assertEqual(
            time["decision_visibility"], "arrival_time_ns <= decision_time_ns"
        )

    def test_nominal_reference_is_literal_byte_identity(self) -> None:
        payload = b'{"z":1,"a":[3,2,1]}\n\x00binary-tail'
        before = hashlib.sha256(payload).hexdigest()
        output = contract.nominal_identity(payload)
        self.assertIs(output, payload)
        self.assertEqual(output, payload)
        self.assertEqual(hashlib.sha256(output).hexdigest(), before)
        with self.assertRaisesRegex(contract.FaultContractError, "immutable bytes"):
            contract.nominal_identity(bytearray(payload))  # type: ignore[arg-type]

    def test_seed_derivation_and_counter_rng_have_fixed_vectors(self) -> None:
        document = contract.load_contract(CONTRACT_PATH)
        self.assertEqual(
            document["seed_derivation"]["preimage"],
            "domain_utf8 || 0x00 || canonical_json_array",
        )
        self.assertEqual(
            document["rng"]["decision_counter_policy"],
            {
                "each_item_and_stream_has_independent_seed_digest": True,
                "bernoulli_counter": 0,
                "normal_start_counter": 0,
            },
        )
        seed_digest = contract.derive_seed_digest(
            base_seed=3407,
            condition_id="independent_packet_loss_0p3",
            scope_id="scene-001",
            source_id="infrastructure-01",
            item_id="message-000042",
            stream_name="independent_packet_drop",
        )
        self.assertEqual(
            seed_digest.hex(),
            "0f5d311268ae2c5417e7d59abb78b50fe95231a729a1d294358ea03fdca0df43",
        )
        self.assertEqual(contract.seed_u64(seed_digest), 1_107_095_038_538_427_476)
        self.assertEqual(
            contract.uniform_open_01(seed_digest, 0).hex(), "0x1.f66175d35254bp-2"
        )
        self.assertEqual(
            contract.uniform_open_01(seed_digest, 1).hex(), "0x1.fc2c3a0b9d156p-1"
        )
        self.assertTrue(contract.bernoulli(seed_digest, 0, 0.5))
        self.assertFalse(contract.bernoulli(seed_digest, 1, 0.5))
        self.assertEqual(
            contract.normal_standard(seed_digest, 0).hex(),
            "0x1.312d10256b6d9p+0",
        )
        self.assertEqual(
            contract.normal_standard(seed_digest, 0),
            contract.normal_standard(seed_digest, 0),
        )
        with self.assertRaisesRegex(contract.FaultContractError, "leave room"):
            contract.normal_standard(seed_digest, 2**64 - 1)
        self.assertTrue(
            math.isfinite(contract.normal_standard(seed_digest, 2**64 - 2))
        )

    def test_keyed_decisions_do_not_depend_on_iteration_order(self) -> None:
        def derive(item_id: str) -> bytes:
            return contract.derive_seed_digest(
                base_seed=4909,
                condition_id="observation_retention_0p5",
                scope_id="sequence-9",
                source_id="roadside",
                item_id=item_id,
                stream_name="tracking_observation_retention",
            )

        forward = {item: derive(item) for item in ("point-1", "point-2", "point-3")}
        reverse = {item: derive(item) for item in ("point-3", "point-2", "point-1")}
        self.assertEqual(forward, reverse)
        self.assertEqual(len(set(forward.values())), 3)

    def test_semantic_change_requires_a_new_contract_version(self) -> None:
        document = contract.load_contract(CONTRACT_PATH)
        mutated = copy.deepcopy(document)
        mutated["conditions"][2]["parameters"]["drop_probability"] = 0.31
        with self.assertRaisesRegex(
            contract.FaultContractError, "create a new version"
        ):
            contract.validate_contract(mutated)

        mutated = copy.deepcopy(document)
        mutated["time_semantics"]["out_of_order_injection"] = True
        with self.assertRaisesRegex(
            contract.FaultContractError, "create a new version"
        ):
            contract.validate_contract(mutated)

    def test_tracking_and_forecasting_protocols_pin_the_same_contract(self) -> None:
        for path in (TRACK_PROTOCOL, FORECAST_PROTOCOL):
            with self.subTest(path=path.name):
                protocol = load_json(path)
                resolved = contract.validate_protocol_reference(protocol, ROOT)
                self.assertEqual(resolved["contract_id"], contract.CONTRACT_ID)
                self.assertEqual(
                    protocol["fault_grid"]["condition_ids"],
                    list(contract.CONDITION_IDS),
                )

    def test_protocol_reference_rejects_hash_or_condition_drift(self) -> None:
        protocol = load_json(TRACK_PROTOCOL)
        protocol["fault_grid"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(contract.FaultContractError, "does not match"):
            contract.validate_protocol_reference(protocol, ROOT)

        protocol = load_json(TRACK_PROTOCOL)
        protocol["fault_grid"]["condition_ids"] = ["nominal"]
        with self.assertRaisesRegex(contract.FaultContractError, "condition ids"):
            contract.validate_protocol_reference(protocol, ROOT)

    def test_strict_loader_rejects_duplicates_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            duplicate = Path(temporary_directory) / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(contract.FaultContractError, "duplicate"):
                contract.load_contract(duplicate)

            nonfinite = Path(temporary_directory) / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(contract.FaultContractError, "non-finite"):
                contract.load_contract(nonfinite)

    def test_seed_inputs_fail_closed(self) -> None:
        valid = {
            "base_seed": 6203,
            "condition_id": "nominal",
            "scope_id": "scope",
            "source_id": "source",
            "item_id": "item",
            "stream_name": "stream",
        }
        for field, value, message in (
            ("base_seed", 1, "fixed seeds"),
            ("condition_id", "unknown", "not in"),
            ("scope_id", "bad\nvalue", "control character"),
        ):
            with self.subTest(field=field):
                arguments = dict(valid)
                arguments[field] = value
                with self.assertRaisesRegex(contract.FaultContractError, message):
                    contract.derive_seed_digest(**arguments)


if __name__ == "__main__":
    unittest.main()
