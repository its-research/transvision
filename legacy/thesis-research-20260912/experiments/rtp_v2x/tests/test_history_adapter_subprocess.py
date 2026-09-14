#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import history_adapter_subprocess as history_process  # noqa: E402


WORKING_ADAPTER = r"""
import os
import sys

class Adapter:
    def __init__(self, context):
        self.context = context

    def runtime_environment(self):
        print("TOKEN=adapter-stderr-value", file=sys.stderr)
        return {
            "framework": "fixture",
            "framework_version": "1",
            "device_name": "NVIDIA A100 fixture",
        }

    def forward(self, sample, training):
        if training:
            assert "ground_truth" in sample
            return {"loss_tensor": "opaque-to-protocol"}
        assert set(sample) == {"scene_id", "target_id", "history", "input_sha256"}
        assert "ground_truth" not in sample
        assert "future" not in sample
        return {
            "scene_id": sample["scene_id"],
            "target_id": sample["target_id"],
            "modes": [
                {
                    "mode_id": "m0",
                    "probability": 1.0,
                    "trajectory": [[1.0, 2.0], [2.0, 3.0]],
                }
            ],
        }

    def backward(self, output):
        assert output["loss_tensor"] == "opaque-to-protocol"
        return {"loss": 1.0, "gradient_norm": 0.5, "backward_completed": True}

    def optimizer_step(self):
        return {"parameter_update_norm": 0.1, "optimizer_step_completed": True}

    def evaluate(self, predictions, targets):
        assert len(predictions) == len(targets)
        assert all("ground_truth" in target for target in targets)
        return {"sample_count": len(predictions), "minADE": 0.0}

    def save_checkpoint(self, path):
        with open(path, "wb") as stream:
            stream.write(b"history-only-checkpoint")

def build_adapter(context):
    return Adapter(context)
""".lstrip()


def history_sample(scene_id: str) -> dict[str, object]:
    history = {
        "decision_time_ns": 10,
        "observations": [
            {
                "actor_id": "actor-1",
                "event_time_ns": 9,
                "arrival_time_ns": 10,
                "position": [1.0, 2.0],
            }
        ],
    }
    return {
        "scene_id": scene_id,
        "target_id": "target-1",
        "history": history,
        "input_sha256": history_process.history_input_sha256(history),
    }


def training_sample() -> dict[str, object]:
    return {
        **history_sample("train-scene"),
        "ground_truth": {
            "timestamps_ns": [11, 12],
            "positions": [[1.0, 2.0], [2.0, 3.0]],
        },
    }


def evaluation_target(scene_id: str) -> dict[str, object]:
    sample = history_sample(scene_id)
    return {
        "scene_id": sample["scene_id"],
        "target_id": sample["target_id"],
        "ground_truth": {
            "timestamps_ns": [11, 12],
            "positions": [[1.0, 2.0], [2.0, 3.0]],
        },
        "input_sha256": sample["input_sha256"],
    }


class HistoryOnlyAdapterProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter.py"
        self.adapter.write_text(WORKING_ADAPTER, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def session(
        self,
        *,
        expected: int = 2,
        timeout: float = 3.0,
        adapter: Path | None = None,
        context: dict[str, object] | None = None,
    ) -> history_process.HistoryOnlyAdapterProcess:
        return history_process.HistoryOnlyAdapterProcess(
            adapter_path=adapter or self.adapter,
            factory="build_adapter",
            context=context
            or {
                "protocol": {"protocol_id": "V2XSEQ-FORECAST-v1"},
                "claim_scope": "diagnostic_only",
                "scientific_claim_allowed": False,
            },
            expected_inference_count=expected,
            timeout_seconds=timeout,
            source_environment={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": "/unsealed/ambient/path",
                "CUDA_VISIBLE_DEVICES": "0",
                "CLEARML_API_SECRET_KEY": "clearml-secret-value",
                "MY_TOKEN": "parent-token-value",
                "CUSTOM_UNSEALED": "not-forwarded",
            },
        )

    def prepare_for_inference(
        self, session: history_process.HistoryOnlyAdapterProcess
    ) -> dict[str, object]:
        runtime = session.start().runtime_environment()
        session.train(training_sample())
        return runtime

    def test_complete_persistent_flow_binds_hashes_and_checkpoint(self) -> None:
        session = self.session()
        runtime = self.prepare_for_inference(session)
        worker = runtime["worker"]
        self.assertEqual(worker["environment_keys"], sorted(worker["environment_keys"]))
        self.assertIn("CUDA_VISIBLE_DEVICES", worker["environment_keys"])
        self.assertNotIn("PYTHONPATH", worker["environment_keys"])
        self.assertFalse(any("CLEARML" in key for key in worker["environment_keys"]))
        self.assertNotIn("python_executable", worker)
        self.assertRegex(worker["python_executable_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(worker["python_cache_tag"], sys.implementation.cache_tag)
        self.assertTrue(worker["temporary_cwd"])

        first = session.infer(history_sample("validation-a"))
        second = session.infer(history_sample("validation-b"))
        self.assertEqual((first.index, second.index), (0, 1))
        self.assertEqual(
            first.prediction_sha256,
            history_process.canonical_json_line_sha256(first.prediction),
        )
        self.assertEqual(session.state, history_process.SessionState.INFERENCE_COMPLETE)

        evaluation = session.evaluate(
            [evaluation_target("validation-a"), evaluation_target("validation-b")]
        )
        self.assertEqual(evaluation.metrics["sample_count"], 2)
        checkpoint_path = self.root / "checkpoint.bin"
        checkpoint = session.save_checkpoint(checkpoint_path)
        self.assertEqual(checkpoint.path, checkpoint_path.resolve())
        self.assertEqual(
            checkpoint.sha256, history_process.sha256_file(checkpoint_path)
        )
        self.assertGreater(checkpoint.size_bytes, 0)
        session.close()

        self.assertEqual(session.state, history_process.SessionState.CLOSED)
        self.assertEqual(
            [record["op"] for record in session.transcript_records],
            [
                "load",
                "runtime_environment",
                "train",
                "infer",
                "infer",
                "evaluate",
                "save_checkpoint",
                "close",
            ],
        )
        self.assertRegex(session.transcript_sha256, r"^[0-9a-f]{64}$")
        self.assertNotEqual(session.transcript_sha256, "0" * 64)
        self.assertGreater(session.stderr_summary["byte_count"], 0)
        self.assertNotIn("adapter-stderr-value", session.redacted_stderr)
        self.assertIn("<redacted>", session.redacted_stderr)

    def test_inference_rejects_future_fields_before_transmission(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        sample = history_sample("validation-a")
        sample["future"] = {"positions": [[99.0, 99.0]]}
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "future/target"
        ):
            session.infer(sample)
        self.assertEqual(session.state, history_process.SessionState.FAILED)
        self.assertNotIn(
            "infer", [record["op"] for record in session.transcript_records]
        )

    def test_nested_ground_truth_is_rejected_from_inference(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        sample = history_sample("validation-a")
        sample["history"]["metadata"] = {"ground_truth": [[9.0, 9.0]]}  # type: ignore[index]
        sample["input_sha256"] = history_process.history_input_sha256(sample["history"])
        with self.assertRaises(history_process.ProtocolValidationError):
            session.infer(sample)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_disguised_target_field_and_host_path_are_rejected(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        sample = history_sample("validation-a")
        sample["history"]["metadata"] = {  # type: ignore[index]
            "target_velocity": [1.0, 0.0]
        }
        sample["input_sha256"] = history_process.history_input_sha256(sample["history"])
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "future/target"
        ):
            session.infer(sample)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "filesystem path"
        ):
            self.session(context={"neutral_name": "/data/hidden-root"})

    def test_evaluate_is_rejected_until_every_prediction_is_confirmed(self) -> None:
        session = self.session(expected=2)
        self.prepare_for_inference(session)
        session.infer(history_sample("validation-a"))
        before = tuple(session.transcript_records)
        with self.assertRaisesRegex(
            history_process.ProtocolStateError, "requires state"
        ):
            session.evaluate(
                [evaluation_target("validation-a"), evaluation_target("validation-b")]
            )
        self.assertEqual(tuple(session.transcript_records), before)
        self.assertEqual(session.state, history_process.SessionState.INFERENCING)
        session.abort()

    def test_evaluation_target_order_is_checked_before_transmission(self) -> None:
        session = self.session(expected=2)
        self.prepare_for_inference(session)
        session.infer(history_sample("validation-a"))
        session.infer(history_sample("validation-b"))
        before = tuple(session.transcript_records)
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "order or identity"
        ):
            session.evaluate(
                [evaluation_target("validation-b"), evaluation_target("validation-a")]
            )
        self.assertEqual(tuple(session.transcript_records), before)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_worker_rejects_out_of_order_inference_index(self) -> None:
        session = self.session(expected=2)
        self.prepare_for_inference(session)
        with self.assertRaisesRegex(
            history_process.RemoteAdapterError, "worker_contract_error"
        ):
            session._request(  # pylint: disable=protected-access
                "infer", {"index": 1, "sample": history_sample("validation-a")}
            )
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_parent_rejects_tampered_prediction_digest(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        sample = history_sample("validation-a")
        prediction = {
            "scene_id": "validation-a",
            "target_id": "target-1",
            "modes": [],
        }
        fake_response = {
            "index": 0,
            "scene_id": "validation-a",
            "target_id": "target-1",
            "input_sha256": sample["input_sha256"],
            "prediction": prediction,
            "prediction_sha256": "0" * 64,
        }
        with mock.patch.object(
            session,
            "_request",
            return_value=(fake_response, history_process.sha256_json(fake_response)),
        ):
            with self.assertRaisesRegex(
                history_process.ProtocolValidationError, "Prediction SHA|prediction SHA"
            ):
                session.infer(sample)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_parent_rejects_tampered_prediction_order(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        sample = history_sample("validation-a")
        prediction = {"scene_id": "validation-a", "target_id": "target-1", "modes": []}
        fake_response = {
            "index": 1,
            "scene_id": "validation-a",
            "target_id": "target-1",
            "input_sha256": sample["input_sha256"],
            "prediction": prediction,
            "prediction_sha256": history_process.canonical_json_line_sha256(prediction),
        }
        with mock.patch.object(
            session,
            "_request",
            return_value=(fake_response, history_process.sha256_json(fake_response)),
        ):
            with self.assertRaisesRegex(
                history_process.ProtocolValidationError, "out of order"
            ):
                session.infer(sample)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_post_confirmation_prediction_mutation_blocks_targets(self) -> None:
        session = self.session(expected=1)
        self.prepare_for_inference(session)
        prediction = session.infer(history_sample("validation-a"))
        prediction.prediction["modes"] = []
        before = tuple(session.transcript_records)
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "mutated before evaluation"
        ):
            session.evaluate([evaluation_target("validation-a")])
        self.assertEqual(tuple(session.transcript_records), before)
        self.assertNotIn("evaluate", [record["op"] for record in before])
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_adapter_stdout_print_is_protocol_pollution(self) -> None:
        noisy = self.root / "noisy.py"
        noisy.write_text("print('adapter-noise')\n" + WORKING_ADAPTER, encoding="utf-8")
        session = self.session(expected=1, adapter=noisy)
        with self.assertRaises(history_process.ProtocolPollutionError):
            session.start()
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_adapter_forward_stdout_print_is_protocol_pollution(self) -> None:
        noisy = self.root / "noisy-forward.py"
        noisy.write_text(
            WORKING_ADAPTER.replace(
                'return {"loss_tensor": "opaque-to-protocol"}',
                'print("forward-noise")\n            return {"loss_tensor": "opaque-to-protocol"}',
            ),
            encoding="utf-8",
        )
        session = self.session(expected=1, adapter=noisy)
        session.start().runtime_environment()
        with self.assertRaises(history_process.ProtocolPollutionError):
            session.train(training_sample())
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_adapter_atexit_stdout_is_final_protocol_pollution(self) -> None:
        noisy = self.root / "noisy-atexit.py"
        noisy.write_text(
            "import atexit\n"
            "atexit.register(lambda: print('atexit-noise'))\n" + WORKING_ADAPTER,
            encoding="utf-8",
        )
        session = self.session(expected=1, adapter=noisy)
        self.prepare_for_inference(session)
        session.infer(history_sample("validation-a"))
        session.evaluate([evaluation_target("validation-a")])
        session.save_checkpoint(self.root / "atexit-checkpoint.bin")
        with self.assertRaises(history_process.ProtocolPollutionError):
            session.close()
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_adapter_exception_is_sanitized_and_fails_session(self) -> None:
        failing = self.root / "failing.py"
        failing.write_text(
            WORKING_ADAPTER.replace(
                'return {"loss_tensor": "opaque-to-protocol"}',
                'raise RuntimeError("TOKEN=do-not-echo-this")',
            ),
            encoding="utf-8",
        )
        session = self.session(expected=1, adapter=failing)
        session.start().runtime_environment()
        with self.assertRaises(history_process.RemoteAdapterError) as captured:
            session.train(training_sample())
        self.assertNotIn("do-not-echo-this", str(captured.exception))
        self.assertNotIn("do-not-echo-this", session.redacted_stderr)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_adapter_timeout_terminates_worker(self) -> None:
        sleeping = self.root / "sleeping.py"
        sleeping.write_text(
            "import time\n"
            + WORKING_ADAPTER.replace(
                'return {"loss_tensor": "opaque-to-protocol"}',
                'time.sleep(2.0)\n            return {"loss_tensor": "opaque-to-protocol"}',
            ),
            encoding="utf-8",
        )
        session = self.session(expected=1, timeout=0.2, adapter=sleeping)
        session.start().runtime_environment()
        started = time.monotonic()
        with self.assertRaises(history_process.WorkerTimeoutError):
            session.train(training_sample())
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_nonzero_worker_exit_is_fatal(self) -> None:
        exiting = self.root / "exiting.py"
        exiting.write_text(
            WORKING_ADAPTER.replace(
                'return {"loss_tensor": "opaque-to-protocol"}',
                "os._exit(19)",
            ),
            encoding="utf-8",
        )
        session = self.session(expected=1, adapter=exiting)
        session.start().runtime_environment()
        with self.assertRaisesRegex(history_process.WorkerExitError, "19"):
            session.train(training_sample())
        self.assertEqual(session.state, history_process.SessionState.FAILED)

    def test_state_machine_rejects_skipped_phases_and_early_close(self) -> None:
        session = self.session(expected=1).start()
        with self.assertRaises(history_process.ProtocolStateError):
            session.train(training_sample())
        with self.assertRaises(history_process.ProtocolStateError):
            session.infer(history_sample("validation-a"))
        with self.assertRaises(history_process.ProtocolStateError):
            session.close()
        self.assertEqual(session.state, history_process.SessionState.LOADED)
        session.abort()

    def test_dataset_root_and_sensitive_context_are_never_loaded(self) -> None:
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "dataset-root"
        ):
            self.session(context={"dataset_root": "/data/private"})
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "sensitive"
        ):
            self.session(context={"api_key": "not-allowed"})

    def test_environment_builder_strips_ambient_pythonpath_and_secret_names(
        self,
    ) -> None:
        environment, secrets = history_process.build_worker_environment(
            {
                "PATH": "/usr/bin",
                "PYTHONPATH": "/unsealed",
                "CUDA_VISIBLE_DEVICES": "4",
                "CLEARML_API_ACCESS_KEY": "access-value",
                "SERVICE_TOKEN": "token-value",
                "HOME": "/private/home",
            },
            self.root,
        )
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "4")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("HOME", environment)
        self.assertFalse(any("CLEARML" in key or "TOKEN" in key for key in environment))
        self.assertEqual(set(secrets), {"access-value", "token-value"})

        empty_environment, empty_secrets = history_process.build_worker_environment(
            {}, self.root
        )
        self.assertEqual(
            set(empty_environment),
            {"TMPDIR", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"},
        )
        self.assertEqual(empty_secrets, ())

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "duplicate"
        ):
            history_process.strict_json_loads('{"x":1,"x":2}')
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "non-finite"
        ):
            history_process.strict_json_loads('{"x":NaN}')
        with self.assertRaises(history_process.ProtocolValidationError):
            history_process.canonical_json_bytes({"x": float("inf")})

    def test_isolation_contract_is_hash_pinned(self) -> None:
        path = history_process.default_isolation_contract_path()
        digest = history_process.sha256_file(path)
        document = history_process.validate_isolation_contract(path, digest)
        self.assertEqual(document["contract_id"], history_process.ISOLATION_CONTRACT_ID)
        with self.assertRaisesRegex(
            history_process.ProtocolValidationError, "mismatch"
        ):
            history_process.validate_isolation_contract(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
