#!/usr/bin/env python3
"""Worker for the RTP-V2X history-only adapter JSON-line protocol.

Stdout is reserved for protocol envelopes.  Adapter output on stdout is not
redirected intentionally: accidental prints become detectable protocol
pollution in the parent.  Stderr remains a separately captured diagnostic
stream.  This worker enforces causal message separation, not hostile-code
containment from the host operating system.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import history_adapter_subprocess as protocol


class WorkerContractError(RuntimeError):
    pass


class WorkerState(str, Enum):
    NEW = "new"
    LOADED = "loaded"
    ENVIRONMENT_REPORTED = "environment_reported"
    TRAINED = "trained"
    INFERENCING = "inferencing"
    INFERENCE_COMPLETE = "inference_complete"
    EVALUATED = "evaluated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CLOSED = "closed"


class Worker:
    def __init__(self) -> None:
        self.state = WorkerState.NEW
        self.adapter: Any = None
        self.adapter_path: Path | None = None
        self.adapter_sha256 = ""
        self.parent_source_sha256 = ""
        self.worker_source_sha256 = ""
        self.isolation_contract_sha256 = ""
        self.context_sha256 = ""
        self.expected_inference_count = 0
        self.predictions: list[dict[str, Any]] = []
        self.confirmations: list[dict[str, Any]] = []

    def dispatch(self, op: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        handlers = {
            "load": self.load,
            "runtime_environment": self.runtime_environment,
            "train": self.train,
            "infer": self.infer,
            "evaluate": self.evaluate,
            "save_checkpoint": self.save_checkpoint,
            "close": self.close,
        }
        if op not in handlers:
            raise WorkerContractError("unsupported protocol operation")
        return handlers[op](payload)

    def _require_state(self, *allowed: WorkerState) -> None:
        if self.state not in allowed:
            raise WorkerContractError(
                "operation is not allowed in the current worker state"
            )

    def load(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.NEW)
        protocol.require_exact_fields(
            payload,
            {
                "adapter_path",
                "adapter_sha256",
                "factory",
                "context",
                "expected_inference_count",
                "isolation_contract_id",
                "isolation_contract_sha256",
                "parent_source_sha256",
                "worker_source_sha256",
            },
            "load payload",
        )
        if payload["isolation_contract_id"] != protocol.ISOLATION_CONTRACT_ID:
            raise WorkerContractError("isolation contract identity mismatch")
        adapter_hash = protocol.require_digest(
            payload["adapter_sha256"], "adapter_sha256"
        )
        parent_hash = protocol.require_digest(
            payload["parent_source_sha256"], "parent_source_sha256"
        )
        worker_hash = protocol.require_digest(
            payload["worker_source_sha256"], "worker_source_sha256"
        )
        isolation_hash = protocol.require_digest(
            payload["isolation_contract_sha256"], "isolation_contract_sha256"
        )
        if protocol.sha256_file(Path(protocol.__file__).resolve()) != parent_hash:
            raise WorkerContractError("parent source binding mismatch")
        if protocol.sha256_file(Path(__file__).resolve()) != worker_hash:
            raise WorkerContractError("worker source binding mismatch")
        adapter_path_value = payload["adapter_path"]
        if (
            not isinstance(adapter_path_value, str)
            or not Path(adapter_path_value).is_absolute()
        ):
            raise WorkerContractError("adapter path must be absolute")
        adapter_path = Path(adapter_path_value)
        if not adapter_path.is_file() or adapter_path.is_symlink():
            raise WorkerContractError("adapter path is not a non-symlink regular file")
        if protocol.sha256_file(adapter_path) != adapter_hash:
            raise WorkerContractError("adapter source binding mismatch")
        factory_name = payload["factory"]
        if (
            not isinstance(factory_name, str)
            or protocol._FACTORY_NAME.fullmatch(factory_name) is None
        ):
            raise WorkerContractError("adapter factory is invalid")
        expected_count = payload["expected_inference_count"]
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise WorkerContractError("expected inference count is invalid")
        context = protocol.require_object(payload["context"], "adapter context")
        protocol.reject_dataset_roots(context)
        protocol.reject_sensitive_keys(context, "adapter context")
        protocol.reject_host_paths(context, "adapter context")
        detached_context = protocol.strict_json_loads(
            protocol.canonical_json_bytes(context), "adapter context"
        )
        context_sha256 = protocol.sha256_json(detached_context)
        adapter_context = protocol.strict_json_loads(
            protocol.canonical_json_bytes(detached_context), "adapter context copy"
        )

        # The adapter directory is the only source path inserted by the worker.
        # A production caller must cover that directory with its package seal.
        adapter_directory = str(adapter_path.parent.resolve())
        if adapter_directory not in sys.path:
            sys.path.insert(0, adapter_directory)
        spec = importlib.util.spec_from_file_location(
            "rtpv2x_history_only_adapter", adapter_path
        )
        if spec is None or spec.loader is None:
            raise WorkerContractError("cannot construct adapter module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise WorkerContractError("adapter factory is not callable")
        adapter = factory(adapter_context)
        for method in (
            "forward",
            "backward",
            "optimizer_step",
            "evaluate",
            "runtime_environment",
            "save_checkpoint",
        ):
            if not callable(getattr(adapter, method, None)):
                raise WorkerContractError("adapter is missing a required method")

        self.adapter = adapter
        self.adapter_path = adapter_path.resolve()
        self.adapter_sha256 = adapter_hash
        self.parent_source_sha256 = parent_hash
        self.worker_source_sha256 = worker_hash
        self.isolation_contract_sha256 = isolation_hash
        self.context_sha256 = context_sha256
        self.expected_inference_count = expected_count
        self.state = WorkerState.LOADED
        return {
            "loaded": True,
            "adapter_sha256": self.adapter_sha256,
            "context_sha256": self.context_sha256,
            "expected_inference_count": self.expected_inference_count,
            "isolation_contract_sha256": self.isolation_contract_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "worker_source_sha256": self.worker_source_sha256,
        }

    def runtime_environment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.LOADED)
        protocol.require_exact_fields(payload, set(), "runtime_environment payload")
        raw_adapter_environment = self.adapter.runtime_environment()
        adapter_environment = protocol.require_object(
            protocol.strict_json_loads(
                protocol.canonical_json_bytes(raw_adapter_environment),
                "adapter runtime environment",
            ),
            "adapter runtime environment",
        )
        protocol.reject_sensitive_keys(
            adapter_environment, "adapter runtime environment"
        )
        executable = Path(sys.executable).resolve()
        temporary_cwd = (
            bool(os.environ.get("TMPDIR"))
            and Path(os.environ["TMPDIR"]).resolve() == Path.cwd().resolve()
        )
        worker_environment = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_cache_tag": sys.implementation.cache_tag,
            "python_executable_basename": executable.name,
            "python_executable_sha256": protocol.sha256_file(executable),
            "platform": platform.platform(),
            "environment_keys": sorted(os.environ),
            "adapter_sha256": self.adapter_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "worker_source_sha256": self.worker_source_sha256,
            "isolation_contract_sha256": self.isolation_contract_sha256,
            "context_sha256": self.context_sha256,
            "temporary_cwd": temporary_cwd,
        }
        self.state = WorkerState.ENVIRONMENT_REPORTED
        return {"adapter": adapter_environment, "worker": worker_environment}

    def train(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.ENVIRONMENT_REPORTED)
        protocol.require_exact_fields(payload, {"sample"}, "train payload")
        sample = protocol.validate_training_sample(payload["sample"])
        detached = protocol.strict_json_loads(
            protocol.canonical_json_bytes(sample), "training sample"
        )
        forward_output = self.adapter.forward(detached, training=True)
        backward = protocol.require_object(
            protocol.strict_json_loads(
                protocol.canonical_json_bytes(self.adapter.backward(forward_output)),
                "backward report",
            ),
            "backward report",
        )
        optimizer = protocol.require_object(
            protocol.strict_json_loads(
                protocol.canonical_json_bytes(self.adapter.optimizer_step()),
                "optimizer report",
            ),
            "optimizer report",
        )
        protocol.reject_sensitive_keys(backward, "backward report")
        protocol.reject_sensitive_keys(optimizer, "optimizer report")
        self.state = WorkerState.TRAINED
        return {"backward_report": backward, "optimizer_report": optimizer}

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.TRAINED, WorkerState.INFERENCING)
        protocol.require_exact_fields(payload, {"index", "sample"}, "infer payload")
        index = payload["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != len(self.predictions)
            or index >= self.expected_inference_count
        ):
            raise WorkerContractError("inference index is out of order")
        sample = protocol.validate_history_sample(payload["sample"])
        detached = protocol.strict_json_loads(
            protocol.canonical_json_bytes(sample), "inference sample"
        )
        prediction = protocol.require_object(
            protocol.strict_json_loads(
                protocol.canonical_json_bytes(
                    self.adapter.forward(detached, training=False)
                ),
                "adapter prediction",
            ),
            "adapter prediction",
        )
        protocol.reject_sensitive_keys(prediction, "adapter prediction")
        prediction_hash = protocol.canonical_json_line_sha256(prediction)
        response = {
            "index": index,
            "scene_id": sample["scene_id"],
            "target_id": sample["target_id"],
            "input_sha256": sample["input_sha256"],
            "prediction": prediction,
            "prediction_sha256": prediction_hash,
        }
        self.predictions.append(prediction)
        self.confirmations.append(
            {
                "index": index,
                "scene_id": sample["scene_id"],
                "target_id": sample["target_id"],
                "input_sha256": sample["input_sha256"],
                "prediction_sha256": prediction_hash,
            }
        )
        self.state = (
            WorkerState.INFERENCE_COMPLETE
            if len(self.predictions) == self.expected_inference_count
            else WorkerState.INFERENCING
        )
        return response

    def evaluate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.INFERENCE_COMPLETE)
        protocol.require_exact_fields(
            payload, {"prediction_confirmations", "targets"}, "evaluate payload"
        )
        confirmations = payload["prediction_confirmations"]
        if not isinstance(confirmations, list) or confirmations != self.confirmations:
            raise WorkerContractError(
                "prediction confirmations do not match worker outputs"
            )
        raw_targets = payload["targets"]
        if not isinstance(raw_targets, list) or len(raw_targets) != len(
            self.confirmations
        ):
            raise WorkerContractError("evaluation target count mismatch")
        targets = [
            protocol.validate_evaluation_target(target, f"evaluation target {index}")
            for index, target in enumerate(raw_targets)
        ]
        for target, confirmation in zip(targets, self.confirmations):
            if (
                target["scene_id"] != confirmation["scene_id"]
                or target["target_id"] != confirmation["target_id"]
                or target["input_sha256"] != confirmation["input_sha256"]
            ):
                raise WorkerContractError(
                    "evaluation target identity or order mismatch"
                )
        detached_predictions = protocol.strict_json_loads(
            protocol.canonical_json_bytes(self.predictions), "evaluation predictions"
        )
        detached_targets = protocol.strict_json_loads(
            protocol.canonical_json_bytes(targets), "evaluation targets"
        )
        metrics = protocol.require_object(
            protocol.strict_json_loads(
                protocol.canonical_json_bytes(
                    self.adapter.evaluate(detached_predictions, detached_targets)
                ),
                "evaluation metrics",
            ),
            "evaluation metrics",
        )
        protocol.reject_sensitive_keys(metrics, "evaluation metrics")
        self.state = WorkerState.EVALUATED
        return {
            "metrics": metrics,
            "prediction_set_sha256": protocol.sha256_json(self.predictions),
            "target_set_sha256": protocol.sha256_json(targets),
        }

    def save_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.EVALUATED)
        protocol.require_exact_fields(
            payload, {"relative_path"}, "save_checkpoint payload"
        )
        if payload["relative_path"] != "checkpoint.bin":
            raise WorkerContractError("checkpoint relative path is not frozen")
        checkpoint = Path.cwd() / "checkpoint.bin"
        if checkpoint.exists() or checkpoint.is_symlink():
            raise WorkerContractError("checkpoint path already exists")
        self.adapter.save_checkpoint(str(checkpoint))
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise WorkerContractError("adapter checkpoint is not a regular file")
        size = checkpoint.stat().st_size
        if size <= 0:
            raise WorkerContractError("adapter checkpoint is empty")
        digest = protocol.sha256_file(checkpoint)
        self.state = WorkerState.CHECKPOINT_SAVED
        return {
            "relative_path": "checkpoint.bin",
            "size_bytes": size,
            "sha256": digest,
        }

    def close(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_state(WorkerState.CHECKPOINT_SAVED)
        protocol.require_exact_fields(payload, set(), "close payload")
        self.state = WorkerState.CLOSED
        return {"closed": True, "final_state": "checkpoint_saved"}


def _write_response(seq: int, op: str, ok: bool, payload: Mapping[str, Any]) -> None:
    normalized = protocol.require_object(
        protocol.strict_json_loads(
            protocol.canonical_json_bytes(dict(payload)), "worker response payload"
        ),
        "worker response payload",
    )
    envelope = {
        "protocol": protocol.PROTOCOL_ID,
        "seq": seq,
        "op": op,
        "ok": ok,
        "payload": normalized,
        "payload_sha256": protocol.sha256_json(normalized),
    }
    encoded = protocol.canonical_json_bytes(envelope) + b"\n"
    if len(encoded) > protocol.MAX_MESSAGE_BYTES:
        raise WorkerContractError("worker response exceeds the message-size limit")
    view = memoryview(encoded)
    while view:
        written = os.write(1, view)
        view = view[written:]


def _error_payload(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, WorkerContractError):
        code = "worker_contract_error"
        error_type = "WorkerContractError"
    elif isinstance(exc, protocol.HistoryAdapterError):
        code = "protocol_validation_error"
        error_type = "ProtocolValidationError"
    else:
        code = "adapter_error"
        error_type = "AdapterException"
    return {
        "error_code": code,
        "error_type": error_type,
        "message": "worker operation failed; details withheld from the protocol",
    }


def main() -> int:
    worker = Worker()
    expected_sequence = 1
    protocol_input = sys.__stdin__.buffer
    while True:
        line = protocol_input.readline(protocol.MAX_MESSAGE_BYTES + 2)
        if not line:
            return 0 if worker.state == WorkerState.CLOSED else 3
        if len(line) > protocol.MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
            return 4
        try:
            request = protocol.require_object(
                protocol.strict_json_loads(line[:-1], "worker request"),
                "worker request",
            )
            protocol.require_exact_fields(
                request, protocol.REQUEST_FIELDS, "worker request"
            )
            if request["protocol"] != protocol.PROTOCOL_ID:
                raise WorkerContractError("request protocol identity mismatch")
            sequence = request["seq"]
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != expected_sequence
            ):
                raise WorkerContractError("request sequence mismatch")
            op = protocol.require_identifier(request["op"], "request op")
            payload = protocol.require_object(request["payload"], "request payload")
            expected_hash = protocol.require_digest(
                request["payload_sha256"], "request payload_sha256"
            )
            if protocol.sha256_json(payload) != expected_hash:
                raise WorkerContractError("request payload SHA-256 mismatch")
            protocol.reject_sensitive_keys(payload, "request payload")
        except BaseException:
            # A malformed envelope cannot be answered without trusting its routing fields.
            return 5
        try:
            response_payload = worker.dispatch(op, payload)
            _write_response(sequence, op, True, response_payload)
        except Exception as exc:
            try:
                _write_response(sequence, op, False, _error_payload(exc))
            except Exception:
                return 6
            return 7
        expected_sequence += 1
        if worker.state == WorkerState.CLOSED:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
