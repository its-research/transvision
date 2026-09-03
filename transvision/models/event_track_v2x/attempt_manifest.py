"""Canonical per-run attempt history for confirmatory EventTrack-V2X jobs.

Every physical run in the confirmatory matrix has one ordered attempt chain.
Only an infrastructure failure that emits no prediction may be retried; a
success or algorithm failure is the unique terminal attempt and is bound to
the corresponding raw-result cell.  External authentication and completeness
of the orchestrator log remain capability-locked in the scientific gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping

from .experiment import ExperimentPlanV1
from .results_registry import (
    RUN_RESULT_STATUS_FAILURE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
    RunResultCellV1,
    RunResultRegistryV1,
)
from .wire import canonical_json_bytes


ATTEMPT_STATUS_SUCCESS_V1 = "success"
ATTEMPT_STATUS_ALGORITHM_FAILURE_V1 = "algorithm_failure"
ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1 = (
    "infrastructure_failure_before_prediction"
)
ATTEMPT_FINAL_STATUSES_V1 = (
    ATTEMPT_STATUS_ALGORITHM_FAILURE_V1,
    ATTEMPT_STATUS_SUCCESS_V1,
)
ATTEMPT_STATUSES_V1 = (
    ATTEMPT_STATUS_ALGORITHM_FAILURE_V1,
    ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1,
    ATTEMPT_STATUS_SUCCESS_V1,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class AttemptManifestError(ValueError):
    """Raised when a run has a hidden, invalid, or unbound retry history."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise AttemptManifestError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise AttemptManifestError(f"{name} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    return None if value is None else _sha256(value, name)


def _optional_identifier(value: object, name: str) -> str | None:
    return None if value is None else _identifier(value, name)


def _utc(value: object, name: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise AttemptManifestError(f"{name} must be canonical RFC3339 UTC")
    try:
        observed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AttemptManifestError(
            f"{name} must be canonical RFC3339 UTC"
        ) from exc
    if (
        observed.utcoffset() is None
        or observed.utcoffset().total_seconds() != 0
        or observed.isoformat().replace("+00:00", "Z") != value
    ):
        raise AttemptManifestError(f"{name} must be canonical RFC3339 UTC")
    return value


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise AttemptManifestError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise AttemptManifestError(f"{name} has missing or unknown fields")
    return value


def _sortable_key(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else f"{type(value).__name__}:{value}" for value in key)


@dataclass(frozen=True, slots=True)
class ConfirmatoryAttemptRecordV1:
    dataset_domain: str
    method_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
    scheduler_id: str
    attempt_index: int
    attempt_id: str
    rerun_of_attempt_id: str | None
    job_id: str
    started_at_utc: str
    first_prediction_at_utc: str | None
    completed_at_utc: str
    terminal_status: str
    prediction_emitted: bool
    parameters_sha256: str
    checkpoint_sha256: str
    environment_sha256: str
    result_cell_sha256: str | None
    evidence_bundle_sha256: str | None
    incident_evidence_sha256: str | None

    _FIELDS = frozenset(
        {
            "attempt_id",
            "attempt_index",
            "budget_bytes_per_second",
            "c9_trace_id",
            "checkpoint_sha256",
            "completed_at_utc",
            "condition_id",
            "dataset_domain",
            "environment_sha256",
            "evidence_bundle_sha256",
            "first_prediction_at_utc",
            "incident_evidence_sha256",
            "job_id",
            "method_id",
            "network_seed",
            "parameters_sha256",
            "prediction_emitted",
            "rerun_of_attempt_id",
            "result_cell_sha256",
            "scheduler_id",
            "sequence_id",
            "started_at_utc",
            "terminal_status",
            "training_seed",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "dataset_domain",
            "method_id",
            "sequence_id",
            "condition_id",
            "scheduler_id",
            "attempt_id",
            "job_id",
            "terminal_status",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "rerun_of_attempt_id",
            _optional_identifier(self.rerun_of_attempt_id, "rerun_of_attempt_id"),
        )
        if self.terminal_status not in ATTEMPT_STATUSES_V1:
            raise AttemptManifestError("terminal_status is not registered")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise AttemptManifestError("training_seed must be non-negative")
        if self.network_seed is not None and (
            type(self.network_seed) is not int or self.network_seed < 0
        ):
            raise AttemptManifestError("network_seed must be non-negative or None")
        if self.c9_trace_id is not None:
            object.__setattr__(
                self, "c9_trace_id", _identifier(self.c9_trace_id, "c9_trace_id")
            )
        if type(self.budget_bytes_per_second) is not int or self.budget_bytes_per_second <= 0:
            raise AttemptManifestError("budget_bytes_per_second must be positive")
        if type(self.attempt_index) is not int or self.attempt_index <= 0:
            raise AttemptManifestError("attempt_index must be positive")
        for name in ("started_at_utc", "completed_at_utc"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if self.first_prediction_at_utc is not None:
            object.__setattr__(
                self,
                "first_prediction_at_utc",
                _utc(self.first_prediction_at_utc, "first_prediction_at_utc"),
            )
        for name in (
            "parameters_sha256",
            "checkpoint_sha256",
            "environment_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "result_cell_sha256",
            "evidence_bundle_sha256",
            "incident_evidence_sha256",
        ):
            object.__setattr__(self, name, _optional_sha256(getattr(self, name), name))
        if type(self.prediction_emitted) is not bool:
            raise TypeError("prediction_emitted must be bool")
        started = datetime.fromisoformat(self.started_at_utc[:-1] + "+00:00")
        completed = datetime.fromisoformat(self.completed_at_utc[:-1] + "+00:00")
        if started >= completed:
            raise AttemptManifestError("attempt must complete after it starts")
        if self.first_prediction_at_utc is not None:
            first = datetime.fromisoformat(
                self.first_prediction_at_utc[:-1] + "+00:00"
            )
            if not started < first <= completed:
                raise AttemptManifestError(
                    "first prediction must occur after start and before completion"
                )
        if self.prediction_emitted != (self.first_prediction_at_utc is not None):
            raise AttemptManifestError(
                "prediction_emitted must match first_prediction_at_utc"
            )
        if self.terminal_status == ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1:
            if (
                self.prediction_emitted
                or self.result_cell_sha256 is not None
                or self.evidence_bundle_sha256 is not None
                or self.incident_evidence_sha256 is None
            ):
                raise AttemptManifestError(
                    "rerunnable infrastructure failure must emit no prediction, have no result, and carry incident evidence"
                )
        elif self.terminal_status == ATTEMPT_STATUS_SUCCESS_V1:
            if (
                not self.prediction_emitted
                or self.result_cell_sha256 is None
                or self.evidence_bundle_sha256 is None
                or self.incident_evidence_sha256 is not None
            ):
                raise AttemptManifestError(
                    "successful terminal attempt requires prediction and bound result evidence"
                )
        elif (
            self.result_cell_sha256 is None
            or self.evidence_bundle_sha256 is None
            or self.incident_evidence_sha256 is None
        ):
            raise AttemptManifestError(
                "algorithm failure must be terminal and bind its zero-score result and incident evidence"
            )

    @property
    def physical_run_key(self) -> tuple[object, ...]:
        return (
            self.dataset_domain,
            self.method_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
            self.scheduler_id,
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "budget_bytes_per_second": self.budget_bytes_per_second,
            "c9_trace_id": self.c9_trace_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "completed_at_utc": self.completed_at_utc,
            "condition_id": self.condition_id,
            "dataset_domain": self.dataset_domain,
            "environment_sha256": self.environment_sha256,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "first_prediction_at_utc": self.first_prediction_at_utc,
            "incident_evidence_sha256": self.incident_evidence_sha256,
            "job_id": self.job_id,
            "method_id": self.method_id,
            "network_seed": self.network_seed,
            "parameters_sha256": self.parameters_sha256,
            "prediction_emitted": self.prediction_emitted,
            "rerun_of_attempt_id": self.rerun_of_attempt_id,
            "result_cell_sha256": self.result_cell_sha256,
            "scheduler_id": self.scheduler_id,
            "sequence_id": self.sequence_id,
            "started_at_utc": self.started_at_utc,
            "terminal_status": self.terminal_status,
            "training_seed": self.training_seed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ConfirmatoryAttemptRecordV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(**item)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ConfirmatoryAttemptManifestV1:
    experiment_plan_sha256: str
    confirmatory_val_unsealed_at_utc: str
    attempt_inventory_closed_at_utc: str
    orchestrator_log_sha256: str
    execution_environment_sha256: str
    attempts: tuple[ConfirmatoryAttemptRecordV1, ...]

    _FIELDS = frozenset(
        {
            "attempt_inventory_closed_at_utc",
            "attempts",
            "content_sha256",
            "execution_environment_sha256",
            "experiment_plan_sha256",
            "kind",
            "orchestrator_log_sha256",
            "schema_version",
            "confirmatory_val_unsealed_at_utc",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "experiment_plan_sha256",
            "orchestrator_log_sha256",
            "execution_environment_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "confirmatory_val_unsealed_at_utc",
            "attempt_inventory_closed_at_utc",
        ):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if not isinstance(self.attempts, (list, tuple)) or not all(
            isinstance(item, ConfirmatoryAttemptRecordV1) for item in self.attempts
        ):
            raise TypeError("attempts must contain ConfirmatoryAttemptRecordV1")
        attempts = tuple(self.attempts)
        if not attempts:
            raise AttemptManifestError("attempt manifest must be non-empty")
        if len({item.attempt_id for item in attempts}) != len(attempts):
            raise AttemptManifestError("attempt IDs must be globally unique")
        if len({item.job_id for item in attempts}) != len(attempts):
            raise AttemptManifestError("job IDs must be globally unique")
        if any(
            item.environment_sha256 != self.execution_environment_sha256
            for item in attempts
        ):
            raise AttemptManifestError("attempt environment does not match manifest")
        expected_order = tuple(
            sorted(
                attempts,
                key=lambda item: (
                    _sortable_key(item.physical_run_key),
                    item.attempt_index,
                ),
            )
        )
        if attempts != expected_order:
            raise AttemptManifestError(
                "attempts must be sorted by physical run key and attempt index"
            )
        groups: dict[tuple[object, ...], list[ConfirmatoryAttemptRecordV1]] = {}
        for item in attempts:
            groups.setdefault(item.physical_run_key, []).append(item)
        for group in groups.values():
            if tuple(item.attempt_index for item in group) != tuple(
                range(1, len(group) + 1)
            ):
                raise AttemptManifestError("attempt indices must be contiguous from one")
            if group[0].rerun_of_attempt_id is not None:
                raise AttemptManifestError("first attempt cannot be a rerun")
            for previous, current in zip(group, group[1:]):
                if current.rerun_of_attempt_id != previous.attempt_id:
                    raise AttemptManifestError("rerun chain is not contiguous")
                if (
                    previous.terminal_status
                    != ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1
                ):
                    raise AttemptManifestError(
                        "only a pre-prediction infrastructure failure may be retried"
                    )
                if datetime.fromisoformat(
                    previous.completed_at_utc[:-1] + "+00:00"
                ) > datetime.fromisoformat(current.started_at_utc[:-1] + "+00:00"):
                    raise AttemptManifestError("attempts for one run must not overlap")
            if group[-1].terminal_status not in ATTEMPT_FINAL_STATUSES_V1:
                raise AttemptManifestError(
                    "each physical run requires one success or algorithm-failure terminal"
                )
        unsealed = datetime.fromisoformat(
            self.confirmatory_val_unsealed_at_utc[:-1] + "+00:00"
        )
        if unsealed >= min(
            datetime.fromisoformat(item.started_at_utc[:-1] + "+00:00")
            for item in attempts
        ):
            raise AttemptManifestError(
                "confirmatory val must be unsealed before execution starts"
            )
        inventory_closed = datetime.fromisoformat(
            self.attempt_inventory_closed_at_utc[:-1] + "+00:00"
        )
        if inventory_closed < max(
            datetime.fromisoformat(item.completed_at_utc[:-1] + "+00:00")
            for item in attempts
        ):
            raise AttemptManifestError("attempt inventory closed before a run completed")
        object.__setattr__(self, "attempts", attempts)

    def payload(self) -> dict[str, Any]:
        return {
            "attempt_inventory_closed_at_utc": self.attempt_inventory_closed_at_utc,
            "attempts": [item.to_primitive() for item in self.attempts],
            "execution_environment_sha256": self.execution_environment_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "kind": "confirmatory_attempt_manifest_v1",
            "orchestrator_log_sha256": self.orchestrator_log_sha256,
            "schema_version": 1,
            "confirmatory_val_unsealed_at_utc": (
                self.confirmatory_val_unsealed_at_utc
            ),
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, Any]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "ConfirmatoryAttemptManifestV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        if (
            item["kind"] != "confirmatory_attempt_manifest_v1"
            or item["schema_version"] != 1
        ):
            raise AttemptManifestError("unsupported attempt manifest schema")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in cls._FIELDS if key != "content_sha256"}
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != observed:
            raise AttemptManifestError("attempt manifest content hash mismatch")
        raw_attempts = item["attempts"]
        if not isinstance(raw_attempts, list):
            raise AttemptManifestError("attempts must be an array")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],  # type: ignore[arg-type]
            confirmatory_val_unsealed_at_utc=item[
                "confirmatory_val_unsealed_at_utc"
            ],  # type: ignore[arg-type]
            attempt_inventory_closed_at_utc=item["attempt_inventory_closed_at_utc"],  # type: ignore[arg-type]
            orchestrator_log_sha256=item["orchestrator_log_sha256"],  # type: ignore[arg-type]
            execution_environment_sha256=item["execution_environment_sha256"],  # type: ignore[arg-type]
            attempts=tuple(
                ConfirmatoryAttemptRecordV1.from_mapping(value)
                for value in raw_attempts
            ),
        )


def physical_result_cell_sha256_v1(cell: RunResultCellV1) -> str:
    if not isinstance(cell, RunResultCellV1):
        raise TypeError("cell must be RunResultCellV1")
    value = cell.to_primitive()
    value.pop("scenario_id")
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_confirmatory_attempt_manifest_v1(
    manifest: ConfirmatoryAttemptManifestV1,
    *,
    experiment_plan: ExperimentPlanV1,
    run_result_registry: RunResultRegistryV1,
) -> None:
    if not isinstance(manifest, ConfirmatoryAttemptManifestV1):
        raise TypeError("manifest must be ConfirmatoryAttemptManifestV1")
    if manifest.experiment_plan_sha256 != experiment_plan.content_sha256:
        raise AttemptManifestError("attempt manifest is not bound to the plan")
    if manifest.digest() != run_result_registry.attempt_manifest_sha256:
        raise AttemptManifestError("attempt manifest digest does not match registry")
    if (
        manifest.confirmatory_val_unsealed_at_utc
        != run_result_registry.confirmatory_val_unsealed_at_utc
    ):
        raise AttemptManifestError(
            "attempt manifest confirmatory-unseal timestamp mismatch"
        )

    cells_by_run: dict[tuple[object, ...], list[RunResultCellV1]] = {}
    for cell in run_result_registry.cells:
        cells_by_run.setdefault(cell.physical_run_key, []).append(cell)
    attempts_by_run: dict[
        tuple[object, ...], list[ConfirmatoryAttemptRecordV1]
    ] = {}
    for attempt in manifest.attempts:
        attempts_by_run.setdefault(attempt.physical_run_key, []).append(attempt)
    if set(attempts_by_run) != set(cells_by_run):
        raise AttemptManifestError(
            "attempt manifest does not cover every physical result run exactly"
        )
    for run_key, cells in cells_by_run.items():
        final = attempts_by_run[run_key][-1]
        if any(cell.attempt_id != final.attempt_id for cell in cells):
            raise AttemptManifestError("result cell does not bind its terminal attempt")
        digests = {physical_result_cell_sha256_v1(cell) for cell in cells}
        if digests != {final.result_cell_sha256}:
            raise AttemptManifestError("terminal attempt result digest mismatch")
        bundles = {cell.evidence_bundle_sha256 for cell in cells}
        if bundles != {final.evidence_bundle_sha256}:
            raise AttemptManifestError("terminal attempt evidence bundle mismatch")
        statuses = {cell.status for cell in cells}
        expected_status = (
            ATTEMPT_STATUS_SUCCESS_V1
            if statuses == {RUN_RESULT_STATUS_SUCCESS_V1}
            else ATTEMPT_STATUS_ALGORITHM_FAILURE_V1
            if statuses == {RUN_RESULT_STATUS_FAILURE_V1}
            else None
        )
        if final.terminal_status != expected_status:
            raise AttemptManifestError("terminal attempt status does not match result")
        expected_parameters_sha256 = experiment_plan.run_config_sha256(
            final.dataset_domain,
            final.method_id,
            final.scheduler_id,
        )
        if any(
            attempt.parameters_sha256 != expected_parameters_sha256
            for attempt in attempts_by_run[run_key]
        ):
            raise AttemptManifestError("attempt parameters do not match frozen plan")
        expected_checkpoint_sha256 = experiment_plan.run_checkpoint_sha256(
            final.dataset_domain,
            final.method_id,
            final.scheduler_id,
            final.training_seed,
        )
        if any(
            attempt.checkpoint_sha256 != expected_checkpoint_sha256
            for attempt in attempts_by_run[run_key]
        ):
            raise AttemptManifestError(
                "attempt checkpoint does not match seed-specific frozen refit"
            )

    starts = tuple(item.started_at_utc for item in manifest.attempts)
    predictions = tuple(
        item.first_prediction_at_utc
        for item in manifest.attempts
        if item.first_prediction_at_utc is not None
    )
    completions = tuple(item.completed_at_utc for item in manifest.attempts)
    if not predictions:
        raise AttemptManifestError("confirmatory matrix emitted no prediction")
    chronology = (
        min(starts),
        min(predictions),
        max(completions),
    )
    expected_chronology = (
        run_result_registry.execution_started_at_utc,
        run_result_registry.first_prediction_at_utc,
        run_result_registry.completed_at_utc,
    )
    if chronology != expected_chronology:
        raise AttemptManifestError("attempt chronology does not match result registry")


def decode_confirmatory_attempt_manifest(
    data: bytes,
) -> ConfirmatoryAttemptManifestV1:
    if type(data) is not bytes:
        raise TypeError("attempt manifest data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise AttemptManifestError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AttemptManifestError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptManifestError("invalid attempt manifest JSON") from exc
    manifest = ConfirmatoryAttemptManifestV1.from_mapping(value)
    if manifest.canonical_bytes != data:
        raise AttemptManifestError("attempt manifest JSON is not canonical")
    return manifest


__all__ = [
    "ATTEMPT_FINAL_STATUSES_V1",
    "ATTEMPT_STATUSES_V1",
    "ATTEMPT_STATUS_ALGORITHM_FAILURE_V1",
    "ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1",
    "ATTEMPT_STATUS_SUCCESS_V1",
    "AttemptManifestError",
    "ConfirmatoryAttemptManifestV1",
    "ConfirmatoryAttemptRecordV1",
    "decode_confirmatory_attempt_manifest",
    "physical_result_cell_sha256_v1",
    "validate_confirmatory_attempt_manifest_v1",
]
