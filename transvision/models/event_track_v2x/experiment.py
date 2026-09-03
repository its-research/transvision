"""Sealed experiment-plan contracts for EventTrack-V2X.

The plan is deliberately independent of ClearML and model frameworks.  It is
the immutable join key between source, data, detector caches, network traces,
evaluators, predictions, and statistical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")

CONFIRMATORY_BYTE_BUDGETS_V1 = (16_000, 32_000, 64_000, 128_000, 256_000)
CONFIRMATORY_ALPHA_V1 = 0.05
CONFIRMATORY_RESAMPLES_V1 = 10_000
CONFIRMATORY_RANDOM_SEED_V1 = 1337
CONFIRMATORY_TRAINING_SEEDS_V1 = (1337, 2027, 3407)
CONFIRMATORY_NETWORK_SEEDS_V1 = tuple(range(1001, 1011))
CONFIRMATORY_CONDITION_IDS_V1 = tuple(f"C{index}" for index in range(10))
CONFIRMATORY_METHOD_IDS_V1 = (
    "vehicle-only-ab3dmot",
    "vehicle-only-simpletrack",
    "vehicle-only-immortaltracker",
    "naive-async-late-fusion",
    "constant-velocity-compensation",
    "fixed-lag-oosm-controlled",
    "eventtrack-v2x",
    "synchronous-oracle",
)
CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1 = tuple(
    sorted(CONFIRMATORY_METHOD_IDS_V1[:6])
)
CONFIRMATORY_SCHEDULER_IDS_V1 = (
    "confidence_top_k",
    "fifo_aoi",
    "full_send",
    "marginal_voi",
    "periodic",
    "random",
)
CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1 = tuple(
    scheduler_id
    for scheduler_id in CONFIRMATORY_SCHEDULER_IDS_V1
    if scheduler_id not in {"full_send", "marginal_voi"}
)
GRIFFIN_25M_VAL_SEQUENCE_IDS_V1 = (
    "scene-0000-Town03-000",
    "scene-0006-Town03-006",
    "scene-0007-Town03-007",
    "scene-0010-Town03-011",
    "scene-0020-Town06-007",
    "scene-0032-Town07-011",
    "scene-0033-Town07-012",
    "scene-0034-Town07-013",
    "scene-0044-Town10HD-012",
    "scene-0046-Town10HD-014",
)


class ExperimentPlanError(ValueError):
    """Raised when an experiment plan is malformed or its seal is invalid."""


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or value != value.strip() or not value:
        raise ExperimentPlanError(f"{name} must be a trimmed non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ExperimentPlanError(f"{name} is not a canonical identifier")
    return result


def _sha256(value: object, name: str) -> str:
    result = _nonempty(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ExperimentPlanError(f"{name} must be a lowercase SHA-256")
    return result


def _unique_strings(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExperimentPlanError(f"{name} must be an array")
    result = tuple(_identifier(value, f"{name} item") for value in values)
    if not result or len(set(result)) != len(result):
        raise ExperimentPlanError(f"{name} must be non-empty and unique")
    return result


def _positive_ints(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExperimentPlanError(f"{name} must be an array")
    result: list[int] = []
    for value in values:
        if type(value) is not int or value <= 0:
            raise ExperimentPlanError(f"{name} items must be positive integers")
        result.append(value)
    if not result or len(set(result)) != len(result):
        raise ExperimentPlanError(f"{name} must be non-empty and unique")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RunConfigBindingV1:
    """Frozen config and checkpoint for one physical run namespace.

    Non-learned trackers still bind an explicit provenance artifact digest in
    ``checkpoint_sha256s``.  This prevents a post-unseal refit or weight swap
    from hiding behind an unchanged configuration file.
    """

    dataset_domain: str
    method_id: str
    scheduler_id: str
    config_sha256: str
    checkpoint_sha256s: tuple[str, ...]

    _FIELDS = frozenset(
        {
            "checkpoint_sha256s",
            "config_sha256",
            "dataset_domain",
            "method_id",
            "scheduler_id",
        }
    )

    def __post_init__(self) -> None:
        for name in ("dataset_domain", "method_id", "scheduler_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "config_sha256", _sha256(self.config_sha256, "config_sha256")
        )
        if not isinstance(self.checkpoint_sha256s, (list, tuple)):
            raise ExperimentPlanError("checkpoint_sha256s must be an array")
        checkpoints = tuple(
            _sha256(value, "checkpoint_sha256s item")
            for value in self.checkpoint_sha256s
        )
        if len(checkpoints) != len(CONFIRMATORY_TRAINING_SEEDS_V1):
            raise ExperimentPlanError(
                "checkpoint_sha256s must contain one digest per training seed"
            )
        object.__setattr__(self, "checkpoint_sha256s", checkpoints)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset_domain, self.method_id, self.scheduler_id)

    def to_primitive(self) -> dict[str, object]:
        return {
            "checkpoint_sha256s": list(self.checkpoint_sha256s),
            "config_sha256": self.config_sha256,
            "dataset_domain": self.dataset_domain,
            "method_id": self.method_id,
            "scheduler_id": self.scheduler_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "RunConfigBindingV1":
        if not isinstance(value, Mapping) or not all(
            type(key) is str for key in value
        ):
            raise ExperimentPlanError(
                "run_config_bindings items must be string-keyed objects"
            )
        if frozenset(value) != cls._FIELDS:
            raise ExperimentPlanError(
                "run_config_bindings items have missing or unknown fields"
            )
        return cls(
            dataset_domain=value["dataset_domain"],
            method_id=value["method_id"],
            scheduler_id=value["scheduler_id"],
            config_sha256=value["config_sha256"],
            checkpoint_sha256s=tuple(value["checkpoint_sha256s"]),  # type: ignore[arg-type]
        )


def confirmatory_run_config_keys_v1(
    *,
    strongest_baseline_id: str,
    strongest_scheduler_id: str,
) -> tuple[tuple[str, str, str], ...]:
    """Return every physical confirmatory parameter namespace in canonical order."""

    baseline = _identifier(strongest_baseline_id, "strongest_baseline_id")
    scheduler = _identifier(strongest_scheduler_id, "strongest_scheduler_id")
    keys = {
        *(
            ("primary", method_id, scheduler)
            for method_id in CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
        ),
        *(
            ("primary", "eventtrack-v2x", scheduler_id)
            for scheduler_id in CONFIRMATORY_SCHEDULER_IDS_V1
        ),
        ("griffin", "eventtrack-v2x", "marginal_voi"),
        ("griffin", baseline, scheduler),
    }
    return tuple(sorted(keys))


def _run_config_bindings(values: object) -> tuple[RunConfigBindingV1, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExperimentPlanError("run_config_bindings must be an array")
    result = tuple(
        item
        if isinstance(item, RunConfigBindingV1)
        else RunConfigBindingV1.from_mapping(item)
        for item in values
    )
    if not result:
        raise ExperimentPlanError("run_config_bindings must be non-empty")
    if len({item.key for item in result}) != len(result):
        raise ExperimentPlanError("run_config_bindings keys must be unique")
    if tuple(item.key for item in result) != tuple(
        sorted(item.key for item in result)
    ):
        raise ExperimentPlanError("run_config_bindings must be sorted by key")
    return result


@dataclass(frozen=True, slots=True)
class ExperimentPlanV1:
    """Decision-complete, hash-sealed confirmatory experiment plan."""

    plan_id: str
    source_tree_sha256: str
    method_config_sha256: str
    run_config_bindings: tuple[RunConfigBindingV1, ...]
    dataset_id: str
    dataset_manifest_sha256: str
    split_name: str
    split_sha256: str
    primary_cohort_sha256: str
    primary_frame_contract_sha256: str
    primary_dataset_release_receipt_sha256: str
    development_dataset_release_receipt_sha256: str
    development_fold_manifest_sha256: str
    primary_sequence_ids: tuple[str, ...]
    detector_cache_sha256: str
    network_trace_manifest_sha256: str
    wire_accounting_config_sha256: str
    heldout_c9_trace_receipt_sha256: str
    c9_trace_ids: tuple[str, ...]
    evaluator_contract_sha256: str
    baseline_qualification_registry_sha256: str
    candidate_selection_registry_sha256: str
    preregistration_provider_id: str
    preregistration_public_key_sha256: str
    verification_provider_id: str
    verification_public_key_sha256: str
    qualified_baseline_ids: tuple[str, ...]
    strongest_qualified_baseline_id: str
    scheduler_ids: tuple[str, ...]
    candidate_scheduler_id: str
    qualified_scheduler_baseline_ids: tuple[str, ...]
    strongest_scheduler_baseline_id: str
    scheduler_token_bucket_burst_seconds: float
    griffin_dataset_id: str
    griffin_dataset_manifest_sha256: str
    griffin_split_name: str
    griffin_cohort_sha256: str
    griffin_frame_contract_sha256: str
    griffin_dataset_release_receipt_sha256: str
    griffin_sequence_ids: tuple[str, ...]
    griffin_detector_cache_sha256: str
    griffin_evaluator_contract_sha256: str
    griffin_strongest_baseline_id: str
    frequency_hz: float
    class_names: tuple[str, ...]
    method_ids: tuple[str, ...]
    network_condition_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    network_seeds: tuple[int, ...]
    byte_budgets_per_second: tuple[int, ...]
    primary_budget_bytes_per_second: int
    primary_metric: str
    decision_deadline_ms: float
    statistical_alpha: float
    statistical_resamples: int
    statistical_random_seed: int
    claim_scope: str

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "dataset_id",
            "split_name",
            "primary_metric",
            "strongest_qualified_baseline_id",
            "candidate_scheduler_id",
            "strongest_scheduler_baseline_id",
            "griffin_dataset_id",
            "griffin_split_name",
            "griffin_strongest_baseline_id",
            "preregistration_provider_id",
            "verification_provider_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "source_tree_sha256",
            "method_config_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "primary_cohort_sha256",
            "primary_frame_contract_sha256",
            "primary_dataset_release_receipt_sha256",
            "development_dataset_release_receipt_sha256",
            "development_fold_manifest_sha256",
            "detector_cache_sha256",
            "network_trace_manifest_sha256",
            "wire_accounting_config_sha256",
            "heldout_c9_trace_receipt_sha256",
            "evaluator_contract_sha256",
            "baseline_qualification_registry_sha256",
            "candidate_selection_registry_sha256",
            "preregistration_public_key_sha256",
            "verification_public_key_sha256",
            "griffin_dataset_manifest_sha256",
            "griffin_cohort_sha256",
            "griffin_frame_contract_sha256",
            "griffin_dataset_release_receipt_sha256",
            "griffin_detector_cache_sha256",
            "griffin_evaluator_contract_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "run_config_bindings",
            _run_config_bindings(self.run_config_bindings),
        )
        for name in (
            "primary_sequence_ids",
            "c9_trace_ids",
            "qualified_baseline_ids",
            "scheduler_ids",
            "qualified_scheduler_baseline_ids",
            "griffin_sequence_ids",
            "class_names",
            "method_ids",
            "network_condition_ids",
        ):
            object.__setattr__(self, name, _unique_strings(getattr(self, name), name))
        for name in (
            "primary_sequence_ids",
            "griffin_sequence_ids",
            "c9_trace_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(values)):
                raise ExperimentPlanError(f"{name} must be lexicographically sorted")
        for name in ("qualified_baseline_ids", "qualified_scheduler_baseline_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(values)):
                raise ExperimentPlanError(f"{name} must be lexicographically sorted")
        for name in (
            "training_seeds",
            "network_seeds",
            "byte_budgets_per_second",
        ):
            object.__setattr__(self, name, _positive_ints(getattr(self, name), name))
        frequency = float(self.frequency_hz)
        deadline = float(self.decision_deadline_ms)
        if isinstance(self.scheduler_token_bucket_burst_seconds, bool) or not isinstance(
            self.scheduler_token_bucket_burst_seconds, (int, float)
        ):
            raise ExperimentPlanError(
                "scheduler_token_bucket_burst_seconds must be numeric"
            )
        burst = float(self.scheduler_token_bucket_burst_seconds)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ExperimentPlanError("frequency_hz must be finite and positive")
        if not np.isfinite(deadline) or deadline <= 0.0:
            raise ExperimentPlanError(
                "decision_deadline_ms must be finite and positive"
            )
        if not np.isfinite(burst) or burst <= 0.0:
            raise ExperimentPlanError(
                "scheduler_token_bucket_burst_seconds must be finite and positive"
            )
        if burst != 1.0:
            raise ExperimentPlanError(
                "scheduler_token_bucket_burst_seconds must equal 1.0"
            )
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "decision_deadline_ms", deadline)
        object.__setattr__(self, "scheduler_token_bucket_burst_seconds", burst)
        if self.plan_id != "eventtrack-v2x-confirmatory-v1":
            raise ExperimentPlanError("plan_id is not the confirmatory V1 contract")
        if self.preregistration_provider_id == self.verification_provider_id:
            raise ExperimentPlanError(
                "preregistration and verification providers must be distinct"
            )
        if (
            self.preregistration_public_key_sha256
            == self.verification_public_key_sha256
        ):
            raise ExperimentPlanError(
                "preregistration and verification public keys must be distinct"
            )
        if self.dataset_id != "v2x-seq-spd" or self.split_name != "val":
            raise ExperimentPlanError(
                "confirmatory plan requires V2X-Seq-SPD val"
            )
        if len(self.primary_sequence_ids) != 21:
            raise ExperimentPlanError(
                "confirmatory V2X-Seq-SPD val requires exactly 21 sequences"
            )
        if (
            self.primary_dataset_release_receipt_sha256
            == self.development_dataset_release_receipt_sha256
        ):
            raise ExperimentPlanError(
                "development train and confirmatory val receipts must be independent"
            )
        if len(self.c9_trace_ids) != 20:
            raise ExperimentPlanError(
                "confirmatory C9 requires exactly 20 held-out trace IDs"
            )
        if frequency != 10.0:
            raise ExperimentPlanError("confirmatory frequency_hz must equal 10")
        if self.class_names != ("car",):
            raise ExperimentPlanError("confirmatory class_names must equal ('car',)")
        if self.network_condition_ids != CONFIRMATORY_CONDITION_IDS_V1:
            raise ExperimentPlanError(
                "confirmatory network conditions must equal C0 through C9"
            )
        if self.training_seeds != CONFIRMATORY_TRAINING_SEEDS_V1:
            raise ExperimentPlanError(
                "confirmatory training seeds do not match preregistration"
            )
        if self.network_seeds != CONFIRMATORY_NETWORK_SEEDS_V1:
            raise ExperimentPlanError(
                "confirmatory network seeds do not match preregistration"
            )
        if self.method_ids != CONFIRMATORY_METHOD_IDS_V1:
            raise ExperimentPlanError(
                "confirmatory method registry does not match preregistration"
            )
        if self.qualified_baseline_ids != CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1:
            raise ExperimentPlanError(
                "confirmatory qualified baselines must include all six controls"
            )
        if self.primary_metric != "robust-assa-at-64k":
            raise ExperimentPlanError(
                "confirmatory primary metric must be robust-assa-at-64k"
            )
        if deadline != 100.0:
            raise ExperimentPlanError(
                "confirmatory decision deadline must equal 100 ms"
            )
        if self.griffin_dataset_id != "griffin-25m" or self.griffin_split_name != "val":
            raise ExperimentPlanError(
                "external validation requires Griffin-25m val"
            )
        if self.griffin_sequence_ids != GRIFFIN_25M_VAL_SEQUENCE_IDS_V1:
            raise ExperimentPlanError(
                "external validation requires the exact official Griffin-25m val cohort"
            )
        if self.byte_budgets_per_second != CONFIRMATORY_BYTE_BUDGETS_V1:
            raise ExperimentPlanError(
                "byte_budgets_per_second must equal the five preregistered budgets"
            )
        if (
            type(self.primary_budget_bytes_per_second) is not int
            or self.primary_budget_bytes_per_second
            not in self.byte_budgets_per_second
        ):
            raise ExperimentPlanError(
                "primary budget must be an integer member of byte budgets"
            )
        if self.primary_budget_bytes_per_second != 64_000:
            raise ExperimentPlanError("primary budget must equal 64000 B/s")
        if (
            type(self.statistical_alpha) is not float
            or self.statistical_alpha != CONFIRMATORY_ALPHA_V1
        ):
            raise ExperimentPlanError("statistical_alpha must equal 0.05")
        if (
            type(self.statistical_resamples) is not int
            or self.statistical_resamples != CONFIRMATORY_RESAMPLES_V1
        ):
            raise ExperimentPlanError("statistical_resamples must equal 10000")
        if (
            type(self.statistical_random_seed) is not int
            or self.statistical_random_seed != CONFIRMATORY_RANDOM_SEED_V1
        ):
            raise ExperimentPlanError("statistical_random_seed must equal 1337")
        if "eventtrack-v2x" not in self.method_ids:
            raise ExperimentPlanError("method_ids must contain eventtrack-v2x")
        if "eventtrack-v2x" in self.qualified_baseline_ids or not set(
            self.qualified_baseline_ids
        ) <= set(self.method_ids):
            raise ExperimentPlanError(
                "qualified baselines must be non-candidate members of method_ids"
            )
        if self.strongest_qualified_baseline_id not in self.qualified_baseline_ids:
            raise ExperimentPlanError(
                "strongest_qualified_baseline_id must be a qualified baseline"
            )
        if self.griffin_strongest_baseline_id not in self.qualified_baseline_ids:
            raise ExperimentPlanError(
                "griffin_strongest_baseline_id must be a qualified baseline"
            )
        if (
            self.griffin_strongest_baseline_id
            != self.strongest_qualified_baseline_id
        ):
            raise ExperimentPlanError(
                "Griffin must reuse the strongest baseline frozen on validation"
            )
        if self.scheduler_ids != CONFIRMATORY_SCHEDULER_IDS_V1:
            raise ExperimentPlanError(
                "scheduler_ids must equal the six preregistered schedulers"
            )
        if self.candidate_scheduler_id != "marginal_voi":
            raise ExperimentPlanError("candidate scheduler must be marginal_voi")
        if (
            self.qualified_scheduler_baseline_ids
            != CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1
        ):
            raise ExperimentPlanError(
                "qualified scheduler baselines must equal the four capped controls; "
                "full_send is an uncapped diagnostic"
            )
        if self.candidate_scheduler_id in self.qualified_scheduler_baseline_ids or not set(
            self.qualified_scheduler_baseline_ids
        ) <= set(self.scheduler_ids):
            raise ExperimentPlanError(
                "qualified scheduler baselines must be non-candidate scheduler_ids"
            )
        if (
            self.strongest_scheduler_baseline_id
            not in self.qualified_scheduler_baseline_ids
        ):
            raise ExperimentPlanError(
                "strongest_scheduler_baseline_id must be a qualified scheduler baseline"
            )
        expected_run_configs = confirmatory_run_config_keys_v1(
            strongest_baseline_id=self.strongest_qualified_baseline_id,
            strongest_scheduler_id=self.strongest_scheduler_baseline_id,
        )
        observed_run_configs = tuple(
            binding.key for binding in self.run_config_bindings
        )
        if observed_run_configs != expected_run_configs:
            raise ExperimentPlanError(
                "run_config_bindings must exactly cover the confirmatory physical "
                "domain/method/scheduler combinations"
            )
        for binding in self.run_config_bindings:
            if binding.method_id == "eventtrack-v2x" and len(
                set(binding.checkpoint_sha256s)
            ) != len(self.training_seeds):
                raise ExperimentPlanError(
                    "EventTrack must freeze a distinct final-refit checkpoint "
                    "for every training seed"
                )
        object.__setattr__(
            self, "claim_scope", _nonempty(self.claim_scope, "claim_scope")
        )

    def run_config_sha256(
        self, dataset_domain: str, method_id: str, scheduler_id: str
    ) -> str:
        """Resolve the sealed parameter artifact for one physical run."""

        key = (dataset_domain, method_id, scheduler_id)
        for binding in self.run_config_bindings:
            if binding.key == key:
                return binding.config_sha256
        raise ExperimentPlanError(f"no frozen run config for {key!r}")

    def run_checkpoint_sha256(
        self,
        dataset_domain: str,
        method_id: str,
        scheduler_id: str,
        training_seed: int,
    ) -> str:
        """Resolve the checkpoint/provenance artifact frozen before val unsealing."""

        key = (dataset_domain, method_id, scheduler_id)
        try:
            seed_index = self.training_seeds.index(training_seed)
        except ValueError as exc:
            raise ExperimentPlanError(
                f"training seed {training_seed!r} is not frozen in the plan"
            ) from exc
        for binding in self.run_config_bindings:
            if binding.key == key:
                return binding.checkpoint_sha256s[seed_index]
        raise ExperimentPlanError(f"no frozen run checkpoint for {key!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "byte_budgets_per_second": list(self.byte_budgets_per_second),
            "claim_scope": self.claim_scope,
            "class_names": list(self.class_names),
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "decision_deadline_ms": self.decision_deadline_ms,
            "development_dataset_release_receipt_sha256": (
                self.development_dataset_release_receipt_sha256
            ),
            "development_fold_manifest_sha256": (
                self.development_fold_manifest_sha256
            ),
            "detector_cache_sha256": self.detector_cache_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "frequency_hz": self.frequency_hz,
            "griffin_cohort_sha256": self.griffin_cohort_sha256,
            "griffin_dataset_release_receipt_sha256": (
                self.griffin_dataset_release_receipt_sha256
            ),
            "griffin_dataset_id": self.griffin_dataset_id,
            "griffin_dataset_manifest_sha256": self.griffin_dataset_manifest_sha256,
            "griffin_detector_cache_sha256": self.griffin_detector_cache_sha256,
            "griffin_evaluator_contract_sha256": self.griffin_evaluator_contract_sha256,
            "griffin_frame_contract_sha256": self.griffin_frame_contract_sha256,
            "griffin_sequence_ids": list(self.griffin_sequence_ids),
            "griffin_split_name": self.griffin_split_name,
            "griffin_strongest_baseline_id": self.griffin_strongest_baseline_id,
            "heldout_c9_trace_receipt_sha256": self.heldout_c9_trace_receipt_sha256,
            "c9_trace_ids": list(self.c9_trace_ids),
            "method_ids": list(self.method_ids),
            "method_config_sha256": self.method_config_sha256,
            "run_config_bindings": [
                item.to_primitive() for item in self.run_config_bindings
            ],
            "network_condition_ids": list(self.network_condition_ids),
            "network_seeds": list(self.network_seeds),
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "plan_id": self.plan_id,
            "baseline_qualification_registry_sha256": (
                self.baseline_qualification_registry_sha256
            ),
            "candidate_selection_registry_sha256": (
                self.candidate_selection_registry_sha256
            ),
            "preregistration_provider_id": self.preregistration_provider_id,
            "preregistration_public_key_sha256": (
                self.preregistration_public_key_sha256
            ),
            "verification_provider_id": self.verification_provider_id,
            "verification_public_key_sha256": (
                self.verification_public_key_sha256
            ),
            "primary_cohort_sha256": self.primary_cohort_sha256,
            "primary_dataset_release_receipt_sha256": (
                self.primary_dataset_release_receipt_sha256
            ),
            "primary_frame_contract_sha256": self.primary_frame_contract_sha256,
            "primary_sequence_ids": list(self.primary_sequence_ids),
            "primary_budget_bytes_per_second": self.primary_budget_bytes_per_second,
            "primary_metric": self.primary_metric,
            "qualified_baseline_ids": list(self.qualified_baseline_ids),
            "qualified_scheduler_baseline_ids": list(
                self.qualified_scheduler_baseline_ids
            ),
            "candidate_scheduler_id": self.candidate_scheduler_id,
            "scheduler_ids": list(self.scheduler_ids),
            "scheduler_token_bucket_burst_seconds": (
                self.scheduler_token_bucket_burst_seconds
            ),
            "statistical_alpha": self.statistical_alpha,
            "statistical_random_seed": self.statistical_random_seed,
            "statistical_resamples": self.statistical_resamples,
            "schema_version": 1,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "strongest_qualified_baseline_id": self.strongest_qualified_baseline_id,
            "strongest_scheduler_baseline_id": self.strongest_scheduler_baseline_id,
            "training_seeds": list(self.training_seeds),
            "wire_accounting_config_sha256": self.wire_accounting_config_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, Any]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())


_PAYLOAD_KEYS = frozenset(
    {
        "byte_budgets_per_second",
        "claim_scope",
        "class_names",
        "dataset_id",
        "dataset_manifest_sha256",
        "decision_deadline_ms",
        "development_dataset_release_receipt_sha256",
        "development_fold_manifest_sha256",
        "detector_cache_sha256",
        "evaluator_contract_sha256",
        "frequency_hz",
        "griffin_cohort_sha256",
        "griffin_dataset_release_receipt_sha256",
        "griffin_dataset_id",
        "griffin_dataset_manifest_sha256",
        "griffin_detector_cache_sha256",
        "griffin_evaluator_contract_sha256",
        "griffin_frame_contract_sha256",
        "griffin_sequence_ids",
        "griffin_split_name",
        "griffin_strongest_baseline_id",
        "heldout_c9_trace_receipt_sha256",
        "c9_trace_ids",
        "method_ids",
        "method_config_sha256",
        "run_config_bindings",
        "network_condition_ids",
        "network_seeds",
        "network_trace_manifest_sha256",
        "plan_id",
        "baseline_qualification_registry_sha256",
        "candidate_selection_registry_sha256",
        "preregistration_provider_id",
        "preregistration_public_key_sha256",
        "verification_provider_id",
        "verification_public_key_sha256",
        "primary_cohort_sha256",
        "primary_dataset_release_receipt_sha256",
        "primary_frame_contract_sha256",
        "primary_sequence_ids",
        "primary_budget_bytes_per_second",
        "primary_metric",
        "qualified_baseline_ids",
        "qualified_scheduler_baseline_ids",
        "candidate_scheduler_id",
        "scheduler_ids",
        "scheduler_token_bucket_burst_seconds",
        "statistical_alpha",
        "statistical_random_seed",
        "statistical_resamples",
        "schema_version",
        "split_name",
        "split_sha256",
        "source_tree_sha256",
        "strongest_qualified_baseline_id",
        "strongest_scheduler_baseline_id",
        "training_seeds",
        "wire_accounting_config_sha256",
    }
)


def plan_from_document(value: Mapping[str, object]) -> ExperimentPlanV1:
    """Validate an exact sealed document and return its immutable plan."""

    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ExperimentPlanError("plan document must be a string-keyed object")
    if frozenset(value) != _PAYLOAD_KEYS | {"content_sha256"}:
        raise ExperimentPlanError("plan document has missing or unknown fields")
    if value["schema_version"] != 1:
        raise ExperimentPlanError("unsupported experiment plan schema version")
    observed = _sha256(value["content_sha256"], "content_sha256")
    raw_payload = {key: value[key] for key in _PAYLOAD_KEYS}
    expected = hashlib.sha256(canonical_json_bytes(raw_payload)).hexdigest()
    if observed != expected:
        raise ExperimentPlanError("experiment plan content SHA-256 mismatch")
    plan = ExperimentPlanV1(
        plan_id=value["plan_id"],
        source_tree_sha256=value["source_tree_sha256"],
        method_config_sha256=value["method_config_sha256"],
        run_config_bindings=value["run_config_bindings"],
        dataset_id=value["dataset_id"],
        dataset_manifest_sha256=value["dataset_manifest_sha256"],
        split_name=value["split_name"],
        split_sha256=value["split_sha256"],
        primary_cohort_sha256=value["primary_cohort_sha256"],
        primary_frame_contract_sha256=value["primary_frame_contract_sha256"],
        primary_dataset_release_receipt_sha256=value[
            "primary_dataset_release_receipt_sha256"
        ],
        development_dataset_release_receipt_sha256=value[
            "development_dataset_release_receipt_sha256"
        ],
        development_fold_manifest_sha256=value[
            "development_fold_manifest_sha256"
        ],
        primary_sequence_ids=value["primary_sequence_ids"],
        detector_cache_sha256=value["detector_cache_sha256"],
        network_trace_manifest_sha256=value["network_trace_manifest_sha256"],
        wire_accounting_config_sha256=value["wire_accounting_config_sha256"],
        heldout_c9_trace_receipt_sha256=value[
            "heldout_c9_trace_receipt_sha256"
        ],
        c9_trace_ids=value["c9_trace_ids"],
        evaluator_contract_sha256=value["evaluator_contract_sha256"],
        baseline_qualification_registry_sha256=value[
            "baseline_qualification_registry_sha256"
        ],
        candidate_selection_registry_sha256=value[
            "candidate_selection_registry_sha256"
        ],
        preregistration_provider_id=value["preregistration_provider_id"],
        preregistration_public_key_sha256=value[
            "preregistration_public_key_sha256"
        ],
        verification_provider_id=value["verification_provider_id"],
        verification_public_key_sha256=value[
            "verification_public_key_sha256"
        ],
        qualified_baseline_ids=value["qualified_baseline_ids"],
        strongest_qualified_baseline_id=value[
            "strongest_qualified_baseline_id"
        ],
        scheduler_ids=value["scheduler_ids"],
        candidate_scheduler_id=value["candidate_scheduler_id"],
        qualified_scheduler_baseline_ids=value[
            "qualified_scheduler_baseline_ids"
        ],
        strongest_scheduler_baseline_id=value[
            "strongest_scheduler_baseline_id"
        ],
        scheduler_token_bucket_burst_seconds=value[
            "scheduler_token_bucket_burst_seconds"
        ],
        griffin_dataset_id=value["griffin_dataset_id"],
        griffin_dataset_manifest_sha256=value[
            "griffin_dataset_manifest_sha256"
        ],
        griffin_split_name=value["griffin_split_name"],
        griffin_cohort_sha256=value["griffin_cohort_sha256"],
        griffin_frame_contract_sha256=value["griffin_frame_contract_sha256"],
        griffin_dataset_release_receipt_sha256=value[
            "griffin_dataset_release_receipt_sha256"
        ],
        griffin_sequence_ids=value["griffin_sequence_ids"],
        griffin_detector_cache_sha256=value[
            "griffin_detector_cache_sha256"
        ],
        griffin_evaluator_contract_sha256=value[
            "griffin_evaluator_contract_sha256"
        ],
        griffin_strongest_baseline_id=value["griffin_strongest_baseline_id"],
        frequency_hz=value["frequency_hz"],
        class_names=value["class_names"],
        method_ids=value["method_ids"],
        network_condition_ids=value["network_condition_ids"],
        training_seeds=value["training_seeds"],
        network_seeds=value["network_seeds"],
        byte_budgets_per_second=value["byte_budgets_per_second"],
        primary_budget_bytes_per_second=value[
            "primary_budget_bytes_per_second"
        ],
        primary_metric=value["primary_metric"],
        decision_deadline_ms=value["decision_deadline_ms"],
        statistical_alpha=value["statistical_alpha"],
        statistical_resamples=value["statistical_resamples"],
        statistical_random_seed=value["statistical_random_seed"],
        claim_scope=value["claim_scope"],
    )
    if observed != plan.content_sha256:
        raise ExperimentPlanError("experiment plan canonical payload mismatch")
    return plan


def decode_plan(data: bytes) -> ExperimentPlanV1:
    """Decode canonical UTF-8 JSON and reject duplicates/non-canonical bytes."""

    if type(data) is not bytes:
        raise TypeError("plan data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ExperimentPlanError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExperimentPlanError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentPlanError("invalid experiment plan JSON") from exc
    plan = plan_from_document(value)
    if plan.canonical_bytes() != data:
        raise ExperimentPlanError("experiment plan JSON is not canonical")
    return plan


__all__ = [
    "CONFIRMATORY_ALPHA_V1",
    "CONFIRMATORY_BYTE_BUDGETS_V1",
    "CONFIRMATORY_CONDITION_IDS_V1",
    "CONFIRMATORY_METHOD_IDS_V1",
    "CONFIRMATORY_NETWORK_SEEDS_V1",
    "CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1",
    "CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1",
    "CONFIRMATORY_RANDOM_SEED_V1",
    "CONFIRMATORY_RESAMPLES_V1",
    "CONFIRMATORY_SCHEDULER_IDS_V1",
    "CONFIRMATORY_TRAINING_SEEDS_V1",
    "GRIFFIN_25M_VAL_SEQUENCE_IDS_V1",
    "ExperimentPlanError",
    "ExperimentPlanV1",
    "RunConfigBindingV1",
    "confirmatory_run_config_keys_v1",
    "decode_plan",
    "plan_from_document",
]
