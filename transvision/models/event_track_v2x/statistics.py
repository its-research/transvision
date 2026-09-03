"""Preregistered robust aggregation and paired sequence-level statistics.

Frames are deliberately absent from these interfaces: the inferential unit is
the sequence.  Network and training seeds are collapsed inside each
sequence-condition cell before conditions and sequences receive equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .network import DEFAULT_NETWORK_SEEDS_V1, NetworkConditionId


DEFAULT_TRAINING_SEEDS_V1 = (1337, 2027, 3407)
ROBUST_SYNTHETIC_CONDITIONS_V1 = tuple(
    condition.value
    for condition in (
        NetworkConditionId.C1,
        NetworkConditionId.C2,
        NetworkConditionId.C3,
        NetworkConditionId.C4,
        NetworkConditionId.C5,
        NetworkConditionId.C6,
        NetworkConditionId.C7,
        NetworkConditionId.C8,
    )
)
ROBUST_CONDITIONS_V1 = ROBUST_SYNTHETIC_CONDITIONS_V1 + ("C9",)


def _nonempty(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _seed(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class MetricObservationV1:
    method: str
    metric: str
    sequence_id: str
    condition_id: str
    network_seed: int | None
    training_seed: int
    value: float | None
    failed: bool = False
    replicate_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("method", "metric", "sequence_id", "condition_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if (self.network_seed is None) == (self.replicate_id is None):
            raise ValueError("exactly one of network_seed and replicate_id is required")
        if self.network_seed is not None:
            object.__setattr__(
                self,
                "network_seed",
                _seed(self.network_seed, "network_seed"),
            )
        if self.replicate_id is not None:
            object.__setattr__(
                self,
                "replicate_id",
                _nonempty(self.replicate_id, "replicate_id"),
            )
        object.__setattr__(
            self,
            "training_seed",
            _seed(self.training_seed, "training_seed"),
        )
        if not isinstance(self.failed, bool):
            raise TypeError("failed must be bool")
        if self.value is None:
            if not self.failed:
                raise ValueError("a successful observation requires a value")
        else:
            value = float(self.value)
            if not np.isfinite(value):
                raise ValueError("metric value must be finite")
            object.__setattr__(self, "value", value)

    @property
    def replicate_key(self) -> str:
        if self.replicate_id is not None:
            return self.replicate_id
        assert self.network_seed is not None
        return f"seed:{self.network_seed}"


@dataclass(frozen=True, slots=True)
class RobustMacroResultV1:
    method: str
    metric: str
    value: float
    sequence_values: tuple[tuple[str, float], ...]
    condition_values: tuple[tuple[str, float], ...]
    failed_runs: int
    observations: int

    def sequence_score_map(self) -> dict[str, float]:
        return dict(self.sequence_values)


def aggregate_robust_macro_v1(
    observations: list[MetricObservationV1] | tuple[MetricObservationV1, ...],
    *,
    method: str,
    metric: str,
    expected_sequence_ids: Sequence[str],
    condition_ids: tuple[str, ...] = ROBUST_CONDITIONS_V1,
    network_seeds: tuple[int, ...] = DEFAULT_NETWORK_SEEDS_V1,
    condition_replicates: Mapping[str, Sequence[str | int]] | None = None,
    training_seeds: tuple[int, ...] = DEFAULT_TRAINING_SEEDS_V1,
    failure_value: float = 0.0,
) -> RobustMacroResultV1:
    """Compute the equal-condition, equal-sequence robust macro metric.

    The requested cohort must be rectangular and complete within each
    condition.  C1--C8 default to the registered synthetic network seeds; C9
    requires an explicit list of held-out trace IDs.  This permits different
    replicate counts without changing the equal weight of each condition.
    Failed runs are present observations whose values are replaced by
    ``failure_value``; missing runs are an error and can never be silently
    treated as failures.
    """

    method = _nonempty(method, "method")
    metric = _nonempty(metric, "metric")
    conditions = tuple(_nonempty(item, "condition_id") for item in condition_ids)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("condition_ids must be non-empty and unique")
    networks = tuple(_seed(item, "network_seed") for item in network_seeds)
    trainings = tuple(_seed(item, "training_seed") for item in training_seeds)
    if not networks or len(set(networks)) != len(networks):
        raise ValueError("network_seeds must be non-empty and unique")
    if not trainings or len(set(trainings)) != len(trainings):
        raise ValueError("training_seeds must be non-empty and unique")
    failure_value = float(failure_value)
    if not np.isfinite(failure_value):
        raise ValueError("failure_value must be finite")

    if condition_replicates is not None and not isinstance(
        condition_replicates, Mapping
    ):
        raise TypeError("condition_replicates must be a mapping or None")
    supplied_replicates = {} if condition_replicates is None else condition_replicates
    for condition in supplied_replicates:
        _nonempty(condition, "condition replicate policy key")
    unknown_policies = set(supplied_replicates) - set(conditions)
    if unknown_policies:
        raise ValueError(
            f"replicate policies supplied for unrequested conditions: {sorted(unknown_policies)}"
        )

    def normalise_replicate(value: str | int, condition: str) -> str:
        if isinstance(value, bool):
            raise ValueError(f"invalid replicate ID for {condition}")
        if isinstance(value, int):
            return f"seed:{_seed(value, 'network_seed')}"
        return _nonempty(value, f"{condition} replicate_id")

    replicate_policy: dict[str, tuple[str, ...]] = {}
    for condition in conditions:
        if condition in supplied_replicates:
            raw_identifiers = supplied_replicates[condition]
            if isinstance(raw_identifiers, (str, bytes)) or not isinstance(
                raw_identifiers, Sequence
            ):
                raise ValueError(f"replicate policy for {condition} must be a sequence")
            if condition == "C9" and any(
                not isinstance(value, str) for value in raw_identifiers
            ):
                raise ValueError("C9 held-out replicate IDs must be strings")
            identifiers = tuple(
                normalise_replicate(value, condition)
                for value in raw_identifiers
            )
        elif condition in ROBUST_SYNTHETIC_CONDITIONS_V1:
            identifiers = tuple(f"seed:{seed}" for seed in networks)
        elif condition == "C9":
            raise ValueError(
                "C9 held-out trace replicate IDs must be supplied explicitly"
            )
        else:
            raise ValueError(f"condition {condition} requires an explicit replicate policy")
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"replicate IDs for {condition} must be non-empty and unique")
        replicate_policy[condition] = identifiers

    sequences = tuple(
        _nonempty(item, "expected_sequence_id") for item in expected_sequence_ids
    )
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("expected_sequence_ids must be non-empty and unique")
    sequences = tuple(sorted(sequences))

    selected = [
        item
        for item in observations
        if item.method == method
        and item.metric == metric
        and item.condition_id in conditions
    ]
    if not selected:
        raise ValueError("no observations match the requested method and metric")
    cells: dict[tuple[str, str, int, str], MetricObservationV1] = {}
    for item in selected:
        key = (
            item.sequence_id,
            item.condition_id,
            item.training_seed,
            item.replicate_key,
        )
        if key in cells:
            raise ValueError(f"duplicate robust metric cell: {key}")
        if (
            item.replicate_key not in replicate_policy[item.condition_id]
            or item.training_seed not in trainings
        ):
            raise ValueError(f"unregistered replicate in robust metric cell: {key}")
        cells[key] = item

    expected = {
        (sequence, condition, training, replicate)
        for sequence in sequences
        for condition in conditions
        for training in trainings
        for replicate in replicate_policy[condition]
    }
    actual = set(cells)
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(map(str, missing[:3]))
        raise ValueError(f"missing robust metric cells ({len(missing)}): {preview}")
    extra = sorted(actual - expected)
    if extra:
        preview = ", ".join(map(str, extra[:3]))
        raise ValueError(f"unexpected robust metric cells ({len(extra)}): {preview}")

    failed_runs = sum(int(item.failed) for item in cells.values())

    def value_for(key: tuple[str, str, int, str]) -> float:
        item = cells[key]
        if item.failed:
            return failure_value
        assert item.value is not None
        return item.value

    sequence_condition: dict[tuple[str, str], float] = {}
    for sequence in sequences:
        for condition in conditions:
            training_values = []
            for training in trainings:
                training_values.append(
                    float(
                        np.mean(
                            [
                                value_for((sequence, condition, training, replicate))
                                for replicate in replicate_policy[condition]
                            ]
                        )
                    )
                )
            sequence_condition[(sequence, condition)] = float(
                np.mean(training_values)
            )

    sequence_values = tuple(
        (
            sequence,
            float(
                np.mean(
                    [sequence_condition[(sequence, condition)] for condition in conditions]
                )
            ),
        )
        for sequence in sequences
    )
    condition_values = tuple(
        (
            condition,
            float(
                np.mean(
                    [sequence_condition[(sequence, condition)] for sequence in sequences]
                )
            ),
        )
        for condition in conditions
    )
    return RobustMacroResultV1(
        method=method,
        metric=metric,
        value=float(np.mean([value for _, value in sequence_values])),
        sequence_values=sequence_values,
        condition_values=condition_values,
        failed_runs=failed_runs,
        observations=len(cells),
    )


def _paired_differences(
    treatment: Mapping[str, float], control: Mapping[str, float]
) -> tuple[tuple[str, ...], np.ndarray]:
    treatment_keys = set(treatment)
    control_keys = set(control)
    if treatment_keys != control_keys:
        missing_treatment = sorted(control_keys - treatment_keys)
        missing_control = sorted(treatment_keys - control_keys)
        raise ValueError(
            "paired sequence cohorts differ: "
            f"missing treatment={missing_treatment}, missing control={missing_control}"
        )
    if len(treatment_keys) < 2:
        raise ValueError("paired sequence inference requires at least two sequences")
    sequence_ids = tuple(sorted(treatment_keys))
    differences = np.asarray(
        [float(treatment[key]) - float(control[key]) for key in sequence_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired sequence scores must be finite")
    return sequence_ids, differences


@dataclass(frozen=True, slots=True)
class PairedBootstrapResultV1:
    estimate: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    resamples: int
    seed: int
    sequence_ids: tuple[str, ...]


def paired_sequence_bca_v1(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 1337,
) -> PairedBootstrapResultV1:
    """BCa confidence interval for the paired mean sequence difference."""

    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    resamples = _seed(resamples, "resamples")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    seed = _seed(seed, "seed")
    sequence_ids, differences = _paired_differences(treatment, control)
    estimate = float(np.mean(differences))
    if np.all(differences == differences[0]):
        return PairedBootstrapResultV1(
            estimate=estimate,
            confidence_lower=estimate,
            confidence_upper=estimate,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
            sequence_ids=sequence_ids,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(resamples, len(differences)),
    )
    bootstrap = np.mean(differences[indices], axis=1)
    below = np.count_nonzero(bootstrap < estimate)
    equal = np.count_nonzero(bootstrap == estimate)
    probability = (below + 0.5 * equal) / resamples
    probability = float(np.clip(probability, 0.5 / resamples, 1.0 - 0.5 / resamples))
    normal = NormalDist()
    bias = normal.inv_cdf(probability)

    total = float(np.sum(differences))
    jackknife = (total - differences) / (len(differences) - 1)
    jackknife_mean = float(np.mean(jackknife))
    centred = jackknife_mean - jackknife
    denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
    acceleration = (
        0.0 if denominator == 0.0 else float(np.sum(centred**3)) / denominator
    )

    alpha = (1.0 - confidence_level) / 2.0

    def adjusted_probability(tail_probability: float) -> float:
        z_alpha = normal.inv_cdf(tail_probability)
        denominator_term = 1.0 - acceleration * (bias + z_alpha)
        if denominator_term == 0.0:
            return tail_probability
        adjusted = normal.cdf(
            bias + (bias + z_alpha) / denominator_term
        )
        return float(np.clip(adjusted, 0.0, 1.0))

    lower_probability = adjusted_probability(alpha)
    upper_probability = adjusted_probability(1.0 - alpha)
    lower, upper = np.quantile(
        bootstrap,
        [lower_probability, upper_probability],
    )
    return PairedBootstrapResultV1(
        estimate=estimate,
        confidence_lower=float(lower),
        confidence_upper=float(upper),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
        sequence_ids=sequence_ids,
    )


class Alternative(str, Enum):
    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two_sided"


@dataclass(frozen=True, slots=True)
class PairedPermutationResultV1:
    estimate: float
    p_value: float
    alternative: Alternative
    permutations: int
    exact: bool
    seed: int
    sequence_ids: tuple[str, ...]


def paired_sequence_permutation_v1(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    alternative: Alternative | str = Alternative.GREATER,
    permutations: int = 10_000,
    seed: int = 1337,
) -> PairedPermutationResultV1:
    """Paired sign-flip permutation test over sequence differences."""

    alternative = Alternative(alternative)
    permutations = _seed(permutations, "permutations")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    seed = _seed(seed, "seed")
    sequence_ids, differences = _paired_differences(treatment, control)
    estimate = float(np.mean(differences))
    exact_count = 1 << len(differences)
    exact = exact_count <= permutations
    if exact:
        indices = np.arange(exact_count, dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(len(differences), dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
        null_statistics = np.mean(signs * differences, axis=1)
        denominator = exact_count
        correction = 0
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(permutations, len(differences)),
        )
        null_statistics = np.mean(signs * differences, axis=1)
        denominator = permutations + 1
        correction = 1

    tolerance = np.finfo(np.float64).eps * max(1.0, abs(estimate)) * 8.0
    if alternative is Alternative.GREATER:
        extreme = np.count_nonzero(null_statistics >= estimate - tolerance)
    elif alternative is Alternative.LESS:
        extreme = np.count_nonzero(null_statistics <= estimate + tolerance)
    else:
        extreme = np.count_nonzero(
            np.abs(null_statistics) >= abs(estimate) - tolerance
        )
    return PairedPermutationResultV1(
        estimate=estimate,
        p_value=float((extreme + correction) / denominator),
        alternative=alternative,
        permutations=exact_count if exact else permutations,
        exact=exact,
        seed=seed,
        sequence_ids=sequence_ids,
    )


@dataclass(frozen=True, slots=True)
class HolmComparisonV1:
    comparison: str
    raw_p_value: float
    adjusted_p_value: float
    reject: bool


def holm_correction_v1(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmComparisonV1, ...]:
    """Return Holm-adjusted p-values in the caller's comparison order."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    original = []
    for name, raw_value in p_values.items():
        name = _nonempty(name, "comparison")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1]")
        original.append((name, value))
    ordered = sorted(original, key=lambda item: (item[1], item[0]))
    adjusted_by_name: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        adjusted_by_name[name] = running
    return tuple(
        HolmComparisonV1(
            comparison=name,
            raw_p_value=value,
            adjusted_p_value=adjusted_by_name[name],
            # The preregistered publication language is strictly p < 0.05.
            # Preserve that open boundary after Holm adjustment as well.
            reject=adjusted_by_name[name] < alpha,
        )
        for name, value in original
    )


__all__ = [
    "Alternative",
    "DEFAULT_TRAINING_SEEDS_V1",
    "HolmComparisonV1",
    "MetricObservationV1",
    "PairedBootstrapResultV1",
    "PairedPermutationResultV1",
    "ROBUST_CONDITIONS_V1",
    "ROBUST_SYNTHETIC_CONDITIONS_V1",
    "RobustMacroResultV1",
    "aggregate_robust_macro_v1",
    "holm_correction_v1",
    "paired_sequence_bca_v1",
    "paired_sequence_permutation_v1",
]
