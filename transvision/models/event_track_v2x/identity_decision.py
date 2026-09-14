"""Conditional Bayes decoding on the full legal partial-matching action space.

The declared loss is the fraction of left rows whose column-or-unmatched choice
is incorrect. The posterior is the retained conditional *factor-model* posterior.
Actions are not restricted to retained leaves: one-to-one Hungarian decoding
maximizes row marginals over every allowed partial assignment. Numerical bounds
remain float64 estimates, not interval certificates or real-world/HOTA bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hypothesis_bank import (
    ActiveBranch, BankSnapshot, LogAssociationFactors, assignment_log_weight, logsumexp,
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    status: str
    decision_us: int
    identity_commit: str
    factors_sha256: str
    choices: tuple[int, ...] | None
    pairs: tuple[tuple[int, int], ...] | None
    unmatched_left: tuple[int, ...] | None
    unmatched_right: tuple[int, ...] | None
    action_in_retained_set: bool | None
    conditional_expected_loss: float | None
    conditional_row_marginals: tuple[tuple[float, ...], ...]
    model_omitted_mass_upper_estimate: float | None
    bank_reported_omitted_mass_estimate: float
    model_truncation_regret_upper_estimate: float | None
    mass_estimate_status: str
    mass_estimate_unavailable_reason: str | None
    retained_hypotheses: int
    loss_range: float
    loss_definition: str = "normalized_left_choice_Hamming; -1 means unmatched; zero rows have zero loss"
    marginal_column_order: str = "right columns 0..M-1, then unmatched"
    action_space: str = "all_allowed_one_to_one_partial_matchings_not_only_retained_hypotheses"
    posterior_scope: str = "conditional_retained_factor_model_not_calibrated_real_posterior"
    numerical_scope: str = "float64_decoder_and_inherited_relaxation_estimate_not_interval_certificates"
    true_posterior_or_tracking_metric_bound: bool = False


def _validate(factors, snapshot):
    if type(factors) is not LogAssociationFactors or type(snapshot) is not BankSnapshot:
        raise TypeError("exact LogAssociationFactors and BankSnapshot instances are required")
    payload = asdict(snapshot)
    commit = payload.pop("commit")
    if commit != _digest(payload):
        raise ValueError("identity snapshot payload hash differs")
    if factors.digest() != snapshot.current_factors_sha256:
        raise ValueError("supplied factors differ from the current identity snapshot digest")
    if not math.isfinite(snapshot.eta_upper) or not 0 <= snapshot.eta_upper <= 1:
        raise ValueError("invalid model omitted-mass estimate")
    if not math.isfinite(snapshot.log_partition_upper):
        raise ValueError("invalid model partition estimate")
    seen = set()
    for leaf in snapshot.active:
        if type(leaf) is not ActiveBranch or not isinstance(leaf.choices, tuple):
            raise ValueError("active hypotheses must be distinct ActiveBranch instances")
        if leaf.choices in seen:
            raise ValueError("active hypotheses must be distinct ActiveBranch instances")
        expected_weight = assignment_log_weight(factors, leaf.choices)
        expected_id = _digest({"original": snapshot.original_factors_sha256, "prefix": leaf.choices})
        expected_parent = (_digest({"original": snapshot.original_factors_sha256, "prefix": leaf.choices[:-1]})
                           if leaf.choices else "0" * 64)
        if (type(leaf.log_weight) is not float or leaf.log_weight != expected_weight
                or leaf.branch_id != expected_id or leaf.parent_id != expected_parent
                or leaf.evidence_history != snapshot.evidence_ids):
            raise ValueError("active hypothesis weight, identity or evidence binding differs")
        seen.add(leaf.choices)
    if snapshot.active:
        log_retained = logsumexp(leaf.log_weight for leaf in snapshot.active)
        if snapshot.log_retained_weight != log_retained or snapshot.log_partition_upper < log_retained:
            raise ValueError("retained weight or partition estimate is inconsistent")
        expected_eta = (0. if snapshot.frontier_count == 0 else
                        min(1., max(0., -math.expm1(min(0., log_retained - snapshot.log_partition_upper)))))
        if snapshot.eta_upper != expected_eta:
            raise ValueError("omitted-mass estimate is inconsistent with retained and upper weights")
    elif snapshot.log_retained_weight is not None or snapshot.eta_upper != 1.:
        raise ValueError("empty active set must have unresolved mass and no normalizer")


def _factor_terms(factors, choices):
    terms = {}
    used = {column for column in choices if column >= 0}
    for row, column in enumerate(choices):
        if column < 0:
            terms[("left", row)] = factors.log_left_unmatched[row]
        else:
            terms[("pair", row, column)] = factors.log_pair[row][column]
    for column, value in enumerate(factors.log_right_unmatched):
        if column not in used:
            terms[("right", column)] = value
    return terms


def _relative_log_weights(factors, active):
    """Cancel common factor occurrences before forming an absolute log weight.

    Computing two large absolute log weights first can erase a small but
    decision-changing difference. math.fsum on signed original terms preserves
    that difference in common-offset examples; this is not interval arithmetic.
    """
    anchor = _factor_terms(factors, active[0].choices)
    result = []
    for leaf in active:
        current = _factor_terms(factors, leaf.choices)
        try:
            relative = math.fsum([*(value for key, value in current.items() if key not in anchor),
                                  *(-value for key, value in anchor.items() if key not in current)])
        except OverflowError as error:
            raise ValueError("relative factor arithmetic exceeds float64 range") from error
        if not math.isfinite(relative):
            raise ValueError("relative factor arithmetic exceeds float64 range")
        result.append(relative)
    return result


def _mass_numeric_domain(factors):
    n, m = factors.shape
    values = [*factors.log_left_unmatched, *factors.log_right_unmatched,
              *(value for row in factors.log_pair for value in row)]
    if n > 64 or m > 64:
        return "component dimensions exceed the tested 64x64 engineering numeric domain"
    if any(abs(value) > 1000. for value in values):
        return "absolute log factors exceed the tested magnitude-1000 engineering numeric domain"
    return None


def decode_conditional_identity(factors: LogAssociationFactors, snapshot: BankSnapshot) -> IdentityDecision:
    """Minimize conditional expected left-choice Hamming loss over full support.

    A complete action may be absent from the retained posterior support. This
    is necessary for marginal/Hamming Bayes optimality. Dedicated dummy columns
    allow every row its unmatched action without sharing an unmatched capacity.

    The bank's mass estimate is inherited after binding/integrity checks; this
    decoder does not independently certify the omitted frontier. L*eta applies
    mathematically to exact common-action Bayes decoding and a valid mass bound;
    this numerical return is not an interval or true-posterior risk guarantee.
    Inherited eta/regret estimates are withheld outside the explicitly tested
    engineering domain (each side <=64; absolute log factors <=1000). This
    domain restriction is NOT a proof of floating-point conservatism within it.
    """
    _validate(factors, snapshot)
    n, m = factors.shape
    unavailable = _mass_numeric_domain(factors)
    shared = dict(decision_us=snapshot.decision_us, identity_commit=snapshot.commit,
                  factors_sha256=factors.digest(),
                  model_omitted_mass_upper_estimate=snapshot.eta_upper if unavailable is None else None,
                  bank_reported_omitted_mass_estimate=snapshot.eta_upper,
                  mass_estimate_status=("inherited_float64_estimate_in_tested_domain_not_certified" if unavailable is None
                                        else "unavailable_outside_tested_numeric_domain"),
                  mass_estimate_unavailable_reason=unavailable,
                  retained_hypotheses=len(snapshot.active), loss_range=1. if n else 0.)
    if not snapshot.active:
        return IdentityDecision(status="unresolved_no_active_hypotheses", choices=None, pairs=None,
            unmatched_left=None, unmatched_right=None, action_in_retained_set=None,
            conditional_expected_loss=None, conditional_row_marginals=(),
            model_truncation_regret_upper_estimate=None, **shared)

    # Normalize differences of original factors, not already-rounded absolute
    # leaf logs. The latter may erase both log(K) and relative evidence itself.
    relative_logs = _relative_log_weights(factors, snapshot.active)
    maximum = max(relative_logs)
    relative = [math.exp(value - maximum) for value in relative_logs]
    normalizer = math.fsum(relative)
    probabilities = [value / normalizer for value in relative]
    cells = [[[] for _ in range(m + 1)] for _ in range(n)]
    for leaf, probability in zip(snapshot.active, probabilities):
        for row, column in enumerate(leaf.choices):
            cells[row][m if column < 0 else column].append(probability)
    marginals = np.asarray([[math.fsum(values) for values in row] for row in cells], dtype=float).reshape(n, m + 1)
    costs = np.full((n, m + n), np.inf)
    for row in range(n):
        for column in range(m):
            if factors.allowed[row][column]:
                costs[row, column] = -marginals[row, column]
        costs[row, m + row] = -marginals[row, m]
    if n:
        rows, columns = linear_sum_assignment(costs)
        if tuple(rows) != tuple(range(n)):
            raise RuntimeError("Bayes action solver did not assign every left row")
        choices = tuple(int(column) if column < m else -1 for column in columns)
    else:
        choices = ()
    assignment_log_weight(factors, choices)  # Independent legality check; action need not be active.
    expected_loss = (math.fsum(probability * sum(left != right for left, right in zip(choices, leaf.choices)) / n
                               for leaf, probability in zip(snapshot.active, probabilities)) if n else 0.)
    used = {column for column in choices if column >= 0}
    return IdentityDecision(status="resolved_conditional_model_bayes", choices=choices,
        pairs=tuple((row, column) for row, column in enumerate(choices) if column >= 0),
        unmatched_left=tuple(row for row, column in enumerate(choices) if column < 0),
        unmatched_right=tuple(column for column in range(m) if column not in used),
        action_in_retained_set=any(choices == leaf.choices for leaf in snapshot.active),
        conditional_expected_loss=expected_loss,
        conditional_row_marginals=tuple(tuple(float(value) for value in row) for row in marginals),
        model_truncation_regret_upper_estimate=((snapshot.eta_upper if n else 0.) if unavailable is None else None),
        **shared)
