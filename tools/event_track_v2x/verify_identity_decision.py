#!/usr/bin/env python3
"""Create-once synthetic verification of full-action conditional Bayes decoding.

The reference enumerates Cartesian legal actions and sums original log factors
as exact rational numbers before 80-digit Decimal normalization. It does not
call the production assignment solver or production exhaustive oracle. Decimal
reference calculations are not formal interval certificates. No real data,
checkpoints, training, network, GPU, or branch-state integration is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from transvision.models.event_track_v2x.hypothesis_bank import HypothesisBank, LogAssociationFactors, logsumexp
from transvision.models.event_track_v2x.identity_decision import decode_conditional_identity


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_log_energies(factors):
    """Legal action enumeration independent of bank/oracle/Hungarian code."""
    n, m = factors.shape
    energies = {}
    for action in itertools.product(range(-1, m), repeat=n):
        used = [column for column in action if column >= 0]
        if len(set(used)) != len(used) or any(column >= 0 and not factors.allowed[row][column]
                                            for row, column in enumerate(action)):
            continue
        terms = [factors.log_left_unmatched[row] if column < 0 else factors.log_pair[row][column]
                 for row, column in enumerate(action)]
        terms += [weight for column, weight in enumerate(factors.log_right_unmatched) if column not in used]
        energies[action] = sum((Fraction.from_float(float(value)) for value in terms), Fraction(0))
    return energies


def normalize(energies):
    with localcontext() as context:
        context.prec = 80
        maximum = max(energies.values())
        weights = {}
        for action, energy in energies.items():
            difference = energy - maximum
            value = Decimal(difference.numerator) / Decimal(difference.denominator)
            weights[action] = value.exp()
        total = sum(weights.values(), Decimal(0))
        return {action: value / total for action, value in weights.items()}


def risk(action, distribution):
    return (math.fsum(float(probability) * sum(left != right for left, right in zip(action, hypothesis)) / len(action)
                      for hypothesis, probability in distribution.items()) if action else 0.)


def signed(snapshot, **changes):
    value = replace(snapshot, **changes)
    payload = asdict(value)
    payload.pop("commit")
    # Bank snapshots use canonical JSON without a trailing newline.
    return replace(value, commit=hashlib.sha256(canonical(payload).rstrip(b"\n")).hexdigest())


def verify_random_cases(seed, trials):
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(trials):
        n, m = index % 5, (index // 5) % 5
        factors = LogAssociationFactors(rng.normal(size=(n, m)), rng.normal(size=n), rng.normal(size=m),
                                        rng.uniform(size=(n, m)) > .25)
        snapshot = HypothesisBank(factors, active_limit=3).advance(decision_us=0,
                                                                   expansion_budget=index % 13)
        decision = decode_conditional_identity(factors, snapshot)
        energies = exact_log_energies(factors)
        posterior = normalize(energies)
        retained = {leaf.choices for leaf in snapshot.active}
        with localcontext() as context:
            context.prec = 80
            eta_decimal = sum((value for action, value in posterior.items() if action not in retained), Decimal(0))
        eta = float(eta_decimal)
        must(eta <= snapshot.eta_upper + 1e-12, "bank omitted-mass estimate underestimates rational/Decimal reference")
        record = {"case": index, "shape": [n, m], "legal_actions": len(energies),
                  "actual_omitted_mass_decimal_reference": str(eta_decimal),
                  "actual_omitted_mass_float_reference": eta, "bank_eta_upper_estimate": snapshot.eta_upper,
                  "decision": asdict(decision)}
        if retained:
            conditional = normalize({action: energies[action] for action in retained})
            conditional_risks = {action: risk(action, conditional) for action in energies}
            full_risks = {action: risk(action, posterior) for action in energies}
            best_q, best_rho = min(conditional_risks.values()), min(full_risks.values())
            actual_regret = full_risks[decision.choices] - best_rho
            must(abs(conditional_risks[decision.choices] - best_q) < 1e-12,
                 "Hungarian action is not conditional Hamming Bayes-optimal over full support")
            must(abs(decision.conditional_expected_loss - best_q) < 1e-12, "reported conditional loss differs")
            must(decision.model_truncation_regret_upper_estimate is not None and
                 actual_regret <= decision.model_truncation_regret_upper_estimate + 1e-12,
                 "common-action model regret exceeds estimated inherited bound")
            record.update(conditional_bayes_risk_reference=best_q, full_model_bayes_risk_reference=best_rho,
                          actual_full_model_regret=actual_regret)
        else:
            must(decision.status == "unresolved_no_active_hypotheses" and decision.choices is None
                 and decision.model_truncation_regret_upper_estimate is None,
                 "empty retained set fabricated a Bayes action or action regret guarantee")
        cases.append(record)
    return cases


def verify_absent_support_action():
    factors = LogAssociationFactors(np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.eye(3, dtype=bool))
    full = HypothesisBank(factors, active_limit=8).advance(decision_us=0, expansion_budget=20)
    retained = {(0, 1, -1), (0, -1, 2), (-1, 1, 2)}
    active = tuple(leaf for leaf in full.active if leaf.choices in retained)
    frontier = tuple(sorted(leaf.choices for leaf in full.active if leaf.choices not in retained))
    log_retained = logsumexp(leaf.log_weight for leaf in active)
    snapshot = signed(full, active=active, active_limit=3, frontier_count=5, frontier_prefix_slots=15,
        frontier_sha256=hashlib.sha256(canonical(frontier).rstrip(b"\n")).hexdigest(),
        log_retained_weight=log_retained, eta_upper=-math.expm1(log_retained - full.log_partition_upper))
    decision = decode_conditional_identity(factors, snapshot)
    energies = exact_log_energies(factors)
    q = normalize({action: energies[action] for action in retained})
    risks = {action: risk(action, q) for action in energies}
    must(decision.choices == (0, 1, 2) and decision.choices not in retained,
         "Hamming Bayes action must be permitted outside retained hypothesis support")
    must(abs(risks[decision.choices] - 1 / 3) < 1e-12 and min(risks.values()) == risks[decision.choices],
         "absent-support action did not attain full-action Bayes risk")
    return {"retained_choices": sorted(retained), "decision": asdict(decision),
            "conditional_hamming_risk": 1 / 3, "any_retained_MAP_hamming_risk": 4 / 9,
            "whole_hypothesis_zero_one_risk": {"retained_MAP": 2 / 3, "absent_support_action": 1.},
            "snapshot_origin": "synthetic valid retained-set receipt from eight fully enumerated legal states; not a search-order claim",
            "state_adapter_boundary": "non-active Bayes action requires separate state replay; do not restrict it back to active"}


def verify_extreme_offset_and_domain():
    factors = LogAssociationFactors([[1e20]], [1.], [1e20])
    snapshot = HypothesisBank(factors, active_limit=2).advance(decision_us=0, expansion_budget=1)
    decision = decode_conditional_identity(factors, snapshot)
    posterior = normalize(exact_log_energies(factors))
    true_best = min(risk(action, posterior) for action in posterior)
    must(len({leaf.log_weight for leaf in snapshot.active}) == 1, "absolute leaf rounding counterexample disappeared")
    must(decision.choices == (-1,) and abs(decision.conditional_expected_loss - true_best) < 1e-12,
         "raw-factor cancellation failed to recover the correct conditional Bayes action")
    must(decision.model_omitted_mass_upper_estimate is None and
         decision.model_truncation_regret_upper_estimate is None and decision.mass_estimate_unavailable_reason,
         "out-of-domain mass/regret claims were not withheld")
    empty = HypothesisBank(factors).advance(decision_us=0, expansion_budget=0)
    unresolved = decode_conditional_identity(factors, empty)
    must(unresolved.choices is None and unresolved.model_omitted_mass_upper_estimate is None
         and unresolved.model_truncation_regret_upper_estimate is None,
         "out-of-domain empty active set exposed an inconsistent bound")
    return {"factors": {"pair": [[1e20]], "left_unmatched": [1.], "right_unmatched": [1e20]},
            "rounded_absolute_leaf_logs": [leaf.log_weight for leaf in snapshot.active],
            "rational_Decimal_reference_unmatched_probability": str(posterior[(-1,)]),
            "legacy_rounded_uniform_tie_if_match_selected_actual_regret": risk((0,), posterior) - true_best,
            "bank_reported_eta_zero_must_not_rescue_inaccurate_decoder": snapshot.eta_upper,
            "corrected_decision": asdict(decision), "unresolved_out_of_domain": asdict(unresolved),
            "declared_engineering_guard": {"max_left_rows": 64, "max_right_columns": 64, "max_abs_log_factor": 1000.},
            "guard_is_not_a_proof_of_float64_conservatism": True}


def assumption_counterexamples():
    # One left/right pair: Hamming is the same as binary-choice 0-1 loss.
    full = {(0,): .9, (-1,): .1}
    restricted_regret = risk((-1,), full) - min(risk(action, full) for action in full)
    model = {(0,): .999, (-1,): .001}
    truth = {(0,): .001, (-1,): .999}
    truth_regret = risk((0,), truth) - min(risk(action, truth) for action in truth)
    zeta = .5 * sum(abs(model[action] - truth[action]) for action in model)
    must(restricted_regret > 0. and truth_regret > .001, "assumption counterexamples did not violate omitted premise")
    return {"removed_action": {"eta": 0., "full_model": [.9, .1],
                                "deleted_action": "matched", "forced_action": "unmatched",
                                "actual_regret": restricted_regret, "invalid_L_eta_claim": 0.},
            "model_miscalibration": {"model_matched_unmatched": [.999, .001],
                "synthetic_truth_matched_unmatched": [.001, .999], "model_eta": .001,
                "actual_truth_regret": truth_regret, "known_synthetic_TV_zeta": zeta,
                "explicit_zeta_loose_bound_capped_at_one": min(1., .001 + 2 * zeta)},
            "scope": "constructed assumption failures; no real-data calibration error is estimated"}


def verification(seed, trials):
    return {"seed": seed, "random_cases": verify_random_cases(seed, trials),
            "absent_support_bayes_action": verify_absent_support_action(),
            "extreme_common_offset": verify_extreme_offset_and_domain(),
            "assumption_counterexamples": assumption_counterexamples()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; verification report is create-once")
    if not 0 <= args.seed < 2**63 or not 1 <= args.trials <= 1024:
        parser.error("seed must be in [0,2**63), trials in [1,1024]")
    paths = [ROOT / "transvision/models/event_track_v2x/hypothesis_bank.py",
             ROOT / "transvision/models/event_track_v2x/identity_decision.py", Path(__file__).resolve(),
             ROOT / "tests/event_track_v2x/test_identity_decision.py",
             ROOT / "tests/event_track_v2x/test_identity_decision_cli.py"]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    first, replay = verification(args.seed, args.trials), verification(args.seed, args.trials)
    must(canonical(first) == canonical(replay), "fresh verification replay is not byte-identical")
    must(sources == {str(path.relative_to(ROOT)): digest(path) for path in paths}, "source changed during verification")
    report = {"kind": "conditional_bayes_identity_decision_verification_v1", "status": "verified",
              "source_hashes": sources, "runtime": {"python": platform.python_version(),
                                                      "numpy": np.__version__, "scipy": scipy.__version__},
              "fresh_verification_replay_byte_equal": True, "verification": first,
              "scope": {"reference": "Cartesian legal actions; exact rational log sums and 80-digit Decimal normalization",
                        "exhaustively_tested_shapes": "left/right counts 0..4, specified random cases only",
                        "loss": "normalized left-choice Hamming, not HOTA/IDF1 or whole-hypothesis 0-1",
                        "input_data": "synthetic finite factor models only", "real_data_read": False,
                        "checkpoint_read": False, "training": False, "network_access": False, "gpu_used": False,
                        "formal_interval_certificate": False, "real_posterior_zeta_estimated": False,
                        "decoder_integrated_with_branch_states": False, "frozen_pipeline_modified": False,
                        "nonactive_action": "must replay its state separately, never restrict actions to retained set",
                        "old_theory_report": "unchanged; this is a separate extreme-numerics and decoder supplement"}}
    with args.output.open("xb") as stream:
        stream.write(canonical(report))
    print(json.dumps({"status": "verified", "output": str(args.output.resolve()), "sha256": digest(args.output),
                      "random_cases": args.trials, "fresh_verification_replay_byte_equal": True}, sort_keys=True))


if __name__ == "__main__":
    main()
