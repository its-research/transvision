#!/usr/bin/env python3
"""Create-once, CPU-only synthetic proof checks for recoverable hypotheses.

No dataset, model checkpoint, GT file, tracking benchmark, or network endpoint
is accepted. A successful report verifies local algebra/inference invariants;
it is not evidence of a trained tracking model or a publishable improvement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from transvision.models.event_track_v2x.hypothesis_bank import (
    EvidenceUpdate, HypothesisBank, LogAssociationFactors, ambiguity_components,
    assignment_log_weight, component_factors, enumerate_assignments,
    logsumexp, posterior_omitted_mass, regret_upper_bound, row_relaxation_log_upper,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cartesian_oracle(factors):
    n, m = factors.shape
    weights = {}
    for choices in itertools.product(range(-1, m), repeat=n):
        used = [column for column in choices if column >= 0]
        if len(used) != len(set(used)):
            continue
        if any(column >= 0 and not factors.allowed[row][column] for row, column in enumerate(choices)):
            continue
        weight = math.prod(math.exp(factors.log_left_unmatched[row] if column < 0
                                    else factors.log_pair[row][column]) for row, column in enumerate(choices))
        weight *= math.prod(math.exp(value) for column, value in enumerate(factors.log_right_unmatched)
                            if column not in used)
        weights[choices] = weight
    return weights


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_partitions(rng, trials):
    results = []
    for number in range(trials):
        n, m = number % 5, (number // 5) % 5
        factors = LogAssociationFactors(rng.normal(size=(n, m)), rng.normal(size=n), rng.normal(size=m),
                                        rng.uniform(size=(n, m)) >= .25)
        weights = cartesian_oracle(factors)
        z = math.fsum(weights.values())
        exact = enumerate_assignments(factors)
        must(set(weights) == {item.choices for item in exact}, "legal assignment coverage differs")
        must(math.isclose(logsumexp(item.log_weight for item in exact), math.log(z), abs_tol=1e-11),
             "recursive and Cartesian oracle partitions differ")
        components = ambiguity_components(factors)
        component_log_z = sum(logsumexp(item.log_weight for item in enumerate_assignments(component_factors(factors, component)))
                              for component in components)
        must(math.isclose(component_log_z, math.log(z), abs_tol=1e-11), "component factorization differs")
        bank = HypothesisBank(factors, active_limit=2)
        states = []
        prior_bound = math.inf
        for decision in range(8):
            snapshot = bank.advance(decision_us=decision, expansion_budget=2)
            active_choices = {item.choices for item in snapshot.active}
            retained = math.fsum(weights[choices] for choices in active_choices)
            eta_exact = max(0., 1. - retained / z)
            must(snapshot.eta_upper + 1e-11 >= eta_exact, "mass certificate underestimates exact omitted mass")
            must(snapshot.log_partition_upper + 1e-11 >= math.log(z), "partition certificate underestimates Z")
            must(snapshot.log_partition_upper <= prior_bound + 1e-11, "refinement increased partition bound")
            prior_bound = snapshot.log_partition_upper
            for choices in weights:
                coverage = int(choices in active_choices) + sum(choices[:len(prefix)] == prefix
                                                                for prefix in bank.frontier_constraints)
                must(coverage == 1, "frontier lost or duplicated an assignment")
            states.append({"decision": decision, "eta_exact": eta_exact, "eta_upper": snapshot.eta_upper,
                           "log_partition_upper": snapshot.log_partition_upper,
                           "expansions": snapshot.expansions, "frontier": snapshot.frontier_count})
        results.append({"case": number, "shape": [n, m], "legal_hypotheses": len(weights),
                        "components": len(components), "exact_log_z": math.log(z),
                        "row_relaxation_log_z_upper": row_relaxation_log_upper(factors), "states": states})
    return results


def verify_algebra(rng, trials):
    max_formula_error = 0.
    min_model_margin = math.inf
    min_approximate_margin = math.inf
    for _ in range(trials):
        eta = rng.uniform(.001, .999)
        u, v = np.exp(rng.uniform(-10, 10, size=2))
        eta_exact = eta * v / ((1 - eta) * u + eta * v)
        eta_computed = posterior_omitted_mass(eta, log_likelihood_ratio=math.log(v / u))
        max_formula_error = max(max_formula_error, abs(eta_exact - eta_computed))
        must(abs(eta_exact - eta_computed) < 1e-12, "Bayes omitted-mass formula failed")
        upper = posterior_omitted_mass(eta, log_likelihood_ratio=math.log(v / u) + 1.)
        must(eta_exact <= upper + 1e-12, "finite likelihood-ratio bound failed")
        model = rng.dirichlet(np.ones(7))
        indices = rng.choice(7, size=3, replace=False)
        omitted = 1. - model[indices].sum()
        conditional = np.zeros(7)
        conditional[indices] = model[indices] / (1 - omitted)
        approximate = .95 * conditional + .05 * rng.dirichlet(np.ones(7))
        truth = .9 * model + .1 * rng.dirichlet(np.ones(7))
        loss = rng.uniform(size=(5, 7))
        eps = .5 * sum(abs(approximate - conditional))
        zeta = .5 * sum(abs(truth - model))
        model_risks = loss @ model
        true_risks = loss @ truth
        conditional_regret = model_risks[np.argmin(loss @ conditional)] - min(model_risks)
        approximate_regret = true_risks[np.argmin(loss @ approximate)] - min(true_risks)
        model_margin = omitted - conditional_regret
        approximate_margin = regret_upper_bound(omitted, epsilon=eps, zeta=zeta) - approximate_regret
        must(model_margin >= -1e-12, "shared-action Bayes regret bound failed")
        must(approximate_margin >= -1e-12, "explicit-TV regret bound failed")
        min_model_margin = min(min_model_margin, float(model_margin))
        min_approximate_margin = min(min_approximate_margin, float(approximate_margin))
    amplified = posterior_omitted_mass(.001, log_likelihood_ratio=math.log(1e6))
    false_ratio_bound = posterior_omitted_mass(.001, log_likelihood_ratio=math.log(2.))
    must(amplified > .99 and amplified > false_ratio_bound, "counterexample was not realized")
    zero_one_loss = np.array([[0., 1.], [1., 0.]])
    certain_model = np.array([1., 0.])
    full_risks = zero_one_loss @ certain_model
    restricted_regret = float(full_risks[1] - min(full_risks))
    must(restricted_regret > 0., "unequal-action counterexample was not realized")
    wrong_model = np.array([.999, .001])
    truth = wrong_model[::-1]
    truth_risks = zero_one_loss @ truth
    wrong_action = int(np.argmin(zero_one_loss @ wrong_model))
    truth_regret = float(truth_risks[wrong_action] - min(truth_risks))
    explicit_zeta = float(.5 * np.abs(truth - wrong_model).sum())
    model_only_bound = .001
    must(truth_regret > model_only_bound
         and truth_regret <= regret_upper_bound(.001, zeta=explicit_zeta),
         "uncalibrated-model counterexample was not realized")
    # Two apparent one-bit components share a cross-component interaction.
    # Dropping it would multiply their independent partitions and falsely
    # certify both the normalizer and any omitted-mass bound derived from it.
    cross_weights = {(0, 0): 10., (0, 1): 1., (1, 0): 1., (1, 1): 10.}
    coupled_z = math.fsum(cross_weights.values())
    false_product_z = 2. * 2.
    must(coupled_z != false_product_z,
         "invalid component-independence premise was not exposed")
    return {"random_cases": trials, "max_bayes_formula_absolute_error": max_formula_error,
            "minimum_regret_bound_slack": min_model_margin,
            "minimum_explicit_tv_regret_bound_slack": min_approximate_margin,
            "counterexamples": {
                "likelihood_amplification": {"initial_omitted": .001, "likelihood_ratio": 1e6,
                                              "updated_omitted": amplified, "invalid_assumed_ratio": 2.,
                                              "invalid_bound": false_ratio_bound},
                "unequal_action_sets": {"eta": 0., "regret_after_deleting_optimal_action": restricted_regret,
                                         "claimed_bound_without_action_assumption": 0.},
                "uncalibrated_model": {"model": wrong_model.tolist(), "truth": truth.tolist(),
                                       "truth_regret": truth_regret, "model_only_bound": model_only_bound,
                                       "explicit_zeta": explicit_zeta,
                                       "valid_with_explicit_zeta": regret_upper_bound(.001, zeta=explicit_zeta)},
                "dependent_components": {"cross_weights": [10., 1., 1., 10.],
                                         "true_partition": coupled_z,
                                         "invalid_independent_product": false_product_z,
                                         "premise_rejected": True},
            }, "epsilon_and_zeta": "computed against synthetic known distributions only; not estimated for real data"}


def verify_recovery(active_limit):
    initial = LogAssociationFactors.from_positive([[1000., 1.], [1., 1000.]], [.01, .01], [.01, .01])
    likelihood = LogAssociationFactors.from_positive([[1., 1e8], [1e8, 1.]], [1., 1.], [1., 1.])
    evidence = EvidenceUpdate("synthetic-late-identity-disambiguation", 20, likelihood)
    transcripts = []
    for _ in range(2):
        bank = HypothesisBank(initial, active_limit=active_limit)
        before = bank.advance(decision_us=10, expansion_budget=2)
        must((1,) in bank.frontier_constraints and (1, 0) not in bank._discovered,
             "reversal assignment should still be unexpanded")
        try:
            bank.advance(decision_us=19, expansion_budget=1, evidence=evidence)
            raise AssertionError("future evidence accepted")
        except ValueError as exc:
            must("future" in str(exc), "unexpected causality failure")
        after = bank.advance(decision_us=20, expansion_budget=1, evidence=evidence)
        must(after.active[0].choices == (1, 0), "unknown branch not recovered")
        must(any(event.choices == (1, 0) and not event.previously_enumerated for event in after.recovery_events),
             "recovery must be from never-enumerated constraints")
        must(bank.advance(decision_us=21, expansion_budget=100, evidence=evidence) is after,
             "duplicate evidence was not idempotent")
        must(bank.commits[0] is before, "historical commit replaced")
        transcripts.append(canonical([asdict(item) for item in bank.commits]))
    must(transcripts[0] == transcripts[1], "replay is not byte-identical")
    updated_factors = initial.add(likelihood)
    fixed = max(before.active, key=lambda item: assignment_log_weight(updated_factors, item.choices))
    exact = enumerate_assignments(updated_factors)
    z = logsumexp(item.log_weight for item in exact)
    must(fixed.choices != (1, 0) and exact[0].choices == after.active[0].choices,
         "fixed retained identities must lose to the recovered oracle identity")
    return {"active_limit": active_limit, "before": asdict(before), "after": asdict(after),
            "fixed_top_k_after_choices": fixed.choices, "exhaustive_map_choices": exact[0].choices,
            "recovered_map_posterior": math.exp(exact[0].log_weight - z),
            "transcript_sha256": hashlib.sha256(transcripts[0]).hexdigest(),
            "replay_byte_equal": True,
            "comparison_scope": "same initial search and subsequent split allowance; not an equal-wall-time or equal-byte benchmark"}


def verify_resource_limits():
    factors = LogAssociationFactors(np.zeros((3, 3)), np.zeros(3), np.zeros(3))
    bank = HypothesisBank(factors, active_limit=1, max_frontier_nodes=2, max_discovered_leaves=2)
    snapshot = bank.advance(decision_us=0, expansion_budget=1000)
    must(snapshot.resource_limited and snapshot.expansions == 0, "resource cap failed closed")
    must(bank.frontier_constraints == ((),), "resource exhaustion dropped unresolved support")
    return {"snapshot": asdict(snapshot), "all_support_retained_as_root_prefix": True,
            "max_frontier_nodes": 2, "max_discovered_leaves": 2,
            "history_limits": "bank defaults: 128 evidence updates, 1024 immutable commits; exhaustion raises before mutation"}


def verify_extremes():
    records = []
    for scale in (-1000., 1000.):
        factors = LogAssociationFactors([[scale, -scale], [-scale, scale]], [scale, -scale], [-scale, scale])
        snapshot = HypothesisBank(factors, active_limit=2).advance(decision_us=0, expansion_budget=5)
        z = logsumexp(item.log_weight for item in enumerate_assignments(factors))
        must(math.isfinite(snapshot.log_partition_upper) and snapshot.log_partition_upper >= z - 1e-10,
             "extreme log potential failed")
        records.append({"log_potential_magnitude": abs(scale), "exact_log_z": z,
                        "log_partition_upper": snapshot.log_partition_upper, "eta_upper": snapshot.eta_upper})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; reports are create-once")
    if not 1 <= args.trials <= 4096 or args.seed < 0:
        parser.error("trials must be in [1,4096] and seed nonnegative")
    sources = [ROOT / "transvision/models/event_track_v2x/hypothesis_bank.py", Path(__file__).resolve(),
               ROOT / "tests/event_track_v2x/test_hypothesis_bank.py"]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    rng = np.random.default_rng(args.seed)
    report = {"kind": "recoverable_hypotheses_theory_verification_v1", "status": "verified",
              "seed": args.seed, "python_version": platform.python_version(), "numpy_version": np.__version__,
              "source_hashes": source_hashes,
              "partitions": verify_partitions(rng, args.trials),
              "algebra": verify_algebra(rng, args.trials * 2),
              "recovery": [verify_recovery(1), verify_recovery(2)],
              "resources": verify_resource_limits(), "extreme_log_potentials": verify_extremes(),
              "scope": {"data": "synthetic local identity factor graphs only", "gt_read": False,
                        "model_checkpoint_read": False, "gpu_used": False, "network_used": False,
                        "complete_tracker": False, "real_dataset_validation": False,
                        "floating_bounds": "deterministic row relaxation with conservative float64 roundoff padding; not interval arithmetic",
                        "likelihood_assumption": "same fixed latent assignment and support, factorized likelihood update",
                        "novelty_claim": False, "publishable_gain_claim": False}}
    must(source_hashes == {str(path.relative_to(ROOT)): digest(path) for path in sources}, "source changed while verifying")
    with args.output.open("xb") as stream:
        stream.write(canonical(report))
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "sha256": digest(args.output), "partition_cases": args.trials,
                      "algebra_cases": args.trials * 2, "recovery_cases": 2}, sort_keys=True))


if __name__ == "__main__":
    main()
