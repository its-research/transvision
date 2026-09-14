#!/usr/bin/env python3
"""Create-once CPU synthetic learned-potential/budget/state pipeline proof.

Only an output path and random initialization seed are accepted. No real data,
ground truth, checkpoints, network access, training, or lifecycle tracking is
performed. Conditional state replay is engineering CI, not a Bayesian posterior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch

from transvision.models.event_track_v2x.hypothesis_bank import (
    EvidenceUpdate, HypothesisBank, LogAssociationFactors, assignment_log_weight,
)
from transvision.models.event_track_v2x.learned_identity import (
    CausalIdentityInput, RecoverableIdentityModel, detached_log_potentials,
)
from transvision.models.event_track_v2x.recoverable_identity import model_digest
from transvision.models.event_track_v2x.recoverable_states import (
    BranchStateWindow, TrackPrior, TrackletObservation,
)
from transvision.models.event_track_v2x.risk_budget import RiskBudgetAllocator


START = 900_000
DECISIONS = (1_000_000, 1_100_000, 1_200_000)
SPLITS = (3, 4, 1)
SHAPES = {"crossing-2x2": (2, 2), "isolated-1x1": (1, 1)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(call, expected):
    try:
        call()
    except ValueError as error:
        must(expected in str(error), "unexpected rejection: " + str(error))
        return True
    raise AssertionError("invalid operation unexpectedly accepted: " + expected)


def physical_mean(x):
    return np.array([x, 0., 1., 2., 4., 2., 0., .2, 0.], dtype=float)


def observation(component, number, side, index, x, information_us, arrival_us):
    return TrackletObservation(f"{component}-observation-{number}", side, f"{side}-{index}",
                               information_us, arrival_us, physical_mean(x), np.eye(9) * .2,
                               np.float32(.8))


def batch_inputs(n, m):
    left, right = torch.randn(n, 203), torch.randn(m, 203)
    history = torch.randn(4, 203)
    batches = []
    for index, decision in enumerate(DECISIONS):
        count = 2 + index
        info = torch.tensor([920_000, 930_000, 970_000, 1_150_000][:count])
        arrival = torch.tensor([950_000, 950_000, 1_080_000, 1_180_000][:count])
        batches.append(CausalIdentityInput(left.clone(), right.clone(), history[:count].clone(),
            torch.full((n,), 940_000), torch.full((m,), 940_000), info,
            torch.full((n,), 960_000), torch.full((m,), 960_000), arrival,
            torch.arange(count) % 2, decision))
    return tuple(batches)


def delta_factors(new, old):
    return LogAssociationFactors(np.asarray(new.log_pair) - np.asarray(old.log_pair),
        np.asarray(new.log_left_unmatched) - np.asarray(old.log_left_unmatched),
        np.asarray(new.log_right_unmatched) - np.asarray(old.log_right_unmatched), old.allowed)


def independence_counterexample():
    """Omitted high-order coupling invalidates a product-of-components claim."""
    local = LogAssociationFactors.from_positive([[1.]], [1.], [1.])
    states = [HypothesisBank(local, active_limit=1).advance(decision_us=0, expansion_budget=1)
              for _ in range(2)]
    must(all(state.active[0].choices == (-1,) for state in states), "expected local unmatched tie-break")
    independent_z = 2. * 2.
    # Each local variable is unmatched/matched. A factor absent from the local
    # model multiplies ONLY the jointly matched configuration by 100.
    coupled_weights = {"unmatched_unmatched": 1., "unmatched_matched": 1.,
                       "matched_unmatched": 1., "matched_matched": 100.}
    true_joint_z = sum(coupled_weights.values())
    invalid_product_eta = 1. - np.prod([1. - state.eta_upper for state in states])
    true_omitted = 1. - coupled_weights["unmatched_unmatched"] / true_joint_z
    must(independent_z == 4. and true_joint_z == 103. and invalid_product_eta < true_omitted,
         "unrecorded dependence counterexample did not violate the product claim")
    return {"kind": "assumption_counterexample_not_algorithm_failure",
            "local_model_potentials": "all one, each component has unmatched/matched alternatives",
            "unrecorded_joint_factor": "100 only when both components are matched; one otherwise",
            "independent_model_partition": independent_z, "true_coupled_partition": true_joint_z,
            "invalid_product_omitted_mass_bound": float(invalid_product_eta),
            "true_coupled_omitted_mass": true_omitted,
            "conclusion": "record all coupling factors before decomposition; local model bounds do not certify this coupled joint"}


def run_once(seed):
    torch.manual_seed(seed)
    model = RecoverableIdentityModel(hidden=16, heads=4, dropout=.1).cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen_hash = model_digest(model)

    def assert_frozen():
        must(not any(module.training for module in model.modules()), "model child entered training mode")
        must(not any(parameter.requires_grad for parameter in model.parameters()), "model parameters became trainable")
        must(model_digest(model) == frozen_hash, "model parameters changed")

    inputs = {key: batch_inputs(*shape) for key, shape in SHAPES.items()}
    current = {}
    banks = {}
    windows = {}
    for key, (n, m) in SHAPES.items():
        with torch.inference_mode():
            current[key] = LogAssociationFactors(*detached_log_potentials(model(inputs[key][0])))
        banks[key] = HypothesisBank(current[key], active_limit=7 if n == 2 else 2,
                                   information_us=DECISIONS[0], max_frontier_nodes=32,
                                   max_discovered_leaves=16, max_evidence_updates=8, max_commits=32)
        priors = tuple(TrackPrior(f"{key}-track-{j}", START, physical_mean(10. * j), np.eye(9),
                                  np.float32(.3)) for j in range(m))
        windows[key] = BranchStateWindow(component_id=key,
            left_node_ids=tuple(f"left-{i}" for i in range(n)),
            right_node_ids=tuple(f"right-{j}" for j in range(m)), priors=priors, start_us=START,
            process_noise=.1, max_observations=16, max_commits=8, max_active_branches=8)
    allocator = RiskBudgetAllocator(banks, loss_ranges={"crossing-2x2": 1., "isolated-1x1": .5}, max_requests=8)
    allocations = []
    state_commits = {key: [] for key in SHAPES}
    first_state_bytes = {}
    checks = {}
    max_rescore_error = 0.

    # Reject future history before accepting any factors or state observations.
    future_batch = replace(inputs["crossing-2x2"][0], history_arrival_us=torch.tensor([1_000_001, 950_000]))
    checks["future_neural_history_rejected"] = rejects(lambda: model(future_batch), "future")

    for index, (decision, budget) in enumerate(zip(DECISIONS, SPLITS)):
        assert_frozen()
        updates = {}
        rescored = {}
        for key in SHAPES:
            with torch.inference_mode():
                rescored[key] = LogAssociationFactors(*detached_log_potentials(model(inputs[key][index])))
            if index:
                updates[key] = EvidenceUpdate(f"{key}-absolute-rescore-{index}", decision,
                                               delta_factors(rescored[key], current[key]))
        if index == 1:
            before = allocator.commits
            future = dict(updates)
            original = future["isolated-1x1"]
            future["isolated-1x1"] = EvidenceUpdate(original.evidence_id, decision + 1, original.factors)
            checks["future_component_update_rejected_atomically"] = rejects(
                lambda: allocator.allocate("rejected-future", decision_us=decision, split_budget=budget,
                                           evidence=future), "future")
            must(allocator.commits is before, "invalid component update changed committed allocation")
        allocation = allocator.allocate(f"decision-{index}", decision_us=decision, split_budget=budget,
                                        evidence=updates)
        allocations.append(asdict(allocation))
        must(allocation.consumed_splits <= budget, "global split budget exceeded")
        must(allocator.allocate(f"decision-{index}", decision_us=decision, split_budget=budget,
                                evidence=updates) is allocation, "allocation retry was not idempotent")
        for component in allocation.components:
            key = component.component_id
            window = windows[key]
            if index == 0:
                messages = [observation(key, f"initial-{i}", "left", i, 10. * i, 950_000, 980_000)
                            for i in range(SHAPES[key][0])]
            elif index == 1:
                # This observation refers to a time BEFORE the first state
                # output, but arrives only at the second decision.
                messages = [observation(key, "late", "left", 0, 1., 970_000, 1_080_000)]
            else:
                messages = [observation(key, "current", "right", 0, .5, 1_150_000, 1_180_000)]
            for message in messages:
                must(window.ingest(message, decision_us=decision), "new observation not ingested")
                before_observations = window.observations
                must(not window.ingest(message, decision_us=decision), "duplicate observation counted twice")
                must(window.observations is before_observations, "duplicate observation mutated storage")
            for branch in component.bank.active:
                error = abs(branch.log_weight - assignment_log_weight(rescored[key], branch.choices))
                max_rescore_error = max(max_rescore_error, error)
                must(error < 1e-10, "absolute rescoring drifted into repeated likelihood multiplication")
            if window.commits:
                path = allocator.component_ancestry(key, window.commits[-1].identity_commit, component.bank.commit)
                must(path and path[-1].commit == component.bank.commit, "ancestry tip differs")
                ancestors = path[:-1]
            else:
                ancestors = ()  # First state output explicitly anchors the current identity commit.
            result = window.commit(component.bank, reference_us=decision, identity_ancestors=ancestors)
            state_commits[key].append(asdict(result))
            must(window.commit(component.bank, reference_us=decision) is result, "state commit retry changed output")
            if index == 0:
                first_state_bytes[key] = canonical(asdict(result))
            else:
                must(canonical(asdict(window.commits[0])) == first_state_bytes[key], "late data rewrote past output")
        current = rescored

    checks["absolute_rescore_max_log_weight_error"] = max_rescore_error
    checks["duplicate_allocation_and_observation_and_state_idempotent"] = True
    checks["late_data_did_not_rewrite_previous_commits"] = True
    checks["all_identity_ancestries_verified_by_state_window"] = True
    final = windows["crossing-2x2"].commits[-1]
    alternatives = {branch.choices: branch for branch in final.branches}
    must((0, 1) in alternatives and (1, 0) in alternatives, "both competing full identities must be present")
    correct = {track.track_id: track for track in alternatives[(0, 1)].tracks}
    swapped = {track.track_id: track for track in alternatives[(1, 0)].tracks}
    track = "crossing-2x2-track-0"
    must(correct[track].mean != swapped[track].mean, "different identities collapsed to shared state")
    distinct = len({hashlib.sha256(canonical([asdict(track) for track in branch.tracks])).hexdigest()
                    for branch in final.branches})
    checks["distinct_conditional_branch_state_histories"] = distinct
    checks["competing_identity_track0_x"] = {"identity": correct[track].mean[0], "swap": swapped[track].mean[0]}

    cap_bank = HypothesisBank(current["crossing-2x2"], active_limit=1, max_frontier_nodes=1)
    capped = RiskBudgetAllocator({"capped": cap_bank})
    limited = capped.allocate("cap", decision_us=DECISIONS[-1], split_budget=100)
    must(limited.components[0].status == "resource_limited" and limited.consumed_splits == 0,
         "resource cap did not stop without dropping unexpanded support")
    checks["resource_cap_fail_closed"] = asdict(limited)
    checks["component_independence_assumption_counterexample"] = independence_counterexample()
    assert_frozen()
    return {"seed": seed, "model_sha256": frozen_hash, "model_initialization": "random_untrained",
            "model_device": "cpu", "all_children_eval": True, "all_parameters_frozen": True,
            "shapes": {key: {"left": shape[0], "right": shape[1], "features": 203,
                              "history_by_step": [len(batch.history) for batch in inputs[key]]}
                       for key, shape in SHAPES.items()},
            "allocations": allocations, "state_commits": state_commits, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; report is create-once")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0,2**63)")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = [ROOT / f"transvision/models/event_track_v2x/{name}.py" for name in
             ("hypothesis_bank", "learned_identity", "recoverable_identity", "risk_budget", "recoverable_states",
              "tracking_v2", "fusion", "prediction_features", "arrays")]
    paths += [Path(__file__).resolve(), ROOT / "tests/event_track_v2x/test_recoverable_pipeline_cli.py"]
    sources = {str(path.relative_to(ROOT)): sha_file(path) for path in paths}
    first, second = run_once(args.seed), run_once(args.seed)
    encoded = canonical(first)
    must(encoded == canonical(second), "fresh instance replay is not byte-identical")
    must(sources == {str(path.relative_to(ROOT)): sha_file(path) for path in paths}, "source changed during verification")
    report = {"kind": "recoverable_pipeline_synthetic_verification_v1", "status": "verified",
              "source_hashes": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                  "torch": torch.__version__, "scipy": scipy.__version__, "torch_threads": torch.get_num_threads()},
              "trace_sha256": hashlib.sha256(encoded).hexdigest(), "fresh_instance_replay_byte_equal": True,
              "trace": first,
              "scope": {"data": "synthetic fixed tracklet universes only", "real_data_read": False,
                        "ground_truth_read": False, "checkpoint_read": False, "network_access": False,
                        "gpu_used": False, "trained_model": False, "complete_lifecycle_tracker": False,
                        "state_update": "conditional engineering CI, not an exact trajectory posterior",
                        "rescore": "absolute learned potential difference, not independent Bayes likelihood",
                        "budget": "prefix splits with explicit storage caps, not wall-time or byte equivalence",
                        "calibrated_real_posterior": False, "paper_metric_or_gain_claim": False}}
    with args.output.open("xb") as stream:
        stream.write(canonical(report))
    print(json.dumps({"status": "verified", "output": str(args.output.resolve()), "sha256": sha_file(args.output),
                      "trace_sha256": report["trace_sha256"], "fresh_instance_replay_byte_equal": True}, sort_keys=True))


if __name__ == "__main__":
    main()
