#!/usr/bin/env python3
"""Create-once finite-model boundary verification; no real data or training.

The independent reference enumerates original parent choices, applies root-wide
source/frame exclusion, and normalizes exact float-log sums using Fraction and
80-digit Decimal arithmetic. Neither reference legality nor normalization calls
the production marginalizer/bank. This is not a formal interval certificate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
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

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_boundary import checkpoint_tracker
from transvision.models.event_track_v2x.forest_boundary_prior import marginalize_boundary, extend_boundary_prior
from transvision.models.event_track_v2x.forest_tracking import (
    CausalForestTracker, ForestTrackingConfig, RawIdentityDetection, replay_forest_states,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode, digest


def _must(condition, message):
    if not condition:
        raise AssertionError(message)


def _observation(name, x=0., *, source=0, time=1_000_000, index=0):
    return RawIdentityDetection('synthetic', IdentityNode(name, source, time, time+10, str(time)),
        index, time, (x, 0., 1., 4., 2., 1.5, 0., .3, 0.), np.eye(9)*.2, .8, np.zeros(203), 'a'*64)


def _checkpoint(raw, *, rows=None, component=False):
    def score(observations, support, decision):
        return ForestFactors(tuple(o.node for o in observations), rows if rows is not None else
                             tuple(tuple((p, 0.) for p in row) for row in support))
    tracker = CausalForestTracker(sequence_id='synthetic', start_us=0, scorer=score,
        config=ForestTrackingConfig(component_mode=component, active_limit=1,
            expansion_budget=40, max_model_regret=1., max_nodes=16))
    result = tracker.step(raw, frame_id='before-boundary', event_id='before-boundary',
                          reference_us=1_400_000, decision_us=1_410_000)
    return tracker, result, checkpoint_tracker(tracker, cutoff_us=1_400_000)


def _reference(factors):
    energies = {}
    for choices in itertools.product(*factors.rows):
        roots, occupied, legal = [], set(), True
        for i, (parent, _) in enumerate(choices):
            root = i if parent < 0 else roots[parent]
            roots.append(root)
            slot = root, factors.nodes[i].source_id, factors.nodes[i].frame_id
            if slot in occupied:
                legal = False
                break
            occupied.add(slot)
        if legal:
            parents = tuple(p for p, _ in choices)
            energies[parents] = (tuple(roots), sum((Fraction.from_float(w) for _, w in choices), Fraction()))
    with localcontext() as context:
        context.prec = 80
        maximum = max(e for _, e in energies.values())
        unnormalized = {}
        for parents, (roots, energy) in energies.items():
            delta = energy-maximum
            unnormalized[parents] = (roots, (Decimal(delta.numerator)/Decimal(delta.denominator)).exp())
        z = sum((v for _, v in unnormalized.values()), Decimal(0))
        by_roots, count = defaultdict(Decimal), defaultdict(int)
        for parents, (roots, weight) in unnormalized.items():
            by_roots[roots] += weight/z
            count[roots] += 1
        history = {p: float(w/z) for p, (_, w) in unnormalized.items()}
        return {r: float(p) for r, p in by_roots.items()}, dict(count), history


def _risks(actions, posterior):
    n = len(next(iter(posterior)))
    marginal = [defaultdict(float) for _ in range(n)]
    for roots, probability in posterior.items():
        for i, root in enumerate(roots):
            marginal[i][root] += probability
    return {action: math.fsum(1-marginal[i][r] for i, r in enumerate(action))/n if n else 0.
            for action in actions}


def _verify_case(seed, trial):
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial]))
    n = 1+trial % 4
    raw = tuple(_observation(str(i), (i % 2)*30. if trial % 3 == 0 else i*.3,
                            source=i % 2, time=1_000_000+i*100_000) for i in range(n))
    _, _, archive = _checkpoint(raw, component=bool(trial % 2))
    old = ForestFactors(archive.factors.nodes,
        tuple(tuple((p, float(rng.normal())) for p, _ in row) for row in archive.factors.rows))
    prior = marginalize_boundary(archive, factors=old)
    m = 1+trial % 2
    newer = tuple(_observation('next-'+str(i), i*3., time=1_500_000, index=i) for i in range(m))
    factors = ForestFactors(old.nodes+tuple(o.node for o in newer), old.rows+tuple(
        tuple((p, float(rng.normal())) for p in range(-1, n)) for _ in newer))
    extension = extend_boundary_prior(prior, factors, decision_us=1_600_000)
    expected, counts, histories = _reference(factors)
    actual = {a.roots: math.exp(a.log_weight-extension.log_partition) for a in extension.atoms}
    _must(set(actual) == set(expected), 'root action support differs')
    _must({a.roots: a.parent_history_count for a in extension.atoms} == counts, 'history multiplicity differs')
    probability_error = max(abs(actual[r]-expected[r]) for r in expected)
    risk_actual, risk_reference = _risks(expected, actual), _risks(expected, expected)
    risk_error = max(abs(risk_actual[a]-risk_reference[a]) for a in expected)
    _must(max(probability_error, risk_error) < 2e-12, 'boundary posterior or decision risk differs')
    action = min(extension.atoms, key=lambda a: (risk_actual[a.roots], a.roots))
    continued = archive.replay(factors=factors, parents=action.representative_parents,
        new_observations=newer, reference_us=1_550_000, decision_us=1_600_000)
    full, _ = replay_forest_states('synthetic', raw+newer, factors,
        action.representative_parents, 1_550_000, archive.config)
    _must(continued.prediction_json == canonical(full), 'conditional state replay differs')
    return {'seed': seed, 'trial': trial, 'old_nodes': n, 'new_nodes': m,
        'old_components': len(prior.components), 'old_parent_histories': prior.parent_history_count,
        'old_joint_root_atoms': extension.old_joint_identity_atoms,
        'original_joint_parent_histories': len(histories), 'joint_root_atoms': len(extension.atoms),
        'enumerated_quotient_leaves': extension.enumerated_quotient_leaves,
        'maximum_probability_error': probability_error, 'maximum_identity_risk_error': risk_error,
        'bayes_action_roots': action.roots, 'selected_conditional_state_replay_equal': True,
        'source_archive_sha256': archive.commit, 'prior_sha256': prior.commit, 'extension_sha256': extension.commit}


def _amplification():
    eta, epsilon = .001, 5e-7
    raw = (_observation('old-a'), _observation('old-b', 4., source=1, time=1_100_000))
    rows = (((-1, 0.),), ((-1, math.log(eta)), (0, math.log1p(-eta))))
    tracker, previous, archive = _checkpoint(raw, rows=rows)
    prior = marginalize_boundary(archive)
    saved = previous.prediction_json
    newer = (_observation('new-a', time=1_500_000),
             _observation('new-b', 4., time=1_500_000, index=1))
    # Two distinct detections in a single future source/frame cannot join the
    # same root. This is an actual forest exclusion, not a hand-coded likelihood.
    factors = ForestFactors(archive.factors.nodes+tuple(o.node for o in newer), rows+(
        ((-1, math.log(epsilon)), (0, 0.)), ((-1, math.log(epsilon)), (1, 0.))))
    posterior = extend_boundary_prior(prior, factors, decision_us=1_600_000)
    full, _, _ = _reference(factors)
    eta_y = math.fsum(p for roots, p in full.items() if roots[:2] == (0, 1))
    u, v = epsilon**2+2*epsilon, (1+epsilon)**2
    formula = eta*v/((1-eta)*u+eta*v)
    _must(abs(eta_y-formula) < 2e-12 and eta_y > .999, 'future omitted-mass amplification failed')
    retained = {r: p/(1-eta_y) for r, p in full.items() if r[:2] == (0, 0)}
    # Both Bayes actions minimize over the SAME complete legal action space.
    risks_q, risks_p = _risks(full, retained), _risks(full, full)
    aq = min(full, key=lambda r: (risks_q[r], r))
    ap = min(full, key=lambda r: (risks_p[r], r))
    regret = risks_p[aq]-risks_p[ap]
    _must(-2e-12 <= regret <= eta_y+2e-12 and regret > .4, 'identity regret test failed')
    atom = next(a for a in posterior.atoms if a.roots == ap)
    projection = archive.project(atom.representative_parents[:2])
    _must(not projection.active_product_member, 'late decision did not leave the old active set')
    recovered = archive.replay(factors=factors, parents=atom.representative_parents,
        new_observations=newer, reference_us=1_550_000, decision_us=1_600_000, projection=projection)
    expected, _ = replay_forest_states('synthetic', raw+newer, factors,
        atom.representative_parents, 1_550_000, archive.config)
    _must(recovered.prediction_json == canonical(expected), 'recovered branch state differs')
    _must(previous.prediction_json == saved and len(tracker.commits) == 1, 'historical output rewritten')
    return {'old_omitted_mass': eta, 'future_retained_average_factor_mass': u,
        'future_omitted_average_factor_mass': v, 'factor_mass_ratio': v/u,
        'updated_omitted_mass_decimal_reference': eta_y, 'updated_omitted_mass_formula': formula,
        'wrong_per_boundary_atom_normalized_mass': eta,
        'conditional_top1_bayes_roots': aq, 'full_model_bayes_roots': ap,
        'normalized_root_identity_regret': regret, 'regret_bound_L1': eta_y,
        'same_legal_action_space': True, 'restored_outside_old_active': True,
        'original_prediction_sha256': previous.prediction['commit_sha256'],
        'conditional_state_replay_equal': True, 'historical_output_unchanged': True,
        'new_track_ids': [p['track_id'] for p in recovered.predictions],
        'true_posterior_or_real_tracking_metric_claim': False}


def verify(seeds=(1337, 2027, 3407), trials=32):
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or type(trials) is not int or not 1 <= trials <= 256):
        raise ValueError('unique nonnegative seeds and 1..256 trials per seed required')
    cases = [_verify_case(seed, index) for seed in seeds for index in range(trials)]
    sources = [Path(__file__)] + [ROOT/'transvision/models/event_track_v2x'/name for name in (
        'forest_boundary.py', 'forest_boundary_prior.py', 'forest_tracking.py', 'forest_components.py',
        'identity_forest.py', 'hypothesis_bank.py', 'recoverable_states.py', 'tracking_v2.py',
        'detection_cache_v2.py', 'prediction_features.py', 'fusion.py', 'arrays.py')]
    return {'kind': 'recoverable_boundary_model_verification_v1', 'status': 'verified',
        'seeds': seeds, 'trials_per_seed': trials, 'random_cases': cases, 'case_count': len(cases),
        'maximum_probability_error': max(c['maximum_probability_error'] for c in cases),
        'maximum_identity_risk_error': max(c['maximum_identity_risk_error'] for c in cases),
        'future_amplification_and_recovery': _amplification(),
        'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'scope': {'class_scope': ['car'], 'synthetic_only': True, 'raw_parent_history_reference': True,
            'reference_decimal_precision': 80, 'formal_interval_certificate': False,
            'finite_model_boundary_prior_exact': True, 'selected_branch_state_continuation_checked': True,
            'bounded_memory_long_sequence_inference_completed': False, 'model_trained': False,
            'real_data_or_test_read': False, 'paper_tracking_performance_evidence': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1337, 2027, 3407])
    parser.add_argument('--trials', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        parser.error('create-once output required, without symlink traversal')
    report = verify(tuple(args.seeds), args.trials)
    with output.open('xb') as stream:
        stream.write(canonical(report))
    print(json.dumps({'status': report['status'], 'case_count': report['case_count'],
        'maximum_probability_error': report['maximum_probability_error'],
        'maximum_identity_risk_error': report['maximum_identity_risk_error'],
        'report_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
