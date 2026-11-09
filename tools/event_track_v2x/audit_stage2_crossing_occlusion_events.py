#!/usr/bin/env python3
"""Independent causal-prefix readback for crossing/occlusion; no production imports."""
import argparse
import decimal
import hashlib
import importlib.util
import json
import math
import sqlite3
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prefix(db, component, handle):
    choices = []
    seen = set()
    while handle:
        if handle in seen:
            raise ValueError('prefix cycle')
        seen.add(handle)
        row = db.execute(f'SELECT parent,depth,choice FROM pc{int(component)}_prefixes WHERE h=?', (handle,)).fetchone()
        if row is None:
            raise ValueError('missing prefix')
        choices.append((row[1], row[2]))
        handle = row[0]
    choices.reverse()
    if [d for d, _ in choices] != list(range(1, len(choices) + 1)):
        raise ValueError('prefix depth discontinuity')
    result = []
    for i, (_, p) in enumerate(choices):
        if p < -1 or p >= i:
            raise ValueError('noncausal parent')
        result.append(p)
    return result


def roots(parents, names):
    result = []
    for i, p in enumerate(parents):
        result.append(names[i] if p < 0 else result[p])
    return tuple(result)


def transport(m, p, dt, q):
    F = np.eye(9)
    F[0, 7] = F[1, 8] = dt
    mean = F @ m
    mean[6] = (mean[6] + math.pi) % (2 * math.pi) - math.pi
    return mean, F @ p @ F.T + np.eye(9) * q * abs(dt)


def state_for_group(raw, timestamp, config):
    mean = cov = None
    clock = None
    for r in sorted(raw, key=lambda x: (x['state_us'], x['node']['source_id'], x['node']['node_id'])):
        new_m, new_p = np.asarray(r['mean']), np.asarray(r['covariance'])
        if mean is None:
            mean, cov = new_m.copy(), new_p.copy()
        else:
            mean, cov = transport(mean, cov, (r['state_us'] - clock) / 1e6, config['process_noise'])
            omega = config['ci_weight']
            I1, I2 = np.linalg.solve(cov, np.eye(9)), np.linalg.solve(new_p, np.eye(9))
            precision = omega * I1 + (1 - omega) * I2
            new_m = new_m.copy()
            new_m[6] = mean[6] + (new_m[6] - mean[6] + math.pi) % (2 * math.pi) - math.pi
            mean = np.linalg.solve(precision, omega * (I1 @ mean) + (1 - omega) * (I2 @ new_m))
            cov = np.linalg.solve(precision, np.eye(9))
        clock = r['state_us']
    mean, cov = transport(mean, cov, (timestamp - clock) / 1e6, config['process_noise'])
    score = max(r['score'] * config['survival_per_second'] ** (max(0, timestamp - r['state_us']) / 1e6) for r in raw)
    return mean, (cov + cov.T) / 2, score


def bayes_update(before, after, factors, active_sets, names):
    """Sum all legal positive suffix weights, then update the old history law.

    Old and new complete enumerations come from the independently qualified
    oracle. Likelihoods here use factor values directly, not production bounds
    or the ratio of already normalized probabilities.
    """
    D = decimal.Decimal
    old_n = len(before['history_probabilities'][0]['parents'])
    new_n = len(after['history_probabilities'][0]['parents'])
    positions = {name: i for i, name in enumerate(names)}
    likelihood = defaultdict(D)
    projected = defaultdict(D)
    for h in after['history_probabilities']:
        weight = Fraction()
        for i in range(old_n, new_n):
            parent = h['parents'][i]
            index = -1 if parent is None else positions[parent]
            weight += Fraction.from_float(float(dict(factors[i])[index]))
        likelihood[tuple(h['parents'][:old_n])] += (D(weight.numerator) / D(weight.denominator)).exp()
        projected[tuple(h['parents'][:old_n])] += D(h['probability_decimal'])
    assert set(likelihood) == {tuple(h['parents']) for h in before['history_probabilities']}
    assert min(likelihood.values()) > 0  # positive births keep every old history legal
    eta_y = sum((D(h['probability_decimal']) * likelihood[tuple(h['parents'])]
        for h in before['history_probabilities']), D(0))
    ratio = (D(after['log_partition_decimal']) - D(before['log_partition_decimal'])).exp()
    assert abs(eta_y - ratio) <= D('1e-50') * max(D(1), abs(ratio))
    maximum = D(0); prior_omitted = D(0); updated_omitted = D(0)
    for h in before['history_probabilities']:
        key = tuple(h['parents'])
        posterior = D(h['probability_decimal']) * likelihood[key] / eta_y
        maximum = max(maximum, abs(posterior - projected[key]))
        if not all(tuple(h['roots'][i] for i in members) in active for members, active in active_sets):
            prior_omitted += D(h['probability_decimal'])
            updated_omitted += posterior
    assert maximum < D('1e-50')
    kappa = max(likelihood.values()) / min(likelihood.values())
    upper = kappa * prior_omitted / (1 - prior_omitted + kappa * prior_omitted)
    assert updated_omitted <= upper + D('1e-50')
    return dict(old_nodes=old_n, new_nodes=new_n, exact_predictive_normalizer_eta_y=str(eta_y),
        min_suffix_likelihood=str(min(likelihood.values())), max_suffix_likelihood=str(max(likelihood.values())),
        exact_likelihood_ratio_kappa=str(kappa), prior_unretained_mass=str(prior_omitted),
        updated_mass_outside_previous_active_classes=str(updated_omitted),
        exact_kappa_amplification_bound=str(upper), maximum_independent_posterior_difference=str(maximum),
        positive_likelihood_for_every_prior_legal_history=True, old_history_support_deleted=False,
        missing_detections_treated_as_existence_likelihood=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(Path(__file__)) == plan['auditor_sha256']
    oracle_path = Path(plan['independent_oracle'])
    assert sha(oracle_path) == '29c038639fb1db1f5604a4f1ac150827752e22f86ccefc3bfcdd9356bc5821c0'
    spec = importlib.util.spec_from_file_location('independent_history_reference', oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    campaign = json.loads(args.campaign.read_text())
    assert campaign['source_plan_sha256'] == sha(args.plan)
    assert len(campaign['runs']) == len(plan['seeds']) * len(plan['cases']) * len(plan['widths'])
    all_checks = []
    decimal.getcontext().prec = 80
    for row in campaign['runs']:
        path = Path(row['path'])
        assert sha(path) == row['sha256']
        data = json.loads(path.read_text())
        dbpath = Path(data['database_path'])
        assert sha(dbpath) == data['database_sha256']
        full_raw, full_factors = data['raw'], data['rows']
        raw, factors = full_raw, full_factors
        names = [r['node']['node_id'] for r in raw]
        nodes = [dict(node_id=r['node']['node_id'], source_id=r['node']['source_id'], frame_id=r['node']['frame_id'],
            information_us=r['node']['information_us'], arrival_us=r['node']['arrival_us'],
            choices=[dict(parent_id=None if p < 0 else names[p], log_potential=w) for p, w in weights])
            for r, weights in zip(raw, factors)]
        state = data['configuration']['state']
        assert not data['truth_read_for_inference']
        assert raw[0]['mean'][0] < raw[1]['mean'][0]
        assert raw[-2]['mean'][0] > raw[-1]['mean'][0]
        assert raw[0]['mean'][1] == raw[-2]['mean'][1] == -2.
        assert raw[1]['mean'][1] == raw[-1]['mean'][1] == 2.
        assert raw[0]['mean'][7] == raw[-2]['mean'][7] == 1.5
        assert raw[1]['mean'][7] == raw[-1]['mean'][7] == -1.5
        assert all(r['features'] == raw[0]['features'] for r in raw)
        if data['case'] == 'occlusion_reappearance':
            assert len(raw) == 6 and raw[3]['node']['information_us'] == 1200000
            assert raw[3]['node']['node_id'] == 'd' and raw[3]['mean'][1] == -2.
            assert [r['node']['node_id'] for r in raw if r['node']['information_us'] == 1200000] == ['d']
        else:
            assert len(raw) == 5
        events = []; previous_reference = None; previous_active_sets = None; updates = []; first_target_status = None; qualified_recoveries = []
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            stored_raw = list(db.execute('SELECT raw,sha FROM observations ORDER BY i'))
            assert len(stored_raw) == len(full_raw)
            for original, (stored, digest) in zip(raw, stored_raw):
                assert canonical(original) == stored and hashlib.sha256(stored).hexdigest() == digest
            assert list(db.execute('SELECT i,p,w FROM potentials ORDER BY i,p')) == sorted((i, p, w) for i, weights in enumerate(factors) for p, w in weights)
            stored_events = list(db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal'))
            assert len(stored_events) == len(data['events']) == plan['compute_pulses'] + (3 if data['case'] == 'occlusion_reappearance' else 2)
            previous = '0' * 64
            for logged, (event_id, pb, ab) in zip(data['events'], stored_events):
                pred, audit = json.loads(pb), json.loads(ab)
                assert pred == logged['prediction'] and audit == logged['audit'] and event_id == logged['event_id']
                assert hashlib.sha256(pb).hexdigest() == logged['prediction_sha256'] and hashlib.sha256(ab).hexdigest() == logged['audit_sha256']
                body = dict(pred); commit = body.pop('commit_sha256')
                assert hashlib.sha256(canonical(body)).hexdigest() == commit and pred['previous_commit_sha256'] == previous
                previous = commit
                n = logged['available_observations']
                raw, factors = full_raw[:n], full_factors[:n]
                assert n == audit['observation_count']
                assert all(r['node']['arrival_us'] <= pred['decision_timestamp_us'] for r in raw)
                loss = audit['decision_indices']
                reference = module.oracle(nodes[:n], [names[i] for i in loss])
                assert reference['production_modules_imported'] is False
                chosen = [None] * n
                component_sets = []
                local_log_partitions = []
                residual_checks = 0
                for component in audit['components']:
                    cid = component['component']
                    members = [i for i, in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (cid,))][:component['nodes']]
                    assert len(members) == component['nodes']
                    local_names = [names[i] for i in members]
                    out_roots = roots(prefix(db, cid, component['output_handle']), local_names)
                    assert len(out_roots) == len(members)
                    for i, root in zip(members, out_roots):
                        assert chosen[i] is None
                        chosen[i] = root
                    active = {roots(prefix(db, cid, r['handle']), local_names) for r in component['active']}
                    component_sets.append((members, active))
                    local = module.oracle([nodes[i] for i in members])
                    z = decimal.Decimal(local['log_partition_decimal'])
                    local_log_partitions.append(z)
                    covered_prob = decimal.Decimal(0)
                    for region in component['frontier']:
                        pref = roots(prefix(db, cid, region['handle']), local_names)
                        excluded = {roots(prefix(db, cid, r['handle']), local_names) for r in region['excluded_leaves']}
                        assert excluded <= active
                        histories = [r for r in local['history_probabilities'] if tuple(r['roots'][:len(pref)]) == pref and tuple(r['roots']) not in excluded]
                        mass = sum((decimal.Decimal(r['probability_decimal']) for r in histories), decimal.Decimal(0))
                        if mass:
                            assert float(z + mass.ln()) <= region['log_upper'] + 1e-11
                        covered_prob += mass
                        residual_checks += 1
                    retained_local = sum((decimal.Decimal(r['probability_decimal']) for r in local['history_probabilities'] if tuple(r['roots']) in active), decimal.Decimal(0))
                    assert abs(covered_prob + retained_local - 1) < decimal.Decimal('1e-11')
                assert all(x is not None for x in chosen)
                assert abs(sum(local_log_partitions) - decimal.Decimal(reference['log_partition_decimal'])) < decimal.Decimal('1e-11')
                action = tuple(chosen[i] for i in loss)
                risk = next(float(x['risk_decimal']) for x in reference['legal_action_posteriors'] if tuple(x['action']) == action)
                regret = risk - float(reference['bayes_risk_decimal'])
                retained = math.fsum(float(r['probability_decimal']) for r in reference['history_probabilities']
                    if all(tuple(r['roots'][i] for i in members) in active for members, active in component_sets))
                omitted = max(0., 1. - retained)
                eta = 1. - math.prod(1. - c['eta_upper'] for c in audit['components'])
                assert omitted <= eta + 1e-11 and regret <= audit['model_regret_upper'] + 1e-11
                assert float(reference['log_partition_decimal']) <= audit['log_partition_upper'] + 1e-11
                tracks = {p['track_id']: p for p in pred['predictions']}
                maximum = 0.
                for root in sorted(set(chosen)):
                    group = [r for r, name in zip(raw, chosen) if name == root]
                    ident = data['sequence_id'] + ':' + hashlib.sha256(canonical(['forest-birth', root])).hexdigest()[:24]
                    target = tracks.pop(ident)
                    mean, cov, score = state_for_group(group, pred['box_reference_timestamp_us'], state)
                    assert np.allclose(mean, target['mean'], atol=1e-8, rtol=1e-8) and np.allclose(cov, target['covariance'], atol=1e-8, rtol=1e-8)
                    assert math.isclose(score, target['score'], abs_tol=1e-8, rel_tol=1e-8)
                    maximum = max(maximum, float(np.max(np.abs(mean - target['mean']))), float(np.max(np.abs(cov - target['covariance']))), abs(score - target['score']))
                assert not tracks
                if previous_reference is not None and len(previous_reference['history_probabilities'][0]['parents']) < n:
                    updates.append(bayes_update(previous_reference, reference, full_factors, previous_active_sets, names))
                if first_target_status is None:
                    target = ('a', 'b', 'b')
                    explicit = all(tuple(target[i] for i in m) in a for m, a in component_sets)
                    covered = True
                    for comp, (members, active) in zip(audit['components'], component_sets):
                        projection = tuple(target[i] for i in members)
                        covered &= any(projection[:len(pref)] == pref and projection not in excluded
                            for pref, excluded in [(roots(prefix(db, comp['component'], reg['handle']), [names[i] for i in members]),
                                {roots(prefix(db, comp['component'], ex['handle']), [names[i] for i in members]) for ex in reg['excluded_leaves']})
                                for reg in comp['frontier']])
                    first_target_status = dict(target_roots=target, initially_explicit=explicit, initially_frontier_covered=bool(covered))
                if tuple(chosen[:3]) == ('a', 'b', 'b') and not first_target_status['initially_explicit'] and first_target_status['initially_frontier_covered']:
                    for comp in audit['components']:
                        for rec in comp['recovery_events']:
                            if rec['previously_unexpanded'] and rec['projection_nodes'] >= 3:
                                qualified_recoveries.append(dict(event_id=event_id, recovery=rec, current_chosen_roots=chosen))
                previous_reference, previous_active_sets = reference, component_sets
                truth = data['evaluator_only_true_roots'][:n]
                true_action = tuple(truth[i] for i in loss)
                truth_supported = any(tuple(x['action']) == true_action for x in reference['legal_action_posteriors'])
                truth_loss = sum(chosen[i] != truth[i] for i in loss) / max(1, len(loss))
                events.append(dict(event_id=event_id, exact_Z_log=reference['log_partition_decimal'], legal_histories=reference['legal_histories'],
                    exact_same_action_model_regret=regret, exact_omitted_mass=omitted, eta_upper=eta,
                    residual_regions_independently_checked=residual_checks, component_partition_product_checked=True,
                    chosen_roots=chosen, evaluator_only_truth_supported=truth_supported, evaluator_only_identity_loss=truth_loss,
                    search_steps=audit['search_steps'], expansion_budget=state['expansion_budget'], full9D_max_abs_difference=maximum,
                    true_identity_guarantee_claimed=False))
        assert sha(dbpath) == data['database_sha256'] and sha(path) == row['sha256']
        all_checks.append(dict(run=row['run'], candidate_sha256=row['sha256'], database_sha256=data['database_sha256'], events=events,
            bayes_updates=updates, initial_target_status=first_target_status, qualifying_current_output_recoveries=qualified_recoveries,
            recovery_claimed=bool(qualified_recoveries), geometric_x_order_swap_verified=True,
            real_missing_observation_frame_verified=data['case'] == 'occlusion_reappearance'))
        print(json.dumps({'independently_accepted_new_runs': len(all_checks), 'total': len(campaign['runs'])}), flush=True)
    result = dict(kind='rbf_stage2_crossing_occlusion_full_exact_and_9D_readback_v1', checks=all_checks,
        all_new_runs_full_events_independently_verified=True, production_modules_imported=False,
        original_reference_reused=True, independent_oracle_sha256=sha(oracle_path), auditor_sha256=sha(Path(__file__)),
        exact_causal_Bayes_updates_checked=True, geometric_x_order_swaps_checked=True,
        absence_used_as_existence_likelihood=False, real_data_performance_claimed=False,
        formal_numeric_interval_certificate=False, learned_priority_executed=False, same_resource_costs_complete=False,
        full_Stage2_complete=False, goal_status='active')
    with args.output.open('xb') as f:
        f.write(canonical(result) + b'\n')


if __name__ == '__main__':
    main()
