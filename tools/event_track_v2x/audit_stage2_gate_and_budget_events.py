#!/usr/bin/env python3
"""Independent full finite-history and9D readback; imports no production code."""
import argparse
import decimal
import hashlib
import importlib.util
import json
import math
import sqlite3
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
        raw, factors = data['raw'], data['rows']
        names = [r['node']['node_id'] for r in raw]
        nodes = [dict(node_id=r['node']['node_id'], source_id=r['node']['source_id'], frame_id=r['node']['frame_id'],
            information_us=r['node']['information_us'], arrival_us=r['node']['arrival_us'],
            choices=[dict(parent_id=None if p < 0 else names[p], log_potential=w) for p, w in weights])
            for r, weights in zip(raw, factors)]
        state = data['configuration']['state']
        assert (data['case'] == 'incorrect_gate') == (1 not in dict(factors[2]))
        assert raw[2]['node']['arrival_us'] <= 1100000
        assert abs(raw[2]['mean'][0] - raw[1]['mean'][0]) < abs(raw[2]['mean'][0] - raw[0]['mean'][0]) < state['gate_distance_m']
        events = []
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            stored_raw = list(db.execute('SELECT raw,sha FROM observations ORDER BY i'))
            assert len(stored_raw) == 3
            for original, (stored, digest) in zip(raw, stored_raw):
                assert canonical(original) == stored and hashlib.sha256(stored).hexdigest() == digest
            assert list(db.execute('SELECT i,p,w FROM potentials ORDER BY i,p')) == sorted((i, p, w) for i, weights in enumerate(factors) for p, w in weights)
            stored_events = list(db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal'))
            assert len(stored_events) == len(data['events']) == plan['events_per_run']
            previous = '0' * 64
            for logged, (event_id, pb, ab) in zip(data['events'], stored_events):
                pred, audit = json.loads(pb), json.loads(ab)
                assert pred == logged['prediction'] and audit == logged['audit'] and event_id == logged['event_id']
                assert hashlib.sha256(pb).hexdigest() == logged['prediction_sha256'] and hashlib.sha256(ab).hexdigest() == logged['audit_sha256']
                body = dict(pred); commit = body.pop('commit_sha256')
                assert hashlib.sha256(canonical(body)).hexdigest() == commit and pred['previous_commit_sha256'] == previous
                previous = commit
                loss = audit['decision_indices']
                reference = module.oracle(nodes, [names[i] for i in loss])
                assert reference['production_modules_imported'] is False
                chosen = [None] * 3
                component_sets = []
                local_log_partitions = []
                residual_checks = 0
                for component in audit['components']:
                    cid = component['component']
                    members = [i for i, in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (cid,))]
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
                    ident = 'finite-stage2-new:' + hashlib.sha256(canonical(['forest-birth', root])).hexdigest()[:24]
                    target = tracks.pop(ident)
                    mean, cov, score = state_for_group(group, pred['box_reference_timestamp_us'], state)
                    assert np.allclose(mean, target['mean'], atol=1e-8, rtol=1e-8) and np.allclose(cov, target['covariance'], atol=1e-8, rtol=1e-8)
                    assert math.isclose(score, target['score'], abs_tol=1e-8, rel_tol=1e-8)
                    maximum = max(maximum, float(np.max(np.abs(mean - target['mean']))), float(np.max(np.abs(cov - target['covariance']))), abs(score - target['score']))
                assert not tracks
                truth = data['evaluator_only_true_roots']
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
        if data['case'] == 'incorrect_gate':
            assert all(not x['evaluator_only_truth_supported'] and x['evaluator_only_identity_loss'] > 0 for x in events)
        else:
            assert events[0]['search_steps'] == state['expansion_budget']
            assert any(x['exact_omitted_mass'] > 0 for x in events)
        all_checks.append(dict(run=row['run'], candidate_sha256=row['sha256'], database_sha256=data['database_sha256'], events=events))
        print(json.dumps({'independently_accepted_new_runs': len(all_checks), 'total': len(campaign['runs'])}), flush=True)
    result = dict(kind='rbf_stage2_new_gate_budget_full_exact_and_9D_readback_v1', checks=all_checks,
        all_new_runs_full_events_independently_verified=True, production_modules_imported=False,
        original_reference_reused=True, independent_oracle_sha256=sha(oracle_path), auditor_sha256=sha(Path(__file__)),
        actual_expansion_exhaustion_witnessed=True, wrong_gate_excluded_truth_and_no_true_guarantee=True,
        formal_numeric_interval_certificate=False, learned_priority_executed=False, same_resource_costs_complete=False,
        full_Stage2_complete=False, goal_status='active')
    with args.output.open('xb') as f:
        f.write(canonical(result) + b'\n')


if __name__ == '__main__':
    main()
