"""Independent learned allocation trajectory replay from immutable SQL.

No production tracker, policy, feature or decoder imports. Reconstructs search
states and operations, then checks every eligible candidate and chosen action.
Dataset admission, fresh continuous states, final Bayes actions and cost remain
separate gates. Passing finite fixtures does not admit a real experiment.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sqlite3
import time

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
REFERENCE = ROOT / 'source-freezes/rbf-independent-exclusive-teacher-capacity-model-bound-v1-20261002/trajectory.py'
REFERENCE_SHA = '37ddf0a7437064d41f6df36a0879eb67b19a430e306c2fb1c8d6e48827c894a9'
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
spec = importlib.util.spec_from_file_location('frozen_search_reference_for_learned', REFERENCE)
search_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_reference)
numeric = search_reference.numeric
SCHEMA = 'persistent_exclusive_completion_learned_allocation_v1'
FEATURES = ('loss_weight', 'eta_upper', 'conditional_risk', 'optimization_gap',
            'model_regret_upper', 'active_entropy', 'top_two_log_gap', 'log_nodes',
            'log_loss_nodes', 'log_active', 'log_frontier', 'frontier_min_depth_fraction',
            'frontier_max_depth_fraction', 'remaining_budget_fraction', 'proposal_operation',
            'log_operation_steps', 'log_prefix_nodes', 'weighted_eta')


def policy_weights(path, expected_sha256):
    """Load fixed 18 -> 32 -> 1 float64 weights without model code."""
    import numpy as np
    assert numeric.sha(path) == expected_sha256, 'priority weight bytes changed'
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {'w0', 'w1', 'w2', 'w3'}
        arrays = [np.array(data[f'w{i}'], dtype='<f8', order='C') for i in range(4)]
    assert [a.shape for a in arrays] == [(32, 18), (32,), (1, 32), (1,)]
    assert all(np.isfinite(a).all() for a in arrays)
    digest = hashlib.sha256(numeric.canonical([
        numeric.FEATURE_RECIPE, FEATURES, numeric.TARGET_RECIPE, [a.shape for a in arrays]]))
    for array in arrays:
        digest.update(array.tobytes(order='C'))
    assert numeric.sha(path) == expected_sha256
    return [a.tolist() for a in arrays], digest.hexdigest()


def score(features, weights):
    """Scalar compensated sums, independent of producer NumPy matrix multiply."""
    a, b, c, d = weights
    assert len(features) == 18 and all(math.isfinite(x) for x in features)
    hidden = [math.tanh(math.fsum(x * y for x, y in zip(features, row)) + bias)
              for row, bias in zip(a, b)]
    result = math.tanh(math.fsum(x * y for x, y in zip(hidden, c[0])) + d[0])
    assert math.isfinite(result)
    return result


def deployed_scores(checked_features, weights):
    """Replay float64 batch operations on independently checked input values.

This additionally checks the *implemented* discrete selection, where tiny
rounding differences can change ties. It does not replace the independent
raw-state feature and compensated-sum score checks. It imports no policy code.
The winner must match exactly; no tolerance is used for ordering or tie breaks.
"""
    import numpy as np
    a, b, c, d = [np.asarray(v, dtype=np.float64) for v in weights]
    x = np.asarray(checked_features, dtype=np.float64)
    with np.errstate(over='raise', invalid='raise'):
        result = np.tanh(np.tanh(x @ a.T + b) @ c.T + d)[:, 0]
    assert np.isfinite(result).all()
    return [float(v) for v in result]


def tree_for_rows(value, rows):
    # Available SQL rows are only a witness. Every consumed new child must be
    # the next consecutively allocated row of the independently derived path.
    assert [row[0] for row in rows] == list(range(len(rows)))
    return numeric.RootTree([tuple(row) for row in rows],
        [tuple(slot) for slot in value['slots']],
        [tuple((p, w) for p, w in row) for row in value['factors']], value['sequence_id'])


class Children:
    def __init__(self, tree, rows, count):
        self.tree, self.rows, self.count = tree, rows, count
        self.children = {(row[1], row[3]): row[0] for row in rows[:count] if row[0]}

    def child(self, parent, choice):
        assert choice in self.tree.allowed[parent], 'illegal prefix child'
        if (parent, choice) not in self.children:
            assert self.count < len(self.rows), 'missing committed operation child'
            row = self.rows[self.count]
            assert (row[0], row[1], row[3]) == (self.count, parent, choice), 'prefix allocation order changed'
            self.children[parent, choice] = self.count
            self.count += 1
        return self.children[parent, choice]


def initial_state(value, old, change, members, memberships, previous, rows, config):
    tree = tree_for_rows(value, rows)
    if old is None:
        count, old_n, active, frontier, fallback = 1, 0, set(), {0}, 0
    else:
        count, old_n = old['prefix_count'], old['nodes']
        active, frontier, fallback = set(old['active']), set(old['frontier']), old['output']
        assert rows[:count] == old['prefixes'], 'committed prefix history changed'
    children = Children(tree, rows, count)
    if change and change['merge']:
        assert old is None
        inverse = {g: i for i, g in enumerate(members)}
        parents = [-1] * value['nodes']
        for c in change['predecessors']:
            prior = previous[c]
            choices = numeric.path_choices(numeric.snapshot(prior), prior['output'])
            assert len(choices) == len(memberships[c])
            for i, p in enumerate(choices):
                parents[inverse[memberships[c][i]]] = -1 if p < 0 else inverse[memberships[c][p]]
        for choice in parents:
            fallback = children.child(fallback, choice)
    else:
        assert tree.p[fallback][2] == old_n
        for _ in range(value['nodes'] - old_n):
            fallback = children.child(fallback, -1)
    if value['nodes'] > old_n:
        frontier.update(h for h in active if not any(tree.ancestor(base, h) for base in frontier))
        active = set()
        if len(frontier) > config['state']['max_frontier']:
            frontier = {0}
    active, frontier = search_reference.promote(tree, active, frontier, value['nodes'], config['state']['active_limit'])
    active.add(fallback)
    active, frontier = search_reference.promote(tree, active, frontier, value['nodes'], config['state']['active_limit'])
    result = dict(value, prefixes=rows[:children.count], prefix_count=children.count,
                  active=sorted(active), frontier=sorted(frontier))
    checked_tree, checked_fallback, proposals = search_reference.activate(
        result, old, change, members, memberships, previous, rows, config)
    assert fallback == checked_fallback
    return result, checked_tree, fallback, proposals


def next_operation(value, tree, config, remaining, prefix_total, frontier_total, proposals, bases):
    h = search_reference.proposal(value, tree, config, remaining, prefix_total, frontier_total, handles=proposals)
    if h is None:
        return dict(base=None, kind='recoverable_prefix_refinement', requested_steps=1)
    reserve = int(not all(any(tree.ancestor(base, a) for base in value['frontier']) for a in value['active']))
    return dict(base=h, requested_steps=value['nodes'] - tree.p[h][2], frontier_reserve=reserve,
                kind='covered_frontier_completion_proposal' if h in bases else 'covered_history_suffix_proposal')


def features(value, tree, scope, weight, operation, remaining, config):
    bound = numeric.decision(value, tree, scope, config)
    logs = bound['weights']
    probabilities = [math.exp(w - bound['retained']) for w in logs]
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0) / math.log(max(2, len(logs)))
    depth = [tree.p[h][2] / value['nodes'] for h in value['frontier']]
    return (weight, bound['eta'], bound['conditional_risk'], bound['optimization_gap'],
        bound['risk_bound'], entropy, min(50., logs[0] - logs[1]) / 50. if len(logs) > 1 else 1.,
        math.log1p(value['nodes']), math.log1p(len(scope)), math.log1p(len(logs)), math.log1p(len(depth)),
        min(depth, default=1.), max(depth, default=1.), remaining / max(1, config['state']['expansion_budget']),
        float(operation['base'] is not None), math.log1p(operation['requested_steps']),
        math.log1p(value['prefix_count']), weight * bound['eta'])


def execute(value, tree, rows, operation, width, prefix_cap, frontier_cap):
    """Reconstruct selected operation; no logged after-state is trusted."""
    available = tree_for_rows(value, rows)
    children = Children(available, rows, value['prefix_count'])
    active, frontier = set(value['active']), set(value['frontier'])
    base = operation['base']
    if base is not None:
        assert operation['requested_steps'] == value['nodes'] - tree.p[base][2] > 0
        handle = base
        while available.p[handle][2] < value['nodes']:
            choice = min(available.allowed[handle], key=lambda p: (-numeric.transition_weight(available, handle, p), p))
            handle = children.child(handle, choice)
        active.add(handle)
        active, frontier = search_reference.promote(available, active, frontier, value['nodes'], width)
    else:
        assert operation == dict(base=None, kind='recoverable_prefix_refinement', requested_steps=1)
        for h in sorted((h for h in frontier if tree.p[h][2] < value['nodes']),
                        key=lambda h: (-tree.upper(h, value['nodes']), tree.p[h][7])):
            allowed = tree.allowed[h]
            missing = sum((h, p) not in children.children for p in allowed)
            if len(frontier) - 1 + len(allowed) > frontier_cap or value['prefix_count'] + missing > prefix_cap:
                continue
            frontier.remove(h)
            frontier.update(children.child(h, p) for p in allowed)
            active, frontier = search_reference.promote(available, active, frontier, value['nodes'], width)
            break
    result = dict(value, prefixes=rows[:children.count], prefix_count=children.count,
                  active=sorted(active), frontier=sorted(frontier))
    result_tree = numeric.snapshot(result)
    done, limited = numeric.operation_result(value, result, tree, result_tree, operation, width, prefix_cap, frontier_cap)
    return result, result_tree, done, limited


def verify_database(path, expected_sha256, weights_path, weights_sha256, expected_policy_signature, progress=lambda value: None):
    path = Path(path)
    assert numeric.sha(path) == expected_sha256
    weights, signature = policy_weights(weights_path, weights_sha256)
    assert signature == expected_policy_signature, "independent policy signature mismatch"
    started = time.monotonic()
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        assert meta['schema'] == SCHEMA
        assert meta['allocation_policy_signature'] == signature
        config = meta['config']
        assert config['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
        assert config['state']['decision_mode'] == 'all-legal-hamming'
        # The frozen final-refit contract disables global regret fallback.
        # Other modes can materialize an unselected action before fallback and
        # require their own action-materialization trajectory admission.
        assert config['state']['max_model_regret'] == 1., 'unsupported non-final-refit regret contract'
        assert config['residual_partition_version'] == config['coverage_admission_version'] == 1
        raw = []
        factors = []
        for i, source, frame, state, arrival, payload, digest in db.execute(
                'SELECT i,source,frame,state_us,arrival_us,raw,sha FROM observations ORDER BY i'):
            assert i == len(raw) and hashlib.sha256(payload).hexdigest() == digest
            value = json.loads(payload)
            assert value['node']['source_id'] == source and value['node']['frame_id'] == frame
            assert value['state_us'] == state and value['node']['arrival_us'] == arrival
            assert 0 <= value['node']['information_us'] <= arrival and 0 <= state <= arrival
            raw.append(dict(slot=[source, frame], state=state, arrival=arrival, sha=digest))
            row = list(db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p', (i,)))
            assert row and row[0][0] == -1 and len(dict(row)) == len(row)
            assert all(-1 <= p < i and math.isfinite(w) for p, w in row)
            factors.append(row)
        final_catalog = {c: dict(live=bool(live), predecessors=json.loads(pred), created_us=clock)
                         for c, live, pred, clock in db.execute('SELECT * FROM component_catalog ORDER BY component')}
        full_prefixes = {}
        sql_members = {}
        for c in final_catalog:
            full_prefixes[c] = [[*r[:6], bytes(r[6]).decode(), r[7]] for r in db.execute(
                f'SELECT h,parent,depth,choice,root,previous_root,jumps,sha FROM pc{c}_prefixes ORDER BY h')]
            sql_members[c] = [g for g, in db.execute(
                'SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (c,))]
        previous, catalog, memberships, owners = {}, {}, {}, {}
        events = candidate_rows = selections = selected_steps = merges = zero_scope_components = archived_max = 0
        rounding_order_differences = []
        zero_charged_operations = resource_limited_operations = 0
        old_n, last_clock, last_reference, maximum = 0, -1, -1, 0.
        factor_hash = hashlib.sha256()
        for ordinal, pb, ab in db.execute('SELECT ordinal,prediction,audit FROM events ORDER BY ordinal'):
            prediction, audit = json.loads(pb), json.loads(ab)
            assert ordinal == events and audit['kind'] == SCHEMA
            assert audit['allocation_policy_signature'] == signature
            assert audit['allocation_policy'] == 'learned_signed_one_operation_model_bound_progress'
            assert audit['learned_allocation_policy'] is True
            assert audit['offline_counterfactual_probes'] is False and audit['priority_is_bound'] is False
            assert audit['priority_row_cap'] == 262144
            assert not audit['rescored_rows']
            n = audit['observation_count']
            clock, reference = prediction['decision_timestamp_us'], prediction['box_reference_timestamp_us']
            assert last_reference < reference <= clock and clock >= last_clock
            assert old_n <= n <= len(raw) and audit['new_observations'] == n - old_n
            assert all(last_clock <= raw[g]['arrival'] <= clock for g in range(old_n, n))
            assert not n or max(r['arrival'] for r in raw[:n]) <= clock
            assert audit['appended_rows'] == [[list(v) for v in r] for r in factors[old_n:n]]
            for g in range(old_n, n):
                factor_hash.update(numeric.canonical([g, raw[g]['sha'], factors[g]]) + b'\n')
            assert factor_hash.hexdigest() == audit['factor_rows_sha256']
            indices = [g for g in range(n) if raw[g]['state'] >= max(0, reference - config['state']['window_us'])]
            assert audit['decision_indices'] == indices, 'default causal loss window changed'
            live, changes = search_reference.derive_partition(previous, owners, memberships, catalog, factors, old_n, n, clock)
            summaries = {s['component']: s for s in audit['components']}
            assert [s['component'] for s in audit['components']] == live
            states, trees, fallbacks, proposals, bases, scopes = {}, {}, {}, {}, {}, {}
            for c in live:
                s = summaries[c]
                members = memberships[c]
                assert members == sql_members[c][:len(members)] and len(members) == s['nodes']
                assert s['predecessors'] == (changes[c]['predecessors'] if c in changes else [c])
                assert s['merge_restart'] == bool(c in changes and changes[c]['merge'])
                scopes[c] = [i for i, g in enumerate(members) if g in indices]
                assert s['decision_indices'] == scopes[c]
                inverse = {g: i for i, g in enumerate(members)}
                expected_factors = [[[(-1 if p < 0 else inverse[p]), w] for p, w in factors[g]] for g in members]
                raw_state = dict(kind='exclusive_raw_kernel_search_snapshot_v1', sequence_id=meta['sequence_id'],
                                 nodes=len(members), revision=0, slots=[raw[g]['slot'] for g in members], factors=expected_factors)
                assert type(s['prefix_nodes']) is int and 1 <= s['prefix_nodes'] <= len(full_prefixes[c])
                states[c], tree, fallback, history = initial_state(raw_state, previous.get(c), changes.get(c),
                    members, memberships, previous, full_prefixes[c][:s['prefix_nodes']], config)
                trees[c], fallbacks[c], proposals[c], bases[c] = tree, fallback, history, set()
                assert s['fallback_handle'] == fallback
                zero_scope_components += not bool(scopes[c])
                merges += bool(c in changes and changes[c]['merge'])
            prefix_total = sum(states[c]['prefix_count'] if c in states else previous[c]['prefix_count'] for c in catalog)
            frontier_total = sum(len(v['frontier']) for v in states.values())
            archived_max = max(archived_max, len(catalog) - len(live))
            assert prefix_total <= config['max_total_prefix_nodes'] and frontier_total <= config['max_total_frontier']
            for c in live:
                if scopes[c]:
                    for _ in range(config['frontier_completions_per_component']):
                        h = search_reference.proposal(states[c], trees[c], config, config['state']['expansion_budget'],
                                     prefix_total, frontier_total, excluded=proposals[c])
                        if h is None:
                            break
                        proposals[c].append(h)
                        bases[c].add(h)
            remaining, blocked = config['state']['expansion_budget'], set()
            spent, proposed = dict.fromkeys(live, 0), dict.fromkeys(live, 0)
            for selection in audit['allocation_trace']:
                candidates = search_reference.eligible(states, trees, scopes, blocked)
                evidence = selection['priority_selection']
                assert evidence['recipe'] == numeric.FEATURE_RECIPE and evidence['priority_is_bound'] is False
                records = evidence['candidates']
                assert evidence['candidate_count'] == len(candidates) > 0 and remaining > 0
                assert [r['component'] for r in records] == candidates, 'learned selection omitted or added eligible candidate'
                operations, scores = {}, {}
                for record in records:
                    c = record['component']
                    operations[c] = next_operation(states[c], trees[c], config, remaining, prefix_total, frontier_total, proposals[c], bases[c])
                    expected_features = features(states[c], trees[c], scopes[c], len(scopes[c]) / len(indices),
                                                 operations[c], remaining, config)
                    assert len(record['features']) == len(expected_features) == 18
                    for got, expected in zip(record['features'], expected_features):
                        maximum = max(maximum, numeric.close(got, expected))
                    scores[c] = score(expected_features, weights)
                    maximum = max(maximum, numeric.close(record['priority_score'], scores[c]))
                    candidate_rows += 1
                replay_scores = dict(zip(candidates, deployed_scores([r['features'] for r in records], weights)))
                for record in records:
                    c = record['component']
                    maximum = max(maximum, numeric.close(replay_scores[c], scores[c]),
                                  numeric.close(record['priority_score'], replay_scores[c]))
                chosen = min(candidates, key=lambda c: (-replay_scores[c], c))
                assert selection['component'] == chosen, 'float64 learned priority order or exact tie break differs'
                reconstructed_chosen = min(candidates, key=lambda c: (-scores[c], c))
                if reconstructed_chosen != chosen:
                    # Explicitly disclose the scope limit rather than equating
                    # numerical agreement with bitwise implementation identity.
                    rounding_order_differences.append(dict(event=ordinal, selection=selections,
                        deployed_chosen=chosen, compensated_reconstruction_chosen=reconstructed_chosen,
                        reconstructed_score_margin=scores[reconstructed_chosen] - scores[chosen],
                        deployed_score_margin=replay_scores[chosen] - replay_scores[reconstructed_chosen]))
                old, tree = states[chosen], trees[chosen]
                maximum = max(maximum, numeric.close(selection['weighted_eta_before'],
                                                    len(scopes[chosen]) / len(indices) * numeric.mass(old, tree)[-1]))
                prefix_cap = min(config['max_prefix_nodes'], old['prefix_count'] + config['max_total_prefix_nodes'] - prefix_total)
                frontier_cap = min(config['state']['max_frontier'], len(old['frontier']) + config['max_total_frontier'] - frontier_total)
                value, after_tree, done, limited = execute(old, tree,
                    full_prefixes[chosen][:summaries[chosen]['prefix_nodes']], operations[chosen],
                    config['state']['active_limit'], prefix_cap, frontier_cap)
                zero_charged_operations += done == 0
                resource_limited_operations += limited
                assert done == selection['charged_search_steps'] and 0 <= done <= remaining
                prefix_total += value['prefix_count'] - old['prefix_count']
                frontier_total += len(value['frontier']) - len(old['frontier'])
                states[chosen], trees[chosen] = value, after_tree
                maximum = max(maximum, numeric.close(selection['eta_after'], numeric.mass(value, after_tree)[-1]))
                assert selection['work_kind'] == operations[chosen]['kind']
                h = operations[chosen]['base']
                if h is not None:
                    proposals[chosen].remove(h)
                    proposed[chosen] += done
                else:
                    spent[chosen] += done
                remaining -= done
                if not done:
                    blocked.add(chosen)
                selected_steps += done
                selections += 1
            assert remaining == 0 or not search_reference.eligible(states, trees, scopes, blocked), 'allocation stopped with available work'
            assert audit['search_steps'] == config['state']['expansion_budget'] - remaining
            assert audit['total_frontier'] == frontier_total
            for c in live:
                value, tree, s = states[c], trees[c], summaries[c]
                assert sorted(r['handle'] for r in s['active']) == value['active']
                assert [r['handle'] for r in s['frontier']] == value['frontier']
                assert s['expansions'] == spent[c] and s['proposal_steps'] == proposed[c]
                # Final action decoding can materialize only its chosen legal path.
                full = dict(value, prefixes=full_prefixes[c][:s['prefix_nodes']], prefix_count=s['prefix_nodes'])
                final_tree = numeric.snapshot(full)
                output = s['output_handle']
                choices = numeric.path_choices(final_tree, output)
                assert len(choices) == value['nodes'] and final_tree.p[output][7] == s['output_sha256']
                children = {(r[1], r[3]): r[0] for r in value['prefixes'] if r[0]}
                count, h = value['prefix_count'], 0
                for p in choices:
                    if (h, p) not in children:
                        r = full['prefixes'][count]
                        assert r[0] == count and r[1] == h and r[3] == p
                        children[h, p] = count
                        count += 1
                    h = children[h, p]
                assert h == output and count == full['prefix_count'], 'unaccounted final action branches'
                assert s['decision']['action_materialization_prefixes'] == count - value['prefix_count']
                previous[c] = dict(full, output=output)
            assert audit['total_prefix_nodes'] == sum(v['prefix_count'] for v in previous.values())
            assert audit['priority_feature_rows'] == sum(len(s['priority_selection']['candidates']) for s in audit['allocation_trace'])
            assert 0 <= audit['priority_feature_rows'] <= 262144
            old_n, last_clock, last_reference = n, clock, reference
            events += 1
            if events % 25 == 0:
                progress(dict(stage='independent_learned_full_search_trajectory', completed_events=events,
                              total_events=meta['state']['events'], ETA_seconds=None,
                              ETA_reason='prefix sizes and candidate counts vary'))
        assert events == meta['state']['events'] and old_n == len(raw)
        assert catalog == final_catalog and memberships == sql_members
        for c, value in previous.items():
            assert value['prefixes'] == full_prefixes[c]
        assert numeric.sha(path) == expected_sha256 and numeric.sha(weights_path) == weights_sha256
        return dict(kind='rbf_independent_exclusive_learned_search_trajectory_checks_v1',
                    weights_sha256=weights_sha256, policy_signature=signature,
                    database_sha256=expected_sha256, events=events, observations=len(raw), candidate_feature_rows=candidate_rows,
                    actual_selected_operations=selections, actual_selected_charged_steps=selected_steps,
                    zero_charged_operations=zero_charged_operations,
                    resource_limited_operations=resource_limited_operations,
                    merge_events=merges, maximum_archived_components=archived_max,
                    zero_loss_scope_component_events=zero_scope_components, max_abs_error=maximum,
                    all_initial_live_states_and_archived_prefix_counts_reconstructed=True,
                    all_raw_parent_component_partitions_and_causal_loss_windows_reconstructed=True,
                    all_candidate_sets_proposal_orders_and_selected_state_transitions_reconstructed=True,
                    all_18_features_and_MLP_scores_independently_recomputed=True,
                    all_selected_operations_and_charged_steps_reconstructed=True,
                    all_float64_selections_exactly_replayed_from_independently_checked_features=True,
                    ordering_tolerance_used=False, feature_bitwise_parity_claimed=False,
                    independent_compensated_arithmetic_order_differences=rounding_order_differences,
                    producer_model_or_tracker_imported=False,
                    full_forest_semantics_or_fresh_state_independently_accepted=False,
                    actual_dataset_and_full_causal_trajectory_accepted=False,
                    learned_Stage2_complete=False,
                    paper_performance_complete=False, formal_interval_certificate=False,
                    atol=numeric.ATOL, rtol=numeric.RTOL, elapsed_seconds=time.monotonic() - started)
    finally:
        db.close()
