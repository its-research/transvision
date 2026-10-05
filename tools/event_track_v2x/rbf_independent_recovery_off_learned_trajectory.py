"""Independent learned ordering with irreversible whole-history restrictions.

The immutable event chain reconstructs each predecessor's active/output root
classes before any new choice. No producer tracker, policy or decoder import.
Existing independently reviewed raw prefix, mass and float64 MLP arithmetic is
source-pinned; restricted activation, coverage and decision terms are separate.
This is a single-database software gate, not full experiment acceptance.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sqlite3
import time

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
REFERENCE = ROOT/'source-freezes/rbf-independent-learned-search-trajectory-v1-20261005/rbf_independent_learned_trajectory.py'
REFERENCE_SHA = '2309b6d9d90007ee56289aa5fc7ce4564bb475709c1eeb5cd290a5ee937c0a47'
assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
spec = importlib.util.spec_from_file_location('independent_unrestricted_learned_primitives_for_recovery_off',REFERENCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
numeric, search_reference = base.numeric, base.search_reference
policy_weights, score, deployed_scores = base.policy_weights, base.score, base.deployed_scores
Children, next_operation = base.Children, base.next_operation
SCHEMA = 'persistent_exclusive_recovery_off_learned_allocation_v1'
SCOPE = 'conditional_on_irreversibly_pruned_support'


def roots(tree, handle):
    result=[]
    for i,p in enumerate(numeric.path_choices(tree,handle)):
        result.append(i if p<0 else result[p])
    return tuple(result)


def allows(clauses, values):
    return all(any(all(values[i]==variant[j] for j,i in enumerate(indices) if i<len(values))
                   for variant in variants) for indices,variants in clauses)


def support_payload(clauses):
    return dict(recipe='previous_commit_active_union_output_root_classes_v1',
                clauses=clauses,cartesian_product_materialized=False)


def derive_clauses(old, change, members, memberships, previous):
    inverse={g:i for i,g in enumerate(members)}
    predecessors=([old] if old is not None else [previous[c] for c in change['predecessors']])
    clauses=[]
    for prior in predecessors:
        if not prior['nodes']:continue
        old_members=prior['members']
        assert set(old_members)<=set(members)
        tree=tree_for_rows(prior,prior['prefixes'])
        variants=set()
        for h in set(prior['active'])|{prior['output']}:
            values=roots(tree,h)
            assert len(values)==len(old_members)
            variants.add(tuple(inverse[old_members[r]] for r in values))
        clauses.append((tuple(inverse[g] for g in old_members),tuple(sorted(variants))))
    return tuple(clauses)


def tree_for_rows(value, rows):
    tree=base.tree_for_rows(value,rows)
    tree.history_clauses=value['history_clauses']
    # Verify archived raw prefix bytes first, then restrict current transitions.
    # Previously discarded archived children remain evidence, not live choices.
    for h, choices in tree.allowed.items():
        values=roots(tree,h)
        tree.allowed[h]=tuple(p for p in choices if allows(tree.history_clauses,values)
            and allows(tree.history_clauses,values+(len(values) if p<0 else values[p],)))
    return tree


def snapshot(value):
    assert value['kind']=='exclusive_raw_kernel_search_snapshot_v1' and value['revision']==0
    assert value['nodes']==len(value['slots'])==len(value['factors'])>0
    assert value['prefix_count']==len(value['prefixes'])
    assert value['active']==sorted(set(value['active'])) and value['frontier']==sorted(set(value['frontier']))
    tree=tree_for_rows(value,value['prefixes'])
    for h in value['active']+value['frontier']:
        assert allows(value['history_clauses'],roots(tree,h)), 'live search recovered discarded history'
    # Independent cut closure now uses independently restricted legal children.
    tree.cut(value['active'],value['frontier'],value['nodes'])
    return tree


def initial_state(value, old, change, members, memberships, previous, rows, config):
    tree=tree_for_rows(value,rows)
    if old is None:
        count,old_n,active,fallback=1,0,set(),0
    else:
        count,old_n=old['prefix_count'],old['nodes']
        active,fallback=set(old['active']),old['output']
        assert rows[:count]==old['prefixes'], 'committed prefix history changed'
    original_active=set(active)
    children=Children(tree,rows,count)
    # The ablation deliberately restarts only the restricted root domain.
    frontier={0}
    if change and change['merge']:
        assert old is None
        inverse={g:i for i,g in enumerate(members)}
        parents=[-1]*value['nodes']
        for c in change['predecessors']:
            prior=previous[c]
            choices=numeric.path_choices(tree_for_rows(prior,prior['prefixes']),prior['output'])
            assert len(choices)==len(memberships[c])
            for i,p in enumerate(choices):
                parents[inverse[memberships[c][i]]]=-1 if p<0 else inverse[memberships[c][p]]
        for choice in parents:fallback=children.child(fallback,choice)
    else:
        assert tree.p[fallback][2]==old_n
        for _ in range(value['nodes']-old_n):fallback=children.child(fallback,-1)
    if value['nodes']>old_n:active=set()
    active,frontier=search_reference.promote(tree,active,frontier,value['nodes'],config['state']['active_limit'])
    active.add(fallback)
    active,frontier=search_reference.promote(tree,active,frontier,value['nodes'],config['state']['active_limit'])
    result=dict(value,prefixes=rows[:children.count],prefix_count=children.count,
                active=sorted(active),frontier=sorted(frontier))
    checked=snapshot(result)
    if old is None:
        depth=value['nodes']-len(change['added'])
        h=fallback
        while checked.p[h][2]>depth:h=checked.p[h][1]
        bases={h}
    elif old_n<value['nodes']:bases=original_active|{old['output']}
    else:bases=set()
    proposals=sorted(bases,key=lambda h:(-checked.weights[h],checked.p[h][7]))
    assert all(allows(value['history_clauses'],roots(checked,h)) for h in proposals)
    return result,checked,fallback,proposals


def decision(value,tree,scope,config):
    active,weights,retained,upper,eta=numeric.mass(value,tree)
    assert all(allows(value['history_clauses'],roots(tree,h)) for h in active)
    if not active or value['nodes']>config['max_decision_nodes']:
        return dict(risk_bound=1.,conditional_risk=0.,optimization_gap=0.,eta=eta,
                    active=active,weights=weights,retained=retained)
    support=tuple(tuple(p for p,w in row) for row in tree.factors)
    # Feature queries have zero expansion budget. Their incumbent is an active
    # legal history, checked against the full correlated restriction above;
    # the relaxed marginal lower bound remains valid in the smaller domain.
    result=numeric.action_module.replay_search(support,tree.slots,
        tuple(numeric.path_choices(tree,h) for h in active),weights,tuple(scope),0,
        config['state']['paper_action_frontier'])
    assert result['action_search_steps'] == 0, 'priority feature attempted action expansion'
    assert allows(value['history_clauses'],result['roots'])
    return dict(risk_bound=min(1.,eta+result['optimization_gap']),
        conditional_risk=result['conditional_risk'],optimization_gap=result['optimization_gap'],
        eta=eta,active=active,weights=weights,retained=retained)


def features(value,tree,scope,weight,operation,remaining,config):
    bound=decision(value,tree,scope,config)
    logs=bound['weights']
    probabilities=[math.exp(w-bound['retained']) for w in logs]
    entropy=-math.fsum(p*math.log(p) for p in probabilities if p>0)/math.log(max(2,len(logs)))
    depth=[tree.p[h][2]/value['nodes'] for h in value['frontier']]
    return (weight,bound['eta'],bound['conditional_risk'],bound['optimization_gap'],
        bound['risk_bound'],entropy,min(50.,logs[0]-logs[1])/50. if len(logs)>1 else 1.,
        math.log1p(value['nodes']),math.log1p(len(scope)),math.log1p(len(logs)),math.log1p(len(depth)),
        min(depth,default=1.),max(depth,default=1.),remaining/max(1,config['state']['expansion_budget']),
        float(operation['base'] is not None),math.log1p(operation['requested_steps']),
        math.log1p(value['prefix_count']),weight*bound['eta'])


def execute(value,tree,rows,operation,width,prefix_cap,frontier_cap):
    available=tree_for_rows(value,rows)
    children=Children(available,rows,value['prefix_count'])
    active,frontier=set(value['active']),set(value['frontier'])
    base_handle=operation['base']
    if base_handle is not None:
        assert operation['requested_steps']==value['nodes']-tree.p[base_handle][2]>0
        h=base_handle
        while available.p[h][2]<value['nodes']:
            choice=min(available.allowed[h],key=lambda p:(-numeric.transition_weight(available,h,p),p))
            h=children.child(h,choice)
        active.add(h)
        active,frontier=search_reference.promote(available,active,frontier,value['nodes'],width)
    else:
        assert operation==dict(base=None,kind='recoverable_prefix_refinement',requested_steps=1)
        for h in sorted((h for h in frontier if tree.p[h][2]<value['nodes']),
                        key=lambda h:(-tree.upper(h,value['nodes']),tree.p[h][7])):
            allowed=tree.allowed[h]
            missing=sum((h,p) not in children.children for p in allowed)
            if len(frontier)-1+len(allowed)>frontier_cap or value['prefix_count']+missing>prefix_cap:continue
            frontier.remove(h)
            frontier.update(children.child(h,p) for p in allowed)
            active,frontier=search_reference.promote(available,active,frontier,value['nodes'],width)
            break
    result=dict(value,prefixes=rows[:children.count],prefix_count=children.count,
                active=sorted(active),frontier=sorted(frontier))
    result_tree=snapshot(result)
    done,limited=numeric.operation_result(value,result,tree,result_tree,operation,width,prefix_cap,frontier_cap)
    return result,result_tree,done,limited


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
        assert type(config['recovery_off_version']) is int and config['recovery_off_version'] == 1
        assert meta['state']['revision'] == 0, 'rescoring requires separate independent replay admission'
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
        restricted_component_events = excluded_archived_prefix_occurrences = 0
        old_n, last_clock, last_reference, maximum = 0, -1, -1, 0.
        factor_hash = hashlib.sha256()
        prediction_head = audit_head = '0' * 64
        configuration_sha = hashlib.sha256(numeric.canonical(config)).hexdigest()
        for ordinal, pb, ab in db.execute('SELECT ordinal,prediction,audit FROM events ORDER BY ordinal'):
            prediction, audit = json.loads(pb), json.loads(ab)
            assert ordinal == events and audit['kind'] == SCHEMA
            assert prediction['previous_commit_sha256'] == prediction_head
            assert audit['previous_audit_sha256'] == audit_head
            assert audit['configuration_sha256'] == configuration_sha
            prediction_head = hashlib.sha256(numeric.canonical({k:v for k,v in prediction.items() if k!='commit_sha256'})).hexdigest()
            assert prediction['commit_sha256'] == audit['prediction_sha256'] == prediction_head
            audit_head = hashlib.sha256(numeric.canonical(audit)).hexdigest()
            assert audit['recovery_enabled'] is False
            assert audit['priority_decision_support_scope'] == SCOPE
            assert audit['priority_action_materialization'] is False
            assert audit['unrestricted_priority_checkpoint_transfer'] is False
            assert audit['original_model_risk_certified'] is False
            assert audit['partition_bound_scope'] == 'current_irreversibly_pruned_support'
            for key in ('model_regret_upper','proposed_model_regret_upper',
                        'weighted_truncation_risk_upper','product_omitted_mass_upper'):
                assert audit[key] == 1., 'restricted posterior falsely certifies original model'

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
                clauses = derive_clauses(previous.get(c),changes.get(c),members,memberships,previous)
                payload = support_payload(clauses)
                support_sha = hashlib.sha256(numeric.canonical(payload)).hexdigest()
                assert numeric.canonical(s['historical_support']) == numeric.canonical(payload), 'historical support differs from prior commits'
                assert s['historical_support_sha256'] == support_sha
                restricted_component_events += bool(clauses)
                assert s['mass_scope'] == SCOPE and s['complete_raw_support_retained'] is False
                assert s['raw_factors_preserved'] is True
                assert s['decision']['support_sha256'] == support_sha and s['decision']['risk_scope'] == SCOPE
                assert s['decision']['original_model_regret_upper'] == 1. and s['decision']['original_model_risk_certified'] is False
                raw_state = dict(history_clauses=clauses, members=list(members),kind='exclusive_raw_kernel_search_snapshot_v1', sequence_id=meta['sequence_id'],
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
                final_tree = snapshot(full)
                excluded_archived_prefix_occurrences += sum(not allows(value['history_clauses'], roots(final_tree,h)) for h in final_tree.p)
                output = s['output_handle']
                choices = numeric.path_choices(final_tree, output)
                assert len(choices) == value['nodes'] and final_tree.p[output][7] == s['output_sha256']
                assert allows(value['history_clauses'],roots(final_tree,output)), 'final output recovered discarded history'
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
                progress(dict(stage='independent_recovery_off_learned_full_search_trajectory', completed_events=events,
                              total_events=meta['state']['events'], ETA_seconds=None,
                              ETA_reason='prefix sizes and candidate counts vary'))
        assert events == meta['state']['events'] and old_n == len(raw)
        assert prediction_head == meta['state']['prediction_sha256'] and audit_head == meta['state']['audit_sha256']
        assert catalog == final_catalog and memberships == sql_members
        for c, value in previous.items():
            assert value['prefixes'] == full_prefixes[c]
        assert numeric.sha(path) == expected_sha256 and numeric.sha(weights_path) == weights_sha256
        return dict(kind='rbf_independent_recovery_off_learned_search_trajectory_checks_v1',
                    historical_support_rebuilt_from_prior_commit_active_union_output=True,
                    restricted_component_events=restricted_component_events,
                    excluded_archived_prefix_occurrences=excluded_archived_prefix_occurrences,
                    correlated_restrictions_preserved_without_cartesian_materialization=True,
                    priority_decision_support_scope=SCOPE, original_model_risk_certified=False,
                    immutable_prediction_and_audit_chain_checked=True,
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
