"""Independent event-boundary support, restricted cuts and mass audit.

No producer imports. Previous immutable event commits define correlated root
classes; current metadata is only compared with that independently derived set.
Raw gross bounds may include forbidden histories and remain conservative. The
scope is the restricted model, never original-posterior regret or performance.
"""
from collections import defaultdict
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import time

ROOT = Path('/Volumes/Data/test/recover-before-fuse')
TREE_PATH = ROOT/'source-freezes/rbf-independent-identity-forest-audit-v1-20261001/oracle.py'
TREE_SHA = '24be6bf660cbecadc3102320e18e5b12ff06063b050e4dfc44307d4defeea8a8'
ATOL = RTOL = 1e-8
EPS = 2.220446049250313e-16
SCHEMA = 'persistent_exclusive_event_boundary_recovery_off_v1'
RECIPE = 'history_restriction_intersect_prefix_minus_retained_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()


assert sha(TREE_PATH) == TREE_SHA
_spec = importlib.util.spec_from_file_location('recovery_off_independent_raw_tree', TREE_PATH)
_tree_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tree_module)
IndependentRootTree = _tree_module.IndependentRootTree
lse = _tree_module.lse


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    error = abs(actual-expected)
    assert error <= ATOL+RTOL*abs(expected), ('independent mass arithmetic', actual, expected)
    return error


def path_roots(tree, handle):
    choices = []
    while handle:
        row = tree.p[handle]
        choices.append(row[3])
        handle = row[1]
    roots = []
    for i, parent in enumerate(reversed(choices)):
        assert -1 <= parent < i
        roots.append(i if parent < 0 else roots[parent])
    return tuple(roots)


class ComponentContext:
    def __init__(self, summary, tree, members, clauses):
        self.summary, self.tree, self.members = summary, tree, tuple(members)
        self.clauses = tuple((tuple(indices), tuple(tuple(v) for v in variants))
                             for indices, variants in clauses)

    def allowed(self, roots):
        # A conjunction of whole predecessor vectors, not marginal root sets.
        return all(any(all(roots[i] == variant[j] for j, i in enumerate(indices)
                           if i < len(roots)) for variant in variants)
                   for indices, variants in self.clauses)

    def payload(self):
        return dict(recipe='previous_commit_active_union_output_root_classes_v1',
                    clauses=self.clauses, cartesian_product_materialized=False)


def _load(db):
    meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
    assert meta['schema'] == SCHEMA
    assert meta['state']['revision'] == 0, 'mutable raw factor histories require a separate oracle'
    config = meta['config']
    assert config['recovery_off_version'] == config['residual_partition_version'] == 1
    assert config['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    catalogs = {c: (live, json.loads(pred), clock) for c, live, pred, clock in
                db.execute('SELECT component,live,predecessors,created_us FROM component_catalog')}
    members, trees = {}, {}
    raw_slots = {i: (source, frame) for i, source, frame in db.execute('SELECT i,source,frame FROM observations')}
    raw_factors = defaultdict(list)
    for i, p, w in db.execute('SELECT i,p,w FROM potentials ORDER BY i,p'):
        raw_factors[i].append((p, w))
    for c in catalogs:
        records = list(db.execute('SELECT local_i,global_i FROM component_members WHERE component=? ORDER BY local_i', (c,)))
        assert [r[0] for r in records] == list(range(len(records)))
        members[c] = tuple(r[1] for r in records)
        assert tuple(sorted(set(members[c]))) == members[c]
        slots = list(db.execute(f'SELECT source,frame FROM pc{c}_observations ORDER BY i'))
        assert slots == [raw_slots[g] for g in members[c]], 'component view changed raw observation identity'
        factors = [[] for _ in slots]
        for i, p, w in db.execute(f'SELECT i,p,w FROM pc{c}_potentials ORDER BY i,p'):
            factors[i].append((p, w))
        assert len(slots) == len(members[c])
        assert all(row and row[0][0] == -1 and len(row) == len(dict(row)) for row in factors)
        inverse = {g: i for i, g in enumerate(members[c])}
        expected_factors = [[(-1 if p < 0 else inverse[p], w) for p, w in raw_factors[g]] for g in members[c]]
        assert factors == expected_factors, 'component view changed immutable raw factor support'
        trees[c] = IndependentRootTree(db.execute(f'SELECT * FROM pc{c}_prefixes').fetchall(),
                                       slots, factors, meta['sequence_id'])
        assert set(trees[c].p) == set(range(len(trees[c].p)))
        for h, revision, value in db.execute(f'SELECT h,revision,value FROM pc{c}_weights'):
            assert revision == 0
            close(value, trees[c].weights[h])
    return meta, catalogs, members, trees


def replay_contexts(db):
    """Yield (ordinal, prediction, audit, contexts) with derived history support.

    The caller must bind the database bytes. Iterating to exhaustion validates
    the entire immutable chain, causal component lineage and final ownership.
    Per-event region/mass validation is provided separately by verify_database.
    """
    meta, catalogs, members, trees = _load(db)
    count = db.execute('SELECT count(*) FROM observations').fetchone()[0]
    assert [i for i, in db.execute('SELECT i FROM observations ORDER BY i')] == list(range(count))
    edges = defaultdict(list)
    for i, p in db.execute('SELECT i,p FROM potentials WHERE p>=0 ORDER BY i,p'):
        assert 0 <= p < i < count
        edges[i].append(p)
    parent = []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prior = {}
    all_last, component_events = {}, defaultdict(int)
    latest = {}
    previous_n, events = 0, 0
    previous_prediction = previous_audit = '0'*64
    last_clock = -1
    for ordinal, event_id, pb, ab in db.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal'):
        p, a = json.loads(pb), json.loads(ab)
        assert ordinal == events and event_id == a['event_id'] and a['sequence_id'] == meta['sequence_id']
        assert a['kind'] == SCHEMA and a['configuration_sha256'] == digest(meta['config'])
        assert p['previous_commit_sha256'] == previous_prediction and a['previous_audit_sha256'] == previous_audit
        assert digest({k: v for k, v in p.items() if k != 'commit_sha256'}) == p['commit_sha256'] == a['prediction_sha256']
        previous_prediction, previous_audit = p['commit_sha256'], digest(a)
        clock = p['decision_timestamp_us']
        assert clock >= last_clock
        last_clock = clock
        n = a['observation_count']
        assert n == previous_n+a['new_observations'] and not a['rescored_rows'] and n <= count
        for i in range(previous_n, n):
            parent.append(i)
            for candidate in edges[i]:
                r, q = root(i), root(candidate)
                parent[max(r, q)] = min(r, q)
        expected_groups = defaultdict(set)
        for i in range(n):
            expected_groups[root(i)].add(i)
        previous_owner = {g: c for c, old in prior.items() for g in old.members}
        scope = tuple(a['decision_indices'])
        assert len(scope) == len(set(scope)) and all(type(i) is int and 0 <= i < n for i in scope)
        contexts, declared_groups, ids = [], set(), set()
        for s in a['components']:
            c = s['component']
            assert c in catalogs and c not in ids
            ids.add(c)
            current = members[c][:s['nodes']]
            assert len(current) == s['nodes'] and all(g < n for g in current)
            group = frozenset(current)
            assert group and group not in declared_groups
            declared_groups.add(group)
            predecessors = sorted({previous_owner[g] for g in current if g in previous_owner})
            assert s['predecessors'] == predecessors, 'advertised predecessor lineage differs from immutable commits'
            assert s['merge_restart'] == (len(predecessors) > 1)
            if len(predecessors) == 1:
                assert c == predecessors[0], 'one-predecessor continuation changed component identity'
            else:
                assert c not in latest and catalogs[c][1] == predecessors and catalogs[c][2] == clock
            inverse = {g: i for i, g in enumerate(current)}
            local_scope = tuple(inverse[g] for g in scope if g in inverse)
            assert tuple(s['decision_indices']) == local_scope and s['indices_count'] == len(local_scope)
            close(s['weight'], len(local_scope)/len(scope) if scope else 0.)
            clauses = []
            for old_c in predecessors:
                old = prior[old_c]
                assert set(old.members) <= group, 'predecessor component split or lost members'
                indices = tuple(inverse[g] for g in old.members)
                variants = set()
                for handle in {r['handle'] for r in old.summary['active']} | {old.summary['output_handle']}:
                    values = path_roots(old.tree, handle)
                    assert len(values) == len(old.members)
                    variants.add(tuple(inverse[old.members[r]] for r in values))
                clauses.append((indices, tuple(sorted(variants))))
            ctx = ComponentContext(s, trees[c], current, clauses)
            assert canonical(s['historical_support']) == canonical(ctx.payload()), 'historical support not derived from prior commits'
            assert s['historical_support_sha256'] == digest(ctx.payload())
            assert latest.get(c, 0) <= s['prefix_nodes'] <= len(trees[c].p)
            latest[c] = s['prefix_nodes']
            contexts.append(ctx)
        assert declared_groups == {frozenset(v) for v in expected_groups.values()}, 'component graph partition differs from raw candidate edges'
        assert a['total_prefix_nodes'] == sum(latest.values()) <= meta['config']['max_total_prefix_nodes']
        assert len(contexts) <= meta['config']['max_components']
        yield ordinal, p, a, contexts
        prior = {c.summary['component']: c for c in contexts}
        all_last.update(prior)
        for c in prior:
            component_events[c] += 1
        previous_n = n
        events += 1
    assert events == meta['state']['events'] and previous_n == count
    assert previous_prediction == meta['state']['prediction_sha256'] and previous_audit == meta['state']['audit_sha256']
    assert set(prior) == {c for c, (live, _, _) in catalogs.items() if live}
    assert dict(db.execute('SELECT global_i,component FROM component_owners')) == {
        g: c for c, context in prior.items() for g in context.members}
    assert latest == {c: len(tree.p) for c, tree in trees.items()}
    assert sum(len(v) for v in members.values()) <= meta['config']['max_member_rows']
    for c, context in all_last.items():
        stored = {k: json.loads(v) for k, v in db.execute(f'SELECT k,v FROM pc{c}_meta')}
        s = context.summary
        assert stored['residual_partition_version'] == 1
        assert canonical(stored['recovery_off_restriction']) == canonical(context.payload())
        state = stored['state']
        assert state['n'] == s['nodes'] and state['revision'] == 0
        assert state['active'] == sorted(r['handle'] for r in s['active'])
        assert state['frontier'] == sorted(r['handle'] for r in s['frontier'])
        assert state['output'] == s['output_handle'] and state['events'] == component_events[c]
        assert canonical(stored['residual_regions']) == canonical(s['frontier'])


def restricted_cut(ctx):
    """Prove restricted raw-support coverage without full leaf enumeration."""
    s, tree = ctx.summary, ctx.tree
    active = [r['handle'] for r in s['active']]
    frontier = [r['handle'] for r in s['frontier']]
    assert len(active) == len(set(active)) and len(frontier) == len(set(frontier))
    assert not set(active) & set(frontier)
    handles = active+frontier+[s['output_handle'], s['fallback_handle']]+[r['handle'] for r in s['branches']]
    for h in handles:
        assert h in tree.p and h < s['prefix_nodes'] and tree.p[h][2] <= s['nodes']
        assert ctx.allowed(path_roots(tree, h)), 'live handle recovered pruned historical identity'
    assert all(tree.p[h][2] == s['nodes'] for h in active+[s['output_handle'], s['fallback_handle']])
    assert {r['handle'] for r in s['branches']} == set(active) | {s['output_handle']}
    assert len(s['branches']) == len(set(active) | {s['output_handle']})
    ordered = sorted(frontier, key=tree.enter.__getitem__)
    assert all(tree.exit[a] <= tree.enter[b] for a, b in zip(ordered, ordered[1:])), 'overlapping restricted prefix regions'
    covered = {h for h in active if any(tree.ancestor(f, h) for f in frontier)}
    terminals = set(frontier) | (set(active)-covered)
    closure, children = set(terminals), defaultdict(set)
    for leaf in terminals:
        h = leaf
        while h:
            p = tree.p[h][1]
            children[p].add(tree.p[h][3])
            if p in closure:
                break
            closure.add(p)
            h = p
    assert 0 in closure
    for h in closure-terminals:
        values = path_roots(tree, h)
        assert len(values) < s['nodes'] and ctx.allowed(values)
        expected = {p for p in tree.allowed[h]
                    if ctx.allowed(values+(len(values) if p < 0 else values[p],))}
        assert children[h] == expected, 'restricted support cut has a missing or forbidden child'
    return covered


def verify_regions(ctx, width):
    s, tree = ctx.summary, ctx.tree
    assert s['representation'] == 'exclusive_recovery_off_prefix_regions_v1'
    assert s['residual_partition_recipe'] == RECIPE
    assert s['mass_scope'] == 'conditional_on_irreversibly_pruned_support'
    assert s['complete_raw_support_retained'] is False and s['raw_factors_preserved'] is True
    assert s['recovery_events'] == []
    active = [r['handle'] for r in s['active']]
    assert len(active) <= width
    covered = restricted_cut(ctx)
    maximum, references = 0., 0
    for row in s['active']:
        assert row['sha256'] == tree.p[row['handle']][7]
        maximum = max(maximum, close(row['log_weight'], tree.weights[row['handle']]))
    for r in s['frontier']:
        h = r['handle']
        assert r['sha256'] == tree.p[h][7] and r['depth'] == tree.p[h][2]
        assert r['nodes'] == s['nodes'] and r['factor_revision'] == 0
        assert r['region_recipe'] == RECIPE and r['base_upper_recipe'] == 'prefix_region_minus_retained_terminal_classes_v1'
        assert r['historical_support_sha256'] == digest(ctx.payload()) and r['gross_upper_may_include_pruned_histories'] is True
        assert r['region_sha256'] == digest({k: v for k, v in r.items() if k != 'region_sha256'})
        expected = {leaf for leaf in active if tree.ancestor(h, leaf)}
        actual = {leaf['handle'] for leaf in r['excluded_leaves']}
        assert len(actual) == len(r['excluded_leaves']) and actual == expected
        references += len(actual)
        for leaf in r['excluded_leaves']:
            assert leaf['sha256'] == tree.p[leaf['handle']][7]
            maximum = max(maximum, close(leaf['log_weight'], tree.weights[leaf['handle']]))
        maximum = max(maximum, close(r['log_gross_upper'], tree.upper(h, s['nodes'])))
        if not actual:
            assert r['log_excluded_mass'] is None and r['gross_padding'] == r['excluded_lower_padding'] == r['output_padding'] == 0.
            maximum = max(maximum, close(r['log_upper'], r['log_gross_upper']))
        else:
            known = lse(tree.weights[h] for h in actual)
            maximum = max(maximum, close(r['log_excluded_mass'], known))
            padding = 64*EPS*max(1., abs(r['log_gross_upper']), abs(known), s['nodes'], len(actual))
            assert r['gross_padding'] == r['excluded_lower_padding'] == padding
            gross = math.nextafter(r['log_gross_upper']+padding, math.inf)
            lower = math.nextafter(known-padding, -math.inf)
            assert gross > lower
            with localcontext() as decimal_context:
                decimal_context.prec = 90
                g = Decimal.from_float(gross)
                residual = float(g+(Decimal(1)-(Decimal.from_float(lower)-g).exp()).ln())
            assert abs(r['output_padding']-32*EPS*max(1., abs(residual), len(actual))) <= 1e-24
            maximum = max(maximum, close(r['log_upper'], residual+r['output_padding']))
    assert references == len(covered) == s['residual_exclusion_references'] <= width
    assert s['residual_region_metadata_bytes'] == len(canonical(s['frontier']))
    retained = lse(tree.weights[h] for h in active) if active else None
    terms = [r['log_upper'] for r in s['frontier']]+[tree.weights[h] for h in active]
    raw = lse(terms)
    upper = raw+32*EPS*max(1., abs(max(terms)), abs(raw), len(terms))
    eta = 0. if not s['frontier'] else 1. if not active else max(0., min(1., -math.expm1(min(0., retained-upper))))
    if active:
        maximum = max(maximum, close(s['log_retained'], retained))
    else:
        assert s['log_retained'] is None
    maximum = max(maximum, close(s['log_partition_upper'], upper), close(s['eta_upper'], eta))
    assert tree.p[s['output_handle']][7] == s['output_sha256']
    for row in s['branches']:
        assert tree.p[row['handle']][2] == s['nodes'] and row['sha256'] == tree.p[row['handle']][7]
    decision = s['decision']
    assert decision['action_space'] == 'legal_extensions_of_previous_retained_and_output_classes'
    assert decision['support_sha256'] == digest(ctx.payload())
    assert decision['risk_scope'] == 'conditional_on_irreversibly_pruned_support'
    assert decision['original_model_regret_upper'] == 1. and decision['original_model_risk_certified'] is False
    assert 0 <= decision['risk_bound'] <= 1.
    return dict(max_abs_error=maximum, references=references, retained=retained, upper=upper, eta=eta, active=bool(active))


def verify_database(path, expected_sha, progress=lambda v: None):
    path = Path(path)
    assert sha(path) == expected_sha
    started = time.monotonic()
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        config, state = meta['config'], meta['config']['state']
        events = components = restriction_clauses = 0
        maximum = 0.
        for ordinal, prediction, a, contexts in replay_contexts(db):
            assert a['recovery_enabled'] is False and a['recovery_policy'] == 'prune_to_previous_commit_active_union_output'
            assert a['historical_raw_evidence_preserved'] is True and a['original_model_risk_certified'] is False
            assert a['partition_bound_scope'] == 'current_irreversibly_pruned_support'
            assert a['formal_numeric_certificate'] is False and a['true_posterior_or_metric_bound'] is False
            values = [verify_regions(c, state['active_limit']) for c in contexts]
            for v in values:
                maximum = max(maximum, v['max_abs_error'])
            assert a['search_steps'] <= state['expansion_budget'] and a['action_search_steps'] <= state['paper_action_budget']
            assert a['expansions'] == sum(c.summary['expansions'] for c in contexts)
            assert a['proposal_steps'] == sum(c.summary['proposal_steps'] for c in contexts)
            assert a['search_steps'] == a['expansions']+a['proposal_steps']
            assert a['state_updates'] <= state['max_replay_operations']
            assert a['total_frontier'] == sum(len(c.summary['frontier']) for c in contexts) <= config['max_total_frontier']
            assert a['residual_exclusion_references'] == sum(v['references'] for v in values)
            assert a['residual_region_metadata_bytes'] == sum(c.summary['residual_region_metadata_bytes'] for c in contexts)
            assert a['restriction_serialized_metadata_bytes'] == sum(len(canonical(c.payload())) for c in contexts)
            assert a['restriction_metadata_counted_in_database'] is True and a['restriction_serialized_bytes_are_not_process_peak_memory'] is True
            maximum = max(maximum, close(a['log_partition_upper'], math.fsum(v['upper'] for v in values)))
            weighted_eta = math.fsum(c.summary['weight']*v['eta'] for c, v in zip(contexts, values))
            risk = math.fsum(c.summary['weight']*c.summary['decision']['risk_bound'] for c in contexts)
            ratio = math.fsum(v['retained']-v['upper'] for v in values) if all(v['active'] for v in values) else None
            omitted = 1. if ratio is None else max(0., min(1., -math.expm1(min(0., ratio))))
            assert state['max_model_regret'] == 1. and a['global_fallback_used'] is False
            for key, value in [('model_regret_upper', risk), ('proposed_model_regret_upper', risk),
                               ('weighted_truncation_risk_upper', weighted_eta), ('product_omitted_mass_upper', omitted)]:
                assert a[key] == 1., 'restricted risk was promoted to original-model risk'
                maximum = max(maximum, close(a['restricted_support_'+key], value))
            components += len(contexts)
            restriction_clauses += sum(len(c.clauses) for c in contexts)
            events += 1
            if events % 25 == 0:
                progress(dict(stage='independent_recovery_off_structure', completed_events=events,
                              total_events=meta['state']['events'], ETA_seconds=None,
                              ETA_reason='remaining prefix closure and predecessor projection work varies'))
        assert sha(path) == expected_sha
        return dict(kind='rbf_independent_recovery_off_history_restricted_structure_mass_v1',
                    database_sha256=expected_sha, sequence_id=meta['sequence_id'], events=events,
                    component_event_snapshots=components, predecessor_clauses=restriction_clauses,
                    historical_restrictions_derived_from_immutable_commits=True,
                    raw_edge_component_membership_and_merge_lineage_verified=True,
                    all_live_handles_within_previous_active_union_output_root_classes=True,
                    restricted_support_cut_coverage_verified=True,
                    independent_restricted_support_coverage_verified=True,
                    restricted_residual_mass_arithmetic_verified=True,
                    original_risk_bounds_conservative_unit_bounds=True,
                    max_abs_arithmetic_error=maximum, atol=ATOL, rtol=RTOL,
                    original_support_coverage_claimed=False, formal_interval_certificate=False,
                    conditional_action_search_verified=False, fresh_branch_states_verified=False,
                    complete_online_method_accepted=False, paper_performance_complete=False,
                    elapsed_seconds=time.monotonic()-started)
    finally:
        db.close()
