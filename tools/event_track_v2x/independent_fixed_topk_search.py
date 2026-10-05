"""Read-only fixed TopK search/pruning audit, without production imports.

The oracle derives append-only components from the raw edge graph, enumerates
each small retained-beam Cartesian product, and enumerates legal root-class
children directly. It never starts a later event from the producer's active
list. This is the arrival-node local beam, not ranked whole-history MHT.

Conditional output selection, continuous states, neural factors, causal/cache
inputs and performance are separate gates. Floating-point agreement is not a
true-posterior certificate. Existing raw-tree validation is reused unchanged.
"""
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

ROOT_ORACLE = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/'
                   'rbf-independent-identity-forest-audit-v1-20261001/oracle.py')
ROOT_SHA = '24be6bf660cbecadc3102320e18e5b12ff06063b050e4dfc44307d4defeea8a8'
POLICY = 'fixed_top_k_root_classes_after_each_arrival_ordered_node'
SCHEMA = 'persistent_irreversible_identity_beam_v1'
ATOL = RTOL = 1e-8


def load_root_oracle():
    import hashlib
    assert hashlib.sha256(ROOT_ORACLE.read_bytes()).hexdigest() == ROOT_SHA
    spec = importlib.util.spec_from_file_location('unchanged_topk_raw_root_tree', ROOT_ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ATOL == module.RTOL == ATOL
    return module


def partition_plans(old_count, count, factors, owners):
    """Independent connected components by graph traversal, without union-find."""
    graph = defaultdict(set)
    for node in range(old_count, count):
        key = ('new', node)
        graph[key]
        for parent, _ in factors[node]:
            if parent < 0:
                continue
            other = ('old', owners[parent]) if parent < old_count else ('new', parent)
            graph[key].add(other)
            graph[other].add(key)
    remaining = set(graph)
    plans = []
    while remaining:
        queue = [min(remaining)]; group = set()
        while queue:
            key = queue.pop()
            if key in group:
                continue
            group.add(key)
            queue.extend(graph[key] - group)
        remaining -= group
        prior = tuple(sorted(v for kind, v in group if kind == 'old'))
        added = tuple(sorted(v for kind, v in group if kind == 'new'))
        assert added
        plans.append((prior, added))
    return sorted(plans, key=lambda value: value[1][0])


def cartesian_topk(groups, width):
    """Exact at-most-K-squared enumeration at each retained-product stage.

    The reported heap work is checked using adjacency of the independently
    selected grid cells: the initial cell and in-bounds right/down neighbours
    of every popped cell. No heap or production search implementation is used.
    """
    assert type(width) is int and width > 0
    retained = [(0., ())]; complete = True; stages = []; evaluations = 0
    for component, options in groups:
        assert options and len(options) <= width
        assert len({handle for _, handle in options}) == len(options)
        left = sorted(retained, key=lambda v: (-v[0], v[1]))
        right = sorted(options, key=lambda v: (-v[0], v[1]))
        candidates = [(math.fsum((a[0], b[0])), a[1]+((component, b[1]),), i, j)
                      for (i, a), (j, b) in itertools.product(enumerate(left), enumerate(right))]
        selected = sorted(candidates, key=lambda v: (-v[0], v[1]))[:width]
        queried = {(0, 0)}
        for _, _, i, j in selected:
            if i+1 < len(left): queried.add((i+1, j))
            if j+1 < len(right): queried.add((i, j+1))
        evaluations += len(queried)
        complete &= len(candidates) <= width
        stages.append(dict(component=component, retained_prefix_product_candidates=len(candidates),
                           kept=len(selected), dropped_at_stage=len(candidates)-len(selected)))
        retained = [(score, selection) for score, selection, _, _ in selected]
    return retained, complete, stages, evaluations


@dataclass(frozen=True)
class Branch:
    handle: int
    choices: tuple
    roots: tuple


def legal_candidates(tree, beam, depth, root_oracle):
    """Group raw parent edges by root identity, summing equivalent paths."""
    result = []
    for branch in sorted(beam, key=lambda b: b.handle):
        assert len(branch.roots) == len(branch.choices) == depth
        assert tree.p[branch.handle][2] == depth
        forbidden = {branch.roots[j] for j in tree.slot_positions[tree.slots[depth]] if j < depth}
        groups = {}
        for parent, weight in tree.factors[depth]:
            root = depth if parent < 0 else branch.roots[parent]
            if parent < 0 or root not in forbidden:
                groups.setdefault(root, []).append((parent, weight))
        canonical = tuple(sorted(min(p for p, _ in rows) for rows in groups.values()))
        assert canonical == tree.allowed[branch.handle], 'raw root-class canonical choices disagree'
        for root, rows in groups.items():
            choice = min(p for p, _ in rows)
            score = math.fsum((tree.weights[branch.handle], root_oracle.lse(w for _, w in rows)))
            sha256 = root_oracle.digest([tree.p[branch.handle][7], depth, choice, root])
            result.append((score, sha256, branch, choice, root))
    return result


class DatabaseAudit:
    def __init__(self, db, root_oracle, meta):
        self.db = db; self.root = root_oracle; self.meta = meta
        self.width = meta['config']['state']['active_limit']
        assert type(self.width) is int and self.width == 4
        assert meta['schema'] == SCHEMA and meta['config']['state']['decision_mode'] == 'retained'
        assert meta['config']['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
        assert meta['config']['state']['prune_score'] == 0.05
        self.slots = []
        self.arrivals = []
        for index, source, frame, arrival in db.execute('SELECT i,source,frame,arrival_us FROM observations ORDER BY i'):
            assert index == len(self.slots)
            self.slots.append((source, frame)); self.arrivals.append(arrival)
        self.factors = [[] for _ in self.slots]
        for index, parent, weight in db.execute('SELECT i,p,w FROM potentials ORDER BY i,p'):
            assert 0 <= index < len(self.slots) and -1 <= parent < index and math.isfinite(weight)
            self.factors[index].append((parent, weight))
        assert all(row and row[0][0] == -1 and len(row) == len(dict(row)) for row in self.factors)
        self.trees = {}; self.created = {}; self.members = {}; self.catalog = {}
        self.beams = {}; self.complete = {}; self.owners = {}; self.live = set()
        self.next_component = 1; self.errors = defaultdict(float)
        self.totals = defaultdict(int)

    def close(self, key, got, expected):
        assert type(got) in (float, int) and math.isfinite(got) and math.isfinite(expected)
        error = abs(got-expected)
        assert error <= ATOL + RTOL*abs(expected), (key, got, expected)
        self.errors[key] = max(self.errors[key], error)

    def tree(self, component):
        if component not in self.trees:
            full_members = tuple(g for g, in self.db.execute(
                'SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (component,)))
            assert len(set(full_members)) == len(full_members)
            inverse = {g: i for i, g in enumerate(full_members)}
            factors = [[(-1 if p < 0 else inverse[p], w) for p, w in self.factors[g]] for g in full_members]
            slots = [self.slots[g] for g in full_members]
            rows = self.db.execute(f'SELECT h,parent,depth,choice,root,previous_root,jumps,sha FROM pc{component}_prefixes ORDER BY h').fetchall()
            tree = self.root.IndependentRootTree(rows, slots, factors, self.meta['sequence_id'])
            tree.child = {(row[1], row[3]): h for h, row in tree.p.items() if h}
            assert len(tree.child) == len(tree.p)-1
            tree.full_members = full_members
            tree.slot_positions = defaultdict(list)
            for i, slot in enumerate(slots): tree.slot_positions[slot].append(i)
            self.trees[component] = tree
        return self.trees[component]

    def materialize(self, component, handle, choice):
        tree = self.tree(component)
        assert (handle, choice) in tree.child, 'selected class missing from stored prefixes'
        result = tree.child[handle, choice]
        assert handle in self.created[component]
        if result not in self.created[component]:
            assert result == len(self.created[component]), 'unexpected prefix materialization order or hidden archive prefix'
            self.created[component].add(result)
        return result

    def lift(self, component, selection, depth, expected_weight):
        members = self.members[component][:depth]; inverse = {g: i for i, g in enumerate(members)}
        choices = [None]*depth
        for previous, handle in selection:
            branches = {b.handle: b for b in self.beams[previous]}
            assert handle in branches, 'merge resurrected a discarded predecessor branch'
            branch = branches[handle]; old_members = self.members[previous]
            assert len(branch.choices) == len(old_members)
            for old_i, choice in enumerate(branch.choices):
                local = inverse[old_members[old_i]]
                assert choices[local] is None, 'predecessor histories overlap'
                choices[local] = -1 if choice < 0 else inverse[old_members[choice]]
        assert all(choice is not None for choice in choices)
        tree = self.tree(component); h = 0; roots = []
        for i, choice in enumerate(choices):
            assert choice in tree.allowed[h]
            h = self.materialize(component, h, choice)
            roots.append(i if choice < 0 else roots[choice])
        self.close('lifted_product_log_weight', tree.weights[h], expected_weight)
        return Branch(h, tuple(choices), tuple(roots))

    def extend(self, component, beam, start, stop):
        tree = self.tree(component); pruning = []; complete = True
        for depth in range(start, stop):
            candidates = legal_candidates(tree, beam, depth, self.root)
            assert candidates
            selected = sorted(candidates, key=lambda v: (-v[0], v[1], v[2].handle, v[3]))[:self.width]
            self.work['beam_expansions'] += len(beam)
            self.work['candidate_evaluations'] += len(candidates)
            generated = self.root.lse(v[0] for v in candidates)
            kept = self.root.lse(v[0] for v in selected)
            pruning.append(dict(depth=depth+1, generated_classes=len(candidates), retained_classes=len(selected),
                dropped_classes=len(candidates)-len(selected),
                generated_prefix_discarded_mass=max(0., min(1., -math.expm1(min(0., kept-generated)))),
                mass_scope='current_generated_prefixes_not_full_posterior'))
            complete &= len(candidates) == len(selected)
            beam = [Branch(self.materialize(component, b.handle, choice), b.choices+(choice,), b.roots+(root,))
                    for _, _, b, choice, root in selected]
        return beam, complete, pruning

    def verify_event(self, ordinal, prediction, audit, old_count):
        count = audit['observation_count']; clock = prediction['decision_timestamp_us']
        assert old_count <= count <= len(self.slots) and audit['new_observations'] == count-old_count
        assert not audit['rescored_rows'] and all(t <= clock for t in self.arrivals[:count])
        assert audit['configuration_sha256'] == self.root.digest(self.meta['config'])
        assert audit['kind'] == SCHEMA and audit['pruning_policy'] == POLICY
        assert audit['beam_width'] == self.width and audit['allocation_policy'] == 'fixed_width_no_adaptive_search'
        assert audit['output_policy'] == 'conditional_bayes_over_retained_classes_no_regret_fallback'
        for key in ('recovery_enabled', 'archived_prefixes_used_for_recovery', 'learned_allocation_policy',
                    'reproduced_classical_mht', 'true_posterior_or_metric_bound', 'formal_numeric_certificate',
                    'global_fallback_used', 'work_counters_are_latency_equivalence'):
            assert audit[key] is False, ('incorrect scope declaration', key)
        assert audit['exact_equivalent_parent_paths_summed'] is True
        assert audit['raw_history_retained_on_disk'] is True
        assert audit['total_frontier'] == 0
        indices = audit['decision_indices']
        assert len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < count for i in indices)
        self.work = defaultdict(int); changes = {}; products = {}
        for prior, added in partition_plans(old_count, count, self.factors, self.owners):
            if len(prior) == 1:
                component = prior[0]; start = len(self.members[component])
                self.members[component] += tuple(added)
            else:
                component = self.next_component; self.next_component += 1
                old = tuple(sorted(g for c in prior for g in self.members[c]))
                self.members[component] = old+tuple(added); start = len(old)
                self.catalog[component] = (prior, clock); self.created[component] = {0}
                self.live.add(component); self.live.difference_update(prior)
            for g in self.members[component]: self.owners[g] = component
            assert self.tree(component).full_members[:len(self.members[component])] == self.members[component]
            changes[component] = (prior, start)
            if len(prior) > 1:
                groups = [(c, [(self.tree(c).weights[b.handle], b.handle) for b in self.beams[c]]) for c in prior]
                combined, unpruned, stages, evaluations = cartesian_topk(groups, self.width)
                self.work['merge_score_evaluations'] += evaluations
                self.work['candidate_evaluations'] += evaluations
                products[component] = (combined, all(self.complete[c] for c in prior) and unpruned, stages)
        summaries = {s['component']: s for s in audit['components']}
        assert len(summaries) == len(audit['components']) and set(summaries) == self.live
        for component in sorted(self.live):
            tree = self.tree(component); n = len(self.members[component]); summary = summaries[component]
            prior, start = changes.get(component, ((component,), n))
            merge_stages = []
            if component in products:
                combined, complete, merge_stages = products[component]
                beam = [self.lift(component, selection, start, score) for score, selection in combined]
                self.work['merge_prefix_steps'] += len(combined)*start
                assert len({b.handle for b in beam}) == len(beam)
            elif not prior:
                beam = [Branch(0, (), ())]; complete = True
            else:
                beam = self.beams[component]; complete = self.complete[component]
            beam, unpruned, pruning = self.extend(component, beam, start, n)
            complete &= unpruned
            active = sorted(beam, key=lambda b: (-tree.weights[b.handle], tree.p[b.handle][7]))
            assert [a['handle'] for a in summary['active']] == [b.handle for b in active], 'retained TopK class selection differs'
            for actual, branch in zip(summary['active'], active):
                assert actual['sha256'] == tree.p[branch.handle][7]
                self.close('active_log_weight', actual['log_weight'], tree.weights[branch.handle])
            assert summary['nodes'] == n and summary['prefix_nodes'] == len(self.created[component])
            inverse = {g: i for i, g in enumerate(self.members[component])}
            scope = [inverse[g] for g in indices if g in inverse]
            assert summary['decision_indices'] == scope and summary['indices_count'] == len(scope)
            self.close('component_loss_scope_weight', summary['weight'], len(scope)/len(indices) if indices else 0.)
            assert summary['predecessors'] == list(prior)
            assert summary['merge_restart'] is False and summary['merge_retained_beams_only'] == (component in products)
            assert summary['merge_pruning_stages'] == merge_stages
            assert summary['frontier'] == summary['recovery_events'] == []
            assert summary['complete_raw_support_retained'] is True
            assert type(summary['full_model_support_still_in_beam']) is bool and summary['full_model_support_still_in_beam'] == complete
            assert len(summary['pruning']) == len(pruning)
            for got, expected in zip(summary['pruning'], pruning):
                for key in ('depth', 'generated_classes', 'retained_classes', 'dropped_classes', 'mass_scope'):
                    assert got[key] == expected[key], ('node-local pruning differs', key)
                self.close('discarded_generated_prefix_mass', got['generated_prefix_discarded_mass'], expected['generated_prefix_discarded_mass'])
            retained = self.root.lse(tree.weights[b.handle] for b in active)
            upper = retained if complete else tree.upper(0, n)
            eta = 0. if complete else max(0., min(1., -math.expm1(min(0., retained-upper))))
            self.close('log_retained', summary['log_retained'], retained)
            self.close('raw_row_relaxation_upper', summary['log_partition_upper'], upper)
            self.close('omitted_mass_upper', summary['eta_upper'], eta)
            # Selection optimality and prediction numerics require separate
            # gates, but an archived/nonretained output is not this baseline.
            assert summary['output_handle'] in {b.handle for b in active}
            assert summary['output_sha256'] == tree.p[summary['output_handle']][7]
            self.beams[component] = active; self.complete[component] = complete
            self.totals['node_pruning_steps'] += len(pruning)
            self.totals['merged_components'] += component in products
        for field in ('beam_expansions', 'candidate_evaluations', 'merge_score_evaluations', 'merge_prefix_steps'):
            assert type(audit[field]) is int and audit[field] == self.work[field], ('work counter differs', field)
            self.totals[field] += self.work[field]
        expected_caps = {key: self.meta['config']['max_'+key] for key in
                         ('beam_expansions', 'candidate_evaluations', 'merge_prefix_steps')}
        assert audit['work_caps'] == expected_caps and all(self.work[k] <= v for k, v in expected_caps.items())
        assert audit['merge_state_cache_queries'] == self.work['merge_prefix_steps']
        assert type(audit['merge_state_cache_copies']) is int and 0 <= audit['merge_state_cache_copies'] <= audit['merge_state_cache_queries']
        assert audit['total_prefix_nodes'] == sum(map(len, self.created.values()))
        assert audit['member_rows'] == sum(map(len, self.members.values()))
        upper_sum = math.fsum(s['log_partition_upper'] for s in summaries.values())
        ratio = math.fsum(s['log_retained']-s['log_partition_upper'] for s in summaries.values())
        self.close('global_log_partition_upper', audit['log_partition_upper'], upper_sum)
        self.close('global_product_omitted_mass_upper', audit['product_omitted_mass_upper'],
                   max(0., min(1., -math.expm1(min(0., ratio)))))
        self.close('weighted_truncation_risk_upper', audit['weighted_truncation_risk_upper'],
                   math.fsum(s['weight']*s['eta_upper'] for s in summaries.values()))
        self.totals['events'] += 1
        return count

    def finish(self, count):
        assert count == len(self.slots) and self.totals['events'] == self.meta['state']['events']
        assert self.totals['node_pruning_steps'] == count
        stored = {c: (bool(live), tuple(json.loads(prior)), created) for c, live, prior, created in
                  self.db.execute('SELECT component,live,predecessors,created_us FROM component_catalog')}
        assert set(stored) == set(self.catalog)
        for c, (prior, created) in self.catalog.items():
            assert stored[c] == (c in self.live, prior, created)
            assert self.tree(c).full_members == self.members[c]
            assert set(self.tree(c).p) == self.created[c], 'unexplained archived or materialized prefixes'
            for handle, revision, value in self.db.execute(f'SELECT h,revision,value FROM pc{c}_weights'):
                assert handle in self.created[c] and revision == 0
                self.close('stored_prefix_log_weight', value, self.tree(c).weights[handle])
        assert dict(self.db.execute('SELECT global_i,component FROM component_owners')) == self.owners
        control = dict(self.db.execute('SELECT k,value FROM component_control'))
        assert json.loads(control['partition_n']) == count
        return dict(kind='rbf_independent_fixed_topK_raw_graph_search_pruning_audit_v1',
            sequence_id=self.meta['sequence_id'], observations=count, beam_width=self.width,
            **dict(self.totals), max_abs_error=dict(self.errors), atol=ATOL, rtol=RTOL,
            raw_edge_components_and_all_member_remaps_verified=True,
            retained_predecessor_products_independently_enumerated=True,
            all_node_local_legal_root_class_topK_choices_verified=True,
            equivalent_parent_path_mass_and_irreversible_pruning_verified=True,
            all_pruning_scope_and_support_completeness_flags_verified=True,
            archived_branches_never_resurrected=True, all_materialized_prefixes_accounted_for=True,
            search_work_counters_and_fixed_caps_verified=True,
            component_loss_scope_weights_and_stored_prefix_weights_verified=True,
            conditional_output_selector_independently_accepted=False,
            continuous_states_independently_accepted=False, neural_factors_independently_accepted=False,
            full_online_method_accepted=False, same_resource_performance_accepted=False,
            true_posterior_certificate=False, paper_performance_complete=False)


def verify_database(path, expected_sha256, progress=lambda value: None):
    root = load_root_oracle(); path = Path(path)
    assert root.sha(path) == expected_sha256
    assert not path.is_symlink()
    started = time.monotonic()
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        audit = DatabaseAudit(db, root, meta); count = 0; events = meta['state']['events']
        for ordinal, pb, ab in db.execute('SELECT ordinal,prediction,audit FROM events ORDER BY ordinal'):
            assert ordinal == audit.totals['events']
            count = audit.verify_event(ordinal, json.loads(pb), json.loads(ab), count)
            progress(dict(stage='independent_fixed_topK_raw_graph_search_and_pruning',
                          completed_events=ordinal+1, total_events=events,
                          completed_observations=count, total_observations=len(audit.slots),
                          ETA_seconds=None, ETA_reason='heterogeneous retained-prefix and component-merge work'))
        result = audit.finish(count)
        assert root.sha(path) == expected_sha256
        result.update(database_sha256=expected_sha256, root_oracle_sha256=ROOT_SHA,
                      elapsed_seconds=time.monotonic()-started)
        return result
    finally:
        db.close()
