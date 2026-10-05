"""Conditional Bayes selector on an independently admitted fixed TopK domain.

The immutable same-byte search proof supplies already verified root classes
and absolute weights. This gate does not repeat search/pruning. It evaluates
pairwise identity Hamming loss with normalized 70-digit Decimal probabilities,
and separately checks the declared float64 marginal reduction and SHA tie rule.
The float64 comparison tolerance remains 1e-8. This is not global full-history
Bayes, a true-posterior certificate, continuous-state or performance admission.
"""
from collections import defaultdict
from decimal import Decimal, localcontext
from functools import reduce
import hashlib
import json
import math
import operator
from pathlib import Path
import sqlite3
import time

ATOL = RTOL = 1e-8
DECIMAL_PRECISION = 70
SEARCH_KIND = 'rbf_independent_fixed_Top1_raw_graph_search_pruning_audit_v1'
SCHEMA = 'persistent_irreversible_identity_beam_v1'
POLICY = 'conditional_bayes_over_retained_classes_no_regret_fallback'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def validate_search_sequence(proof, expected_sha, sequence, events, nodes):
    assert proof['kind'] == SEARCH_KIND
    assert proof['database_sha256'] == expected_sha
    assert proof['sequence_id'] == sequence and proof['events'] == events and proof['observations'] == nodes
    assert proof['beam_width'] == 1 and proof['atol'] == proof['rtol'] == ATOL
    for key in ('raw_edge_components_and_all_member_remaps_verified',
                'retained_predecessor_products_independently_enumerated',
                'all_node_local_legal_root_class_Top1_choices_verified',
                'equivalent_parent_path_mass_and_irreversible_pruning_verified',
                'all_pruning_scope_and_support_completeness_flags_verified',
                'archived_branches_never_resurrected', 'all_materialized_prefixes_accounted_for',
                'search_work_counters_and_fixed_caps_verified',
                'component_loss_scope_weights_and_stored_prefix_weights_verified'):
        assert proof[key] is True, ('missing search prerequisite', key)
    assert proof['conditional_output_selector_independently_accepted'] is False
    assert proof['paper_performance_complete'] is False


def pairwise_risks(roots, weights):
    """Alternative loss formulation, independent of a marginal-map decoder."""
    assert 1 <= len(roots) == len(weights) <= 4
    size = len(roots[0]); assert all(len(r) == size for r in roots)
    assert all(math.isfinite(w) for w in weights)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        top = max(weights)
        exponentials = [(Decimal.from_float(w)-Decimal.from_float(top)).exp() for w in weights]
        partition = sum(exponentials, Decimal(0))
        probabilities = [e/partition for e in exponentials]
        losses = [[sum(a != b for a, b in zip(action, branch)) for branch in roots] for action in roots]
        risks = [sum((p*loss for p, loss in zip(probabilities, row)), Decimal(0))/size
                 if size else Decimal(0) for row in losses]
        return tuple(float(v) for v in risks), tuple(float(p) for p in probabilities)


def normative_float64(roots, weights, retained, shas, *, complete, eta):
    """Frozen arithmetic/tie contract, evaluated from admitted loss inputs.

    Class indices are grouped per query, then reduced in the admitted active
    order. This fixes reduction order independently of a producer hash-map.
    A second pairwise Decimal check is required in addition to this contract.
    """
    assert len(roots) == len(weights) == len(shas) and 1 <= len(roots) <= 4
    size = len(roots[0]); assert all(len(r) == size for r in roots)
    assert math.isfinite(retained) and math.isfinite(eta) and 0 <= eta <= 1
    probabilities = [math.exp(w-retained) for w in weights]
    if not size:
        return 0, dict(risk_bound=0., conditional_risk=0., relaxed_bayes_lower=0.,
            optimization_gap=0., empty_loss_scope=True,
            conditional_bayes_lower_kind='complete_supported_action_enumeration' if complete else 'independent_root_relaxation'), (0.,)*len(roots)
    grouped = []
    for i in range(size):
        groups = defaultdict(list)
        for index, branch in enumerate(roots): groups[branch[i]].append(index)
        grouped.append({identity: reduce(operator.add, (probabilities[j] for j in members), 0.)
                        for identity, members in groups.items()})
    risks = tuple(math.fsum(1.-marginal[identity] for marginal, identity in zip(grouped, branch))/size
                  for branch in roots)
    chosen = min(range(len(roots)), key=lambda i: (risks[i], shas[i]))
    relaxation = max(0., math.fsum(1.-max(marginal.values()) for marginal in grouped)/size)
    lower = risks[chosen] if complete else relaxation
    gap = max(0., risks[chosen]-lower)
    return chosen, dict(risk_bound=min(1., gap+(0. if complete else eta)),
        conditional_risk=risks[chosen], relaxed_bayes_lower=lower, optimization_gap=gap,
        empty_loss_scope=False,
        conditional_bayes_lower_kind='complete_supported_action_enumeration' if complete else 'independent_root_relaxation'), risks


class AdmittedPrefixIndex:
    """Binary ancestor lookup from prefix structure already checked by search."""
    def __init__(self, db, component):
        assert type(component) is int and component > 0
        self.rows = {h: (parent, depth, root, tuple(json.loads(jumps)), digest)
            for h, parent, depth, root, jumps, digest in db.execute(
                f'SELECT h,parent,depth,root,jumps,sha FROM pc{component}_prefixes')}

    def at(self, handle, depth):
        assert 0 < depth <= self.rows[handle][1]
        distance = self.rows[handle][1]-depth
        bit = 0
        while distance:
            if distance & 1: handle = self.rows[handle][3][bit]
            distance >>= 1; bit += 1
        assert self.rows[handle][1] == depth
        return self.rows[handle][2]


def verify_database(path, expected_sha, search_proof, progress=lambda value: None):
    path = Path(path); assert not path.is_symlink() and sha(path) == expected_sha
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    started = time.monotonic(); errors = defaultdict(float); counters = defaultdict(int)
    def close(label, got, expected):
        assert type(got) in (float, int) and math.isfinite(got) and math.isfinite(expected)
        error = abs(got-expected)
        assert error <= ATOL+RTOL*abs(expected), (label, got, expected)
        errors[label] = max(errors[label], error)
    try:
        meta = {key: json.loads(value) for key, value in db.execute('SELECT k,v FROM meta')}
        nodes = db.execute('SELECT count(*) FROM observations').fetchone()[0]
        validate_search_sequence(search_proof, expected_sha, meta['sequence_id'], meta['state']['events'], nodes)
        assert meta['schema'] == SCHEMA and meta['config']['state']['decision_mode'] == 'retained'
        assert meta['config']['state']['active_limit'] == 1
        prefixes = {}; members = {}
        for ordinal, blob in db.execute('SELECT ordinal,audit FROM events ORDER BY ordinal'):
            assert ordinal == counters['events']
            value = json.loads(blob)
            assert value['kind'] == SCHEMA and value['output_policy'] == POLICY
            assert value['global_fallback_used'] is False
            component_bounds = []
            for summary in value['components']:
                c = summary['component']
                if c not in prefixes:
                    prefixes[c] = AdmittedPrefixIndex(db, c)
                    members[c] = tuple(g for g, in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (c,)))
                index = prefixes[c]; indices = summary['decision_indices']; n = summary['nodes']
                actual_members = members[c][:n]
                inverse = {g: i for i, g in enumerate(actual_members)}
                scope = [inverse[g] for g in value['decision_indices'] if g in inverse]
                assert indices == scope and summary['indices_count'] == len(indices)
                active = summary['active']; handles = [a['handle'] for a in active]
                assert len(set(handles)) == len(handles) and len(handles) == 1
                assert all(index.rows[h][1] == n for h in handles)
                shas = [index.rows[h][4] for h in handles]
                assert shas == [a['sha256'] for a in active]
                weights = [a['log_weight'] for a in active]
                assert handles == [h for _, _, h in sorted((-w, s, h) for h, w, s in zip(handles, weights, shas))]
                roots = [tuple(index.at(h, i+1) for i in indices) for h in handles]
                chosen, expected, risks = normative_float64(roots, weights, summary['log_retained'], shas,
                    complete=summary['full_model_support_still_in_beam'], eta=summary['eta_upper'])
                decimal_risks, _ = pairwise_risks(roots, weights)
                for a, b in zip(risks, decimal_risks): close('pairwise_70_digit_conditional_risk', a, b)
                close('selected_vs_pairwise_minimum', decimal_risks[chosen], min(decimal_risks))
                assert summary['output_handle'] == handles[chosen], 'output differs from frozen conditional loss/SHA tie rule'
                assert summary['output_sha256'] == shas[chosen]
                decision = summary['decision']
                assert set(decision) == set(expected), 'wrong conditional-decision contract'
                assert type(decision['empty_loss_scope']) is bool
                for key in ('empty_loss_scope', 'conditional_bayes_lower_kind'): assert decision[key] == expected[key]
                for key in ('conditional_risk', 'relaxed_bayes_lower', 'optimization_gap', 'risk_bound'):
                    close(key, decision[key], expected[key])
                if indices:
                    counters['nonempty_component_decisions'] += 1
                    counters['posterior_MAP_and_output_differ'] += chosen != 0
                    counters['scoped_identity_queries'] += len(indices)
                else: counters['empty_component_decisions'] += 1
                counters['component_decisions'] += 1
                component_bounds.append(summary['weight']*expected['risk_bound'])
            close('global_weighted_model_regret', value['model_regret_upper'], math.fsum(component_bounds))
            counters['events'] += 1
            progress(dict(stage='independent_fixed_Top1_conditional_output_selector',
                          completed_events=counters['events'], total_events=meta['state']['events'],
                          ETA_seconds=None, ETA_reason='unequal component scopes and prefix-ancestor work'))
        assert counters['events'] == meta['state']['events'] and sha(path) == expected_sha
        return dict(kind='rbf_independent_fixed_Top1_conditional_selector_audit_v1',
            sequence_id=meta['sequence_id'], database_sha256=expected_sha, observations=nodes,
            **dict(counters), max_abs_error=dict(errors), atol=ATOL, rtol=RTOL, decimal_precision=DECIMAL_PRECISION,
            upstream_same_byte_search_proof_required=True, search_pruning_rerun=False,
            pairwise_conditional_risks_independently_verified=True,
            all_outputs_match_frozen_float64_minimum_and_SHA_tie_rule=True,
            complete_and_truncated_conditional_lower_semantics_verified=True,
            no_MAP_substitution_or_regret_fallback=True,
            conditional_output_selector_independently_accepted=True,
            continuous_states_independently_accepted=False, full_online_method_accepted=False,
            global_full_history_Bayes_accepted=False, true_posterior_certificate=False,
            same_resource_performance_accepted=False, paper_performance_complete=False,
            elapsed_seconds=time.monotonic()-started)
    finally: db.close()
