"""Independent bounded action search in the event's irreversible history domain.

The raw-support Hamming objective and capacity checks reuse a hash-pinned
independent oracle. Historical restrictions come from preceding immutable
commits, never production search helpers or the advertised current restriction.
No claim about the unrestricted model's optimum or physical identity is made.
"""
import copy
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import time

CAPACITY = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/'
    'rbf-independent-action-capacity-undecided-v2-20261002/decoder_capacity.py')
CAPACITY_SHA = '2f05714726a48280f64bdb559a330e1fb3582cf33d58d51ffc7a7a2e82c3d6cc'
assert hashlib.sha256(CAPACITY.read_bytes()).hexdigest() == CAPACITY_SHA
spec = importlib.util.spec_from_file_location('recovery_off_original_independent_action_primitives', CAPACITY)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
sha, close = legacy.sha, legacy.close
ATOL = RTOL = 1e-8


def replay_search(support, slots, active, log_weights, indices, budget, frontier_limit, allowed):
    assert type(budget) is int and budget >= 0
    assert type(frontier_limit) is int and frontier_limit >= 1
    labels, _, _, previous, bound = legacy.make_problem(support,slots,active,log_weights,indices)
    assert allowed(()) and all(allowed(roots) for roots in labels)
    selected = min(range(len(active)),key=lambda i:(bound(labels[i]),active[i]))
    chosen, chosen_roots = active[selected], labels[selected]
    risk = bound(chosen_roots)
    queue = [(bound(()),(),())]
    steps, maximum_queue = 0, 1
    while queue and steps < budget:
        minimum,prefix,roots = queue[0]
        if minimum >= risk:
            queue.clear()
            break
        i=len(prefix)
        assert i < len(support)
        forbidden = set() if slots[i][0] == -1 else {roots[j] for j in previous[i]}
        children=[]
        for parent in support[i]:
            root=i if parent == -1 else roots[parent]
            labels=roots+(root,)
            if root not in forbidden and allowed(labels):
                children.append((prefix+(parent,),labels))
        # Empty subtrees can be discarded; this is distinct from dropping a
        # nonempty subtree because the declared frontier capacity was reached.
        if len(queue)-1+len(children) > frontier_limit:
            break
        heapq.heappop(queue)
        steps+=1
        for action,labels in children:
            lower=bound(labels)
            if len(action)==len(support):
                if (lower,action)<(risk,chosen):chosen,chosen_roots,risk=action,labels,lower
            elif lower<risk:heapq.heappush(queue,(lower,action,labels))
        maximum_queue=max(maximum_queue,len(queue))
    lower=min(risk,queue[0][0]) if queue else risk
    return dict(action=chosen,roots=chosen_roots,conditional_risk=risk,
        relaxed_bayes_lower=lower,optimization_gap=max(0.,risk-lower),action_search_steps=steps,
        action_search_complete=not queue,selected_outside_active=chosen not in active,
        maximum_queue=maximum_queue,unresolved_prefixes=len(queue))


def verify_summary(summary, support, slots, prefixes, budget, frontier_limit, max_decision_nodes, allowed):
    n=summary['nodes']; assert n==len(support)==len(slots)
    active=tuple(legacy.path_from_prefixes(prefixes,r['handle']) for r in summary['active'])
    weights=tuple(r['log_weight'] for r in summary['active'])
    actual=legacy.path_from_prefixes(prefixes,summary['output_handle'])
    fallback=legacy.path_from_prefixes(prefixes,summary['fallback_handle'])
    for path in (*active,actual,fallback):
        assert len(path)==n and allowed(legacy.legal_roots(path,support,slots)), 'action recovers discarded history'
    decision=summary['decision']
    assert decision['action_space']=='legal_extensions_of_previous_retained_and_output_classes'
    assert decision['risk_scope']=='conditional_on_irreversibly_pruned_support'
    assert decision['original_model_regret_upper']==1. and decision['original_model_risk_certified'] is False
    assert decision['support_sha256']==hashlib.sha256(legacy.canonical(summary['historical_support'])).hexdigest()
    if not active or n>max_decision_nodes:
        # Only vocabulary is translated for the unchanged capacity/no-active
        # verifier; actual selected support has already been checked above.
        translated=copy.deepcopy(summary)
        translated['decision']['action_space']='all_supported_legal_histories'
        return legacy.verify_summary(translated,support,slots,prefixes,budget,frontier_limit,
                                     summary['eta_upper'],max_decision_nodes)
    expected=replay_search(support,slots,active,weights,tuple(summary['decision_indices']),
                           budget,frontier_limit,allowed)
    assert decision['decision_rule']=='conditional_Hamming' and decision['loss']=='mean_identity_root_Hamming'
    assert decision['numeric_certificate'] is False
    assert actual==expected['action'], 'bounded restricted action differs'
    for key in ('action_search_steps','action_search_complete','selected_outside_active'):
        assert decision[key]==expected[key], key
    maximum=max(close(decision[k],expected[k]) for k in ('conditional_risk','relaxed_bayes_lower','optimization_gap'))
    maximum=max(maximum,close(decision['risk_bound'],min(1.,summary['eta_upper']+expected['optimization_gap'])))
    assert decision['empty_loss_scope']==(not summary['decision_indices'])
    assert decision['action_materialization_steps']==n
    assert summary['output_handle'] in {x['handle'] for x in summary['branches']}
    return dict(action_search_steps=expected['action_search_steps'],outside_active=expected['selected_outside_active'],
        complete=expected['action_search_complete'],max_abs_error=maximum,fallback=False,
        chosen_action_own_state_export=True)


def verify_database(path, expected_sha, progress=lambda value:None):
    from recovery_off_structure_oracle import replay_contexts
    path=Path(path); assert sha(path)==expected_sha
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    start=time.monotonic()
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        meta={k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')}
        assert meta['schema']=='persistent_exclusive_event_boundary_recovery_off_v1'
        assert meta['state']['revision']==0, 'rescored raw history needs a separately versioned input contract'
        state=meta['config']['state']
        assert state['decision_mode']=='all-legal-hamming' and state['max_model_regret']==1.
        events=checks=steps=capacity=complete=outside=0; maximum=0.
        for ordinal,prediction,audit,contexts in replay_contexts(db):
            assert ordinal==events and audit['global_fallback_used'] is False
            assert audit['action_search_budget']==state['paper_action_budget']
            assert [x.summary['component'] for x in contexts]==sorted(x.summary['component'] for x in contexts)
            remaining=state['paper_action_budget']; values=[]
            total_scope=len(audit['decision_indices'])
            for context in contexts:
                summary=context.summary;n=summary['nodes'];tree=context.tree
                support=tuple(tuple(p for p,w in row) for row in tree.factors[:n])
                prefixes={h:(r[1],r[2],r[3]) for h,r in tree.p.items()}
                value=verify_summary(summary,support,tree.slots[:n],prefixes,remaining,
                    state['paper_action_frontier'],meta['config']['max_decision_nodes'],context.allowed)
                remaining-=value['action_search_steps'];assert remaining>=0
                weight=len(summary['decision_indices'])/total_scope if total_scope else 0.
                maximum=max(maximum,close(summary['weight'],weight),value['max_abs_error'])
                values.append(weight*summary['decision']['risk_bound'])
                checks+=1;outside+=value['outside_active'];complete+=value['complete']
                capacity+=value.get('capacity_undecided',False)
            event_steps=state['paper_action_budget']-remaining
            assert audit['action_search_steps']==event_steps
            conditional_bound=math.fsum(values)
            for key in ('model_regret_upper','proposed_model_regret_upper'):
                assert audit[key]==1., 'restricted posterior bound mislabeled as original model guarantee'
                maximum=max(maximum,close(audit['restricted_support_'+key],conditional_bound))
            events+=1;steps+=event_steps
            if events%25==0:progress(dict(stage='independent_recovery_off_action_search',completed_events=events,
                total_events=meta['state']['events'],ETA_seconds=None,ETA_reason='heterogeneous restricted component searches'))
        assert events==meta['state']['events'] and sha(path)==expected_sha
        return dict(kind='rbf_independent_recovery_off_restricted_action_search_v1',database_sha256=expected_sha,
            events=events,component_decisions=checks,shared_action_search_steps=steps,
            restricted_action_search_or_capacity_fallback_verified=True,
            all_selected_actions_raw_support_and_historical_restriction_legal=True,
            capacity_undecided_component_decisions=capacity,capacity_undecided_counted_as_optimal=False,
            complete_component_searches=complete,incomplete_component_searches=checks-complete,
            actions_outside_current_active_classes=outside,max_abs_arithmetic_error=maximum,
            unrestricted_model_regret_accepted=False,fresh_selected_state_numeric_admission=False,
            complete_online_method_accepted=False,paper_performance_complete=False,
            atol=ATOL,rtol=RTOL,elapsed_seconds=time.monotonic()-start)
    finally:db.close()
