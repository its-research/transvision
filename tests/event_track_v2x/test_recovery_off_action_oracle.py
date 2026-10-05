"""Exact tiny domains expose pruning, bounded search and risk-label mistakes."""
import itertools
import math
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
from recovery_off_action_oracle import replay_search, legacy


def independent_roots(action, support, slots):
    roots=[]; used=set()
    for i,parent in enumerate(action):
        if parent not in support[i]: return None
        root=i if parent==-1 else roots[parent]
        key=(root,*slots[i])
        if slots[i][0]!=-1 and key in used:return None
        used.add(key);roots.append(root)
    return tuple(roots)


def allows(variants):
    return lambda roots:any(roots[:min(len(roots),len(v))]==v[:min(len(roots),len(v))] for v in variants)


@pytest.mark.parametrize('seed',[1337,2027,3407])
@pytest.mark.parametrize('budget,limit',[(0,4096),(1,4096),(4096,1),(4096,4096)])
def test_bounded_result_against_exhaustive_raw_parent_domain(seed,budget,limit):
    support=tuple(tuple(range(-1,i)) for i in range(5))
    slots=((0,'a'),(1,'b'),(0,'c'),(1,'d'),(0,'a'))
    allowed=allows(((0,0,2),(0,1,1)))
    legal={}
    for action in itertools.product(*support):
        roots=independent_roots(action,support,slots)
        if roots is not None and allowed(roots):legal.setdefault(roots,action)
    rng=random.Random(seed)
    root_paths=rng.sample(list(legal),min(4,len(legal)))
    active=tuple(legal[r] for r in root_paths)
    logs=tuple(rng.uniform(-2,2) for _ in active)
    masses=[math.exp(x-max(logs)) for x in logs];total=sum(masses)
    indices=(1,2,3,4)
    def risk(roots):
        return sum(m*sum(roots[i]!=r[i] for i in indices)/len(indices)
                   for r,m in zip(root_paths,masses))/total
    optimum=min(map(risk,legal))
    result=replay_search(support,slots,active,logs,indices,budget,limit,allowed)
    assert result['action'] in list(itertools.product(*support))
    roots=independent_roots(result['action'],support,slots)
    assert roots in legal and allowed(roots)
    assert result['conditional_risk']==pytest.approx(risk(roots),abs=1e-12)
    assert result['relaxed_bayes_lower'] <= optimum + 1e-12
    assert optimum <= result['conditional_risk'] + 1e-12
    assert result['action_search_steps']<=budget and result['maximum_queue']<=limit
    if result['action_search_complete']:
        assert result['conditional_risk']==pytest.approx(optimum,abs=1e-12)


def test_unrestricted_oracle_would_admit_a_better_but_discarded_history():
    support=tuple(tuple(range(-1,i)) for i in range(4));slots=tuple((i%2,str(i)) for i in range(4))
    active=((-1,-1,0,0),(-1,0,-1,0),(-1,0,0,-1))
    variants=tuple(independent_roots(p,support,slots) for p in active)
    allowed=allows(variants)
    unrestricted=legacy.replay_search(support,slots,active,(0.,0.,0.),(1,2,3),4096,4096)
    restricted=replay_search(support,slots,active,(0.,0.,0.),(1,2,3),4096,4096,allowed)
    assert unrestricted['roots']==(0,0,0,0) and not allowed(unrestricted['roots'])
    assert allowed(restricted['roots']) and restricted['action_search_complete']
    assert unrestricted['conditional_risk']<restricted['conditional_risk']


def test_invalid_active_history_cannot_seed_restricted_search():
    with pytest.raises(AssertionError):
        replay_search(((-1,),(-1,0)),((0,'a'),(1,'b')),((-1,0),),(0.,),(0,1),10,10,
                      allows(((0,1),)))


@pytest.mark.parametrize('budget,width,max_nodes',[(0,1,4096),(1,1,4096),(100,1,4096),(100,2,4096),(100,2,2)])
def test_frozen_database_action_scope(tmp_path,budget,width,max_nodes):
    from test_recovery_off_structure_oracle import fixture
    from recovery_off_action_oracle import verify_database
    path=tmp_path/'frozen-action.sqlite'
    expected=fixture(path,budget=budget,width=width,max_nodes=max_nodes)
    result=verify_database(path,expected)
    assert result['events']==4
    assert result['restricted_action_search_or_capacity_fallback_verified'] is True
    assert result['all_selected_actions_raw_support_and_historical_restriction_legal'] is True
    assert not result['unrestricted_model_regret_accepted']
