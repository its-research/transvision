import importlib.util
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[2]/'tools/event_track_v2x';sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('stage1_discrete_replay_oracle',ROOT/'stage1_discrete_replay_oracle.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)


def node(x=0.,score=.9,cls=0):
    m=np.zeros(9);m[0]=x;m[3:6]=[4.,2.,1.]
    return dict(mean=m,covariance=np.eye(9),score=score,birth_score=score,class_index=cls)


def test_global_assignment_corrects_greedy_collision_and_has_dual_witness():
    cols,proof=o.assignment([[1.,2.,12.,np.inf],[1.01,10.,np.inf,12.]])
    assert cols.tolist()==[1,0] and proof['primal']==pytest.approx(3.01)
    assert proof['duality_gap']<1e-12


def test_assignment_against_exhaustive_legal_injections_including_forbidden_edges():
    rng=np.random.default_rng(1337)
    for n in [1,2,3,4]:
        for _ in range(20):
            a=rng.normal(size=(n,n+2));a[rng.random(a.shape)<.25]=np.inf
            for i in range(n):a[i,i]=rng.normal()
            cols,p=o.assignment(a)
            exact=min(math.fsum(a[i,j] for i,j in enumerate(js)) for js in itertools.permutations(range(n+2),n))
            assert p['primal']==pytest.approx(exact,abs=1e-12)
            assert len(set(cols))==n


def test_empty_assignment_and_infeasible_support_are_explicit():
    cols,p=o.assignment(np.empty((0,3)));assert len(cols)==0 and p['primal']==0
    with pytest.raises(ValueError,match='finite full row'):o.assignment([[np.inf]])


def test_full_scratch_replay_creates_updates_misses_and_prunes_state():
    r=o.Replay('s');out,m,e,p=r.step(1000000,[node()]);assert list(out)==['s:000001'] and not m
    out,m,e,p=r.step(1100000,[node(.2)]);assert m==[dict(track_id='s:000001',node_index=0)]
    # Analytic x/v block includes the propagated .1 cross-covariance.
    expected=.2*(1.02*2.01-.1**2)/(2.02*2.01-.1**2)
    assert out['s:000001']['mean'][0]==pytest.approx(expected,abs=1e-14)
    out,m,e,p=r.step(1200000,[]);assert e==[dict(event='miss',track_id='s:000001')]
    out,m,e,p=r.step(3200001,[]);assert not out and e==[dict(event='kill_before_assignment',track_id='s:000001')]


def test_frozen_class_gate_prevents_cross_class_temporal_update():
    r=o.Replay('s');r.step(1000000,[node(cls=0)])
    out,m,e,p=r.step(1100000,[node(cls=1)]);assert not m and len(out)==2
    assert any(x['event']=='birth' and x['track_id']=='s:000002' for x in e)


def test_node_score_mass_and_M4_birth_only_effect_are_distinct():
    v=node(score=.9);road=node(score=.8);left=[(v['mean'],v['covariance'],.9,0)];right=[(road['mean'],road['covariance'],.8,0)]
    h=[dict(pairs=[[0,0]],unmatched_left=[],unmatched_right=[],weight=.8),dict(pairs=[],unmatched_left=[0],unmatched_right=[0],weight=.2)]
    m0=o.nodes(left,right,h,'M0');m3=o.nodes(left,right,h,'M3');m4=o.nodes(left,right,h,'M4')
    assert m0[1]['score']==pytest.approx(.16) and m3[1]['score']==.8
    assert m4[1]['score']==pytest.approx(.16) and m4[1]['birth_score']==.8
    r=o.Replay('s');out,_,_,_=r.step(1000000,m0);assert len(out)==1
    r=o.Replay('s');out,_,_,_=r.step(1000000,m4);assert len(out)==2


def test_bad_sealed_support_and_fabricated_M1_pairs_are_rejected():
    h=dict(pairs=[[0,0]],unmatched_left=[0],unmatched_right=[],weight=1.)
    with pytest.raises(ValueError,match='unmatched complement'):o.hypotheses([h],1,1,'M0')
    h['unmatched_left']=[]
    with pytest.raises(ValueError,match='M1'):o.hypotheses([h],1,1,'M1')


def test_causal_timestamps_cannot_be_rewritten():
    r=o.Replay('s');r.step(1000000,[])
    with pytest.raises(ValueError,match='increasing'):r.step(1000000,[])
