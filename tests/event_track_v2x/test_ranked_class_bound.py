"""Independent parent-path enumeration for a diagnostic class-ranking bound."""
from collections import defaultdict
import itertools
import math

import numpy as np
import pytest

from tools.event_track_v2x.ranked_class_bound import SingleClassBoundModel


def oracle(rows,slots=None):
    classes=defaultdict(float)
    for edges in itertools.product(*rows):
        roots=[];occupied=set();legal=True
        for index,(parent,_) in enumerate(edges):
            root=index if parent<0 else roots[parent]
            roots.append(root)
            if slots is not None:
                key=root,slots[index]
                if key in occupied: legal=False;break
                occupied.add(key)
        if legal: classes[tuple(roots)]+=math.exp(math.fsum(w for _,w in edges))
    return dict(classes)


@pytest.mark.parametrize('seed',range(6))
@pytest.mark.parametrize('with_exclusion',[False,True])
def test_every_legal_prefix_bounds_exhaustive_full_identity_classes(seed,with_exclusion):
    rng=np.random.default_rng(seed)
    rows=tuple(tuple((p,float(rng.uniform(-2,2))) for p in range(-1,index)) for index in range(6))
    slots=tuple((index%2,str(index//3)) for index in range(6)) if with_exclusion else None
    model=SingleClassBoundModel(rows,slots=slots);complete=oracle(rows,slots)
    for depth in range(7):
        prefixes=oracle(rows[:depth],None if slots is None else slots[:depth])
        for prefix,weight in prefixes.items():
            result=model.for_prefix(prefix)
            candidates=[w for roots,w in complete.items() if roots[:depth]==prefix]
            assert result.log_prefix_class_weight==pytest.approx(math.log(weight))
            assert math.log(max(candidates))<=result.log_single_class_upper+1e-11
            assert result.log_single_class_upper<=result.log_row_partition_upper+1e-11
            assert result.inspected_prefix_factor_terms+result.inspected_suffix_factor_terms==model.catalog_terms
            assert not result.partition_mass_bound and not result.formal_numeric_certificate
            if depth:
                assert result.log_single_class_upper<=model.for_prefix(prefix[:-1]).log_single_class_upper+1e-10


def test_parent_aliases_must_sum_not_maximize_single_edges():
    rows=(((-1,0.),),((-1,0.),(0,math.log(.9))),
          ((-1,math.log(.01)),(0,math.log(.6)),(1,math.log(.7))))
    model=SingleClassBoundModel(rows);result=model.for_prefix((0,0))
    exact=oracle(rows)[(0,0,0)]
    assert exact==pytest.approx(.9*(.6+.7))
    assert math.exp(result.log_single_class_upper)==pytest.approx(exact)
    assert .9*max(.6,.7)<exact  # This tempting edge-wise bound would be invalid.


def test_single_class_upper_cannot_replace_a_partition_bound():
    model=SingleClassBoundModel((((-1,0.),),((-1,0.),(0,0.))))
    result=model.for_prefix((0,));weights=oracle(model.rows)
    assert math.exp(result.log_single_class_upper)==pytest.approx(max(weights.values()))
    assert math.exp(result.log_single_class_upper)<sum(weights.values())
    assert not result.partition_mass_bound


def test_unmaterialized_predecessor_groups_bound_every_prior_identity_class():
    rows=(((-1,0.),),((-1,-1.),(0,.2)),((-1,.1),),((-1,-.4),(2,.3)),
          ((-1,-1.),(0,.2),(1,.4),(2,.5)),((-1,-2.),(1,.2),(3,-.1),(4,.3)))
    model=SingleClassBoundModel(rows);groups=(0,0,1,1)
    result=model.for_predecessor_partition(groups);old=oracle(rows[:4]);full=oracle(rows)
    assert result.log_prefix_class_weight is None and result.log_single_class_upper is None
    for prefix,weight in old.items():
        maximum=max(w for roots,w in full.items() if roots[:4]==prefix)
        assert math.log(maximum/weight)<=result.log_suffix_single_class_factor_upper+1e-11
        assert model.for_prefix(prefix).log_suffix_single_class_factor_upper<=result.log_suffix_single_class_factor_upper+1e-11
    with pytest.raises(ValueError,match='support edge'):
        model.for_predecessor_partition((0,1,2,3))


@pytest.mark.parametrize('error',['future','duplicate','missing_birth','nonfinite','boolean','work_limit','nodes'])
def test_invalid_or_unbounded_catalog_rejected(error):
    rows=[((-1,0.),),((-1,0.),(0,0.))];kwargs={}
    if error=='future': rows[1]=((-1,0.),(1,0.))
    elif error=='duplicate': rows[1]=((-1,0.),(0,0.),(0,0.))
    elif error=='missing_birth': rows[1]=((0,0.),)
    elif error=='nonfinite': rows[1]=((-1,float('nan')),)
    elif error=='boolean': rows[1]=((-1,False),)
    elif error=='work_limit': kwargs['max_factor_terms']=2
    else: kwargs['max_nodes']=1
    with pytest.raises(ValueError): SingleClassBoundModel(rows,**kwargs)


def test_invalid_prefix_or_duplicate_source_frame_identity_rejected():
    model=SingleClassBoundModel((((-1,0.),),((-1,0.),(0,0.))),slots=((0,'f'),(0,'f')))
    with pytest.raises(ValueError,match='source/frame'): model.for_prefix((0,0))
    with pytest.raises(ValueError,match='canonical'): model.for_prefix((0,2))
    with pytest.raises(ValueError,match='canonical'): model.for_prefix((True,))
    isolated=SingleClassBoundModel((((-1,0.),),((-1,0.),)))
    with pytest.raises(ValueError,match='supported parent'): isolated.for_prefix((0,0))


def test_log_domain_extremes_remain_finite_without_exponentiating_weights():
    rows=(((-1,1000.),),((-1,-1000.),(0,1000.)),((-1,-1000.),(0,1000.),(1,1000.)))
    result=SingleClassBoundModel(rows).for_prefix((0,0))
    assert math.isfinite(result.log_single_class_upper)
    assert result.log_single_class_upper==pytest.approx(3000.+math.log(2))
