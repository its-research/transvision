import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.relaxed_identity_risk import independent_parent_messages, action_risk_certificate
from tools.event_track_v2x.diagnose_history_truncation import star_model


def unconstrained(factors):
    histories = []
    for choices in itertools.product(*factors.rows):
        roots = []
        parents = tuple(parent for parent, _ in choices)
        for i, parent in enumerate(parents):
            roots.append(i if parent < 0 else roots[parent])
        histories.append((parents, tuple(roots), math.exp(math.fsum(value for _, value in choices))))
    z = math.fsum(record[2] for record in histories)
    return [(parents, roots, weight/z) for parents, roots, weight in histories]


@pytest.mark.parametrize('seed', range(12))
def test_root_and_equality_messages_and_action_risk_against_full_parent_enumeration(seed):
    rng = np.random.default_rng(seed)
    nodes = tuple(IdentityNode(str(i), i % 2, i, i, str(i // 3)) for i in range(6))
    rows = tuple(tuple((parent, float(rng.normal())) for parent in range(-1, i)) for i in range(6))
    factors = ForestFactors(nodes, rows)
    q0 = unconstrained(factors)
    roots, equality = independent_parent_messages(factors)
    actual_roots, actual_equal = np.zeros((6, 6)), np.zeros((6, 6))
    legal, excluded = [], 0.
    for parents, labels, weight in q0:
        for i, label in enumerate(labels):
            actual_roots[i, label] += weight
            for j in range(6):
                actual_equal[i, j] += weight*(labels[i] == labels[j])
        try:
            factors.roots(parents)
        except ValueError:
            excluded += weight
        else:
            legal.append((parents, labels, weight))
    np.testing.assert_allclose(roots, actual_roots, atol=1e-12)
    np.testing.assert_allclose(equality, actual_equal, atol=1e-12)
    z = math.fsum(weight for _, _, weight in legal)
    for scope in ((5,), (0, 2, 4, 5)):
        def risk(labels):
            return math.fsum(weight*sum(labels[i] != other[i] for i in scope)/len(scope)
                             for _, other, weight in legal)/z
        optimum = min(risk(labels) for _, labels, _ in legal)
        for parents, labels, _ in legal[::max(1, len(legal)//8)]:
            report = action_risk_certificate(factors, parents, scope=scope)
            assert excluded <= report['excluded_mass_upper']+1e-12
            assert risk(labels)-optimum <= report['model_regret_upper']+1e-12
            assert not report['formal_numeric_certificate'] and report['model_only']


def test_current_action_can_be_certified_without_retaining_exponential_past_histories():
    raw, rows = star_model(24)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    action = (-1,)*25+(0,)
    result = action_risk_certificate(factors, action, scope=(25,))
    assert result['model_regret_upper'] < 1e-8
    assert result['collision_pairs'] == 0 and result['independent_optimization_gap'] == 0
    assert result['selected_root_probabilities'][0] == pytest.approx(1/1.001)
    bad = action_risk_certificate(factors, (-1,)*26, scope=(25,))
    assert bad['model_regret_upper'] > .99


def test_unknown_or_illegal_actions_and_resource_exhaustion_are_not_certified():
    nodes = tuple(IdentityNode(str(i), 0, i, i, 'same') for i in range(3))
    factors = ForestFactors(nodes, (((-1, 0.),), ((-1, -1.), (0, 1.)), ((-1, -1.), (0, 1.), (1, 1.))))
    with pytest.raises(ValueError, match='collision'):
        action_risk_certificate(factors, (-1, 0, -1))
    with pytest.raises(ValueError, match='capacity exceeded'):
        action_risk_certificate(factors, (-1, -1, -1), maximum_nodes=2)
    with pytest.raises(ValueError, match='complete legal'):
        action_risk_certificate(factors, (-1,))
    with pytest.raises(ValueError, match='unique in-range'):
        action_risk_certificate(factors, (-1, -1, -1), scope=(1, 1))
    report = action_risk_certificate(factors, (-1, -1, -1))
    assert report['excluded_mass_upper'] == report['model_regret_upper'] == 1.
    assert action_risk_certificate(factors, (-1, -1, -1), scope=())['model_regret_upper'] == 0
