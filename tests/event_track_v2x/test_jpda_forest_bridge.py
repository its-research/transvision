"""Bridge preserves summed raw parent weights, conditional on past identities."""
import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import assignment_log_weight
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.jpda_forest_bridge import condition_forest_scan
from transvision.models.event_track_v2x.jpda_marginals import exact_jpda


def fixture():
    nodes = tuple(IdentityNode(str(i), source, stamp, stamp, frame) for i, (source, stamp, frame) in enumerate([
        (0, 1, 'old'), (0, 1, 'old'), (1, 2, 'other'), (0, 3, 'new'), (0, 3, 'new')]))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, 0.), (0, math.log(2.))),
            ((-1, math.log(3.)), (0, math.log(2.)), (1, math.log(4.)), (2, math.log(5.))),
            ((-1, 0.), (0, math.log(3.)), (1, math.log(2.)), (2, math.log(6.)), (3, math.log(100.))))
    return ForestFactors(nodes, rows)


def test_equivalent_parent_weights_are_summed_not_maximised():
    forest = fixture()
    bridge = condition_forest_scan(forest, (-1, -1, 0), decision_us=3)
    assert bridge.track_roots == (0, 1)
    np.testing.assert_allclose(np.exp(bridge.factors.log_pair), [[7., 9.], [4., 2.]])
    assert bridge.factors.log_left_unmatched == (0., 0.)
    np.testing.assert_allclose(np.exp(bridge.factors.log_right_unmatched), [3., 1.])
    assert bridge.forest_sha256 == forest.digest()


@pytest.mark.parametrize('prefix', [(-1, -1, 0), (-1, -1, -1)])
def test_matching_weights_equal_aggregate_legal_forest_extensions(prefix):
    forest = fixture()
    bridge = condition_forest_scan(forest, prefix, decision_us=3)
    accumulated = {}
    for choices in itertools.product(*(tuple(p for p, _ in row) for row in forest.rows[len(prefix):])):
        parents = prefix + choices
        try:
            roots = forest.roots(parents)
        except ValueError:
            continue
        assignment = tuple(next((j for j, child in enumerate(bridge.detection_indices) if roots[child] == root), -1)
                           for root in bridge.track_roots)
        weight = math.exp(forest.log_weight(parents) - forest.log_weight(prefix))
        accumulated[assignment] = accumulated.get(assignment, 0.) + weight
    for assignment, weight in accumulated.items():
        assert math.exp(assignment_log_weight(bridge.factors, assignment)) == pytest.approx(weight)
    inferred = exact_jpda(bridge.factors)
    assert math.exp(inferred.log_partition) == pytest.approx(sum(accumulated.values()))


def test_already_occupied_root_cannot_take_second_observation_from_scan():
    forest = fixture()
    bridge = condition_forest_scan(forest, (-1, -1, 0, 0), decision_us=3)
    assert not bridge.factors.allowed[0][0]
    assert bridge.factors.allowed[1][0]


def test_all_new_same_scan_nodes_become_distinct_births_without_past_tracks():
    nodes = (IdentityNode('a', 0, 1, 1, 'x'), IdentityNode('b', 0, 1, 1, 'x'))
    forest = ForestFactors(nodes, (((-1, 1.),), ((-1, 2.), (0, 100.))))
    bridge = condition_forest_scan(forest, (), decision_us=1)
    assert bridge.factors.shape == (0, 2)
    assert exact_jpda(bridge.factors).right_unmatched == (1., 1.)


def test_mixed_source_batch_and_future_evidence_are_rejected():
    forest = fixture()
    with pytest.raises(ValueError, match='one real source/frame'):
        condition_forest_scan(forest, (-1, -1), decision_us=3)
    with pytest.raises(ValueError, match='arrived'):
        condition_forest_scan(forest, (-1, -1, 0), decision_us=2)
