import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parents[2] / 'tools/event_track_v2x'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


audit = load('audit_stage1_identity_episodes')
geometry = load('stage1_independent_evaluation')
oracle = load('stage1_identity_episode_oracle')


def box(tid, x, y=0., yaw=0., length=4., width=2.):
    return dict(track_id=tid, class_label='car', mean=[x, y, 0., length, width, 1.5, yaw, 0., 0.])


@pytest.mark.parametrize('offset', [0., 700000.])
def test_conservative_AABB_pruning_preserves_direct_rotated_clip_kernel(offset):
    left = [box('a', offset), box('b', offset + 3.2, .4, math.pi / 3),
            box('c', offset - .2, .1, .7, .03, .02), box('d', offset + 200.)]
    right = [box('x', offset, yaw=math.pi / 2), box('y', offset + 3., .7, -.8),
             box('z', offset - .2, .1, .7, .03, .02), box('q', offset + 4.)]
    fast = audit.similarities(geometry, left, right)
    direct = np.array([[geometry.bev_iou(a['mean'], b['mean']) for b in right] for a in left])
    np.testing.assert_array_equal(fast, direct)
    assert audit.similarities(geometry, [], right).shape == (0, 4)
    assert audit.similarities(geometry, left, []).shape == (4, 0)


def record(side, index, obj):
    return dict(side=side, selected_index=index, class_index=0, mean=obj['mean'])


def node(side, index, obj, kind='vehicle_anchored_mixture', mass=1., components=None):
    return dict(node_index=index, kind=kind, source_side=side, source_raw_index=index,
                source_selected_index=index, source_frame_id='native', class_index=0,
                score=.8, birth_effective_score=.7, mean=obj['mean'], covariance_sha256='c' * 64,
                matched_mass=mass, unmatched_mass=1. - mass, components=components or [])


def component(index, weight, obj):
    return dict(partner_selected_index=index, partner_raw_index=index, hypothesis_index=index,
                weight=weight, kind='actual_CI', mean=obj['mean'], covariance_sha256='d' * 64)


def test_actual_pair_weights_keep_wrong_same_and_unknown_source_mass_separate():
    a, b, distant = box('A', 0.), box('B', 20.), box('none', 200.)
    records = [record('vehicle-side', 0, a), record('infrastructure-side', 0, a),
               record('infrastructure-side', 1, b), record('infrastructure-side', 2, distant)]
    n = node('vehicle-side', 0, a, components=[component(0, .2, a), component(1, .3, b), component(2, .5, distant)])
    result = audit.source_and_nodes(geometry, oracle, dict(source_records=records, nodes=[n]), [a, b])[0]
    assert result['qualified_same_GT_pair_mass'] == .2
    assert result['qualified_different_GT_pair_mass'] == .3
    assert result['unknown_pair_mass'] == .5
    assert result['active_component_pair_mass'] == 1.
    assert result['birth_effective_score'] == .7
    assert result['nonexecuted_pair_mass_for_residual'] == 0.


def test_road_residual_matched_mass_is_not_executed_pair_or_error_mass():
    obj = box('A', 0.)
    n = node('infrastructure-side', 0, obj, kind='road_residual', mass=.8)
    result = audit.source_and_nodes(geometry, oracle,
        dict(source_records=[record('infrastructure-side', 0, obj)], nodes=[n]), [obj])[0]
    assert result['active_component_pair_mass'] == 0.
    assert result['qualified_different_GT_pair_mass'] == 0.
    assert result['nonexecuted_pair_mass_for_residual'] == .8
    assert result['unmatched_mass'] == pytest.approx(.2)


def test_duplicate_source_correspondences_remain_unknown_without_forcing_anchors():
    obj = box('A', 0.)
    n = node('vehicle-side', 0, obj, components=[component(0, 1., obj)])
    result = audit.source_and_nodes(geometry, oracle, dict(source_records=[
        record('vehicle-side', 0, obj), record('vehicle-side', 1, obj), record('infrastructure-side', 0, obj)], nodes=[n]), [obj])[0]
    assert result['anchor_correspondence']['status'] == 'ambiguous_geometric_correspondence'
    assert result['unknown_pair_mass'] == 1.
    assert result['qualified_same_GT_pair_mass'] == result['qualified_different_GT_pair_mass'] == 0.


def test_incomplete_pair_mass_is_rejected_instead_of_renormalized():
    obj = box('A', 0.)
    n = node('vehicle-side', 0, obj, components=[component(0, .8, obj)])
    with pytest.raises(ValueError, match='cover actual matched mass'):
        audit.source_and_nodes(geometry, oracle, dict(source_records=[record('vehicle-side', 0, obj),
            record('infrastructure-side', 0, obj)], nodes=[n]), [obj])


def test_full_identity_counts_are_required_before_diagnostic_owners_are_used():
    obj = box('A', 0.); pred = box('p', 0.)
    gt = [dict(sid='seq', fid='f', boxes=[obj])]; predictions = [dict(boxes=[pred])]
    expected = dict(seq=dict(Identity=dict(IDTP=1, IDFN=0, IDFP=0)))
    matches, owners, checks = audit.full_id_correspondences(geometry, oracle, gt, predictions, expected)
    assert owners['seq']['p']['owner'] == 'A' and checks[0]['original_Identity_counts_exact']
    assert matches['seq', 'f']['A']['other'] == 'p'
    expected['seq']['Identity']['IDTP'] = 2
    with pytest.raises(ValueError, match='objective counts differ'):
        audit.full_id_correspondences(geometry, oracle, gt, predictions, expected)
