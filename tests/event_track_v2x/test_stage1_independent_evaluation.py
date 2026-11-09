import importlib.util
from pathlib import Path
import math
import pytest

spec = importlib.util.spec_from_file_location('independent_evaluation', Path(__file__).parents[2] / 'tools/event_track_v2x/stage1_independent_evaluation.py')
evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)


def box(x=0, y=0, yaw=0, length=4, width=2):
    return [x, y, 0., length, width, 1.5, yaw, 0., 0.]


def test_intersection_has_exact_physical_length_width_rotation():
    assert evaluation.bev_iou(box(), box()) == pytest.approx(1.)
    assert evaluation.bev_iou(box(), box(yaw=math.pi / 2)) == pytest.approx(1 / 3)
    assert evaluation.bev_iou(box(), box(x=2)) == pytest.approx(1 / 3)
    assert evaluation.bev_iou(box(), box(x=4)) == 0.


def test_bev_overlap_ignores_height_and_is_translation_symmetric():
    a, b = box(yaw=.7), box(.7, -.2, -.3)
    c = list(b); c[2] = 90; c[5] = 20
    expected = evaluation.bev_iou(a, b)
    assert evaluation.bev_iou(b, a) == pytest.approx(expected, abs=1e-12)
    assert evaluation.bev_iou(a, c) == expected
    for v in (a, b): v[0] += 4096; v[1] -= 4096
    assert evaluation.bev_iou(a, b) == pytest.approx(expected, abs=1e-12)


def test_containment_rotated_disjoint_and_square_area():
    a, b = box(length=2, width=2), box(length=1, width=1, yaw=.7)
    assert evaluation.bev_iou(a, b) == pytest.approx(.25)
    assert evaluation.bev_iou(box(yaw=.5), box(100, 100, -.5)) == 0.
    assert evaluation.intersection_area([(0, 0), (2, 0), (2, 2), (0, 2)], [(1, 1), (3, 1), (3, 3), (1, 3)]) == 1.


def test_strict_ego_roi_class_filter_and_original_identity_order():
    boxes = [dict(track_id='boundary', class_label='car', mean=box(50)),
             dict(track_id='inside', class_label='car', mean=box(49.999), score=.4),
             dict(track_id='bike', class_label='bicycle', mean=box())]
    result = evaluation.car_rows(boxes, [0, 0, 0])
    assert [b['track_id'] for b in result] == ['inside']
    assert result[0]['score'] == .4
    assert result[0]['mean'] == boxes[1]['mean']


def test_duplicate_ids_are_rejected_even_outside_evaluated_class_roi():
    boxes = [dict(track_id='duplicate', class_label='bicycle', mean=box(80)),
             dict(track_id='duplicate', class_label='car', mean=box())]
    with pytest.raises(ValueError, match='duplicate'): evaluation.car_rows(boxes, [0, 0, 0])


def test_bad_geometry_and_unknown_class_cannot_be_filtered_silently():
    for name, mean in [('unmapped', box(80)), ('car', box(length=-2)), ('car', box(yaw=float('nan')))]:
        with pytest.raises(ValueError, match='class/world box'):
            evaluation.car_rows([dict(track_id='id', class_label=name, mean=mean)], [0, 0, 0])


def test_actual_counts_require_exact_equality_and_float_tolerance_is_fixed():
    assert evaluation.compare({'AssA': .5 + 5e-13}, {'AssA': .5}) < 1e-12
    with pytest.raises(ValueError, match='metric differs'):
        evaluation.compare({'HOTA_TP': [3. + 5e-13]}, {'HOTA_TP': [3.]})
    with pytest.raises(ValueError, match='metric differs'):
        evaluation.compare({'HOTA': .5 + 2e-12}, {'HOTA': .5})


def test_coverage_and_null_arrays_cannot_be_invented():
    with pytest.raises(ValueError, match='structure'): evaluation.compare({}, {'IDF1': .5})
    with pytest.raises(ValueError, match='coverage'): evaluation.compare([1], [1, 2])
    with pytest.raises(ValueError, match='value differs'): evaluation.compare(0., None)
    assert evaluation.compare({'nuscenes': {'eval_time': 99., 'amota': .2}}, {'nuscenes': {'eval_time': 1., 'amota': .2}}) == 0.
