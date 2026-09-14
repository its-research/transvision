import itertools

import pytest

from tools.event_track_v2x.audit_forest_cache_capacity import audit, window_load
from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from test_detection_cache_v2 import _sources


def test_simultaneous_arrival_and_inclusive_window_are_order_invariant():
    rows = [(0, 2), (10, 3), (10, 7), (11, 1)]
    for order in itertools.permutations(rows):
        result = window_load(order, window_us=10, max_nodes=11)
        assert result == {'information_time_groups': 3, 'frames': 4, 'peak_nodes': 12,
            'frames_at_over_capacity_decisions': 2, 'load_quantiles_0_50_95_100': [2., 11., 11.9, 12.]}


def test_train_capacity_audit_is_not_tracking_evidence(tmp_path):
    roots, calibration, inputs, output = _sources(tmp_path, split='train')
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    result = audit(output, sha, window_us=100, max_nodes=3)
    assert result['frames'] == 4 and result['selected_car_detections'] == 6
    assert result['peak_information_time_window_nodes'] == 4
    assert result['frames_at_over_capacity_decisions'] == 2
    assert result['source_counts']['vehicle-side']['selected_car'] == 2
    assert result['selected_car_with_valid_appearance'] == 6
    assert not result['paper_performance_evidence'] and not result['identity_tracking_executed']


def test_capacity_design_rejects_validation_before_full_payload_load(tmp_path, monkeypatch):
    roots, calibration, inputs, output = _sources(tmp_path)
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    monkeypatch.setattr('tools.event_track_v2x.audit_forest_cache_capacity.load_manifest',
                        lambda *a: pytest.fail('validation payloads inspected'))
    with pytest.raises(ValueError, match='train only'):
        audit(output, sha)


@pytest.mark.parametrize('rows,window,max_nodes', [([(0, -1)], 1, 1), ([(1.5, 0)], 1, 1), ([], 0, 1), ([], 1, 0)])
def test_invalid_capacity_input_rejected(rows, window, max_nodes):
    with pytest.raises(ValueError):
        window_load(rows, window_us=window, max_nodes=max_nodes)
