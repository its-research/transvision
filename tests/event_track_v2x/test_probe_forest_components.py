import pytest

from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from tools.event_track_v2x.probe_forest_components import probe
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from test_detection_cache_v2 import _sources


def test_full_train_probe_counts_every_car_even_when_components_exceed_capacity(tmp_path):
    roots, calibration, inputs, output = _sources(tmp_path, split='train')
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    result = probe(output, sha, max_component_nodes=1)
    assert result['frames'] == 4 and result['sequences'] == 2
    assert result['observations'] == 6
    assert sum(w['observations'] for w in result['windows']) == 6
    assert result['over_capacity_components'] == 1
    assert result['over_capacity_observations'] == 4
    assert result['largest_component'] == 4
    assert result['window_boundary_edges_omitted_for_sizing_only']
    assert not result['identity_preserving_handoff_executed']
    assert not result['paper_performance_evidence']


def test_probe_rejects_val_before_payload_audit(tmp_path, monkeypatch):
    roots, calibration, inputs, output = _sources(tmp_path)
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    monkeypatch.setattr('tools.event_track_v2x.probe_forest_components.load_manifest',
                        lambda *a: pytest.fail('validation payload accessed'))
    with pytest.raises(ValueError, match='train only'):
        probe(output, sha)
