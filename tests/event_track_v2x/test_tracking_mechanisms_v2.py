"""Single-factor and immutable prediction-only diagnostic regressions."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x import tracking_mechanisms_v2 as mechanism
from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.tracking_v2 import physical_world
from test_tracking_v2 import FixedModel, calibration, frame, tracker
from test_source_mask_v2 import short_sequence


def diagnostic(mode='M0', model=None):
    return mechanism.MechanismDiagnosticTrackerV2(model or FixedModel(), calibration(),
        'seq01', 1_000_000, mode=mode, plan_sha256='b' * 64)


def offset_pair(number=1, time_us=1_000_000):
    return (frame(number=number, time_us=time_us, scores=[.4]),
        frame('infrastructure-side', number=number, time_us=time_us,
              states=[[3., 0., 1., 2., 4., 2., -np.pi/2, 0., 0.]], scores=[.9],
              covariance=np.eye(9)[None] * 3))


@pytest.mark.parametrize('pair_value,dustbin', [(2., -4.), (0., 0.), (-8., 8.)])
def test_m0_prediction_and_association_bytes_equal_sealed_path(pair_value, dustbin):
    original = tracker(FixedModel(pair_value, dustbin))
    controlled = diagnostic(model=FixedModel(pair_value, dustbin))
    for pair in short_sequence():
        actual = controlled.step(*pair)
        expected = original.step(*pair)
        assert tuple(canonical(x) for x in actual[:2]) == tuple(canonical(x) for x in expected)
    for a, b in zip(original.model.inputs, controlled.model.inputs):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['M0', 'M1', 'M2', 'M3'])
def test_diagnostic_chain_predictions_and_association_are_bound(mode):
    item = diagnostic(mode)
    previous = '0' * 64
    for pair in short_sequence():
        result, association, audit = item.step(*pair)
        assert audit['previous_diagnostic_commit_sha256'] == previous
        previous = audit['diagnostic_commit_sha256']
        payload = {k: v for k, v in audit.items() if k != 'diagnostic_commit_sha256'}
        assert hashlib.sha256(canonical(payload)).hexdigest() == previous
        assert audit['prediction_commit_sha256'] == result['commit_sha256']
        assert audit['association_frame_sha256'] == hashlib.sha256(canonical(association)).hexdigest()
        assert audit['plan_sha256'] == 'b' * 64
        assert audit['tracks_after'] == len(result['predictions'])
        assert sum(audit['selected_detections']) == len(audit['source_records'])


def test_m1_keeps_both_original_scores_and_does_not_call_pair_model():
    pair = offset_pair()
    encoded = []
    for values in [(100., -100.), (-100., 100.)]:
        item = diagnostic('M1', FixedModel(*values))
        result, association, audit = item.step(*pair)
        assert item.model.inputs == []
        assert association['hypotheses'] == [{'pairs': [], 'unmatched_left': [0],
            'unmatched_right': [0], 'weight': 1., 'energy': 0.}]
        assert [n['score'] for n in audit['nodes']] == [.4, .9]
        assert len(result['predictions']) == 2
        assert audit['nodes'][0]['components'][0]['kind'] == 'vehicle_unmatched'
        encoded.append(tuple(canonical(x) for x in (result, association, audit)))
    assert encoded[0] == encoded[1]


def test_m2_only_changes_matched_component_state_and_covariance():
    pair = offset_pair()
    base = diagnostic('M0').step(*pair)
    actual = diagnostic('M2').step(*pair)
    assert canonical(base[1]) == canonical(actual[1])
    road_mean, road_cov = physical_world(pair[1], [0], 1_000_000)
    expected_state = mechanism.state_record(road_mean[0], road_cov[0])
    for original, changed in zip(base[2]['nodes'], actual[2]['nodes']):
        assert original['score'] == changed['score']
        assert original['matched_mass'] == changed['matched_mass']
        assert original['unmatched_mass'] == changed['unmatched_mass']
        if original['kind'] == 'road_residual':
            assert original == changed
        else:
            for old, new in zip(original['components'], changed['components']):
                assert old['weight'] == new['weight'] and old['score'] == new['score']
                assert old['partner_selected_index'] == new['partner_selected_index']
                if old['partner_selected_index'] is None:
                    assert old == new
                else:
                    assert new['kind'] == 'road_state_replacement'
                    for field, value in expected_state.items():
                        assert new[field] == value
                    assert old['mean'] != new['mean']


def test_m3_only_changes_positive_road_residual_scores_before_temporal_assignment():
    pair = offset_pair()
    base = diagnostic('M0').step(*pair)
    actual = diagnostic('M3').step(*pair)
    assert canonical(base[1]) == canonical(actual[1])
    assert base[2]['source_records'] == actual[2]['source_records']
    for original, changed in zip(base[2]['nodes'], actual[2]['nodes']):
        expected = copy.deepcopy(original)
        if original['kind'] == 'road_residual':
            assert 0 < original['unmatched_mass'] < 1
            assert original['score'] == original['unmatched_mass'] * .9
            expected['score'] = .9
        assert expected == changed
    assert any(e['event'] == 'birth_rejected_score' for e in base[2]['events'])
    assert sum(e['event'] == 'birth' for e in actual[2]['events']) == 2


@pytest.mark.parametrize('mode', ['M0', 'M2', 'M3'])
@pytest.mark.parametrize('empty_side', ['left', 'right', 'both'])
def test_no_association_mechanism_has_zero_effect_with_empty_sides(mode, empty_side):
    pair = (frame(states=[] if empty_side in ['left', 'both'] else None),
            frame('infrastructure-side', states=[] if empty_side in ['right', 'both'] else None))
    original = tracker().step(*pair)
    controlled = diagnostic(mode).step(*pair)
    assert tuple(canonical(x) for x in original) == tuple(canonical(x) for x in controlled[:2])


@pytest.mark.parametrize('mode', ['M0', 'M1', 'M2', 'M3'])
def test_duplicate_and_late_failures_leave_both_commit_chains_unchanged(mode):
    item = diagnostic(mode)
    pair = offset_pair()
    item.step(*pair)
    commits = item.commit_hash, item.diagnostic_commit_hash, item.last_reference_us
    snapshot = copy.deepcopy(item.tracks)
    for invalid in [pair, offset_pair(2, 999_999)]:
        with pytest.raises(ValueError):
            item.step(*invalid)
        assert (item.commit_hash, item.diagnostic_commit_hash, item.last_reference_us) == commits
        for tid, track in snapshot.items():
            np.testing.assert_array_equal(track['mean'], item.tracks[tid]['mean'])
            np.testing.assert_array_equal(track['cov'], item.tracks[tid]['cov'])


@pytest.mark.parametrize('mode', ['M0', 'M1', 'M2', 'M3'])
def test_inputs_weights_and_returned_audit_are_detached(mode):
    item = diagnostic(mode)
    pair = offset_pair()
    sources = [x.digest() for x in pair]
    state = {k: v.clone() for k, v in item.model.state_dict().items()}
    result, _, audit = item.step(*pair)
    audit['nodes'][0]['mean'][0] = 1e9
    audit['source_records'][0]['score'] = -1
    result['predictions'][0]['mean'][0] = 1e9
    assert [x.digest() for x in pair] == sources
    for k, value in state.items():
        torch.testing.assert_close(item.model.state_dict()[k], value, rtol=0, atol=0)
    assert all(track['mean'][0] != 1e9 for track in item.tracks.values())


def test_lifecycle_and_residual_cost_are_auditable():
    item = diagnostic('M1')
    all_events, assignments = [], []
    for pair in short_sequence():
        _, _, audit = item.step(*pair)
        all_events.extend(audit['events'])
        assignments.extend(audit['temporal_assignments'])
    kinds = {x['event'] for x in all_events}
    assert {'birth', 'miss', 'kill_before_assignment'} <= kinds
    assert assignments
    for match in assignments:
        expected = .5 * match['mahalanobis_squared'] - np.log(max(match['node_score'], 1e-12))
        assert match['assigned_cost'] == pytest.approx(expected)
        assert len(match['innovation_xyz']) == 3


@pytest.mark.parametrize('mode', ['M0', 'M1', 'M2', 'M3'])
def test_future_source_is_rejected_before_feature_and_diagnostic_access(mode):
    result, _, audit = diagnostic(mode).step(frame(image_us=1_100_001), frame('infrastructure-side'))
    assert result['source_available'] == [False, True]
    assert result['selected_detections'] == [0, 1]
    assert {s['side'] for s in audit['source_records']} == {'infrastructure-side'}


@pytest.mark.parametrize('bad', [None, 0, True, 'M4', [], {}])
def test_invalid_mode_rejected(bad):
    with pytest.raises(ValueError, match='unknown mechanism'):
        diagnostic(bad)


def test_frozen_model_required():
    model = FixedModel()
    model.anchor.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        diagnostic(model=model)


def test_gzip_is_replay_deterministic_and_runner_matrix_is_fixed():
    payload = canonical(diagnostic().step(*offset_pair())[2]) + b'\n'
    encoded = []
    for _ in range(2):
        sink = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=sink, mtime=0, compresslevel=6) as stream:
            stream.write(payload)
        encoded.append(sink.getvalue())
    assert encoded[0] == encoded[1]
    assert gzip.decompress(encoded[0]) == payload
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('mechanism_runner', root/'tools/event_track_v2x/run_mechanism_diagnostics_v2.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert len(runner.RUNS) == 10
    assert len({x['run_id'] for x in runner.RUNS}) == 10
    assert [x['mode'] for x in runner.RUNS].count('M1') == 1
    for mode in ['M0', 'M2', 'M3']:
        assert [x['seed'] for x in runner.RUNS if x['mode'] == mode] == [1337, 2027, 3407]
