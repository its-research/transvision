"""Synthetic identity timelines; no real data or full validation claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.event_track_v2x import evaluate_full_identity_events as tool


def box(tid):
    return dict(track_id=tid, class_label='car', mean=[0., 0., 0., 4., 2., 1., 0., 0., 0.])


@pytest.fixture
def timeline(tmp_path):
    gt, predictions, audits = [], [], []
    for sequence, ids in [('a', ['A', 'B', 'B', 'A']), ('b', ['Z', 'Z'])]:
        previous = '0' * 64
        for i, tid in enumerate(ids):
            frame = dict(sequence_id=sequence, frame_id=str(i), box_reference_timestamp_us=1_000_000+100_000*i)
            gt.append(dict(frame, ego_translation_world=[0., 0., 0.], objects=[box('same-gt-id')]))
            prediction = dict(frame, predictions=[box(tid)], commit_sha256=f'commit-{sequence}-{i}')
            predictions.append(prediction)
            audit = dict(sequence_id=sequence, event_id=str(i), prediction_sha256=prediction['commit_sha256'],
                         previous_audit_sha256=previous)
            previous = tool.mht.ledger.digest(audit); audits.append(dict(tracking=audit))
    tracking = tmp_path / 'tracking.jsonl'
    tracking.write_bytes(b''.join(tool.native.canonical(r)+b'\n' for r in audits))
    adapter = SimpleNamespace(canonical=tool.native.canonical,
        _roi=lambda boxes, ego, cls: [b for b in boxes if b['class_label'] == cls and np.linalg.norm(b['mean'][:2]) < 50],
        oriented_bev_iou=lambda a, b: np.ones((len(a), len(b))))
    return gt, predictions, audits, tracking, adapter


def diagnose(timeline, tmp_path):
    gt, predictions, _, tracking, adapter = timeline
    return tool.diagnose_sequences(adapter, tool.load_helper(), gt, predictions, tracking, tmp_path)


def test_complete_sequences_reset_identity_and_preserve_observed_spans(timeline, tmp_path):
    details = diagnose(timeline, tmp_path); summary = tool.summarize(details)
    assert summary['frames'] == 6 and summary['sequences'] == 2
    assert summary['counts']['anchored_gt_identities'] == 2
    assert summary['counts']['anchor_disagreement_gt_frames'] == 2
    assert summary['anchor_disagreement_given_unique'] == pytest.approx(2/6)
    assert summary['observed_error_span_seconds_quantiles'] == pytest.approx([.1]*4)
    episode = details['a']['episodes'][0]
    assert episode['recovery_observation_elapsed_seconds'] == pytest.approx(.2)
    assert episode['left_censored'] is False and episode['right_censored'] is False
    assert not details['b']['episodes']  # Reused GT ID across sequences never inherits an anchor.
    assert not list(tmp_path.glob('identity-event-slices-*'))
    assert summary['unknowns_are_not_assumed_correct']
    assert summary['observed_span_is_not_continuous_error_duration']
    for flag in ('gt_error_calibration_computed', 'official_ids_computed', 'causal_effect_identified'):
        assert summary[flag] is False


def test_unknown_mapping_censors_instead_of_interpolating_identity(timeline, tmp_path):
    timeline[1][2]['predictions'] = []
    details = diagnose(timeline, tmp_path); summary = tool.summarize(details)
    episode = details['a']['episodes'][0]
    assert episode['right_censored'] and episode['closure_reason'] == 'mapping_unknown'
    assert episode['observed_error_span_seconds'] == 0.
    assert episode['recovery_observation_elapsed_seconds'] is None
    assert summary['unknown_fraction_of_roi_gt_observations'] == pytest.approx(1/6)
    assert summary['anchor_disagreement_given_unique'] == pytest.approx(1/5)


def test_all_unknown_keeps_rates_and_spans_missing_not_zero(timeline, tmp_path):
    for p in timeline[1]: p['predictions'] = []
    summary = tool.summarize(diagnose(timeline, tmp_path))
    assert summary['anchor_disagreement_given_unique'] is None
    assert summary['unknown_fraction_of_roi_gt_observations'] == 1.
    assert summary['observed_error_span_seconds_quantiles'] is None


def test_duplicate_overlap_is_unknown_not_forced_to_a_track(timeline, tmp_path):
    timeline[1][1]['predictions'] = [box('A'), box('B')]
    summary = tool.summarize(diagnose(timeline, tmp_path))
    assert summary['counts']['duplicate_excess_predictions'] == 1
    assert summary['counts']['frames_with_duplicates'] == 1
    assert summary['unknown_fraction_of_roi_gt_observations'] == pytest.approx(1/6)


@pytest.mark.parametrize('bad', ['missing_gt', 'missing_prediction', 'missing_audit', 'extra_audit',
                               'clock', 'frame', 'chain', 'interleaved', 'sequence_order'])
def test_misaligned_or_incomplete_input_is_rejected(timeline, tmp_path, bad):
    gt, predictions, audits, tracking, _ = timeline
    if bad == 'missing_gt': gt.pop()
    elif bad == 'missing_prediction': predictions.pop()
    elif bad == 'missing_audit': audits.pop()
    elif bad == 'extra_audit': audits.append(audits[-1])
    elif bad == 'clock': predictions[0]['box_reference_timestamp_us'] += 1
    elif bad == 'frame': predictions[0]['frame_id'] = 'other'
    elif bad == 'chain': audits[0]['tracking']['previous_audit_sha256'] = 'bad'
    elif bad == 'interleaved':
        gt[:], predictions[:], audits[:] = ([v[i] for i in (0, 4, 1, 2, 3, 5)] for v in (gt, predictions, audits))
    elif bad == 'sequence_order':
        gt[:], predictions[:], audits[:] = ([v[i] for i in (4, 5, 0, 1, 2, 3)] for v in (gt, predictions, audits))
    tracking.write_bytes(b''.join(tool.native.canonical(r)+b'\n' for r in audits))
    with pytest.raises(ValueError): diagnose(timeline, tmp_path)
    assert not list(tmp_path.glob('identity-event-slices-*'))


def test_summary_pools_episodes_instead_of_averaging_sequence_percentiles(timeline, tmp_path):
    details = diagnose(timeline, tmp_path)
    a = details['a']; b = details['b']
    a['episodes'] = [dict(observed_error_span_seconds=1.), dict(observed_error_span_seconds=2.)]
    b['episodes'] = [dict(observed_error_span_seconds=100.)]
    a['counts']['anchor_disagreement_episodes'] = 2; b['counts']['anchor_disagreement_episodes'] = 1
    assert tool.summarize(details)['observed_error_span_seconds_quantiles'][0] == 2.


@pytest.mark.parametrize('bad', ['protocol', 'frame_count', 'denominator', 'episodes', 'negative_span'])
def test_invalid_summary_inputs_are_rejected(timeline, tmp_path, bad):
    d = diagnose(timeline, tmp_path)
    if bad == 'protocol': d['b']['protocol'] = {}
    elif bad == 'frame_count': d['a']['counts']['frames'] += 1
    elif bad == 'denominator': d['a']['counts']['identity_unique_gt_frames'] += 1
    elif bad == 'episodes': d['a']['counts']['anchor_disagreement_episodes'] += 1
    elif bad == 'negative_span': d['a']['episodes'][0]['observed_error_span_seconds'] = -1
    with pytest.raises(ValueError): tool.summarize(d)


@pytest.mark.parametrize('backend', ['probabilistic', 'scan-mht'])
def test_full_audit_required_before_any_gt_or_metric_engine(tmp_path, monkeypatch, backend):
    def reject(*a): raise ValueError('not a complete audit')
    monkeypatch.setattr(tool.common, 'inspect_run', reject)
    monkeypatch.setattr(tool.mht, 'inspect_audited_run', reject)
    monkeypatch.setattr(tool.native, 'load_adapter', lambda: pytest.fail('partial input reached GT evaluator'))
    with pytest.raises(ValueError, match='complete audit'):
        tool.evaluate(backend, 'run', 'sha', 'audit', 'sha', 'GT', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_diagnostic_and_existing_helper_do_not_import_torch_or_predictor():
    p = subprocess.run([sys.executable, '-c',
        'import sys; from tools.event_track_v2x import evaluate_full_identity_events as e; e.load_helper(); '
        'assert "torch" not in sys.modules; '
        'assert "tools.event_track_v2x.persistent_mht_tracking" not in sys.modules'],
        cwd=tool.ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stderr


def test_empty_summary_is_rejected():
    with pytest.raises(ValueError, match='nonempty'): tool.summarize({})
