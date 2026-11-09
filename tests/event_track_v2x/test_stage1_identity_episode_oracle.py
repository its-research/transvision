import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('episode_oracle', Path(__file__).parents[2] / 'tools/event_track_v2x/stage1_identity_episode_oracle.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)


def test_global_owner_follows_whole_GT_identity_history_not_first_anchor():
    frames = [(['A'], ['pred'], np.array([[1.]]))] + [(['B'], ['pred'], np.array([[1.]])) for _ in range(3)]
    result = oracle.global_owners(frames, ['A', 'B'], ['pred'])
    assert result['owners']['pred']['owner'] == 'B'
    assert (result['IDTP'], result['IDFN'], result['IDFP']) == (3, 1, 1)
    label = oracle.identity_label(dict(status='unique', other='pred', IoU=1.), result['owners'], 'A')
    assert label['label'] == 'wrong_under_unique_global_ID_assignment'


def test_equal_optimal_GT_owners_stay_unknown():
    frames = [(['A'], ['pred'], np.array([[1.]])), (['B'], ['pred'], np.array([[1.]]))]
    result = oracle.global_owners(frames, ['A', 'B'], ['pred'])
    owner = result['owners']['pred']
    assert owner['status'] == 'ambiguous_optimal_owner' and owner['owner'] is None
    assert oracle.identity_label(dict(status='unique', other='pred', IoU=1.), result['owners'], 'A')['label'] == 'unknown'


def test_duplicate_spatial_detections_are_not_forced_into_an_identity_label():
    result = oracle.unique_correspondences(np.array([[1., .8]]), ['GT'], ['p1', 'p2'])
    assert result['GT']['status'] == 'ambiguous_geometric_correspondence'
    assert result['GT']['other'] is None
    result = oracle.unique_correspondences(np.array([[.9], [.8]]), ['GT1', 'GT2'], ['p'])
    assert all(x['status'] != 'unique' for x in result.values())


def test_zero_overlap_owner_is_not_a_claimed_correct_identity():
    result = oracle.global_owners([(['GT'], ['pred'], np.array([[0.]]))], ['GT'], ['pred'])
    assert (result['IDTP'], result['IDFN'], result['IDFP']) == (0, 1, 1)
    assert result['owners']['pred']['owner'] is None


def wrong(): return dict(label='wrong_under_unique_global_ID_assignment', prediction_id='p', global_GT_owner='other')


def test_native_error_interval_does_not_impute_an_unobserved_frame_period():
    e = oracle.Episodes(); e.step('seq', 0, 1000000, {'GT': wrong()}); e.step('seq', 1, 1300100, {'GT': wrong()})
    e.step('seq', 2, 2000000, {'GT': dict(label='correct')})
    assert len(e.finished) == 1
    interval = e.finished[0]
    assert interval['observed_span_us'] == 300100
    assert interval['observed_native_intervals_us'] == [300100]
    assert interval['possible_end_time_upper_us'] == 2000000
    assert interval['hidden_time_continuity_claimed'] is False


def test_missing_GT_and_ambiguity_censor_and_split_errors():
    e = oracle.Episodes(); e.step('seq', 0, 1, {'GT': wrong()}); e.step('seq', 1, 3, {})
    e.step('seq', 2, 9, {'GT': wrong()}); e.step('seq', 3, 10, {'GT': dict(label='unknown', reason='ambiguous')})
    assert len(e.finished) == 2 and all(x['right_censored'] for x in e.finished)
    assert all(x['possible_end_time_upper_us'] is None for x in e.finished)


def test_sequence_end_is_censored_and_unknown_labels_are_not_zero_duration_errors():
    e = oracle.Episodes(); e.step('seq', 0, 1, {'unknown': dict(label='unknown', reason='no_correspondence')})
    assert not e.active and not e.finished
    e.step('seq', 1, 5, {'GT': wrong()}); e.end_sequence()
    assert e.finished[0]['right_censored'] and e.finished[0]['end_reason'] == 'sequence_ended'


def test_nonconsecutive_frame_index_is_not_silently_bridged():
    e = oracle.Episodes(); e.step('seq', 0, 1, {'GT': wrong()})
    with pytest.raises(ValueError, match='continuity'): e.step('seq', 2, 4, {'GT': wrong()})
