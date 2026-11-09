import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sequence_statistics', Path(__file__).parents[2] / 'tools/event_track_v2x/stage1_sequence_statistics.py')
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)


def row(tp, fn, fp, assa, itp, ifn, ifp):
    return dict(HOTA={**{k: [v] * 19 for k, v in zip(statistics.COUNTS, (tp, fn, fp))},
                      **{k: [v] * 19 for k, v in [('AssA', assa), ('AssPr', assa), ('AssRe', assa), ('LocA', 1.)]}},
                Identity=dict(IDTP=itp, IDFN=ifn, IDFP=ifp), gt_objects=tp + fn, predicted_objects=tp + fp)


def test_large_and_small_sequences_use_detection_counts_not_macro_mean():
    # Nine correct associations and one wrong association: weighted AssA=.9.
    data = statistics.sufficient({'large': row(9, 0, 0, 1., 9, 0, 0), 'small': row(1, 0, 0, 0., 0, 1, 1)}, ['large', 'small'])
    result = statistics.aggregate(data, [[1, 1]])
    assert result['AssA'][0, 0] == pytest.approx(.9)
    assert result['HOTA'][0, 0] == pytest.approx(np.sqrt(.9))
    assert result['IDF1'][0] == pytest.approx(.9)
    assert result['AssA'][0, 0] != pytest.approx(.5)


def test_resampled_duplicate_sequence_has_separate_identity_namespace():
    data = statistics.sufficient({'a': row(3, 1, 2, .5, 2, 2, 3), 'b': row(1, 0, 0, 1., 1, 0, 0)}, ['a', 'b'])
    result = statistics.aggregate(data, [[2, 0]])
    assert result['HOTA_TP'][0, 0] == 6
    assert result['HOTA_FN'][0, 0] == 2
    assert result['HOTA_FP'][0, 0] == 4
    assert result['AssA'][0, 0] == .5
    assert result['IDF1'][0] == pytest.approx(4 / 9)


def test_zero_true_positive_tail_retains_author_localization_convention():
    data = statistics.sufficient({'empty': row(0, 5, 2, 0., 0, 5, 2)}, ['empty'])
    result = statistics.aggregate(data, [[1]])
    for name in statistics.METRICS:
        assert np.count_nonzero(result[name]) == 0
    assert np.all(result['LocA'] == 1.)


def test_corrupt_sequence_counts_cannot_be_hidden_in_full_aggregate():
    x = row(4, 1, 2, .5, 4, 1, 2); x['HOTA']['HOTA_FN'][3] = 1.1
    with pytest.raises(ValueError, match='integer counts'):
        statistics.sufficient({'a': x}, ['a'])
    x = row(4, 1, 2, .5, 3, 1, 2)
    with pytest.raises(ValueError, match='identity count conservation'):
        statistics.sufficient({'a': x}, ['a'])


def test_shared_bootstrap_is_exactly_10000_fixed_whole_sequence_draws():
    indices, weights = statistics.draws(21)
    assert indices.shape == (10000, 21)
    assert weights.shape == (10000, 21)
    assert np.all(weights.sum(1) == 21)
    assert np.array_equal(indices, statistics.draws(21)[0])
    assert np.array_equal(weights[0], np.bincount(indices[0], minlength=21))
    assert not np.array_equal(indices[0], indices[1])


def test_one_deterministic_control_has_unknown_sample_standard_deviation():
    summary = statistics.descriptive([dict(HOTA=.3, IDF1=.2)])
    assert summary['HOTA']['sample_std'] is None
    assert summary['HOTA']['mean'] == .3
    with pytest.raises(ValueError, match='missing-field zeros'):
        statistics.descriptive([dict(HOTA=.3, IDF1=.2), dict(HOTA=.4)])


@pytest.mark.parametrize('weights', [[[0, 0]], [[1.5, .5]], [[1, -1]], [[float('nan'), 1]]])
def test_invalid_resampling_weights_rejected(weights):
    data = statistics.sufficient({'a': row(1, 0, 0, 1., 1, 0, 0), 'b': row(1, 0, 0, 1., 1, 0, 0)}, ['a', 'b'])
    with pytest.raises(ValueError, match='multiplicities'):
        statistics.aggregate(data, weights)
