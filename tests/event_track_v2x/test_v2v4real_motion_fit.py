import numpy as np
import pytest

from tools.event_track_v2x.v2v4real_motion_fit import (
    legacy_targets, wrap, offline_matches, prior, fit_model, apply_model, select_gate)


def targets():
    return [dict(group=g, side=s, velocity_source=np.tile([2., -3.], (20, 1)),
                 velocity_valid=np.ones(20, dtype=bool))
            for g in ('a', 'b') for s in ('vehicle-side', 'infrastructure-side')]


def rows():
    out = []
    for group in ('a', 'b'):
        for side in ('vehicle-side', 'infrastructure-side'):
            for ordinal in (0, 1):
                xyz = np.stack([np.arange(12)*10 + ordinal*.2, np.zeros(12), np.zeros(12)], axis=1)
                state = np.column_stack([xyz, np.tile([2., 4., 1., 0.], (12, 1))])
                truth = np.column_stack([xyz, np.tile([4., 2., 1., -np.pi/2], (12, 1))])
                out.append(dict(group=group, sequence_id=group, side=side, frame_ordinal=ordinal,
                    source_to_world=np.eye(4), states_source_legacy7=state,
                    class_indices=np.zeros(12, dtype=np.int64), row_sha256=f'{ordinal:064x}',
                    gt_matches=np.arange(12), target_states_source=truth,
                    target_velocity_source=np.tile([2., 0.], (12, 1)),
                    target_velocity_valid=np.ones(12, dtype=bool)))
    return out


def test_axis_yaw_velocity_conversion_and_wrap_boundary():
    a = legacy_targets(np.array([[1, 2, 3, 4, 5, 6, np.pi/2]]), np.array([[7, 8]]))
    np.testing.assert_allclose(a, [[1, 2, 3, 5, 4, 6, -np.pi, 7, 8]])
    np.testing.assert_allclose(wrap([np.pi, -np.pi, 3*np.pi]), [-np.pi]*3)


def test_offline_match_score_order_one_to_one_tie_and_strict_boundary():
    b = np.zeros((3, 7)); b[:, 0] = [.1, .2, 2]
    g = np.zeros((1, 7))
    assert offline_matches(b, np.array([.5, .9, .1]), g).tolist() == [-1, 0, -1]
    g = np.zeros((2, 7)); g[:, 0] = [-1, 1]
    assert offline_matches(np.zeros((1, 7)), np.ones(1), g).tolist() == [-1]


def test_holdout_prior_excludes_held_out_and_invalid_samples():
    t = targets()
    for row in t:
        if row['group'] == 'b': row['velocity_source'] *= 100
    p = prior(t, {'a'})
    assert p['vehicle-side']['samples'] == 20
    np.testing.assert_array_equal(p['vehicle-side']['velocity_source'], [2, -3])
    with pytest.raises(ValueError, match='insufficient'): prior(t, {'missing'})


def test_full_second_moment_cross_terms_and_initial_prior_distinction():
    source_rows = rows()
    for row in source_rows:
        row['target_states_source'][:, 0] -= .2
    model, motion = fit_model(source_rows, targets(), ['a', 'b'],
                              dict(max_speed_mps=5., max_gap_ordinals=1))
    # Unavailable prior differs from target in vy; retained covariance includes it.
    cov = np.array(model['sides']['vehicle-side']['statuses']['unavailable']['covariance'])
    assert cov[8, 8] > 8.9
    np.testing.assert_allclose(cov[0, 8], -.57, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > 0
    output = apply_model({'side': 'vehicle-side'}, motion[0], model)
    assert output['initial_prior_applied'].all() and not output['velocity_measured_valid'].any()
    np.testing.assert_array_equal(output['velocity_source'], np.tile([2, -3], (12, 1)))
    # Measured static or moving observations remain measured; population prior is not substituted.
    output = apply_model({'side': 'vehicle-side'}, motion[1], model)
    assert output['velocity_measured_valid'].all()
    np.testing.assert_allclose(output['velocity_source'], np.tile([2, 0], (12, 1)))


def test_gate_selection_fixed_scope_and_minimum_sample_rejection():
    result = select_gate({1337: rows()}, targets(), ['a', 'b'])
    assert len(result['scores']) == 12
    assert result['selected'] == dict(max_speed_mps=5., max_gap_ordinals=1)
    assert len({f['samples'] for x in result['scores'] for f in x['folds']}) == 1
    with pytest.raises(ValueError, match='two'): select_gate({1337: rows()}, targets(), ['a'])
    with pytest.raises(ValueError, match='coverage'): select_gate({1337: rows()}, targets(), ['a', 'c'])
    with pytest.raises(ValueError, match='declared'): fit_model(rows(), targets(), ['a'], result['selected'])
    with pytest.raises(ValueError, match='insufficient actual'):
        fit_model(rows(), targets(), ['a', 'b'], dict(max_speed_mps=.001, max_gap_ordinals=1))
