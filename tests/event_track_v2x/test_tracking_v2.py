"""Prediction-only V2 integration checks; no dataset or GT fixtures are read."""
from __future__ import annotations

from dataclasses import asdict
import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transvision.models.event_track_v2x.detection_cache_v2 import (
    BOX_LAYOUT, FEATURE_METHOD, META_FIELDS, DetectionCacheV2, canonical,
)
from transvision.models.event_track_v2x import predicted_association_v2 as association
from transvision.models.event_track_v2x.prediction_features import wrap_angle
from transvision.models.event_track_v2x import tracking_v2 as tracking


def frame(side="vehicle-side", *, sequence="seq01", number=1, time_us=1_000_000,
          image_us=None, states=None, scores=None, rotation=None, translation=None,
          covariance=None, classes=None):
    states = np.asarray([[0., 0., 1., 2., 4., 2., -np.pi / 2, 0., 0.]]
                        if states is None else states, dtype=float).reshape(-1, 9)
    n = len(states)
    meta = {key: "a" * 64 for key in META_FIELDS if key.endswith("_sha256")}
    meta.update(kind="detection_cache_v2", schema_version=2, sequence_id=sequence,
                frame_id=f"frame{number:04d}", side=side,
                box_reference_timestamp_us=time_us,
                source_image_timestamp_us=time_us if image_us is None else image_us,
                coordinate_system="source_lidar", box_layout=BOX_LAYOUT,
                lidar_to_world_row_rotation=(np.eye(3) if rotation is None else rotation).tolist(),
                lidar_to_world_translation=(np.zeros(3) if translation is None else translation).tolist(),
                agent_mask=1 if side == "vehicle-side" else 2,
                dataset_split="val", feature_method=FEATURE_METHOD,
                calibration_fit_split="train", calibration_sha256=association.CALIBRATION_SHA)
    scores = np.full(n, .9) if scores is None else np.asarray(scores, float)
    appearance = np.zeros((n, 128), np.float32)
    if n:
        appearance[:, 0] = 1.
    return DetectionCacheV2(
        canonical(meta), states, scores, scores,
        np.zeros(n, np.int64) if classes is None else np.asarray(classes, np.int64),
        np.tile(np.eye(9)[None], (n, 1, 1)) if covariance is None else covariance,
        appearance, np.ones(n, bool),
    )


def calibration():
    score = {"slope": 1., "intercept": 0., "logit_clip": 1e-6}
    return {"sides": {side: {name: {"score": score.copy()}
                            for name in ("car", "bicycle", "pedestrian")}
                      for side in ("vehicle-side", "infrastructure-side")}}


class FixedModel(nn.Module):
    """A deterministic pair/dustbin model which also audits its actual inputs."""
    def __init__(self, pair=2., dustbin=-4.):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.pair_value, self.dustbin_value = pair, dustbin
        self.inputs = []
        self.eval()

    def forward(self, left, right):
        self.inputs.append((left.detach().clone(), right.detach().clone()))
        b, n, _ = left.shape
        m = right.shape[1]
        return (left.new_full((b, n, m), self.pair_value),
                left.new_full((b, n), self.dustbin_value),
                right.new_full((b, m), self.dustbin_value))


def tracker(model=None, config=None):
    return tracking.LearnedPairTrackerV2(model or FixedModel(), calibration(), "seq01",
                                         1_000_000, config)


def exhaustive(cost):
    n, m = cost.shape
    solutions = []
    for columns in itertools.permutations(range(m), n):
        value = float(cost[np.arange(n), columns].sum())
        if np.isfinite(value):
            solutions.append((value, columns))
    return sorted(solutions)


def test_murty_rectangular_with_unique_dummies_matches_exhaustive():
    rng = np.random.default_rng(44)
    for n, m in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        costs = np.full((n, m + n), np.inf)
        costs[:, :m] = rng.normal(size=(n, m))
        costs[np.arange(n), m + np.arange(n)] = rng.normal(size=n)
        wanted = exhaustive(costs)
        actual = association.k_best_assignments(costs, len(wanted) + 2)
        assert [x[1] for x in actual] == [x[1] for x in wanted]
        np.testing.assert_allclose([x[0] for x in actual], [x[0] for x in wanted])
        assert len(set(x[1] for x in actual)) == len(actual)


def test_murty_ties_empty_and_infeasible_are_deterministic():
    costs = np.zeros((3, 4))
    a = association.k_best_assignments(costs, 30)
    assert a == association.k_best_assignments(costs, 30)
    assert set(x[1] for x in a) == set(x[1] for x in exhaustive(costs))
    assert association.k_best_assignments(np.empty((0, 4)), 3) == [(0., ())]
    assert association.k_best_assignments(np.full((2, 3), np.inf), 3) == []
    for invalid in [np.zeros((2, 1)), np.full((1, 2), np.nan)]:
        with pytest.raises(ValueError):
            association.k_best_assignments(invalid)


def test_joint_top_h_has_one_to_one_alternatives_and_explicit_unmatched_mass():
    hypotheses = association.association_hypotheses(
        np.array([[2., 1.5], [1.8, 2.]]), np.zeros(2), np.zeros(2), [0, 0], [0, 0], 3)
    assert 3 <= len(hypotheses) <= 4
    assert sum(h["weight"] for h in hypotheses) == pytest.approx(1.)
    assert any(not h["pairs"] and h["unmatched_left"] == [0, 1] and
               h["unmatched_right"] == [0, 1] and h["weight"] > 0 for h in hypotheses)
    assert len({tuple(h["pairs"]) for h in hypotheses}) == len(hypotheses)
    for h in hypotheses:
        assert len({a for a, _ in h["pairs"]}) == len(h["pairs"])
        assert len({b for _, b in h["pairs"]}) == len(h["pairs"])
        assert h["weight"] > 0
    # A missing source and a class-incompatible source cannot manufacture edges.
    for n, m in [(0, 0), (0, 2), (2, 0)]:
        h = association.association_hypotheses(np.zeros((n, m)), np.zeros(n), np.zeros(m),
                                              np.zeros(n), np.zeros(m))[0]
        assert h["pairs"] == [] and h["weight"] == 1.
    h = association.association_hypotheses(np.full((1, 1), 100.), [0.], [0.], [0], [1])
    assert len(h) == 1 and h[0]["pairs"] == []


def test_joint_energy_includes_both_dustbins_and_matches_direct_enumeration():
    from scipy.special import logsumexp
    pair = np.array([[2., -1.], [.4, 1.3]])
    ld, rd = np.array([.1, .6]), np.array([.7, -.3])
    row = np.c_[pair, ld]; row -= logsumexp(row, axis=1)[:, None]
    col = np.c_[pair.T, rd]; col -= logsumexp(col, axis=1)[:, None]
    h = association.association_hypotheses(pair, ld, rd, [0, 0], [0, 0], 20)
    assert len(h) == 7
    for item in h:
        expected = -sum(row[i, j] + col[j, i] for i, j in item["pairs"])
        expected -= sum(row[i, -1] for i in item["unmatched_left"])
        expected -= sum(col[j, -1] for j in item["unmatched_right"])
        assert item["energy"] == pytest.approx(expected)


def test_203_feature_groups_exact_values_selection_and_no_physical_axis_swap():
    f = frame(time_us=1_300_000, image_us=1_350_000,
              states=[[10., 20., 4., 2., 4., 3., np.pi / 4, 6., -3.]])
    selected, actual = association.frame_features(
        f, f.metadata, 1_000_000, -.2, calibration()["sides"]["vehicle-side"])
    expected = np.concatenate([
        [ .1, .2, .2, .1, .2, .3, np.sqrt(.5), np.sqrt(.5)],
        [.2, -.1], np.r_[1., np.zeros(127)], [1., 0., 0.],
        [.003, .05, -.2, 0.], np.eye(9)[np.triu_indices(9)] / 100,
        np.zeros(6), [1., np.log(2) / 8, 0., 0.], [.9, .9, 1.],
    ]).astype(np.float32)
    assert selected.tolist() == [0] and actual.shape == (1, 203)
    np.testing.assert_array_equal(actual[0], expected)
    many = frame(states=np.repeat(f.states, 70, axis=0), scores=[.04] + [.7] * 69)
    chosen, values = association.frame_features(many, many.metadata, 1_000_000, 0.,
                                               calibration()["sides"]["vehicle-side"])
    assert chosen.tolist() == list(range(1, 65)) and values.shape == (64, 203)


def test_network_matches_frozen_training_architecture_and_empty_dimensions():
    torch.manual_seed(912)
    model = association.PredictedAssociation().eval()
    state = model.state_dict()
    assert state["encoder.0.weight"].shape == (128, 203)
    assert state["encoder.4.weight"].shape == (128, 128)
    assert state["pair.0.weight"].shape == (128, 512)
    assert state["left_dustbin.weight"].shape == state["right_dustbin.weight"].shape == (1, 128)
    left, right = torch.randn(1, 2, 203), torch.randn(1, 3, 203)
    def encoder(x):
        x = F.linear(x, state["encoder.0.weight"], state["encoder.0.bias"])
        x = F.layer_norm(x, (128,), state["encoder.1.weight"], state["encoder.1.bias"])
        x = F.gelu(x)
        return F.gelu(F.linear(x, state["encoder.4.weight"], state["encoder.4.bias"]))
    l, r = encoder(left), encoder(right)
    a, b = l[:, :, None].expand(-1, -1, 3, -1), r[:, None].expand(-1, 2, -1, -1)
    pairs = torch.cat([a, b, torch.abs(a-b), a*b], -1)
    expected = F.linear(F.gelu(F.linear(pairs, state["pair.0.weight"], state["pair.0.bias"])),
                        state["pair.3.weight"], state["pair.3.bias"]).squeeze(-1)
    output = model(left, right)
    torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(output[1], F.linear(l, state["left_dustbin.weight"], state["left_dustbin.bias"]).squeeze(-1))
    torch.testing.assert_close(output[2], F.linear(r, state["right_dustbin.weight"], state["right_dustbin.bias"]).squeeze(-1))
    for n, m in [(0, 0), (2, 0), (0, 3)]:
        outputs = model(torch.empty(1, n, 203), torch.empty(1, m, 203))
        assert [tuple(x.shape) for x in outputs] == [(1, n, m), (1, n), (1, m)]


def test_checkpoint_loader_pins_all_three_seeds_weights_only_and_frozen_state(monkeypatch):
    assert association.CHECKPOINTS == {
        1337: "de2363f4269f807e50200f5742ab55a6ed113110cf4764b5623f3000fb7696b0",
        2027: "a8a785b41d539390e284c9fc191993eeb0592c6a0f21cd5239f32526b24f04f8",
        3407: "55ff4f9ca449e91fd155e7b7403a8103556ccb1c5e69fd4c43de65f0b7fd28c7",
    }
    state = association.PredictedAssociation().state_dict()
    for seed in association.CHECKPOINTS:
        monkeypatch.setattr(association, "sha_file", lambda p, seed=seed: association.CHECKPOINTS[seed])
        def safe_load(path, *, map_location, weights_only, seed=seed):
            assert weights_only is True and map_location == "cpu"
            return {"kind": "eventtrack_predicted_association_checkpoint_v1", "seed": seed,
                    "epoch": 24, "feature_schema_sha256": association.SCHEMA_SHA,
                    "plan_sha256": association.PLAN_SHA, "data_manifest_sha256": association.DATA_SHA,
                    "model": state}
        monkeypatch.setattr(association.torch, "load", safe_load)
        model = association.load_frozen_model(Path("synthetic-checkpoint.pt"), seed)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert asdict(tracking.TrackingConfigV2()) == asdict(tracker(model).config)
    monkeypatch.setattr(association, "sha_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="changed final checkpoint"):
        association.load_frozen_model("changed.pt", 1337)
    with pytest.raises(ValueError, match="unrecognized"):
        association.load_frozen_model("changed.pt", 1)


def test_physical_world_legacy_yaw_dimensions_rotation_velocity_and_covariance():
    theta = .6
    row = np.array([[np.cos(theta), np.sin(theta), 0.],
                    [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    state = np.array([3., 2., 1., 2., 5., 3., .2, 4., -1.])
    a = np.arange(81).reshape(9, 9) / 200
    cov = a @ a.T + np.eye(9)
    f = frame(states=[state], covariance=cov[None], rotation=row, translation=np.array([10., 20., 3.]))
    actual, uncertainty = tracking.physical_world(f, [0], 1_000_000)
    def physical(x):
        result = x.copy()
        result[:3] = x[:3] @ row + [10., 20., 3.]
        result[[3, 4]] = x[[4, 3]]
        result[6] = wrap_angle(-x[6] - np.pi / 2 + theta)
        result[7:9] = x[7:9] @ row[:2, :2]
        return result
    np.testing.assert_allclose(actual[0], physical(state), atol=1e-12)
    eps, jac = 1e-5, np.zeros((9, 9))
    for i in range(9):
        offset = np.eye(9)[i] * eps
        jac[:, i] = (physical(state + offset) - physical(state - offset)) / (2 * eps)
    np.testing.assert_allclose(uncertainty[0], jac @ cov @ jac.T, atol=1e-8)
    future, future_cov = tracking.physical_world(f, [0], 1_500_000)
    np.testing.assert_allclose(future[0, :2], actual[0, :2] + actual[0, 7:9] * .5)
    assert np.trace(future_cov[0]) > np.trace(uncertainty[0])


def test_ci_duplicate_information_preserves_covariance_and_wraps_yaw():
    mean = np.zeros(9); mean[3:6] = [4., 2., 2.]; mean[6] = np.pi - .01
    cov = np.diag(np.arange(1., 10.))
    equal, equal_cov = tracking.ci(mean, cov, mean, cov)
    np.testing.assert_allclose(equal, mean, atol=1e-12)
    np.testing.assert_allclose(equal_cov, cov, atol=1e-12)
    other = mean.copy(); other[6] = -np.pi + .01
    wrapped, _ = tracking.ci(mean, cov, other, cov)
    assert abs(wrapped[6]) == pytest.approx(np.pi)


def test_mixture_keeps_between_hypothesis_variance_and_circular_mean():
    means = np.zeros((2, 9)); means[:, 3:6] = [4., 2., 2.]
    means[:, 0] = [0., 4.]; means[:, 6] = [np.pi-.1, -np.pi+.1]
    mixed, cov = tracking.moment_match(means, np.tile(np.eye(9), (2, 1, 1)), [.5, .5])
    assert mixed[0] == pytest.approx(2.) and cov[0, 0] == pytest.approx(5.)
    assert abs(mixed[6]) == pytest.approx(np.pi)
    assert cov[6, 6] == pytest.approx(1.01)
    with pytest.raises(ValueError, match="mixture mass"):
        tracking.moment_match(means, np.tile(np.eye(9), (2, 1, 1)), [.1, .2])


def test_future_image_excluded_before_features_deadline_inclusive():
    model = FixedModel()
    t = tracker(model)
    result, _ = t.step(frame(), frame("infrastructure-side", image_us=1_100_001))
    assert result["source_available"] == [True, False]
    assert result["selected_detections"] == [1, 0]
    assert model.inputs[0][1].shape == (1, 0, 203)
    allowed, _ = tracker().step(frame(), frame("infrastructure-side", image_us=1_100_000))
    assert allowed["source_available"] == [True, True]
    blocked, _ = tracker().step(frame(image_us=1_100_001), frame("infrastructure-side", image_us=1_100_001))
    assert blocked["selected_detections"] == [0, 0] and blocked["predictions"] == []


def test_continuous_track_id_missed_detection_retention_then_expiry():
    t = tracker()
    first, _ = t.step(frame(), frame("infrastructure-side", states=[]))
    moved = frame(number=2, time_us=1_100_000, states=[[.1, 0., 1., 2., 4., 2., -np.pi/2, 0., 0.]])
    second, _ = t.step(moved, frame("infrastructure-side", number=2, time_us=1_100_000, states=[]))
    tid = first["predictions"][0]["track_id"]
    assert [p["track_id"] for p in second["predictions"]] == [tid]
    empty, _ = t.step(frame(number=3, time_us=1_200_000, states=[]),
                      frame("infrastructure-side", number=3, time_us=1_200_000, states=[]))
    assert [p["track_id"] for p in empty["predictions"]] == [tid]
    assert empty["predictions"][0]["score"] < second["predictions"][0]["score"]
    expired, _ = t.step(frame(number=4, time_us=3_200_001, states=[]),
                        frame("infrastructure-side", number=4, time_us=3_200_001, states=[]))
    assert expired["predictions"] == []
    assert first["previous_commit_sha256"] == "0" * 64
    assert second["previous_commit_sha256"] == first["commit_sha256"]
    assert empty["previous_commit_sha256"] == second["commit_sha256"]


def test_invalid_duplicate_cross_sequence_and_old_time_leave_commit_unchanged():
    t = tracker()
    t.step(frame(), frame("infrastructure-side", states=[]))
    digest, last, snapshots = t.commit_hash, t.last_reference_us, copy.deepcopy(t.tracks)
    cases = [
        (frame(), frame("infrastructure-side", states=[])),
        (frame(sequence="seq02", number=2, time_us=1_100_000), frame("infrastructure-side", sequence="seq02", number=2, time_us=1_100_000)),
        (frame(number=2, time_us=999_999), frame("infrastructure-side", number=2, time_us=999_999)),
        (frame(number=1, time_us=1_100_000), frame("infrastructure-side", number=1, time_us=1_100_000)),
    ]
    for pair in cases:
        with pytest.raises(ValueError):
            t.step(*pair)
        assert t.commit_hash == digest and t.last_reference_us == last
        assert t.tracks.keys() == snapshots.keys()
        for tid in snapshots:
            np.testing.assert_array_equal(t.tracks[tid]["mean"], snapshots[tid]["mean"])
            np.testing.assert_array_equal(t.tracks[tid]["cov"], snapshots[tid]["cov"])
            assert t.tracks[tid]["score"] == snapshots[tid]["score"]


def test_failed_numerical_update_is_atomic_and_retry_matches_clean_run(monkeypatch):
    t, clean = tracker(), tracker()
    for item in [t, clean]:
        item.step(frame(), frame("infrastructure-side", states=[]))
    pair = (frame(number=2, time_us=1_500_000),
            frame("infrastructure-side", number=2, time_us=1_500_000, states=[]))
    committed, last, state = t.commit_hash, t.last_reference_us, copy.deepcopy(t.tracks)
    original_ci = tracking.ci
    def fail(*args, **kwargs):
        raise FloatingPointError("injected update failure")
    monkeypatch.setattr(tracking, "ci", fail)
    with pytest.raises(FloatingPointError, match="injected"):
        t.step(*pair)
    assert t.commit_hash == committed and t.last_reference_us == last
    for tid in state:
        np.testing.assert_array_equal(t.tracks[tid]["mean"], state[tid]["mean"])
        np.testing.assert_array_equal(t.tracks[tid]["cov"], state[tid]["cov"])
        assert t.tracks[tid]["score"] == state[tid]["score"]
    monkeypatch.setattr(tracking, "ci", original_ci)
    assert t.step(*pair) == clean.step(*pair)


def test_tracker_rejects_trainable_models_and_outputs_are_detached_from_live_state():
    with pytest.raises(ValueError, match="frozen"):
        tracker(association.PredictedAssociation())
    t = tracker()
    result, _ = t.step(frame(), frame("infrastructure-side", states=[]))
    saved = copy.deepcopy(result)
    t.step(frame(number=2, time_us=1_100_000),
           frame("infrastructure-side", number=2, time_us=1_100_000, states=[]))
    assert result == saved


def test_mutating_returned_identity_record_cannot_change_live_track():
    t = tracker()
    result, _ = t.step(frame(), frame("infrastructure-side", states=[]))
    tid = result["predictions"][0]["track_id"]
    result["predictions"][0]["identity_hypotheses"][0]["probability"] = .001
    assert t.tracks[tid]["identity_hypotheses"][0]["probability"] == 1.
