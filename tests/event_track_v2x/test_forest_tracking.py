from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x.forest_tracking import (
    CausalForestTracker, ForestTrackingConfig, RawIdentityDetection, cache_detections,
    replay_forest_states,
)
from transvision.models.event_track_v2x.forest_potentials import (
    GeometryForestScorer, LearnedForestScorer, neural_parent_logits, parent_supervision_loss,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode, validate_snapshot
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.tracking_evaluation_v2 import validate_predictions
from test_detection_cache_v2 import _frame


def observation(name, x=0., *, source=0, state_us=1_000_000, arrival_us=None, frame=None, index=0, score=.8):
    features = np.zeros(203)
    features[0], features[138], features[200], features[201] = x/100, 1, score, score
    return RawIdentityDetection('0003', IdentityNode(name, source, state_us,
        arrival_us if arrival_us is not None else state_us+10, frame or str(state_us)), index, state_us,
        [x, 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9)*.2, score, features, 'a'*64)


def tracker(**options):
    return CausalForestTracker(sequence_id='0003', start_us=0,
        scorer=GeometryForestScorer(birth_logit=-20),
        config=ForestTrackingConfig(**dict(active_limit=8, expansion_budget=100, max_model_regret=1., **options)))


def step(t, observations=(), *, reference=1_000_000, decision=None, event='first'):
    return t.step(observations, frame_id=event, reference_us=reference,
                  decision_us=decision or reference+100_000, event_id=event)


def test_birth_branch_replay_and_existence_are_not_posterior_moment_matching():
    t = tracker()
    a, b = observation('left', 0., score=.9), observation('right', .4, source=1, score=.6)
    c = step(t, [a, b])
    assert len(c.prediction['predictions']) == 1
    result = c.prediction['predictions'][0]
    assert result['mean'][0] == pytest.approx(.2)
    assert result['score'] == .9
    np.testing.assert_allclose(result['covariance'], np.eye(9)*.2)
    assert {len(branch['tracks']) for branch in c.audit['branches']} == {1, 2}
    assert c.audit['output_model_regret_estimate'] == 0.
    assert c.audit['raw_local_factors']['nodes'][0]['node_id'] == 'left'
    assert all('parent_branch_id' in branch for branch in c.audit['branches'])
    validate_snapshot(t.bank.factors, t.bank.commits[-1])


def test_temporal_identity_continues_and_late_evidence_replays_current_state_only():
    t = tracker(gate_distance_m=20.)
    first = step(t, [observation('current', 1.)])
    saved = first.prediction_json
    late = observation('late', 0., source=1, state_us=900_000, arrival_us=1_200_000)
    second = step(t, [late], reference=1_150_000, decision=1_250_000, event='late')
    assert second.prediction['predictions'][0]['track_id'] == first.prediction['predictions'][0]['track_id']
    # Independent information-form CI reference, including CV/process noise
    # between the two source state times and final propagation to output time.
    f = np.eye(9); f[0, 7] = f[1, 8] = .1
    earlier_cov = f @ (np.eye(9)*.2) @ f.T + np.eye(9)*.01
    later_cov = np.eye(9)*.2
    covariance = np.linalg.inv(.5*np.linalg.inv(earlier_cov)+.5*np.linalg.inv(later_cov))
    mean = covariance @ (.5*np.linalg.solve(earlier_cov, late.mean)
                         + .5*np.linalg.solve(later_cov, t.observations[0].mean))
    output_f = np.eye(9); output_f[0, 7] = output_f[1, 8] = .15
    np.testing.assert_allclose(second.prediction['predictions'][0]['mean'], output_f @ mean)
    assert first.prediction_json == saved
    assert first.prediction['predictions'][0]['mean'][0] == 1.
    assert second.prediction['previous_commit_sha256'] == first.prediction['commit_sha256']


def test_raw_detections_from_one_source_frame_never_share_an_identity():
    t = tracker()
    same_source = [observation('a', 0., index=0), observation('b', .01, index=1)]
    c = step(t, same_source)
    assert len(c.prediction['predictions']) == 2
    assert len({p['track_id'] for p in c.prediction['predictions']}) == 2
    assert t.support == ((-1,), (-1,))


def test_low_detection_score_suppresses_birth_but_does_not_delete_identity_factors():
    t = tracker()
    c = step(t, [observation('low', score=.1)])
    assert c.prediction['predictions'] == [] and c.audit['suppressed']
    assert len(t.bank.factors.nodes) == 1 and len(t.bank.commits[-1].active) == 1


def test_no_detection_frames_are_explicit_and_tracks_age_out():
    t = tracker(max_age_us=100_000)
    first = step(t, [observation('a')])
    second = step(t, reference=1_050_000, event='missed')
    third = step(t, reference=1_200_000, event='expired')
    assert first.prediction['predictions'][0]['track_id'] == second.prediction['predictions'][0]['track_id']
    assert second.prediction['predictions'][0]['score'] < first.prediction['predictions'][0]['score']
    assert third.prediction['predictions'] == [] and third.audit['removed_output_ids']
    assert len(t.observations) == 1  # Aging outputs is not deleting recoverable factors.


def test_empty_scene_has_a_valid_prediction_commit():
    t = tracker()
    c = step(t)
    assert c.prediction['predictions'] == []
    assert c.audit['output_model_regret_estimate'] == 0.


def test_zero_search_budget_outputs_unmatched_without_claiming_a_risk_bound():
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=GeometryForestScorer(),
        config=ForestTrackingConfig(expansion_budget=0))
    c = step(t, [observation('a'), observation('b', source=1)])
    assert len(c.prediction['predictions']) == 2
    assert c.audit['output_model_regret_estimate'] is None
    assert c.audit['forest']['frontier'] == [[]]
    assert c.audit['forest']['eta_upper'] == 1.
    assert c.audit['output_parents'] == [-1, -1]


def test_duplicate_events_and_observations_are_idempotent_and_detached():
    t = tracker()
    a = observation('a')
    first = step(t, [a, a])
    assert first.audit['new_observations'] == 1 and first.audit['duplicate_observations'] == 1
    assert step(t, [a, a]) is first
    second = step(t, [a], reference=1_200_000, event='again')
    assert second.audit['new_observations'] == 0
    assert step(t, [a, a]) is first  # Retry original after a later commit.
    result = first.prediction
    result['predictions'][0]['mean'][0] = 999
    assert first.prediction['predictions'][0]['mean'][0] == 0.
    assert len(t.commits) == 2


@pytest.mark.parametrize('kind', ['future', 'sequence', 'conflict', 'alias', 'old_reference', 'late_order'])
def test_invalid_updates_leave_original_state_and_prediction_unchanged(kind):
    t = tracker()
    a = observation('a')
    first = step(t, [a])
    before = (t.observations, t.support, t.bank, t.commits)
    new, reference = [observation('b', source=1, state_us=1_200_000)], 1_200_000
    if kind == 'future':
        new = [replace(new[0], node=replace(new[0].node, arrival_us=1_400_000))]
    elif kind == 'sequence':
        new = [replace(new[0], sequence_id='other')]
    elif kind == 'conflict':
        new = [replace(a, score=.5)]
    elif kind == 'alias':
        new = [replace(a, node=replace(a.node, node_id='alias'))]
    elif kind == 'old_reference':
        reference = 1_000_000
    elif kind == 'late_order':
        new = [observation('b', source=1, state_us=900_000, arrival_us=950_000)]
    with pytest.raises(ValueError):
        step(t, new, reference=reference, event='bad')
    assert (t.observations, t.support, t.bank, t.commits) == before
    assert t.commits[0] is first


def test_replay_failure_is_atomic_and_retry_matches_fresh_tracker(monkeypatch):
    import transvision.models.event_track_v2x.forest_tracking as module
    t = tracker()
    first = step(t, [observation('a')])
    real = module.replay_forest_states
    def fail(*args, **kwargs):
        raise ValueError('simulated numerical failure')
    monkeypatch.setattr(module, 'replay_forest_states', fail)
    with pytest.raises(ValueError, match='simulated'):
        step(t, [observation('b', source=1, state_us=1_200_000)], reference=1_200_000, event='second')
    assert t.commits == (first,) and len(t.bank.commits) == 1 and len(t.observations) == 1
    monkeypatch.setattr(module, 'replay_forest_states', real)
    second = step(t, [observation('b', source=1, state_us=1_200_000)], reference=1_200_000, event='second')
    fresh = tracker()
    step(fresh, [observation('a')])
    assert second == step(fresh, [observation('b', source=1, state_us=1_200_000)], reference=1_200_000, event='second')


def test_limits_fail_before_scoring_or_destroying_original_history():
    calls = []
    def scorer(*args):
        calls.append(args)
        return GeometryForestScorer()(*args)
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=scorer,
                           config=ForestTrackingConfig(max_nodes=1, max_commits=1))
    with pytest.raises(ValueError, match='before scoring'):
        step(t, [observation('a'), observation('b', source=1)])
    assert not calls and not t.commits
    first = step(t, [observation('a')])
    with pytest.raises(ValueError, match='commit budget'):
        step(t, reference=1_200_000, event='next')
    with pytest.raises(ValueError, match='window expired'):
        step(t, reference=3_000_000, event='outside')
    assert t.commits == (first,)


def test_previous_identity_can_be_restored_without_changing_old_output():
    def scorer(obs, support, decision):
        target = (-1, -1, 0, 1) if decision < 1_400_000 else (-1, -1, 1, 0)
        return ForestFactors(tuple(o.node for o in obs), tuple(tuple((p, 10. if p == target[i] else -20.)
                             for p in parents) for i, parents in enumerate(support)))
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=scorer,
        config=ForestTrackingConfig(active_limit=1, expansion_budget=2, max_model_regret=1., gate_distance_m=20.))
    first = step(t, [observation('a', 0.), observation('b', 10., index=1)])
    second = step(t, [observation('c', 2., source=1, state_us=1_200_000),
                     observation('d', 8., source=1, state_us=1_200_000, index=1)],
                  reference=1_200_000, event='ambiguous')
    saved = second.prediction_json
    third = step(t, reference=1_400_000, event='rescore')
    assert second.audit['output_parents'] == [-1, -1, 0, 1]
    assert third.audit['output_parents'] == [-1, -1, 1, 0]
    assert third.audit['forest']['never_enumerated_recoveries'] == [[-1, -1, 1, 0]]
    assert second.prediction_json == saved
    assert {p['track_id'] for p in first.prediction['predictions']} == {p['track_id'] for p in third.prediction['predictions']}


def test_prediction_output_uses_existing_evaluator_format_and_commit_chain():
    t = tracker()
    records = [step(t, [observation('a')]).prediction, step(t, reference=1_200_000, event='next').prediction]
    gt = [{k: p[k] for k in ('sequence_id', 'frame_id', 'box_reference_timestamp_us')} | {'objects': []} for p in records]
    validate_predictions(records, gt)


def test_newly_arrived_evidence_can_restore_an_unenumerated_old_prefix():
    def scorer(obs, support, decision):
        target = (-1, -1, 0, 1) if len(obs) < 5 else (-1, -1, 1, 0, -1)
        return ForestFactors(tuple(o.node for o in obs), tuple(tuple((p, 10. if p == target[i] else -20.)
                             for p in parents) for i, parents in enumerate(support)))
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=scorer,
        config=ForestTrackingConfig(active_limit=1, expansion_budget=2, max_model_regret=1., gate_distance_m=20.))
    step(t, [observation('a', 0.), observation('b', 10., index=1)])
    old = step(t, [observation('c', 2., source=1, state_us=1_200_000),
                  observation('d', 8., source=1, state_us=1_200_000, index=1)],
               reference=1_200_000, event='ambiguous')
    assert (-1, -1, 1, 0) not in t.bank.discovered
    late = observation('new-evidence', 5., state_us=1_250_000, arrival_us=1_450_000)
    recovered = step(t, [late], reference=1_400_000, decision=1_500_000, event='arrival')
    assert recovered.audit['forest']['restored_ancestral_prefixes'] == [[-1, -1, 1, 0]]
    assert recovered.audit['output_parents'] == [-1, -1, 1, 0, -1]
    assert recovered.audit['forest']['never_enumerated_recoveries'] == []
    assert old.audit['output_parents'] == [-1, -1, 0, 1]


def test_cache_adapter_uses_only_car_and_preserves_image_information_time():
    f = replace(_frame(3), class_indices=np.array([0, 1, 2]))
    detections = cache_detections(f, arrival_us=1_150_000, decision_us=1_200_000, origin_us=0)
    assert len(detections) == 1
    d = detections[0]
    assert d.detection_index == 0 and d.state_us == 1_000_000 and d.node.information_us == 1_100_000
    assert d.node.arrival_us == 1_150_000 and len(d.features) == 203
    np.testing.assert_allclose(d.features[10:138], f.appearance[0])
    assert d.mean[3:6] == (2., 4., 1.)  # Invert the legacy width/length convention.
    assert cache_detections(_frame(0), arrival_us=1_100_000, decision_us=1_100_000, origin_us=0) == ()


def test_future_cache_is_rejected_before_physical_state_or_feature_access(monkeypatch):
    import transvision.models.event_track_v2x.forest_tracking as module
    def forbidden(*args):
        pytest.fail('future arrays were accessed')
    monkeypatch.setattr(module, 'physical_world', forbidden)
    with pytest.raises(ValueError, match='future'):
        cache_detections(_frame(), arrival_us=1_090_000, decision_us=1_090_000, origin_us=0)


def test_neural_potentials_affect_joint_history_and_train_without_target_inputs():
    torch.manual_seed(17)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.)
    obs = (observation('a'), observation('b', .2, source=1))
    support = ((-1,), (-1, 0))
    rows = neural_parent_logits(model, obs, support, 1_100_000)
    loss = parent_supervision_loss(rows, support, (-1, 0))
    loss['loss'].backward()
    for module in (model.encoder, model.pair_scorer, model.temporal_scorer, model.temporal_attention):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())
    model.requires_grad_(False).eval()
    scorer = LearnedForestScorer(model)
    f = scorer(obs, support, 1_100_000)
    assert all(sum(np.exp(w) for _, w in row) == pytest.approx(1.) for row in f.rows)
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=scorer,
        config=ForestTrackingConfig(active_limit=4, max_model_regret=1.))
    c = step(t, obs)
    assert c.audit['scorer_signature'] == scorer.signature
    assert c.audit['scorer_configuration_bound']
    # This test does not produce or claim a real trained checkpoint.


def test_untrained_model_mutation_and_neural_budget_fail_before_state_update(monkeypatch):
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).requires_grad_(False).eval()
    scorer = LearnedForestScorer(model, max_nodes=1)
    def forbidden(*args, **kwargs):
        pytest.fail('neural forward ran despite an exceeded budget')
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError, match='budget'):
        scorer((observation('a'), observation('b', source=1)), ((-1,), (-1, 0)), 1_100_000)
    with torch.no_grad():
        next(model.parameters()).add_(1.)
    with pytest.raises(ValueError, match='model changed'):
        scorer((observation('a'),), ((-1,),), 1_100_000)


def test_supervision_outside_support_is_not_mislabeled_as_birth():
    rows = (torch.tensor([0.], requires_grad=True), torch.tensor([0., 1.], requires_grad=True))
    support = ((-1,), (-1, 0))
    result = parent_supervision_loss(rows, support, (-1, None))
    assert result['positive_outside_support_rows'] == 1 and result['supervised_rows'] == 1
    with pytest.raises(ValueError, match='no supervised'):
        parent_supervision_loss(rows, support, (None, None))
    with pytest.raises(ValueError, match='outside support'):
        parent_supervision_loss(rows, support, (-1, 10))


def test_cache_through_neural_forest_to_existing_evaluation_format():
    torch.manual_seed(23)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).requires_grad_(False).eval()
    t = CausalForestTracker(sequence_id='0003', start_us=0, scorer=LearnedForestScorer(model),
        config=ForestTrackingConfig(active_limit=4, expansion_budget=100, max_model_regret=1.))
    frames = [replace(_frame(2, side=side), class_indices=np.array([0, 2]))
              for side in ('vehicle-side', 'infrastructure-side')]
    inputs = tuple(d for i, frame in enumerate(frames) for d in cache_detections(frame,
        arrival_us=1_150_000+i*10_000, decision_us=1_200_000, origin_us=0))
    c = t.step(inputs, frame_id='official-frame', reference_us=1_000_000,
               decision_us=1_200_000, event_id='cache-tick')
    assert c.audit['new_observations'] == 2 and len(c.audit['source_cache_sha256']) == 2
    assert c.prediction['predictions'] and all(p['class_label'] == 'car' for p in c.prediction['predictions'])
    validate_predictions([c.prediction], [{'sequence_id': '0003', 'frame_id': 'official-frame',
                                           'box_reference_timestamp_us': 1_000_000, 'objects': []}])


def test_changed_scorer_configuration_cannot_silently_change_the_method():
    t = tracker()
    first = step(t, [observation('a')])
    t.scorer.birth_logit = -1.
    with pytest.raises(ValueError, match='scorer configuration'):
        step(t, reference=1_200_000, event='changed')
    assert t.commits == (first,)
