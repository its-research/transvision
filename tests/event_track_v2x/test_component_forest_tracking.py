from dataclasses import replace

import pytest
import torch

from transvision.models.event_track_v2x.forest_components import validate_component_snapshot
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer, LearnedForestScorer
from transvision.models.event_track_v2x.forest_tracking import CausalForestTracker, ForestTrackingConfig
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from test_forest_tracking import observation, step


def tracker(**options):
    return CausalForestTracker(sequence_id='0003', start_us=1_000_000,
        scorer=GeometryForestScorer(birth_logit=-20.), config=ForestTrackingConfig(**{**dict(
            component_mode=True, max_nodes=512, max_component_nodes=128,
            active_limit=4, expansion_budget=2000, max_model_regret=1.), **options}))


def cars(number=2):
    return tuple(observation(f'{side}-{i}', i*30.+side*.2, source=side, index=i,
        arrival_us=1_000_010+side*10) for side in (0, 1) for i in range(number))


def test_two_components_equal_exhaustive_monolithic_geometry_output():
    t = tracker()
    original = CausalForestTracker(sequence_id=t.sequence_id, start_us=t.start_us, scorer=t.scorer,
                                  config=replace(t.config, component_mode=False))
    partitioned = step(t, cars())
    monolithic = step(original, cars())
    assert partitioned.prediction == monolithic.prediction
    assert len(partitioned.prediction['predictions']) == 2
    assert len(partitioned.audit['forest']['components']) == 2
    assert not partitioned.audit['decoded_action']['global_active_cartesian_product_materialized']
    assert partitioned.audit['output_model_regret_estimate'] == pytest.approx(0.)
    validate_component_snapshot(t.bank.factors, t.bank.commits[-1])


def test_more_than_128_global_nodes_with_two_node_local_limits_and_shared_work():
    t = tracker(max_component_nodes=2)
    # This explicitly exercises a scene larger than the previous global cap.
    result = step(t, cars(80))
    assert len(t.observations) == 160
    assert len(result.prediction['predictions']) == 80
    assert len(result.audit['forest']['components']) == 80
    assert all(len(c['component']['indices']) == 2 for c in result.audit['forest']['components'])
    assert result.audit['forest']['expansions'] == 160
    assert result.audit['forest']['expansions'] <= t.config.expansion_budget
    assert result.audit['decoded_action']['action_expansions'] <= t.config.action_budget
    assert len(result.audit['branches']) == 160  # Not 2**80 global hypotheses.


def test_contextual_network_is_scored_locally_under_its_own_token_cap():
    torch.manual_seed(1337)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).eval()
    model.requires_grad_(False)
    scorer = LearnedForestScorer(model, max_nodes=2, max_pairs=4)
    t = CausalForestTracker(sequence_id='0003', start_us=1_000_000, scorer=scorer,
        config=tracker().config)
    result = step(t, cars(3))
    assert len(t.observations) == 6
    assert result.audit['decoded_action']['potential_context'] == 'component'
    assert len(result.audit['forest']['components']) == 3
    assert result.audit['scorer_configuration_bound']


def test_neural_monolithic_baseline_can_use_identical_component_scored_potentials():
    torch.manual_seed(3407)
    model = RecoverableIdentityModel(hidden=8, heads=2, dropout=0.).eval()
    model.requires_grad_(False)
    scorer = LearnedForestScorer(model, max_nodes=2, max_pairs=4)
    config = replace(tracker().config, potential_context='component')
    partitioned = CausalForestTracker(sequence_id='0003', start_us=1_000_000, scorer=scorer, config=config)
    monolithic = CausalForestTracker(sequence_id='0003', start_us=1_000_000, scorer=scorer,
                                    config=replace(config, component_mode=False))
    first, other = step(partitioned, cars()), step(monolithic, cars())
    assert first.audit['raw_local_factors'] == other.audit['raw_local_factors']
    assert first.audit['potential_context'] == other.audit['potential_context'] == 'component'
    assert first.prediction == other.prediction


def test_bridge_merges_original_factors_but_does_not_rewrite_old_output():
    t = tracker()
    first = step(t, [observation('a', 0., index=0), observation('b', 12., index=1)])
    first_bytes = first.prediction_json
    bridge = observation('bridge', 6., source=1, state_us=1_100_000)
    second = step(t, [bridge], reference=1_100_000, event='bridge')
    record = second.audit['forest']['components'][0]
    assert len(second.audit['forest']['components']) == 1 and record['merge_restart']
    assert len(record['predecessors']) == 2
    assert first.prediction_json == first_bytes
    assert second.prediction['previous_commit_sha256'] == first.prediction['commit_sha256']
    assert all(len(branch['parent_branches']) == 2 for branch in second.audit['branches'])
    # Two same-source/same-frame cars can never collapse into one identity,
    # even when the other source provides an ambiguous connecting observation.
    assert len(second.prediction['predictions']) >= 2


def test_insufficient_budget_holds_prior_ids_and_keeps_entire_merged_support():
    t = tracker(expansion_budget=0, max_model_regret=0.)
    first = step(t, [observation('a', 0., index=0), observation('b', 12., index=1)])
    second = step(t, [observation('bridge', 6., source=1, state_us=1_100_000)],
                  reference=1_100_000, event='bridge')
    assert len(second.prediction['predictions']) == 3
    assert {p['track_id'] for p in first.prediction['predictions']} <= {
        p['track_id'] for p in second.prediction['predictions']}
    assert second.audit['forest']['components'][0]['posterior']['frontier'] == [[]]
    decision = second.audit['decoded_action']['components'][0]
    assert decision['fallback_bound_is_trivial_loss_range']
    assert second.audit['output_model_regret_estimate'] == 1.
    assert not second.audit['true_posterior_or_tracking_metric_bound']


def test_large_component_rejected_before_neural_scoring_without_silent_filtering(monkeypatch):
    t = tracker(max_component_nodes=2)
    first = step(t, [observation('a', 0., index=0), observation('b', 12., index=1)])
    monkeypatch.setattr(t, '_score', lambda *a: pytest.fail('over-capacity component scored'))
    with pytest.raises(ValueError, match='component capacity'):
        step(t, [observation('bridge', 6., source=1, state_us=1_100_000)],
             reference=1_100_000, event='bridge')
    assert t.commits == (first,) and len(t.observations) == 2
    assert len(t.bank.commits[-1].components) == 2


def test_component_state_replay_failure_is_atomic_and_retry_matches_fresh(monkeypatch):
    import transvision.models.event_track_v2x.forest_tracking as module
    t = tracker()
    original = module.replay_forest_states
    calls = [0]
    def fail_late(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 5:
            raise ValueError('late state replay failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'replay_forest_states', fail_late)
    with pytest.raises(ValueError, match='late state'):
        step(t, cars())
    assert t.observations == () and t.bank.commits == () and t.commits == ()
    monkeypatch.setattr(module, 'replay_forest_states', original)
    result = step(t, cars())
    assert result == step(tracker(), cars())


def test_component_window_is_still_explicit_not_silent_history_expiration():
    t = tracker(window_us=100)
    first = step(t, cars(), decision=1_000_100)
    with pytest.raises(ValueError, match='window expired'):
        step(t, [], reference=1_000_101, event='expired')
    assert t.commits == (first,)


def test_empty_component_scene_and_event_idempotence():
    t = tracker()
    empty = step(t)
    assert empty.prediction['predictions'] == []
    assert empty.audit['output_model_regret_estimate'] == 0.
    assert step(t) is empty
    following = step(t, cars(), reference=1_100_000, event='arrived')
    assert len(following.prediction['predictions']) == 2
    assert step(t) is empty


def test_tracking_append_under_frontier_pressure_keeps_raw_history_and_prior_receipt():
    t = tracker(active_limit=1, max_frontier=5)
    initial = tuple(observation(str(i), .1*i, state_us=1_000_000+i*10_000) for i in range(3))
    first = step(t, initial, reference=1_020_000)
    assert len(next(iter(t.bank.banks.values())).frontier) == 5
    old_bytes = first.prediction_json
    after = step(t, [observation('next', .3, state_us=1_030_000)], reference=1_030_000, event='next')
    assert after.audit['forest']['components'][0]['restart_reason'] == 'append_frontier_pressure'
    assert len(t.observations) == 4
    assert first.prediction_json == old_bytes
    assert after.prediction['previous_commit_sha256'] == first.prediction['commit_sha256']
