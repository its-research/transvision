from dataclasses import replace
import math

import pytest

from transvision.models.event_track_v2x.covered_completion_tracking import (
    PersistentCoveredCompletionConfig, CoveredCompletionTracker, CoveredCompletionTeacher, CoveredCompletionLearned,
)
from transvision.models.event_track_v2x.completion_component_tracking import CompletionComponentTracker, PersistentCompletionConfig
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from test_covered_proposal_capacity import inputs
from test_persistent_component_tracking import step, joint_action
from test_persistent_forest import brute
from test_learned_component_allocation import policy
from test_forest_tracking import observation


@pytest.mark.parametrize('cls', [CoveredCompletionTracker, CoveredCompletionTeacher, CoveredCompletionLearned])
@pytest.mark.parametrize('seed', range(3))
def test_covered_admission_is_budgeted_legal_and_resumable_without_rewriting_old_ids(tmp_path, cls, seed):
    cfg = PersistentCoveredCompletionConfig(state=ForestTrackingConfig(
        active_limit=2, expansion_budget=4, max_frontier=1), max_total_frontier=1)
    options = dict(allocation_policy=policy(seed)) if cls is CoveredCompletionLearned else {}
    path = tmp_path/'tracker.sqlite'
    tracker = cls(path, sequence_id='0003', config=cfg, **options)
    raw, rows = inputs(seed)
    first = step(tracker, raw, rows)
    assert first.audit['search_steps'] == 4
    assert first.audit['coverage_aware_proposal_admission']
    assert first.audit['total_frontier'] == 1
    assert first.audit['allocation_trace'][0]['work_kind'] == 'covered_history_suffix_proposal'
    result = step(tracker, time=1_200_000, event='second')
    assert result.audit['search_steps'] <= 4
    probabilities, roots, z = brute(ForestFactors(tuple(o.node for o in raw), rows))
    chosen = joint_action(tracker, result)
    def risk(action):
        return math.fsum(p*sum(a != b for a, b in zip(roots[action], roots[h]))/len(raw)
                         for h, p in probabilities.items())
    assert risk(chosen)-min(risk(h) for h in probabilities) <= result.audit['model_regret_upper']+1e-12
    assert math.log(z) <= result.audit['log_partition_upper']+1e-12
    assert step(tracker, raw, rows).prediction_json == first.prediction_json
    digest = tracker.close()
    with pytest.raises(ValueError, match='schema'):
        CompletionComponentTracker.open(path, expected_prediction_sha256=result.prediction['commit_sha256'],
                                         expected_database_sha256=digest)
    reopened = cls.open(path, expected_prediction_sha256=result.prediction['commit_sha256'],
                        expected_database_sha256=digest, **options)
    assert step(reopened, raw, rows).prediction_json == first.prediction_json
    reopened.close()


def test_covered_config_requires_known_explicit_version():
    with pytest.raises(ValueError, match='unsupported coverage'):
        replace(PersistentCoveredCompletionConfig(), coverage_admission_version=2)
    with pytest.raises(ValueError, match='positive integer'):
        replace(PersistentCoveredCompletionConfig(), coverage_admission_version=True)


def test_full_frontier_does_not_block_confident_temporal_identity_for_120_events(tmp_path):
    state = ForestTrackingConfig(active_limit=2, expansion_budget=1, max_frontier=1)
    covered = CoveredCompletionTracker(tmp_path/'covered.sqlite', sequence_id='0003',
        config=PersistentCoveredCompletionConfig(state=state, max_total_frontier=1))
    control = CompletionComponentTracker(tmp_path/'control.sqlite', sequence_id='0003',
        config=PersistentCompletionConfig(state=state, max_total_frontier=1))
    identity, control_ids, first = None, set(), None
    for index in range(120):
        time = 1_000_000+index*10_000
        raw = observation(str(index), x=index*.001, state_us=time, arrival_us=time)
        row = ((-1, -20.),) if index == 0 else ((-1, -20.), (index-1, 0.))
        result = step(covered, (raw,), (row,), time=time, event=str(index))
        baseline = step(control, (raw,), (row,), time=time, event=str(index))
        assert result.audit['search_steps'] <= 1 and result.audit['total_frontier'] <= 1
        assert len(result.prediction['predictions']) == 1
        current = result.prediction['predictions'][0]['track_id']
        identity = identity or current
        assert current == identity
        control_ids.update(prediction['track_id'] for prediction in baseline.prediction['predictions'])
        first = first or result.prediction_json
    assert len(control_ids) == 120  # Conservative reserve blocks both search operations at capacity one.
    raw = observation('0', state_us=1_000_000, arrival_us=1_000_000)
    assert step(covered, (raw,), (((-1, -20.),),), time=1_000_000, event='0').prediction_json == first
    digest = covered.close()
    reopened = CoveredCompletionTracker.open(tmp_path/'covered.sqlite',
        expected_prediction_sha256=result.prediction['commit_sha256'], expected_database_sha256=digest)
    assert reopened.meta['events'] == 120
    reopened.close()
    control.close()
