from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import HypothesisBank, LogAssociationFactors, EvidenceUpdate
from transvision.models.event_track_v2x.recoverable_states import BranchStateWindow, TrackPrior, TrackletObservation


def state(x):
    return np.array([x, 0, 0, 4, 2, 1.5, 0, 0, 0.])


def setup(**limits):
    priors = [TrackPrior('id-A', 0, state(-3), np.eye(9), .7),
              TrackPrior('id-B', 0, state(3), np.eye(9), .8)]
    window = BranchStateWindow(component_id='scene-window', left_node_ids=('v0', 'v1'),
        right_node_ids=('r0', 'r1'), priors=priors, start_us=0, **limits)
    factors = LogAssociationFactors.from_positive([[1000., 1.], [1., 1000.]], [.01, .01], [.01, .01])
    return window, HypothesisBank(factors, active_limit=2), priors


def observation(message='one', side='left', node='v0', info=100, arrival=200, x=8):
    return TrackletObservation(message, side, node, info, arrival, state(x), np.eye(9), .9)


def test_input_arrays_copied_and_immutable():
    mean, cov = state(1), np.eye(9)
    value = TrackletObservation('id', 'left', 'v0', 1, 2, mean, cov, .8)
    mean[0] = 900
    cov[0, 0] = 100
    assert value.mean[0] == 1 and value.covariance[0][0] == 1
    with pytest.raises(FrozenInstanceError):
        value.mean = tuple(mean)


def test_duplicate_future_and_capacity_fail_closed():
    window, _, _ = setup(max_observations=1)
    obs = observation()
    with pytest.raises(ValueError, match='future'):
        window.ingest(obs, decision_us=199)
    assert window.ingest(obs, decision_us=200)
    assert not window.ingest(obs, decision_us=200)
    with pytest.raises(ValueError, match='conflicting'):
        window.ingest(replace(obs, mean=state(2)), decision_us=200)
    with pytest.raises(ValueError, match='double count'):
        window.ingest(replace(obs, message_id='alias'), decision_us=200)
    with pytest.raises(ValueError, match='budget'):
        window.ingest(observation('two', node='v1'), decision_us=200)
    assert window.observations == (obs,)


def test_each_identity_has_own_state_and_raw_existence():
    window, bank, priors = setup()
    window.ingest(observation(), decision_us=200)
    snapshot = bank.advance(decision_us=200, expansion_budget=20)
    output = window.commit(snapshot, reference_us=200)
    assert len(output.branches) == 2
    by_choice = {b.choices: {s.track_id: s for s in b.tracks} for b in output.branches}
    assert (0, 1) in by_choice and (1, 0) in by_choice
    assert by_choice[(0, 1)]['id-A'].mean[0] > priors[0].mean[0]
    assert by_choice[(1, 0)]['id-A'].mean[0] == priors[0].mean[0]
    assert by_choice[(1, 0)]['id-B'].existence_score == .9
    assert window.commit(snapshot, reference_us=200) is output


def test_recovered_identity_replays_original_states_without_history_rewrite():
    window, bank, _ = setup()
    window.ingest(observation(), decision_us=200)
    before = window.commit(bank.advance(decision_us=200, expansion_budget=2), reference_us=200)
    late = observation('late', node='v1', info=50, arrival=300, x=-8)
    window.ingest(late, decision_us=300)
    likelihood = LogAssociationFactors.from_positive([[1, 1e8], [1e8, 1]], [1, 1], [1, 1])
    identity = bank.advance(decision_us=300, expansion_budget=2, evidence=EvidenceUpdate('reverse', 300, likelihood))
    after = window.commit(identity, reference_us=300)
    assert after.branches[0].choices == (1, 0)
    assert window.commits[0] is before and before.branches[0].choices == (0, 1)
    assert after.previous_commit == before.commit
    # A fresh window with the same raw observations produces the same branch states.
    replay, _, _ = setup()
    replay.ingest(late, decision_us=300)
    replay.ingest(observation(), decision_us=300)
    replayed = replay.commit(identity, reference_us=300)
    assert replayed.branches == after.branches


def test_new_reference_cannot_regress_or_repeat_and_fork_rejected():
    window, bank, _ = setup()
    before = window.commit(bank.advance(decision_us=200, expansion_budget=2), reference_us=200)
    next_identity = bank.advance(decision_us=300, expansion_budget=0)
    for reference in (100, 200):
        with pytest.raises(ValueError, match='increasing'):
            window.commit(next_identity, reference_us=reference)
    with pytest.raises(ValueError, match='ancestry'):
        window.commit(replace(next_identity, previous_commit='a'*64), reference_us=300)
    assert window.commits == (before,)


def test_unmatched_ids_are_stable_and_no_weight_scaling():
    window, _, _ = setup()
    factors = LogAssociationFactors([[0., 0.], [0., 0.]], [0., 0.], [0., 0.], [[False, False], [False, False]])
    bank = HypothesisBank(factors)
    window.ingest(observation(), decision_us=200)
    first = window.commit(bank.advance(decision_us=200, expansion_budget=2), reference_us=200)
    second = window.commit(bank.advance(decision_us=300, expansion_budget=0), reference_us=300)
    a = [x for x in first.branches[0].tracks if x.track_id.startswith('unmatched:')]
    b = [x for x in second.branches[0].tracks if x.track_id.startswith('unmatched:')]
    assert len(a) == len(b) == 1 and a[0].track_id == b[0].track_id
    assert a[0].existence_score == b[0].existence_score == .9


@pytest.mark.parametrize('override', [{'mean': state(0) * float('nan')}, {'covariance': -np.eye(9)},
                                    {'existence_score': 2}, {'arrival_us': 0}])
def test_invalid_observation_rejected(override):
    values = dict(message_id='x', side='left', node_id='v0', information_us=1, arrival_us=2,
                  mean=state(0), covariance=np.eye(9), existence_score=.5)
    with pytest.raises(ValueError):
        TrackletObservation(**(values | override))


def test_numpy_scores_and_configuration_hash():
    obs = replace(observation(), existence_score=np.float32(.8))
    assert type(obs.existence_score) is float
    window, bank, priors = setup(ci_weight=np.float32(.5))
    window.ingest(obs, decision_us=200)
    first = window.commit(bank.advance(decision_us=200, expansion_budget=2), reference_us=200)
    window.ci_weight = .8
    with pytest.raises(ValueError, match='configuration'):
        window.commit(bank.advance(decision_us=300, expansion_budget=0), reference_us=300)
    assert window.commits == (first,)


def test_verified_intermediate_identity_search_commits():
    window, bank, _ = setup()
    first_identity = bank.advance(decision_us=200, expansion_budget=2)
    window.commit(first_identity, reference_us=200)
    intermediate = bank.advance(decision_us=300, expansion_budget=1)
    final = bank.advance(decision_us=300, expansion_budget=1)
    with pytest.raises(ValueError, match='ancestry'):
        window.commit(final, reference_us=300)
    result = window.commit(final, reference_us=300, identity_ancestors=(intermediate,))
    assert result.identity_ancestors == (intermediate.commit,)
