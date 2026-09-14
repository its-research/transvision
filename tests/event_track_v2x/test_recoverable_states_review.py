"""Independent conditional-state replay checks; synthetic fixed tracklets only."""
from dataclasses import FrozenInstanceError, asdict
import copy
import hashlib
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import HypothesisBank, LogAssociationFactors
from transvision.models.event_track_v2x.recoverable_states import (
    BranchStateWindow, TrackPrior, TrackletObservation,
)


START = 1_000_000


def mean(x, velocity=0.):
    return np.array([x, 0., 1., 2., 4., 2., 0., velocity, 0.])


def observation(message, side, node, x, information, arrival, score=.8):
    return TrackletObservation(message, side, node, information, arrival,
                               mean(x), np.eye(9) * .1, score)


def setup(max_observations=4096):
    factors = LogAssociationFactors(np.zeros((2, 2)), np.zeros(2), np.zeros(2))
    bank = HypothesisBank(factors, active_limit=7, information_us=START)
    priors = (TrackPrior("track-0", START, mean(0.), np.eye(9), .2),
              TrackPrior("track-1", START, mean(10.), np.eye(9), .3))
    window = BranchStateWindow(component_id="crossing", left_node_ids=("left-0", "left-1"),
        right_node_ids=("right-0", "right-1"), priors=priors, start_us=START,
        process_noise=0., max_observations=max_observations)
    return bank, window, priors


def branch_states(commit):
    return {branch.choices: {track.track_id: track for track in branch.tracks} for branch in commit.branches}


def test_competing_identity_branches_have_distinct_state_histories_and_no_shared_pollution():
    bank, window, priors = setup()
    original = copy.deepcopy(priors)
    first = observation("left-message-0", "left", "left-0", 0., START + 10_000, START + 20_000)
    second = observation("left-message-1", "left", "left-1", 10., START + 10_000, START + 20_000)
    window.ingest(first, decision_us=START + 100_000)
    window.ingest(second, decision_us=START + 100_000)
    snapshot = bank.advance(decision_us=START + 100_000, expansion_budget=10)
    commit = window.commit(snapshot, reference_us=START + 100_000)
    branches = branch_states(commit)
    assert branches[(0, 1)]["track-0"].mean[0] == pytest.approx(0.)
    assert branches[(0, 1)]["track-1"].mean[0] == pytest.approx(10.)
    assert branches[(1, 0)]["track-0"].mean[0] > 8.
    assert branches[(1, 0)]["track-1"].mean[0] < 2.
    assert priors == original and window.priors == original
    assert branches[(0, 1)]["track-0"] is not branches[(1, 0)]["track-0"]
    with pytest.raises(FrozenInstanceError):
        branches[(0, 1)]["track-0"].mean = tuple(mean(999.))
    with pytest.raises(TypeError):
        branches[(0, 1)]["track-0"].covariance[0][0] = 999.
    # Physical existence/detection evidence is not multiplied by identity mass.
    assert all(track.existence_score == .8 for key in ((0, 1), (1, 0)) for track in branches[key].values())


def test_unmatched_left_ids_are_stable_across_competing_branches():
    bank, window, _ = setup()
    window.ingest(observation("left0", "left", "left-0", 0., START + 10_000, START + 20_000),
                  decision_us=START + 100_000)
    snapshot = bank.advance(decision_us=START + 100_000, expansion_budget=10)
    branches = branch_states(window.commit(snapshot, reference_us=START + 100_000))
    first_ids = {key for key in branches[(-1, -1)] if key.startswith("unmatched:")}
    second_ids = {key for key in branches[(-1, 0)] if key.startswith("unmatched:")}
    assert len(first_ids) == 1 and first_ids == second_ids


def test_late_information_recomputes_from_originals_and_never_mutates_committed_history():
    bank, window, _ = setup()
    early = observation("early", "right", "right-0", 2., START + 80_000, START + 90_000)
    late = observation("late", "left", "left-0", 1., START + 40_000, START + 200_000)
    later = observation("later", "right", "right-1", 9., START + 150_000, START + 200_000)
    window.ingest(early, decision_us=START + 100_000)
    first_identity = bank.advance(decision_us=START + 100_000, expansion_budget=10)
    first = window.commit(first_identity, reference_us=START + 100_000)
    original_bytes = json.dumps(asdict(first), sort_keys=True).encode()
    # Make two copies with identical committed history, then vary only ingest order.
    reverse = copy.deepcopy(window)
    for target, messages in ((window, (late, later)), (reverse, (later, late))):
        for message in messages:
            assert target.ingest(message, decision_us=START + 300_000)
    second_identity = bank.advance(decision_us=START + 300_000, expansion_budget=0)
    second = window.commit(second_identity, reference_us=START + 300_000)
    other = reverse.commit(second_identity, reference_us=START + 300_000)
    assert second == other  # Information-time replay, not arrival-order filtering.
    assert json.dumps(asdict(window.commits[0]), sort_keys=True).encode() == original_bytes
    assert window.commits[0] is first
    assert second.previous_commit == first.commit
    assert second.observation_sha256 != first.observation_sha256
    first_track = branch_states(first)[(0, 1)]["track-0"]
    second_track = branch_states(second)[(0, 1)]["track-0"]
    assert first_track.reference_us < second_track.reference_us
    assert first_track.mean != second_track.mean
    assert window.commit(second_identity, reference_us=START + 300_000) is second


def test_source_duplicate_future_and_budget_errors_leave_evidence_unchanged():
    bank, window, _ = setup(max_observations=1)
    item = observation("original", "left", "left-0", 1., START + 10_000, START + 20_000)
    with pytest.raises(ValueError, match="future"):
        window.ingest(item, decision_us=START + 19_999)
    assert window.observations == ()
    assert window.ingest(item, decision_us=START + 20_000)
    assert not window.ingest(item, decision_us=START + 20_000)
    alias = observation("different-id", "left", "left-0", 1., START + 10_000, START + 20_000)
    with pytest.raises(ValueError, match="double count"):
        window.ingest(alias, decision_us=START + 20_000)
    extra = observation("extra", "right", "right-1", 9., START + 11_000, START + 20_000)
    with pytest.raises(ValueError, match="budget"):
        window.ingest(extra, decision_us=START + 20_000)
    assert window.observations == (item,)


def test_state_commit_digest_covers_every_branch_and_replays_identically():
    transcripts = []
    for _ in range(2):
        bank, window, _ = setup()
        snapshot = bank.advance(decision_us=START + 100_000, expansion_budget=10)
        window.ingest(observation("m", "left", "left-0", 2., START + 50_000, START + 80_000),
                      decision_us=START + 100_000)
        result = window.commit(snapshot, reference_us=START + 100_000)
        content = asdict(result)
        digest = content.pop("commit")
        assert hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == digest
        transcripts.append(asdict(result))
    assert transcripts[0] == transcripts[1]
