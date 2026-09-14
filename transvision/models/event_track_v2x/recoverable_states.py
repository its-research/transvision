"""Replay conditional 3D states from immutable observations, never a mixture.

This fixed-tracklet window connects identity leaves to separate state histories.
It is not a birth/death tracker: the producer must provide stable, prediction-only
tracklet IDs on both sides. CI replay is a conditional engineering state update,
not an exact Bayesian trajectory posterior or a source of identity likelihoods.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .hypothesis_bank import BankSnapshot
from .tracking_v2 import ci, propagate


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')


def _state(mean, covariance, score):
    m, c = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    if m.shape != (9,) or c.shape != (9, 9) or not np.isfinite(m).all() or not np.isfinite(c).all():
        raise ValueError('finite mean9 and covariance9x9 required')
    if not np.allclose(c, c.T, atol=1e-10, rtol=1e-10):
        raise ValueError('covariance must be symmetric')
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:
        raise ValueError('positive definite covariance required') from error
    if not math.isfinite(score) or not 0 <= score <= 1 or np.any(m[3:6] <= 0):
        raise ValueError('physical dimensions and detection score required')
    return tuple(map(float, m)), tuple(tuple(map(float, row)) for row in c)


@dataclass(frozen=True)
class TrackPrior:
    track_id: str
    reference_us: int
    mean: tuple
    covariance: tuple
    existence_score: float

    def __post_init__(self):
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError('nonempty stable track ID required')
        _integer(self.reference_us, 'reference_us')
        m, c = _state(self.mean, self.covariance, self.existence_score)
        object.__setattr__(self, 'mean', m)
        object.__setattr__(self, 'covariance', c)
        object.__setattr__(self, 'existence_score', float(self.existence_score))


@dataclass(frozen=True)
class TrackletObservation:
    message_id: str
    side: str
    node_id: str
    information_us: int
    arrival_us: int
    mean: tuple
    covariance: tuple
    existence_score: float

    def __post_init__(self):
        if (self.side not in ('left', 'right') or not isinstance(self.node_id, str) or not self.node_id
                or not isinstance(self.message_id, str) or not self.message_id):
            raise ValueError('side, stable tracklet ID and unique message ID required')
        _integer(self.information_us, 'information_us')
        _integer(self.arrival_us, 'arrival_us')
        if self.arrival_us < self.information_us:
            raise ValueError('arrival cannot precede information')
        m, c = _state(self.mean, self.covariance, self.existence_score)
        object.__setattr__(self, 'mean', m)
        object.__setattr__(self, 'covariance', c)
        object.__setattr__(self, 'existence_score', float(self.existence_score))


@dataclass(frozen=True)
class ConditionalBranchStates:
    branch_id: str
    choices: tuple[int, ...]
    log_weight: float
    tracks: tuple[TrackPrior, ...]


@dataclass(frozen=True)
class StateWindowCommit:
    decision_us: int
    reference_us: int
    identity_commit: str
    identity_ancestors: tuple[str, ...]
    prior_sha256: str
    observation_sha256: str
    branches: tuple[ConditionalBranchStates, ...]
    previous_commit: str
    commit: str


class BranchStateWindow:
    """Reconstruct every active leaf separately from the same original evidence.

    No state feeds back into the identity bank here. Joint temporal association,
    window handoff and data-derived tracklet creation remain outer-adapter work.
    The right ID universe stays fixed; unmatched left tracklets get stable local
    IDs, and association mass never scales existence_score.
    """

    def __init__(self, *, component_id, left_node_ids, right_node_ids, priors,
                 start_us, window_us=2_000_000, max_observations=4096, max_commits=1024,
                 max_active_branches=64, ci_weight=.5, process_noise=.1):
        _integer(start_us, 'start_us')
        for name, value in (('window_us', window_us), ('max_observations', max_observations),
                            ('max_commits', max_commits), ('max_active_branches', max_active_branches)):
            _integer(value, name)
            if value == 0:
                raise ValueError(name + ' must be positive')
        if not isinstance(component_id, str) or not component_id:
            raise ValueError('component_id required')
        left, right, priors = tuple(left_node_ids), tuple(right_node_ids), tuple(priors)
        for ids in (left, right):
            if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('stable node IDs must be unique within each side')
        if (len(priors) != len(right) or any(type(x) is not TrackPrior for x in priors)
                or len({x.track_id for x in priors}) != len(priors)
                or any(x.track_id.startswith('unmatched:') for x in priors)
                or any(x.reference_us != start_us for x in priors)):
            raise ValueError('one distinct initial prior per right node at window start required')
        if not math.isfinite(ci_weight) or not 0 < ci_weight < 1 or not math.isfinite(process_noise) or process_noise < 0:
            raise ValueError('invalid CI/process configuration')
        self.component_id, self.left, self.right, self.priors = component_id, left, right, priors
        self.start_us, self.window_us = start_us, window_us
        self.max_observations, self.max_commits = max_observations, max_commits
        self.max_active_branches = max_active_branches
        self.ci_weight, self.process_noise = float(ci_weight), float(process_noise)
        self.observations = ()
        self.commits = ()
        self._identity_origin = None
        self.prior_sha256 = self._config_digest()

    def _config_digest(self):
        return _hash({'component': self.component_id, 'left': self.left, 'right': self.right,
            'priors': [asdict(p) for p in self.priors], 'start_us': self.start_us, 'window_us': self.window_us,
            'ci_weight': self.ci_weight, 'process_noise': self.process_noise,
            'limits': [self.max_observations, self.max_commits, self.max_active_branches]})

    def _time(self, decision_us):
        if self._config_digest() != self.prior_sha256:
            raise ValueError('window configuration or original priors changed')
        _integer(decision_us, 'decision_us')
        if not self.start_us <= decision_us <= self.start_us + self.window_us:
            raise ValueError('outside fixed window; no silent expiration')
        if self.commits and decision_us < self.commits[-1].decision_us:
            raise ValueError('decision cannot rewrite committed history')

    def ingest(self, observation, *, decision_us):
        self._time(decision_us)
        if type(observation) is not TrackletObservation:
            raise TypeError('prediction-only tracklet observation required')
        if (observation.arrival_us > decision_us or observation.information_us < self.start_us
                or observation.node_id not in (self.left if observation.side == 'left' else self.right)):
            raise ValueError('future/out-of-window evidence or unknown node')
        for previous in self.observations:
            if previous.message_id == observation.message_id:
                if previous != observation:
                    raise ValueError('conflicting duplicate message')
                return False
            if (previous.side, previous.node_id, previous.information_us) == (
                    observation.side, observation.node_id, observation.information_us):
                raise ValueError('same source tracklet/time under different ID would double count evidence')
        if len(self.observations) >= self.max_observations:
            raise ValueError('observation budget exhausted; original factors not removed')
        self.observations += (observation,)
        return True

    def _states(self, choices, reference_us):
        # All arrays are fresh per branch, so updates cannot contaminate siblings.
        states = {p.track_id: p for p in self.priors}
        left_ids = {node: (self.priors[j].track_id if j >= 0 else
                           'unmatched:' + _hash([self.component_id, node])) for node, j in zip(self.left, choices)}
        right_ids = {node: p.track_id for node, p in zip(self.right, self.priors)}
        for obs in sorted(self.observations, key=lambda x: (x.information_us, x.side, x.node_id, x.message_id)):
            track_id = (left_ids if obs.side == 'left' else right_ids)[obs.node_id]
            if track_id in states:
                prior = states[track_id]
                mean, cov = propagate(np.array(prior.mean), np.array(prior.covariance),
                                      (obs.information_us-prior.reference_us)/1e6, self.process_noise)
                mean, cov = ci(mean, cov, np.array(obs.mean), np.array(obs.covariance), self.ci_weight)
                score = max(prior.existence_score, obs.existence_score)
            else:
                mean, cov, score = obs.mean, obs.covariance, obs.existence_score
            states[track_id] = TrackPrior(track_id, obs.information_us, mean, cov, score)
        result = []
        for track_id, prior in sorted(states.items()):
            mean, cov = propagate(np.array(prior.mean), np.array(prior.covariance),
                                  (reference_us-prior.reference_us)/1e6, self.process_noise)
            result.append(TrackPrior(track_id, reference_us, mean, cov, prior.existence_score))
        return tuple(result)

    def commit(self, snapshot, *, reference_us, identity_ancestors=()):
        """Commit current states; skipped search commits require a verified chain.

        identity_ancestors contains intermediate immutable bank snapshots, not
        old state outputs. Their hashes and direct-parent links are rechecked.
        """
        if type(snapshot) is not BankSnapshot:
            raise TypeError('identity bank snapshot required')
        self._time(snapshot.decision_us)
        _integer(reference_us, 'reference_us')
        if not self.start_us <= reference_us <= snapshot.decision_us:
            raise ValueError('output reference must lie inside causal window')
        if any(x.arrival_us > snapshot.decision_us or x.information_us > reference_us for x in self.observations):
            raise ValueError('state input is later than decision/reference')
        if self._identity_origin not in (None, snapshot.original_factors_sha256):
            raise ValueError('identity support origin changed')
        if len(snapshot.active) > self.max_active_branches:
            raise ValueError('active-state branch budget exceeded')
        obs_hash = _hash([asdict(o) for o in sorted(self.observations, key=lambda x: x.message_id)])
        if self.commits and self.commits[-1].identity_commit == snapshot.commit:
            prior = self.commits[-1]
            if prior.reference_us != reference_us or prior.observation_sha256 != obs_hash:
                raise ValueError('same identity commit cannot bind different state evidence')
            return prior
        if self.commits and reference_us <= self.commits[-1].reference_us:
            raise ValueError('new state outputs must have strictly increasing reference times')
        ancestors = tuple(identity_ancestors)
        if not self.commits and ancestors:
            raise ValueError('first state commit anchors its identity snapshot without intermediate ancestry')
        parent = self.commits[-1].identity_commit if self.commits else None
        for item in ancestors + (snapshot,):
            if type(item) is not BankSnapshot:
                raise ValueError('identity ancestry requires bank snapshots')
            payload = asdict(item)
            digest = payload.pop('commit')
            if (digest != _hash(payload) or item.original_factors_sha256 != snapshot.original_factors_sha256
                    or item.decision_us > snapshot.decision_us or (parent is not None and item.previous_commit != parent)):
                raise ValueError('identity ancestry hash, causal time or parent chain differs')
            parent = item.commit
        if len(self.commits) >= self.max_commits:
            raise ValueError('commit budget exhausted; no history rewrite')
        branches = []
        for leaf in snapshot.active:
            used = [j for j in leaf.choices if j >= 0]
            if (len(leaf.choices) != len(self.left) or len(set(used)) != len(used)
                    or any(type(j) is not int or j < -1 or j >= len(self.right) for j in leaf.choices)):
                raise ValueError('identity configuration violates fixed one-to-one universe')
            branches.append(ConditionalBranchStates(leaf.branch_id, leaf.choices, leaf.log_weight,
                                                    self._states(leaf.choices, reference_us)))
        payload = dict(decision_us=snapshot.decision_us, reference_us=reference_us,
            identity_commit=snapshot.commit, identity_ancestors=tuple(a.commit for a in ancestors),
            prior_sha256=self.prior_sha256, observation_sha256=obs_hash,
            branches=tuple(branches), previous_commit=self.commits[-1].commit if self.commits else '0'*64)
        serial = {**payload, 'branches': [asdict(b) for b in branches]}
        result = StateWindowCommit(**payload, commit=_hash(serial))
        self._identity_origin = snapshot.original_factors_sha256
        self.commits += (result,)
        return result
