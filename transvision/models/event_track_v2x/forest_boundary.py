"""Recoverable boundary checkpoints and exact conditional CI continuation.

No MAP identity is selected to form this archive. It stores all original raw
observations/factors and a complete recoverable support cover. State projection
can materialize ANY legal history, including leaves never enumerated before.
This supplies boundary/state recovery, not yet the next-window identity prior
marginalization or a complete bounded-memory sequence inference algorithm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np

from .detection_cache_v2 import canonical
from .forest_components import split_forest, validate_component_snapshot
from .forest_tracking import CausalForestTracker, ForestTrackingConfig, RawIdentityDetection
from .identity_forest import ForestFactors, ForestSnapshot, IdentityNode, digest, validate_snapshot
from .recoverable_states import _state
from .tracking_v2 import ci, propagate


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _order(obs):
    return obs.state_us, obs.node.source_id, obs.node.node_id


def _replay(group, config, *, mean=None, covariance=None, state_us=None):
    """Leave the result at its LAST observation time, never at a cut boundary."""
    for obs in sorted(group, key=_order):
        if mean is None:
            mean, covariance = np.asarray(obs.mean), np.asarray(obs.covariance)
        else:
            mean, covariance = propagate(np.asarray(mean), np.asarray(covariance),
                (obs.state_us-state_us)/1e6, config.process_noise)
            mean, covariance = ci(mean, covariance, np.asarray(obs.mean), np.asarray(obs.covariance), config.ci_weight)
        state_us = obs.state_us
    return mean, covariance, state_us


@dataclass(frozen=True)
class BoundaryGroup:
    group_id: str
    indices: tuple[int, ...]
    factors: ForestFactors
    snapshot: ForestSnapshot
    discovered: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BoundaryCarry:
    root_index: int
    track_id: str
    last_order_key: tuple
    mean: tuple
    covariance: tuple
    archived_indices: tuple[int, ...]
    occupied_source_frames: tuple[tuple[int, str], ...]
    # Keep exact score/time witnesses; association weights never scale existence.
    score_witnesses: tuple[tuple[float, int], ...]
    first_state_us: int
    last_state_us: int


@dataclass(frozen=True)
class BoundaryProjection:
    archive_sha256: str
    parents: tuple[int, ...]
    roots: tuple[int, ...]
    separator_indices: tuple[int, ...]
    separator_roots: tuple[int, ...]
    carries: tuple[BoundaryCarry, ...]
    log_weight: float
    active_product_member: bool
    all_component_leaves_previously_discovered: bool
    recovered_group_ids: tuple[str, ...]
    commit: str


@dataclass(frozen=True)
class BoundaryReplay:
    archive_sha256: str
    projection_sha256: str
    factors_sha256: str
    decision_us: int
    reference_us: int
    prediction_json: bytes
    audit_json: bytes

    @property
    def predictions(self):
        return json.loads(self.prediction_json)

    @property
    def audit(self):
        return json.loads(self.audit_json)


@dataclass(frozen=True)
class ForestBoundaryArchive:
    sequence_id: str
    cutoff_us: int
    decision_us: int
    reference_us: int
    config: ForestTrackingConfig
    observations: tuple[RawIdentityDetection, ...]
    factors: ForestFactors
    groups: tuple[BoundaryGroup, ...]
    emitted_parents: tuple[int, ...]
    source_prediction_sha256: str
    source_audit_sha256: str
    source_cache_manifest_sha256: str | None
    inference_mode: str
    commit: str

    def payload(self):
        result = asdict(self)
        result.pop('commit')
        return dict(kind='recoverable_forest_boundary_archive_v1', **result)

    def validate(self):
        if (not isinstance(self.sequence_id, str) or not self.sequence_id
                or type(self.config) is not ForestTrackingConfig
                or any(type(v) is not int or v < 0 for v in (self.cutoff_us, self.decision_us, self.reference_us))
                or not self.cutoff_us <= self.reference_us <= self.decision_us
                or not all(_hash(v) for v in (self.source_prediction_sha256, self.source_audit_sha256, self.commit))
                or self.source_cache_manifest_sha256 is not None and not _hash(self.source_cache_manifest_sha256)
                or self.inference_mode not in {'component', 'monolithic'}):
            raise ValueError('invalid boundary identity/time/configuration/hash')
        if (type(self.observations) is not tuple or type(self.groups) is not tuple
                or type(self.factors) is not ForestFactors
                or len(self.observations) > self.config.max_nodes
                or (self.inference_mode == 'component') != self.config.component_mode):
            raise ValueError('invalid boundary storage, factors or inference mode')
        if self.commit != digest(self.payload()):
            raise ValueError('boundary payload hash differs')
        if (any(type(o) is not RawIdentityDetection or o.sequence_id != self.sequence_id
                or o.node.arrival_us > self.decision_us for o in self.observations)
                or self.factors.nodes != tuple(o.node for o in self.observations)
                or len({(o.node.source_id, o.node.frame_id, o.detection_index) for o in self.observations}) != len(self.observations)
                or len(self.emitted_parents) != len(self.observations)):
            raise ValueError('future, duplicate or unbound boundary observations/action')
        self.factors.roots(self.emitted_parents)
        expected = (split_forest(self.factors) if self.inference_mode == 'component' else None)
        if expected is None:
            if (len(self.groups) != 1 or self.groups[0].group_id != 'global'
                    or self.groups[0].indices != tuple(range(len(self.observations)))
                    or self.groups[0].factors != self.factors):
                raise ValueError('monolithic boundary coverage differs')
        elif (len(expected) != len(self.groups) or any((c.component_id, c.indices, c.factors) !=
                (g.group_id, g.indices, g.factors) for c, g in zip(expected, self.groups))):
            raise ValueError('boundary components no longer cover the original factors')
        if self.config.component_mode and (len(self.groups) > self.config.max_components
                or sum(len(g.snapshot.frontier) for g in self.groups) > self.config.max_total_frontier
                or sum(len(g.discovered) for g in self.groups) > self.config.max_total_discovered):
            raise ValueError('boundary shared storage cap exceeded')
        for group in self.groups:
            if (len(group.snapshot.active) > self.config.active_limit
                    or len(group.snapshot.frontier) > self.config.max_frontier
                    or len(group.discovered) > self.config.max_discovered
                    or self.config.component_mode and len(group.indices) > self.config.max_component_nodes):
                raise ValueError('boundary component storage cap exceeded')
            validate_snapshot(group.factors, group.snapshot)
            if (group.snapshot.decision_us != self.decision_us
                    or len(set(group.discovered)) != len(group.discovered)
                    or any(len(p) != len(group.indices) for p in group.discovered)
                    or not set(group.snapshot.active) <= set(group.discovered)):
                raise ValueError('boundary discovery history or decision differs')
            for parents in group.discovered:
                group.factors.roots(parents)

    @property
    def separator_indices(self):
        return tuple(i for i, o in enumerate(self.observations) if o.node.information_us >= self.cutoff_us)

    def project(self, parents):
        """Materialize a legal history, not a posterior sample or a new weight fit."""
        self.validate()
        if type(parents) is not tuple or len(parents) != len(self.observations):
            raise ValueError('complete original parent configuration required')
        roots = self.factors.roots(parents)
        kept = self.separator_indices
        archived = set(range(len(roots))) - set(kept)
        if len(archived) > self.config.max_replay_operations:
            raise ValueError('boundary projection state-work cap exceeded before replay')
        by_root = {}
        for i in sorted(archived):
            by_root.setdefault(roots[i], []).append(i)
        carries = []
        for root, indices in sorted(by_root.items()):
            raw = tuple(self.observations[i] for i in indices)
            mean, covariance, last = _replay(raw, self.config)
            mean, covariance = _state(mean, covariance, max(o.score for o in raw))
            carries.append(BoundaryCarry(root,
                self.sequence_id+':'+digest(['forest-birth', self.factors.nodes[root].node_id])[:24],
                max(map(_order, raw)), mean, covariance, tuple(indices),
                tuple(sorted({(o.node.source_id, o.node.frame_id) for o in raw})),
                tuple((o.score, o.state_us) for o in raw), min(o.state_us for o in raw), last))
        active, discovered, recovered = True, True, []
        for group in self.groups:
            inverse = {original: local for local, original in enumerate(group.indices)}
            local = tuple(-1 if parents[i] < 0 else inverse[parents[i]] for i in group.indices)
            active &= local in group.snapshot.active
            discovered &= local in group.discovered
            if local not in group.discovered:
                recovered.append(group.group_id)
        payload = dict(archive_sha256=self.commit, parents=parents, roots=roots,
            separator_indices=kept, separator_roots=tuple(roots[i] for i in kept), carries=tuple(carries),
            log_weight=self.factors.log_weight(parents), active_product_member=active,
            all_component_leaves_previously_discovered=discovered, recovered_group_ids=tuple(recovered))
        return BoundaryProjection(**payload, commit=digest(dict(payload, carries=[asdict(c) for c in carries])))

    def replay(self, *, factors, parents, new_observations=(), reference_us, decision_us, projection=None):
        """Continue a selected identity history using carries or late raw replay.

The caller supplies the next legal identity action and its current factors.
This routine DOES NOT compute the next-window marginal prior/posterior. Old row
weights may be rescored; old node identity and candidate support cannot change.
"""
        self.validate()
        new = tuple(new_observations)
        n = len(self.observations)
        if (type(factors) is not ForestFactors or type(parents) is not tuple
                or len(parents) != n+len(new)
                or any(type(t) is not int for t in (reference_us, decision_us))
                or not self.reference_us <= reference_us <= decision_us or decision_us < self.decision_us
                or any(type(o) is not RawIdentityDetection or o.sequence_id != self.sequence_id
                       or not self.decision_us < o.node.arrival_us <= decision_us for o in new)):
            raise ValueError('invalid/future/withheld continuation evidence, action or time')
        observations = self.observations + new
        # Charge a conservative upper bound including rebuilding the projection
        # (even if supplied) and the worst case of replaying every raw node.
        work_upper = n-len(self.separator_indices)+len(observations)
        if len(observations) > self.config.max_nodes or work_upper > self.config.max_replay_operations:
            raise ValueError('boundary continuation node/state-work cap exceeded before replay')
        if (factors.nodes != tuple(o.node for o in observations)
                or any(tuple(p for p, _ in a) != tuple(p for p, _ in b)
                       for a, b in zip(self.factors.rows, factors.rows))
                or len({(o.node.source_id, o.node.frame_id, o.detection_index) for o in observations}) != len(observations)):
            raise ValueError('continuation rewrote original support, raw nodes or source slots')
        roots = factors.roots(parents)
        expected = self.project(parents[:n])
        if projection is not None and projection != expected:
            raise ValueError('stale or altered conditional boundary projection')
        projection = expected
        kept = set(projection.separator_indices)
        carries = {c.root_index: c for c in projection.carries}
        groups = {}
        for index, root in enumerate(roots):
            groups.setdefault(root, []).append(index)
        predictions, suppressed, modes, operations = [], [], [], 0
        for root, indices in sorted(groups.items()):
            raw = tuple(observations[i] for i in indices)
            carry = carries.get(root)
            pending = tuple(observations[i] for i in indices if i >= n or i in kept)
            can_continue = carry is not None and all(_order(o) > carry.last_order_key for o in pending)
            mode = 'conditional_carry' if can_continue else 'raw_replay_late_or_interleaved' if carry else 'raw_new_root'
            track_id = self.sequence_id+':'+digest(['forest-birth', factors.nodes[root].node_id])[:24]
            age = max(0, reference_us-max(o.state_us for o in raw))
            score = max(o.score*self.config.survival_per_second**(max(0, reference_us-o.state_us)/1e6) for o in raw)
            if (max(o.score for o in raw) < self.config.birth_score or score < self.config.prune_score
                    or age > self.config.max_age_us):
                suppressed.append(track_id)
                continue
            if can_continue:
                mean, covariance, state_us = _replay(pending, self.config,
                    mean=np.asarray(carry.mean), covariance=np.asarray(carry.covariance), state_us=carry.last_state_us)
                operations += len(pending)
            else:
                mean, covariance, state_us = _replay(raw, self.config)
                operations += len(raw)
            mean, covariance = propagate(mean, covariance, (reference_us-state_us)/1e6, self.config.process_noise)
            mean, covariance = _state(mean, covariance, score)
            predictions.append({'track_id': track_id, 'class_label': 'car', 'mean': mean,
                'covariance': covariance, 'score': score, 'observation_ids': tuple(o.node.node_id for o in raw),
                'birth_state_us': min(o.state_us for o in raw), 'last_update_us': max(o.state_us for o in raw)})
            modes.append({'track_id': track_id, 'mode': mode, 'root_index': root})
        audit = {'kind': 'recoverable_boundary_conditional_state_replay_v1', 'archive_sha256': self.commit,
            'projection_sha256': projection.commit, 'factors_sha256': factors.digest(),
            'parents': parents, 'reference_us': reference_us, 'decision_us': decision_us,
            'modes': modes, 'suppressed': suppressed, 'state_updates_after_projection': operations,
            'projection_build_state_updates': sum(len(c.archived_indices) for c in projection.carries),
            'state_work_upper': work_upper,
            'source_prediction_sha256': self.source_prediction_sha256,
            'recovered_group_ids': projection.recovered_group_ids,
            'new_posterior_or_missing_mass_bound_computed': False,
            'full_sequence_identity_handoff_completed': False,
            'historical_output_rewritten': False, 'ci_mixture_of_identity_branches': False}
        return BoundaryReplay(self.commit, projection.commit, factors.digest(), decision_us, reference_us,
                              canonical(predictions), canonical(audit))

    def save(self, path, *, max_bytes=128*1024**2):
        """Exclusive creation; return the archive commit, NOT the file SHA-256."""
        self.validate()
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('positive archive byte cap required')
        payload = canonical(dict(self.payload(), commit=self.commit))
        if len(payload) > max_bytes:
            raise ValueError('boundary archive byte cap exceeded; no file created')
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('boundary archive path may not traverse symlinks')
        with path.open('xb') as stream:
            stream.write(payload)
        return self.commit

    @classmethod
    def load(cls, path, *, expected_archive_sha256, max_bytes=128*1024**2):
        """Verify the canonical payload commit; it is not the whole-file hash."""
        path = Path(path).absolute()
        if (not _hash(expected_archive_sha256) or type(max_bytes) is not int or max_bytes < 1
                or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file() or path.stat().st_size > max_bytes):
            raise ValueError('invalid regular archive file, expected hash or byte cap')
        raw = path.read_bytes()
        payload = json.loads(raw)
        allowed = {f.name for f in fields(cls)} | {'kind'}
        if (raw != canonical(payload) or set(payload) != allowed
                or payload.pop('kind') != 'recoverable_forest_boundary_archive_v1'
                or payload['commit'] != expected_archive_sha256):
            raise ValueError('noncanonical, unknown or incorrectly bound boundary archive')
        payload['config'] = ForestTrackingConfig(**payload['config'])
        payload['observations'] = tuple(RawIdentityDetection(**dict(o, node=IdentityNode(**o['node'])))
                                        for o in payload['observations'])
        payload['factors'] = _load_factors(payload['factors'])
        groups = []
        for group in payload['groups']:
            snapshot = dict(group['snapshot'])
            for name in ('active', 'frontier', 'never_enumerated_recoveries', 'restored_ancestral_prefixes'):
                snapshot[name] = tuple(tuple(p) for p in snapshot[name])
            snapshot['active_log_weights'] = tuple(snapshot['active_log_weights'])
            groups.append(BoundaryGroup(**dict(group, indices=tuple(group['indices']),
                factors=_load_factors(group['factors']), snapshot=ForestSnapshot(**snapshot),
                discovered=tuple(tuple(p) for p in group['discovered']))))
        payload['groups'] = tuple(groups)
        payload['emitted_parents'] = tuple(payload['emitted_parents'])
        result = cls(**payload)
        result.validate()
        return result


def _load_factors(value):
    if set(value) != {'nodes', 'rows'}:
        raise ValueError('unknown raw factor fields')
    return ForestFactors(tuple(IdentityNode(**n) for n in value['nodes']), tuple(value['rows']))


def checkpoint_tracker(tracker, *, cutoff_us, source_cache_manifest_sha256=None):
    """Bind a committed tracker without changing it or selecting a single MAP."""
    if type(tracker) is not CausalForestTracker or not tracker.commits:
        raise ValueError('committed causal forest tracker required')
    prediction, audit = tracker.commits[-1].prediction, tracker.commits[-1].audit
    payload = dict(prediction)
    recorded = payload.pop('commit_sha256')
    if (digest(payload) != recorded or audit['prediction_sha256'] != recorded
            or digest([asdict(o) for o in tracker.observations]) != audit['observations_sha256']
            or tuple(tuple(p for p, _ in row) for row in tracker.bank.factors.rows) != tracker.support
            or tuple(audit['output_parents']) != tracker.output_parents
            or canonical(asdict(tracker.bank.factors)) != canonical(audit['raw_local_factors'])
            or canonical(asdict(tracker.bank.commits[-1])) != canonical(audit['forest'])
            or tracker.config_hash != audit['config_sha256']
            or tracker.scorer_signature != audit['scorer_signature']
            or getattr(tracker.scorer, 'signature', None) != tracker.scorer_signature
            or digest([tracker.sequence_id, tracker.start_us, asdict(tracker.config)]) != audit['config_sha256']):
        raise ValueError('tracker changed since its last immutable output')
    factors, groups = tracker.bank.factors, []
    if tracker.config.component_mode:
        validate_component_snapshot(factors, tracker.bank.commits[-1])
        for c in tracker.bank.commits[-1].components:
            key = c.component.component_id
            groups.append(BoundaryGroup(key, c.component.indices, c.component.factors, c.posterior,
                                       tuple(sorted(tracker.bank.banks[key].discovered))))
    else:
        groups.append(BoundaryGroup('global', tuple(range(len(factors.nodes))), factors,
            tracker.bank.commits[-1], tuple(sorted(tracker.bank.discovered))))
    values = dict(sequence_id=tracker.sequence_id, cutoff_us=cutoff_us,
        decision_us=prediction['decision_timestamp_us'], reference_us=prediction['box_reference_timestamp_us'],
        config=tracker.config, observations=tracker.observations, factors=factors, groups=tuple(groups),
        emitted_parents=tuple(audit['output_parents']), source_prediction_sha256=recorded,
        source_audit_sha256=digest(audit), source_cache_manifest_sha256=source_cache_manifest_sha256,
        inference_mode='component' if tracker.config.component_mode else 'monolithic')
    prototype = ForestBoundaryArchive(**values, commit='0'*64)
    archive = ForestBoundaryArchive(**values, commit=digest(prototype.payload()))
    archive.validate()
    return archive
