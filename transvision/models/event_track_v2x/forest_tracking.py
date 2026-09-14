"""Causal raw-detection identity histories with branch-specific 3D replay.

This connects the forest to dynamic detections, births, aging and immutable prediction-format outputs. It is one bounded inference window: no silent MAP handoff or history
expiration. Long-sequence component/window management belongs to the outer runner and must retain an explicit boundary approximation audit.
"""
from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from .detection_cache_v2 import DetectionCacheV2, canonical
from .identity_forest import ForestFactors, IdentityNode, RecoverableForestBank, decode_identity_roots, digest
from .recoverable_states import _state
from .tracking_v2 import ci, physical_world, propagate

FEATURE_RECIPE = 'recover-before-fuse-physical-world-cache203-v1'


@dataclass(frozen=True)
class RawIdentityDetection:
    sequence_id: str
    node: IdentityNode
    detection_index: int
    state_us: int
    mean: tuple
    covariance: tuple
    score: float
    features: tuple
    source_cache_sha256: str

    @property
    def class_index(self):
        return int(np.argmax(self.features[138:141]))

    def __post_init__(self):
        if (not isinstance(self.sequence_id, str) or not self.sequence_id or type(self.node) is not IdentityNode or self.node.source_id not in (0, 1)
                or type(self.detection_index) is not int or self.detection_index < 0 or type(self.state_us) is not int or not 0 <= self.state_us <= self.node.information_us
                or not isinstance(self.source_cache_sha256, str) or len(self.source_cache_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_cache_sha256)):
            raise ValueError('invalid prediction-only detection identity/time/provenance')
        mean, covariance = _state(self.mean, self.covariance, self.score)
        features = np.asarray(self.features, dtype=float)
        if features.shape != (203, ) or not np.isfinite(features).all():
            raise ValueError('finite 203-dimensional frozen features required')
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'covariance', covariance)
        object.__setattr__(self, 'features', tuple(map(float, features)))
        object.__setattr__(self, 'score', float(self.score))


def cache_detections(frame, *, arrival_us, decision_us, origin_us, minimum_raw_score=.05, maximum_detections=64, candidate_protocol='rbf-car-first-top64-v1'):
    """Protocol-versioned V2 adapter; availability is checked BEFORE reading
    arrays.

    The pretrained appearance vector is unchanged. This new model's physical-world 203-feature recipe is distinct from the sealed legacy 203-feature checkpoint. State time and
    feature information time remain separate.
    """
    if not isinstance(frame, DetectionCacheV2):
        raise TypeError('DetectionCacheV2 required')
    meta = frame.metadata
    if (type(arrival_us) is not int or type(decision_us) is not int or type(origin_us) is not int or origin_us < 0
            or not frame.information_timestamp_us <= arrival_us <= decision_us or not math.isfinite(minimum_raw_score) or not 0 <= minimum_raw_score <= 1
            or type(maximum_detections) is not int or maximum_detections < 1):
        raise ValueError('future/unavailable cache or invalid selection configuration')
    from .paper_protocol import LEGACY, PAPER
    if candidate_protocol not in {PAPER, LEGACY}:
        raise ValueError('unknown candidate protocol')
    mask = frame.raw_scores >= minimum_raw_score
    if candidate_protocol == LEGACY:
        mask &= frame.class_indices == 0
    selected = np.flatnonzero(mask)
    selected = sorted(selected.tolist(), key=lambda i: (-float(frame.raw_scores[i]), i))[:maximum_detections]
    # Array indexing requires an integer dtype even when the frame is empty.
    selected = np.asarray(selected, dtype=np.int64)
    state_time = meta['box_reference_timestamp_us']
    means, covariances = physical_world(frame, selected, state_time)
    source = 0 if meta['side'] == 'vehicle-side' else 1
    frame_hash = frame.digest()
    result = []
    from scipy.spatial.transform import Rotation
    pose = np.r_[np.asarray(meta['lidar_to_world_translation']) / [100, 100, 20], Rotation.from_matrix(np.asarray(meta['lidar_to_world_row_rotation']).T).as_euler('xyz') / np.pi]
    for local, index in enumerate(selected):
        mean, covariance = means[local], covariances[local]
        features = np.r_[mean[:6] / [100, 100, 20, 20, 20, 10],
                         np.sin(mean[6]),
                         np.cos(mean[6]), mean[7:9] / 30, frame.appearance[index],
                         np.eye(3)[frame.class_indices[index]], [(state_time - origin_us) / 1e8, (meta['source_image_timestamp_us'] - state_time) / 1e6,
                                                                 (arrival_us - frame.information_timestamp_us) / 1e6, source], covariance[np.triu_indices(9)] / 100, pose,
                         [1, np.log(2) / 8, 0, 0], [frame.raw_scores[index], frame.scores[index], frame.appearance_valid[index]]]
        node_id = digest([meta['sequence_id'], source, meta['frame_id'], int(index), frame_hash])
        result.append(
            RawIdentityDetection(meta['sequence_id'], IdentityNode(node_id, source, frame.information_timestamp_us, arrival_us, meta['frame_id']), int(index), state_time, mean,
                                 covariance, float(frame.scores[index]), features, frame_hash))
    return tuple(result)


@dataclass(frozen=True)
class ForestTrackingConfig:
    window_us: int = 2_000_000
    max_nodes: int = 128
    max_commits: int = 64
    active_limit: int = 4
    max_frontier: int = 4096
    max_discovered: int = 65536
    max_replay_operations: int = 65536
    expansion_budget: int = 256
    action_budget: int = 256
    action_frontier: int = 4096
    parent_limit: int = 8
    max_parent_gap_us: int = 2_000_000
    gate_distance_m: float = 8.
    birth_score: float = .3
    prune_score: float = .05
    max_age_us: int = 2_000_000
    survival_per_second: float = .95
    ci_weight: float = .5
    process_noise: float = .1
    max_model_regret: float = .05
    candidate_protocol: ClassVar[str] = 'rbf-car-first-top64-v1'
    decision_mode: ClassVar[str] = 'retained'
    paper_action_budget: ClassVar[int] = 4096
    paper_action_frontier: ClassVar[int] = 8192
    component_mode: bool = False
    potential_context: str = 'auto'
    max_component_nodes: int = 128
    max_components: int = 2048
    max_total_frontier: int = 65536
    max_total_discovered: int = 262144

    def __post_init__(self):
        integers = ('window_us', 'max_nodes', 'max_commits', 'active_limit', 'max_frontier', 'max_discovered', 'max_replay_operations', 'action_frontier', 'parent_limit',
                    'max_parent_gap_us', 'max_age_us', 'max_component_nodes', 'max_components', 'max_total_frontier', 'max_total_discovered')
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError('positive integer resource/time limits required')
        from .paper_protocol import LEGACY, PAPER
        if self.candidate_protocol not in {PAPER, LEGACY}:
            raise ValueError('unknown candidate protocol')
        if self.decision_mode not in {'retained', 'all-legal-hamming', 'map'}:
            raise ValueError('unknown identity action mode')
        if any(type(v) is not int or v < 1 for v in (self.paper_action_budget, self.paper_action_frontier)):
            raise ValueError('positive paper action limits required')
        if type(self.component_mode) is not bool:
            raise ValueError('component_mode must be boolean')
        if not isinstance(self.potential_context, str) or self.potential_context not in {'auto', 'global', 'component'}:
            raise ValueError('potential_context must be auto, global or component')
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 0 for key in ('expansion_budget', 'action_budget')):
            raise ValueError('nonnegative expansion/action budgets required')
        if (any(not math.isfinite(getattr(self, key))
                for key in ('gate_distance_m', 'birth_score', 'prune_score', 'survival_per_second', 'ci_weight', 'process_noise', 'max_model_regret'))
                or not 0 <= self.prune_score <= self.birth_score <= 1 or not 0 < self.survival_per_second <= 1 or not 0 < self.ci_weight < 1 or self.process_noise < 0
                or self.gate_distance_m <= 0 or not 0 <= self.max_model_regret <= 1):
            raise ValueError('invalid motion, existence or decision configuration')


@dataclass(frozen=True)
class PaperForestTrackingConfig(ForestTrackingConfig):
    """New protocol fields never alter the historical default configuration
    hash."""
    candidate_protocol: str = 'rbf-all-class-top64-v1'
    decision_mode: str = 'retained'
    paper_action_budget: int = 4096
    paper_action_frontier: int = 8192


def load_tracking_config(values):
    cls = PaperForestTrackingConfig if 'candidate_protocol' in values else ForestTrackingConfig
    return cls(**values)


def append_parent_support(observations, old_support, config):
    """Prediction-only support gating fixed on first arrival, never on GT.

    The omitted-mass bound is conditional on this support. Excluding a candidate by the distance/time/top-neighbor gate is NOT accounted for by that bound.
    """
    support = list(old_support)
    for i in range(len(support), len(observations)):
        current = observations[i]
        candidates = []
        for j, previous in enumerate(observations[:i]):
            if ((current.node.source_id, current.node.frame_id) == (previous.node.source_id, previous.node.frame_id)):
                continue
            delta = (current.state_us - previous.state_us) / 1e6
            if abs(current.state_us - previous.state_us) > config.max_parent_gap_us:
                continue
            mean, _ = propagate(np.asarray(previous.mean), np.asarray(previous.covariance), delta, config.process_noise)
            distance = float(np.linalg.norm(np.asarray(current.mean[:2]) - mean[:2]))
            if distance <= config.gate_distance_m:
                candidates.append((distance, j))
        parents = tuple(sorted(j for _, j in sorted(candidates)[:config.parent_limit]))
        support.append((-1, ) + parents)
    return tuple(support)


def replay_forest_states(sequence_id, observations, factors, parents, reference_us, config):
    """Fresh conditional CI replay for ANY legal action, including outside
    active."""
    roots = factors.roots(parents)
    groups = {}
    for index, root in enumerate(roots):
        groups.setdefault(root, []).append(observations[index])
    predictions, suppressed = [], []
    for root, group in sorted(groups.items()):
        track_id = sequence_id + ':' + digest(['forest-birth', factors.nodes[root].node_id])[:24]
        age = max(0, reference_us - max(obs.state_us for obs in group))
        score = max(obs.score * config.survival_per_second**(max(0, reference_us - obs.state_us) / 1e6) for obs in group)
        if max(obs.score for obs in group) < config.birth_score or score < config.prune_score or age > config.max_age_us:
            suppressed.append(track_id)
            continue
        mean, covariance, state_time = None, None, None
        # Replay by state time, not arrival time: late evidence never propagates
        # an already averaged sibling state backwards to guess its identity.
        for obs in sorted(group, key=lambda o: (o.state_us, o.node.source_id, o.node.node_id)):
            if mean is None:
                mean, covariance = np.asarray(obs.mean), np.asarray(obs.covariance)
            else:
                mean, covariance = propagate(mean, covariance, (obs.state_us - state_time) / 1e6, config.process_noise)
                mean, covariance = ci(mean, covariance, np.asarray(obs.mean), np.asarray(obs.covariance), config.ci_weight)
            state_time = obs.state_us
        mean, covariance = propagate(mean, covariance, (reference_us - state_time) / 1e6, config.process_noise)
        mean, covariance = _state(mean, covariance, score)
        predictions.append({
            'track_id': track_id,
            'class_label': ('car', 'bicycle', 'pedestrian')[observations[root].class_index],
            'mean': mean,
            'covariance': covariance,
            'score': score,
            'observation_ids': tuple(o.node.node_id for o in group),
            'birth_state_us': min(o.state_us for o in group),
            'last_update_us': max(o.state_us for o in group)
        })
    return tuple(predictions), tuple(suppressed)


@dataclass(frozen=True)
class ForestTrackingCommit:
    prediction_json: bytes
    audit_json: bytes

    @property
    def prediction(self):
        return json.loads(self.prediction_json)

    @property
    def audit(self):
        return json.loads(self.audit_json)


class CausalForestTracker:
    """Dynamic raw-observation tracker within an explicit recovery window.

    Scorer returns absolute ForestFactors over immutable nodes/support. Every mutation is staged until inference, decision and all state replays succeed. The bank never consumes
    the emitted fallback action as its posterior.
    """

    def __init__(self, *, sequence_id, start_us, scorer, config=None):
        if not isinstance(sequence_id, str) or not sequence_id or type(start_us) is not int or start_us < 0:
            raise ValueError('sequence and nonnegative window start required')
        self.sequence_id, self.start_us, self.scorer = sequence_id, start_us, scorer
        self.config = config or ForestTrackingConfig()
        if type(self.config) is not ForestTrackingConfig or not callable(scorer):
            raise ValueError('validated configuration and factor scorer required')
        self.observations, self.support, self.output_parents = (), (), ()
        c = self.config
        if c.component_mode:
            from .forest_components import ComponentForestBank
            self.bank = ComponentForestBank(
                **{
                    key: getattr(c, key)
                    for key in ('active_limit', 'max_frontier', 'max_nodes', 'max_commits', 'max_discovered', 'max_component_nodes', 'max_components', 'max_total_frontier',
                                'max_total_discovered')
                })
        else:
            self.bank = RecoverableForestBank(
                ForestFactors((), ()), active_limit=c.active_limit, max_frontier=c.max_frontier, max_nodes=c.max_nodes, max_commits=c.max_commits, max_discovered=c.max_discovered)
        self.commits, self.events = (), {}
        self.config_hash = digest([sequence_id, start_us, asdict(c)])
        self.scorer_signature = getattr(scorer, 'signature', None)

    def _score(self, observations, support, decision_us):
        factors = self.scorer(observations, support, decision_us)
        if (type(factors) is not ForestFactors or factors.nodes != tuple(o.node for o in observations) or tuple(tuple(p for p, _ in row) for row in factors.rows) != support):
            raise ValueError('scorer changed identity nodes or original support')
        return factors

    @property
    def potential_context(self):
        return (('component' if self.config.component_mode else 'global') if self.config.potential_context == 'auto' else self.config.potential_context)

    def _score_factors(self, candidate, support, decision_us):
        from .forest_components import split_forest
        if self.potential_context == 'global' and not self.config.component_mode:
            return self._score(candidate, support, decision_us)
        zero = ForestFactors(tuple(o.node for o in candidate), tuple(tuple((p, 0.) for p in row) for row in support))
        structure = split_forest(zero)
        if (len(structure) > self.config.max_components or any(len(c.indices) > self.config.max_component_nodes for c in structure)):
            raise ValueError('component capacity exhausted before neural scoring; no edge was dropped')
        if self.potential_context == 'global':
            return self._score(candidate, support, decision_us)
        scored = [None] * len(candidate)
        for component in structure:
            local = tuple(candidate[i] for i in component.indices)
            local_support = tuple(tuple(p for p, _ in row) for row in component.factors.rows)
            factors = self._score(local, local_support, decision_us)
            for index, row in zip(component.indices, factors.rows):
                scored[index] = tuple((-1 if p < 0 else component.indices[p], w) for p, w in row)
        return ForestFactors(zero.nodes, tuple(scored))

    def _infer_components(self, candidate, support, accepted_count, reference_us, decision_us, event_id):
        from .forest_components import join_actions
        factors = self._score_factors(candidate, support, decision_us)
        bank = self.bank.fork()
        snapshot = bank.advance(factors=factors, decision_us=decision_us, expansion_budget=self.config.expansion_budget, message_id=event_id)
        decoded = {
            c.component.component_id: decode_identity_roots(c.component.factors, c.posterior, expansion_budget=0, max_frontier=self.config.action_frontier)
            for c in snapshot.components
        }
        # The action budget is GLOBAL too. Initial gap computations consume no
        # search splits; each selected component is subsequently searched once.
        order = sorted(snapshot.components, key=lambda c: (-len(c.component.indices) * (decoded[c.component.component_id]['action_search_gap'] or 0.), c.component.component_id))
        remaining = self.config.action_budget
        for c in order:
            key = c.component.component_id
            if remaining and decoded[key]['action_search_gap']:
                decoded[key] = decode_identity_roots(c.component.factors, c.posterior, expansion_budget=remaining, max_frontier=self.config.action_frontier)
                remaining -= decoded[key]['action_expansions']
        actions, branches, decisions, risks = [], [], [], []
        fallback = self.output_parents + (-1, ) * accepted_count
        for record in snapshot.components:
            component = record.component
            key, indices = component.component_id, component.indices
            local = tuple(candidate[i] for i in indices)
            inverse = {global_index: i for i, global_index in enumerate(indices)}
            bound = decoded[key]['model_regret_upper_estimate']
            use_decoded = bound is not None and bound <= self.config.max_model_regret
            action = decoded[key]['parents'] if use_decoded else tuple(-1 if fallback[i] < 0 else inverse[fallback[i]] for i in indices)
            actions.append(action)
            risks.append(len(indices) * (bound if use_decoded else 1.))
            decisions.append({
                'component_id': key,
                'decoded': decoded[key],
                'output_parents': action,
                'used_decoded_action': use_decoded,
                'output_model_regret_estimate': bound if use_decoded else 1.,
                'fallback_bound_is_trivial_loss_range': not use_decoded
            })
            location = {obs.node.node_id: i for i, obs in enumerate(local)}
            for parent, weight in zip(record.posterior.active, record.posterior.active_log_weights):
                states, suppressed = replay_forest_states(self.sequence_id, local, component.factors, parent, reference_us, self.config)
                ancestors = []
                for old_key, old_commit in record.predecessors:
                    old = self.bank.banks[old_key]
                    old_indices = tuple(location[n.node_id] for n in old.factors.nodes)
                    old_inverse = {current: before for before, current in enumerate(old_indices)}
                    prefix = tuple(-1 if parent[i] < 0 else old_inverse[parent[i]] for i in old_indices)
                    ancestors.append({
                        'component_id': old_key,
                        'posterior_commit': old_commit,
                        'branch_id': digest([tuple(n.node_id for n in old.factors.nodes), prefix]),
                        'materialized_previous': prefix in old.active
                    })
                branches.append({
                    'component_id': key,
                    'global_indices': indices,
                    'branch_id': digest([tuple(o.node.node_id for o in local), parent]),
                    'parent_branches': ancestors,
                    'parents': parent,
                    'log_weight': weight,
                    'tracks': states,
                    'suppressed': suppressed
                })
        parents = join_actions(factors, tuple(c.component for c in snapshot.components), tuple(actions))
        bound = math.fsum(risks) / len(candidate) if candidate else 0.
        result = {
            'status': 'componentwise',
            'components': decisions,
            'parents': parents,
            'model_regret_upper_estimate': bound,
            'action_expansions': self.config.action_budget - remaining,
            'potential_context': self.potential_context,
            'global_active_cartesian_product_materialized': False
        }
        return (bank, snapshot, factors, parents, branches, result, bound, 'componentwise_risk_acceptance_with_previous_action_unmatched_fallback')

    def _infer_monolithic(self, candidate, support, accepted_count, reference_us, decision_us, event_id):
        factors = self._score_factors(candidate, support, decision_us)
        bank = self.bank.fork()
        snapshot = bank.advance(decision_us=decision_us, expansion_budget=self.config.expansion_budget, factors=factors, message_id=event_id)
        decoded = decode_identity_roots(factors, snapshot, expansion_budget=self.config.action_budget, max_frontier=self.config.action_frontier)
        bound = decoded['model_regret_upper_estimate']
        use_decoded_action = bound is not None and bound <= self.config.max_model_regret
        if use_decoded_action:
            parents = decoded['parents']
            policy = 'conditional_bayes_action_with_reported_model_risk_estimate'
        else:
            parents = self.output_parents + (-1, ) * accepted_count
            policy = 'hold_previous_action_new_detections_unmatched_bank_remains_recoverable'
        branches = []
        for parent, weight in zip(snapshot.active, snapshot.active_log_weights):
            states, suppressed = replay_forest_states(self.sequence_id, candidate, factors, parent, reference_us, self.config)
            old_size = len(self.observations)
            branches.append({
                'branch_id': digest([tuple(o.node.node_id for o in candidate), parent]),
                'parent_branch_id': digest([tuple(o.node.node_id for o in self.observations), parent[:old_size]]),
                'parent_materialized_previous': parent[:old_size] in self.bank.active,
                'parents': parent,
                'log_weight': weight,
                'tracks': states,
                'suppressed': suppressed
            })
        return bank, snapshot, factors, parents, branches, decoded, bound if use_decoded_action else None, policy

    def step(self, observations, *, frame_id, reference_us, decision_us, event_id):
        new = tuple(observations)
        if any(type(o) is not RawIdentityDetection for o in new):
            raise TypeError('prediction-only raw detections required')
        if not isinstance(event_id, str) or not event_id or not isinstance(frame_id, str) or not frame_id:
            raise ValueError('nonempty event and frame IDs required')
        if (type(reference_us) is not int or type(decision_us) is not int or not self.start_us <= reference_us <= decision_us <= self.start_us + self.config.window_us):
            raise ValueError('invalid/future time or recovery window expired; explicit handoff required')
        if self.config_hash != digest([self.sequence_id, self.start_us, asdict(self.config)]):
            raise ValueError('tracker configuration changed')
        if self.scorer_signature != getattr(self.scorer, 'signature', None):
            raise ValueError('scorer configuration changed')
        request = digest([frame_id, reference_us, decision_us, [asdict(o) for o in new]])
        if event_id in self.events:
            previous, commit = self.events[event_id]
            if previous != request:
                raise ValueError('conflicting duplicate event')
            return commit
        if self.commits:
            previous = self.commits[-1].prediction
            if reference_us <= previous['box_reference_timestamp_us'] or decision_us < previous['decision_timestamp_us']:
                raise ValueError('output reference times must increase; historical output cannot be rewritten')
        if len(self.commits) >= self.config.max_commits:
            raise ValueError('commit budget exhausted before mutation')
        seen = {obs.node.node_id: obs for obs in self.observations}
        slots = {(obs.node.source_id, obs.node.frame_id, obs.detection_index): obs.node.node_id for obs in self.observations}
        accepted = []
        for obs in new:
            if obs.sequence_id != self.sequence_id or obs.node.arrival_us > decision_us or obs.state_us < self.start_us:
                raise ValueError('cross-sequence, future or out-of-window raw detection')
            if obs.node.node_id in seen:
                if seen[obs.node.node_id] != obs:
                    raise ValueError('conflicting duplicate raw detection')
                continue
            slot = (obs.node.source_id, obs.node.frame_id, obs.detection_index)
            if slot in slots:
                raise ValueError('same source detection under another ID would double count evidence')
            seen[obs.node.node_id], slots[slot] = obs, obs.node.node_id
            accepted.append(obs)
        accepted.sort(key=lambda o: (o.node.arrival_us, o.node.source_id, o.node.frame_id, o.detection_index))
        candidate = self.observations + tuple(accepted)
        if (len(candidate) > self.config.max_nodes or len(candidate) * (self.config.active_limit + 1) > self.config.max_replay_operations):
            raise ValueError('node/state-replay budget exhausted before scoring')
        if self.observations and accepted and accepted[0].node.arrival_us < self.observations[-1].node.arrival_us:
            raise ValueError('arrival ordering cannot rewrite old node indices')
        support = append_parent_support(candidate, self.support, self.config)
        infer = self._infer_components if self.config.component_mode else self._infer_monolithic
        bank, snapshot, factors, parents, branches, decoded, bound, policy = infer(candidate, support, len(accepted), reference_us, decision_us, event_id)
        states, suppressed = replay_forest_states(self.sequence_id, candidate, factors, parents, reference_us, self.config)
        previous_ids = set() if not self.commits else {t['track_id'] for t in self.commits[-1].prediction['predictions']}
        current_ids = {t['track_id'] for t in states}
        prediction = {
            'sequence_id': self.sequence_id,
            'frame_id': frame_id,
            'box_reference_timestamp_us': reference_us,
            'decision_timestamp_us': decision_us,
            'coordinate_frame': 'world',
            'state_layout': 'gravity_xyz_length_width_height_yaw_vxy',
            'predictions': [{k: v
                             for k, v in state.items() if k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for state in states],
            'previous_commit_sha256': self.commits[-1].prediction['commit_sha256'] if self.commits else '0' * 64
        }
        prediction['commit_sha256'] = digest(prediction)
        audit = {
            'kind': ('recoverable_component_forest_tracking_window_v1' if self.config.component_mode else 'recoverable_forest_tracking_window_v1'),
            'sequence_id': self.sequence_id,
            'frame_id': frame_id,
            'config_sha256': self.config_hash,
            'feature_recipe': FEATURE_RECIPE,
            'observations_sha256': digest([asdict(o) for o in candidate]),
            'source_cache_sha256': sorted({o.source_cache_sha256
                                           for o in candidate}),
            'forest': asdict(snapshot),
            'raw_local_factors': asdict(factors),
            'scorer_signature': self.scorer_signature,
            'scorer_configuration_bound': self.scorer_signature is not None,
            'potential_context': self.potential_context,
            'decoded_action': decoded,
            'output_parents': parents,
            'output_policy': policy,
            'output_model_regret_estimate': bound,
            'branches': branches,
            'selected_states': states,
            'suppressed': suppressed,
            'new_output_ids': sorted(current_ids - previous_ids),
            'removed_output_ids': sorted(previous_ids - current_ids),
            'new_observations': len(accepted),
            'duplicate_observations': len(new) - len(accepted),
            'numerical_certificate': False,
            'true_posterior_or_tracking_metric_bound': False,
            'mass_bound_includes_support_gating_error': False,
            'continuous_state_update': 'branch_conditional_CI_replay',
            'previous_audit_sha256': digest(self.commits[-1].audit) if self.commits else '0' * 64,
            'prediction_sha256': prediction['commit_sha256']
        }
        commit = ForestTrackingCommit(canonical(prediction), canonical(audit))
        self.observations, self.support, self.output_parents, self.bank = candidate, support, parents, bank
        self.commits += (commit, )
        self.events[event_id] = (request, commit)
        return commit
