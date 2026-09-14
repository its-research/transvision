"""Durable single-history JPDA/PKF adapters over the formal raw V2 row stream.

Raw arrival factors are frozen and unchanged. Each source/frame scan conditions
on the previously committed hard identity map; Gaussian states are collapsed.
Measurements are projected to the current output reference, while a collapsed
prior is NEVER propagated backwards. This declared approximation is not the
recoverable backend's state-time ordered branch replay or an exact OOSM filter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import sqlite3

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection_cache_v2 import canonical
from .forest_tracking import ForestTrackingConfig
from .hypothesis_bank import LogAssociationFactors, logsumexp
from .identity_forest import digest
from .jpda_filter import GaussianBelief, gaussian_moment_match, kalman_condition
from .jpda_marginals import JPDALimits, exact_jpda, lbp_jpda
from .persistent_forest import PersistentForestConfig, PersistentForestTracker, _path, _file_hash
from .pkf_filter import pkf_condition
from .tracking_v2 import ci, propagate


@dataclass(frozen=True)
class PersistentProbabilisticConfig(PersistentForestConfig):
    association_algorithm: str = 'lbp'
    update_rule: str = 'jpda-ci'
    anchor_decoder: str = 'joint-map'
    max_scan_tracks: int = 2048
    inference: JPDALimits = JPDALimits()

    def __post_init__(self):
        # The base class's validation assumes its own fields are all integers.
        PersistentForestConfig(**{k: getattr(self, k) for k in PersistentForestConfig.__dataclass_fields__})
        if (self.association_algorithm not in ('exact', 'lbp')
                or self.update_rule not in ('jpda-ci', 'jpda-kalman', 'pkf')
                or self.anchor_decoder not in ('joint-map', 'marginal-bayes')
                or type(self.max_scan_tracks) is not int or self.max_scan_tracks < 1
                or type(self.inference) is not JPDALimits):
            raise ValueError('explicit probabilistic algorithm, update rule and positive limits required')
        defaults = ForestTrackingConfig()
        for key in ('active_limit', 'max_frontier', 'expansion_budget', 'max_model_regret'):
            if getattr(self.state, key) != getattr(defaults, key):
                raise ValueError('probabilistic backend does not use '+key)


class PersistentProbabilisticTracker(PersistentForestTracker):
    CONFIG_TYPE = PersistentProbabilisticConfig
    SCHEMA = 'persistent_single_history_probabilistic_tracker_v1'

    def __init__(self, path, *, sequence_id, config=None):
        config = config or self.CONFIG_TYPE()
        if type(config) is not self.CONFIG_TYPE:
            raise TypeError('probabilistic tracker configuration required')
        super().__init__(path, sequence_id=sequence_id, config=config)
        self.db.executescript('''
            CREATE TABLE identity_anchors(i INTEGER PRIMARY KEY, root INTEGER NOT NULL);
            CREATE INDEX identity_anchor_roots ON identity_anchors(root,i);
            CREATE TABLE gaussian_tracks(root INTEGER PRIMARY KEY, reference_us INTEGER NOT NULL, payload BLOB NOT NULL);
        ''')
        self._set('schema', self.SCHEMA)

    def _bounds(self):
        # No full-history partition bound is claimed or needed by this backend.
        self.suffix, self.suffix_abs = np.zeros(1), np.zeros(1)

    def _load(self):
        super()._load()
        self.scan_audits = []
        self.chosen_anchor = self.meta['output']
        self.state_work = {}

    @classmethod
    def open(cls, path, *, expected_prediction_sha256, expected_database_sha256):
        path = _path(path)
        if (not path.is_file() or _file_hash(path) != expected_database_sha256
                or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                       for s in (expected_prediction_sha256, expected_database_sha256))):
            raise ValueError('sealed probabilistic database and output hashes required')
        connection = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
        try:
            values = {k: json.loads(v) for k, v in connection.execute('SELECT k,v FROM meta')}
        finally:
            connection.close()
        if values.get('schema') != cls.SCHEMA or values['state']['prediction_sha256'] != expected_prediction_sha256:
            raise ValueError('probabilistic database schema or output head differs')
        obj = cls.__new__(cls)
        obj.path, obj.sequence_id = path, values['sequence_id']
        config = values['config']
        obj.config = cls.CONFIG_TYPE(**dict(config, state=ForestTrackingConfig(**config['state']),
                                           inference=JPDALimits(**config['inference'])))
        obj._connect()
        try:
            obj._load()
            if (obj.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok'
                    or obj.db.execute('SELECT count(*) FROM identity_anchors').fetchone()[0] != obj.n):
                raise ValueError('probabilistic database integrity or identity count differs')
            if obj.meta['events']:
                raw = obj.db.execute('SELECT prediction,audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()
                prediction, audit = (json.loads(v) for v in raw)
                if (digest({k: v for k, v in prediction.items() if k != 'commit_sha256'}) != expected_prediction_sha256
                        or digest(audit) != obj.meta['audit_sha256'] or audit['factor_rows_sha256'] != obj.factor_digest()
                        or audit['configuration_sha256'] != obj.configuration_sha256
                        or audit['identity_anchor_sha256'] != obj._anchor_digest()
                        or audit['gaussian_state_sha256'] != obj._gaussian_digest()):
                    raise ValueError('probabilistic output, factors, configuration or anchors differ')
            return obj
        except BaseException:
            obj.close()
            raise

    def step(self, observations, rows, **kwargs):
        if kwargs.get('rescored_rows') or kwargs.get('decision_indices') is not None:
            raise ValueError('single-history baseline does not rescore old rows or use recoverable decision scopes')
        self.current_reference = kwargs['reference_us']
        self.scan_audits = []
        self.state_work = {}
        return super().step(observations, rows, **kwargs)

    def _promote(self, decision):
        pass

    def _state_at(self, root, reference):
        raw = self.db.execute('SELECT reference_us,payload FROM gaussian_tracks WHERE root=?', (root,)).fetchone()
        if raw is None or reference < raw[0]:
            raise ValueError('missing prior or attempted backward propagation of collapsed state')
        state = json.loads(raw[1])
        self._work('prior_or_output_projections')
        mean, cov = propagate(np.asarray(state['mean']), np.asarray(state['covariance']),
                              (reference-raw[0])/1e6, self.config.state.process_noise)
        return GaussianBelief(mean, cov)

    def _save_state(self, root, state):
        self._work('gaussian_writes')
        self.db.execute('INSERT OR REPLACE INTO gaussian_tracks VALUES(?,?,?)',
                        (root, self.current_reference, canonical(asdict(state))))

    def _work(self, kind, count=1):
        self.state_work[kind] = self.state_work.get(kind, 0)+count
        self.state_updates += count
        if self.state_updates > self.config.state.max_replay_operations:
            raise ValueError('probabilistic state work budget exceeded')

    def _scan(self, indices, decision):
        observations = tuple(self._observation(i) for i in indices)
        slot = observations[0].node.source_id, observations[0].node.frame_id
        if any((o.node.source_id, o.node.frame_id) != slot for o in observations):
            raise ValueError('single-source scan required')
        # Store no per-parent state histories. Raw parent identities are looked
        # up in the one immutable hard map from previous scans.
        parent_roots, occupied_roots, grouped_rows, births = {}, {}, [], []
        for index in indices:
            groups = {}
            for parent, weight in self._row(index):
                if parent < 0:
                    births.append(weight)
                    continue
                parent_slot = self.db.execute('SELECT source,frame FROM observations WHERE i=?', (parent,)).fetchone()
                if parent_slot == slot:
                    continue  # A structurally illegal edge in the original forest.
                if parent not in parent_roots:
                    found = self.db.execute('SELECT root FROM identity_anchors WHERE i=?', (parent,)).fetchone()
                    if found is None:
                        raise ValueError('parent is not in the already processed hard identity history')
                    parent_roots[parent] = found[0]
                root = parent_roots[parent]
                if root not in occupied_roots:
                    occupied_roots[root] = bool(self.db.execute('SELECT 1 FROM identity_anchors a JOIN observations o ON a.i=o.i '
                        'WHERE a.root=? AND o.source=? AND o.frame=? LIMIT 1', (root, *slot)).fetchone())
                if not occupied_roots[root]:
                    groups.setdefault(root, []).append((parent, weight))
            grouped_rows.append(groups)
        roots = tuple(sorted({root for row in grouped_rows for root in row}))
        if len(roots) > self.config.max_scan_tracks:
            raise ValueError('probabilistic scan track capacity exceeded')
        if len(roots)*len(indices)+len(roots)+len(indices) > self.config.inference.max_factor_cells:
            raise ValueError('probabilistic factor-cell capacity exceeded before matrix allocation')
        rindex = {r: i for i, r in enumerate(roots)}
        pair, allowed = np.zeros((len(roots), len(indices))), np.zeros((len(roots), len(indices)), dtype=bool)
        for j, groups in enumerate(grouped_rows):
            for root, values in groups.items():
                i = rindex[root]
                pair[i, j], allowed[i, j] = logsumexp(w for _, w in values), True
        factors = LogAssociationFactors(pair, np.zeros(len(roots)), births, allowed)
        marginal = (exact_jpda if self.config.association_algorithm == 'exact' else lbp_jpda)(
            factors, limits=self.config.inference)
        self._work('raw_measurement_projections', len(observations))
        projected = tuple(GaussianBelief(*propagate(np.asarray(o.mean), np.asarray(o.covariance),
            (self.current_reference-o.state_us)/1e6, self.config.state.process_noise)) for o in observations)
        for i, root in enumerate(roots):
            prior = self._state_at(root, self.current_reference)
            if self.config.update_rule == 'pkf':
                columns = np.flatnonzero(allowed[i]).tolist()
                self._work('pkf_measurement_information_terms', len(columns))
                state = pkf_condition(prior, [projected[j].mean for j in columns], np.eye(9),
                    [projected[j].covariance for j in columns], [marginal.pair[i][j] for j in columns],
                    angular_state_indices=(6,), angular_measurement_indices=(6,))
            else:
                components, weights = [prior], [marginal.left_unmatched[i]]
                for j, detection in enumerate(projected):
                    if not allowed[i, j]:
                        continue
                    self._work('conditional_gaussian_updates')
                    if self.config.update_rule == 'jpda-ci':
                        state = GaussianBelief(*ci(prior.mean, np.asarray(prior.covariance), detection.mean,
                            np.asarray(detection.covariance), self.config.state.ci_weight))
                    else:
                        state = kalman_condition(prior, detection.mean, np.eye(9), detection.covariance,
                            angular_state_indices=(6,), angular_measurement_indices=(6,))
                    components.append(state)
                    weights.append(marginal.pair[i][j])
                state = gaussian_moment_match(components, weights, angular_state_indices=(6,))
            self._save_state(root, state)
        # Explicit lifecycle anchor decoder. Joint MAP maximises the summed
        # parent-class weight. The optional Bayes decoder minimises NEW root
        # 0/1 loss and can prefer all births under symmetric ambiguity.
        assigned = {}
        assignment_cells = 0
        if roots:
            benefits = (np.asarray(births)[None, :]-pair if self.config.anchor_decoder == 'joint-map'
                        else np.asarray(marginal.right_unmatched)[None, :]-np.asarray(marginal.pair))
            # Assign the smaller side. Zero-cost dummy choices represent the
            # all-birth baseline; benefits are relative to that same baseline.
            if len(roots) <= len(indices):
                costs = np.zeros((len(roots), len(indices)+len(roots)))
                costs[:, :len(indices)] = np.where(allowed, benefits, np.inf)
                rr, cc = linear_sum_assignment(costs)
                assigned = {j: roots[i] for i, j in zip(rr, cc) if j < len(indices)}
            else:
                costs = np.zeros((len(indices), len(roots)+len(indices)))
                costs[:, :len(roots)] = np.where(allowed, benefits, np.inf).T
                rr, cc = linear_sum_assignment(costs)
                assigned = {j: roots[i] for j, i in zip(rr, cc) if i < len(roots)}
            assignment_cells = costs.size
        anchors = []
        for j, index in enumerate(indices):
            root = assigned.get(j, index)
            # Within an identical root class choose a canonical representative
            # parent. Its individual weight is NOT used as class probability.
            parent = min(p for p, _ in grouped_rows[j][root]) if j in assigned else -1
            if parent not in self._choices(self.chosen_anchor):
                raise ValueError('decoded scan action violates original forest identity constraints')
            self.chosen_anchor = self._child(self.chosen_anchor, parent)
            self.db.execute('INSERT INTO identity_anchors VALUES(?,?)', (index, root))
            if j not in assigned:
                self._save_state(root, projected[j])
            anchors.append(dict(index=index, root=root, representative_parent=parent))
        self.scan_audits.append(dict(source_slot=slot, indices=indices, conditioned_track_roots=roots,
            factors_sha256=factors.digest(), log_pair=factors.log_pair, allowed=factors.allowed,
            log_birth=factors.log_right_unmatched, marginals=asdict(marginal), anchors=anchors,
            negative_raw_projection_count=sum(o.state_us > self.current_reference for o in observations),
            reference_us=self.current_reference, decision_us=decision))
        self.scan_audits[-1]['anchor_assignment_cells'] = assignment_cells

    def _refine(self, decision):
        self.chosen_anchor = self.meta['output']
        start = self._prefix(self.chosen_anchor).depth
        seen = set()
        indices = range(start, self.n)
        key = lambda i: self.db.execute('SELECT source,frame FROM observations WHERE i=?', (i,)).fetchone()
        for slot, group in itertools.groupby(indices, key=key):
            if slot in seen:
                raise ValueError('interleaved source/frame observations require explicit scan order')
            seen.add(slot)
            self._scan(tuple(group), decision)
        self.active, self.frontier = {self.chosen_anchor}, set()
        return 0, False

    def _mass(self):
        # Placeholder interface values are removed from the final sealed audit.
        return [self.chosen_anchor], [self._weight(self.chosen_anchor)], None, None, None

    def _decode(self, *args, **kwargs):
        return self.chosen_anchor, dict(rule=self.config.anchor_decoder,
                                       full_history_risk_bound=None)

    def _predict(self, handle, reference):
        if handle != self.chosen_anchor:
            raise ValueError('probabilistic tracker cannot restore another identity history')
        config = self.config.state
        result = []
        for root, first, last, maximum in self.db.execute(
            'SELECT a.root,MIN(o.state_us),MAX(o.state_us),MAX(o.score) FROM identity_anchors a '
            'JOIN observations o ON a.i=o.i GROUP BY a.root ORDER BY a.root'):
            if maximum < config.birth_score or max(0, reference-last) > config.max_age_us:
                continue
            rows = tuple(self.db.execute('SELECT o.node_id,o.state_us,o.score FROM identity_anchors a '
                'JOIN observations o ON a.i=o.i WHERE a.root=? ORDER BY a.i', (root,)))
            score = max(s * config.survival_per_second**(max(0, reference-t)/1e6) for _, t, s in rows)
            if score < config.prune_score:
                continue
            prior = self._state_at(root, reference)
            node_id = self.db.execute('SELECT node_id FROM observations WHERE i=?', (root,)).fetchone()[0]
            result.append(dict(track_id=self.sequence_id+':'+digest(['forest-birth', node_id])[:24], class_label='car',
                mean=prior.mean, covariance=prior.covariance, score=score,
                observation_ids=tuple(r[0] for r in rows), birth_state_us=first, last_update_us=last))
        return tuple(result)

    def _anchor_digest(self):
        h = hashlib.sha256()
        for row in self.db.execute('SELECT i,root FROM identity_anchors ORDER BY i'):
            h.update(canonical(row)+b'\n')
        return h.hexdigest()

    def _gaussian_digest(self):
        h = hashlib.sha256()
        for root, reference, payload in self.db.execute('SELECT root,reference_us,payload FROM gaussian_tracks ORDER BY root'):
            h.update(canonical([root, reference, json.loads(payload)])+b'\n')
        return h.hexdigest()

    def _finalize_audit(self, audit):
        for key in ('active', 'frontier', 'log_partition_upper', 'eta_upper', 'decision_indices',
                    'restored_ancestors', 'branches'):
            audit.pop(key)
        audit.update(kind=self.SCHEMA, conditional_scans=self.scan_audits,
            association_algorithm=self.config.association_algorithm, update_rule=self.config.update_rule,
            anchor_decoder=self.config.anchor_decoder,
            identity_anchor_sha256=self._anchor_digest(), historical_identity_map_compressed=True,
            gaussian_state_sha256=self._gaussian_digest(),
            retains_alternative_identity_histories=False, recovery_enabled=False,
            state_policy='project_raw_measurements_to_output_reference_then_sequential_scan_updates',
            same_state_time_protocol_as_recoverable=False, full_history_posterior_bound=None,
            unmatched_mass_scales_detection_score=False, lifecycle_uses_hard_anchor_observations=True,
            cross_track_covariances_retained=False, paper_eligible=False)
        audit['state_work_breakdown'] = self.state_work
        audit['state_work_units_are_not_cross_backend_flop_equivalents'] = True
        return audit
