"""Global scan-oriented MHT using frozen V2 raw-node identity potentials.

Murty ranking is conditioned on EACH surviving identity history, followed by
global top-K selection after each contiguous source/frame scan. Equivalent raw
parents with the same identity are summed, not duplicated as physical tracks.
Each surviving history has its own conditional CI state chain. Pruned histories
never re-enter inference; archival factors are for replay/audit only.

This is a declared local MHT adaptation, not a reproduction of a published
appearance model or an exact complete-history posterior. In particular the
frozen learned row scores use raw observations, not branch-filtered states.
It stays outside the model package while other experiments seal that package.
"""
from __future__ import annotations
import heapq
import json
import math
import sqlite3
from dataclasses import dataclass

import numpy as np

from tools.event_track_v2x.ranked_partial_assignment import RankedAssignmentLimits, k_best_partial_assignments
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, load_tracking_config
from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors, logsumexp
from transvision.models.event_track_v2x.identity_forest import digest
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker, _file_hash, _path


@dataclass(frozen=True)
class PersistentMHTConfig(PersistentForestConfig):
    max_assignment_solves: int = 10000
    max_assignment_frontier: int = 10000
    max_assignment_matrix_cells: int = 1000000
    max_generated_candidates: int = 100000

    def __post_init__(self):
        super().__post_init__()
        if self.state.active_limit > min(4096, self.state.max_frontier):
            raise ValueError('MHT width exceeds ranked output or persistent frontier capacity')
        defaults = ForestTrackingConfig()
        for key in ('expansion_budget', 'max_model_regret'):
            if getattr(self.state, key) != getattr(defaults, key):
                raise ValueError('scan MHT does not use ' + key)


class PersistentMHTTracker(PersistentForestTracker):
    CONFIG_TYPE = PersistentMHTConfig
    SCHEMA = 'persistent_global_scan_mht_v1'

    def __init__(self, path, *, sequence_id, config=None):
        config = config or self.CONFIG_TYPE()
        if type(config) is not self.CONFIG_TYPE:
            raise TypeError('scan MHT configuration required')
        super().__init__(path, sequence_id=sequence_id, config=config)
        self._set('schema', self.SCHEMA)

    def _bounds(self):
        # No valid full-history mass certificate remains after irreversible pruning.
        self.suffix, self.suffix_abs = np.zeros(1), np.zeros(1)

    def _load(self):
        super()._load()
        self.event_start = self.n
        self.scan_audits = []
        self.assignment_solves = self.generated_candidates = 0

    def _weight(self, handle):
        if handle == 0:
            return 0.
        row = self.db.execute('SELECT revision,value FROM weights WHERE h=?', (handle, )).fetchone()
        if row is None or row[0] != self.revision:
            raise ValueError('MHT retained identity-class weight is missing or stale')
        return row[1]

    @classmethod
    def open(cls, path, *, expected_prediction_sha256, expected_database_sha256):
        path = _path(path)
        if (not path.is_file() or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                                      for s in (expected_prediction_sha256, expected_database_sha256)) or _file_hash(path) != expected_database_sha256):
            raise ValueError('sealed MHT database and output hashes required')
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            values = {k: json.loads(v) for k, v in connection.execute('SELECT k,v FROM meta')}
        finally:
            connection.close()
        if values.get('schema') != cls.SCHEMA or values['state']['prediction_sha256'] != expected_prediction_sha256:
            raise ValueError('MHT schema or output head differs')
        obj = cls.__new__(cls)
        obj.path, obj.sequence_id = path, values['sequence_id']
        raw = values['config']
        obj.config = cls.CONFIG_TYPE(**dict(raw, state=load_tracking_config(raw['state'])))
        obj._connect()
        try:
            obj._load()
            if obj.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('MHT database integrity failed')
            if obj.meta['events']:
                result = obj.db.execute('SELECT prediction,audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()
                prediction, audit = (json.loads(v) for v in result)
                if (digest({k: v
                            for k, v in prediction.items() if k != 'commit_sha256'}) != expected_prediction_sha256 or prediction['commit_sha256'] != expected_prediction_sha256
                        or digest(audit) != obj.meta['audit_sha256'] or audit['configuration_sha256'] != obj.configuration_sha256
                        or audit['factor_rows_sha256'] != obj.factor_digest() or audit['retained_history_sha256'] != obj._retained_digest()):
                    raise ValueError('MHT output, configuration, factors or retained histories changed')
            return obj
        except BaseException:
            obj.close()
            raise

    def step(self, observations, rows, **kwargs):
        if kwargs.get('rescored_rows') or kwargs.get('decision_indices') is not None:
            raise ValueError('irreversible MHT rejects old-row rescoring and custom decision scopes')
        self.event_start = self.n
        self.scan_audits = []
        self.assignment_solves = self.generated_candidates = 0
        return super().step(observations, rows, **kwargs)

    def _promote(self, decision):
        pass  # Only whole scans, never per-node promotion or raw-history recovery.

    def _scan_factors(self, handle, indices):
        start = indices[0]
        source, frame = self.db.execute('SELECT source,frame FROM observations WHERE i=?', (start, )).fetchone()

        def root(i):
            return self._prefix(self.ancestor(handle, i + 1)).root

        occupied = {root(i) for i, in self.db.execute('SELECT i FROM observations WHERE source=? AND frame=? AND i<?', (source, frame, start))}
        groups, births, roots = [], [], set()
        for i in indices:
            group = {}
            for parent, weight in self._row(i):
                if parent < 0:
                    births.append(weight)
                elif parent < start:
                    identity = root(parent)
                    if identity not in occupied:
                        group.setdefault(identity, []).append((parent, weight))
                # A parent inside this same source/frame scan is mutually exclusive.
            groups.append(group)
            roots.update(group)
        roots = tuple(sorted(roots))
        n, m = len(roots), len(indices)
        if n * (m + n) + n + m > self.config.max_assignment_matrix_cells:
            raise ValueError('MHT matrix capacity exhausted before dense factor allocation')
        pair = np.zeros((len(roots), len(indices)))
        allowed = np.zeros_like(pair, dtype=bool)
        for j, group in enumerate(groups):
            for i, identity in enumerate(roots):
                if identity in group:
                    pair[i, j] = logsumexp(w for _, w in group[identity])
                    allowed[i, j] = True
        factors = LogAssociationFactors(pair, np.zeros(len(roots)), births, allowed=allowed)
        return roots, groups, factors

    def _refine(self, decision):
        beam = self.frontier | self.active
        if not beam or any(self._prefix(h).depth != self.event_start for h in beam):
            raise ValueError('MHT can extend only the immediately retained predecessor histories')
        scans = []
        seen = set()
        for i, source, frame in self.db.execute('SELECT i,source,frame FROM observations WHERE i>=? ORDER BY i', (self.event_start, )):
            slot = source, frame
            if not scans or scans[-1][0] != slot:
                if slot in seen:
                    raise ValueError('MHT requires contiguous source/frame scans')
                seen.add(slot)
                scans.append((slot, []))
            scans[-1][1].append(i)
        width = self.config.state.active_limit
        for slot, indices in scans:
            candidates, parents_audit = [], []
            for handle in sorted(beam):
                roots, groups, factors = self._scan_factors(handle, indices)
                ranked = k_best_partial_assignments(
                    factors,
                    width,
                    limits=RankedAssignmentLimits(
                        max_solves=self.config.max_assignment_solves - self.assignment_solves,
                        max_frontier=self.config.max_assignment_frontier,
                        max_matrix_cells=self.config.max_assignment_matrix_cells))
                self.assignment_solves += ranked.assignment_solves
                if not (ranked.requested_k_reached or ranked.support_exhausted):
                    raise ValueError('MHT ranked-assignment budget exhausted; no partial event committed')
                parent_weight = self._weight(handle)
                for hypothesis in ranked.hypotheses:
                    self.generated_candidates += 1
                    if self.generated_candidates > self.config.max_generated_candidates:
                        raise ValueError('MHT candidate capacity exhausted; no partial event committed')
                    assigned = {j: roots[i] for i, j in enumerate(hypothesis.choices) if j >= 0}
                    choices, increments = [], []
                    signature = self._prefix(handle).sha256
                    for j, depth in enumerate(indices):
                        identity = assigned.get(j)
                        choice = -1 if identity is None else min(p for p, _ in groups[j][identity])
                        increment = (factors.log_right_unmatched[j] if identity is None else logsumexp(w for _, w in groups[j][identity]))
                        choices.append(choice)
                        increments.append(increment)
                        signature = digest([signature, depth, choice, depth if identity is None else identity])
                    score = math.fsum([parent_weight, hypothesis.log_weight])
                    candidates.append((-score, signature, handle, tuple(choices), tuple(increments)))
                parents_audit.append(
                    dict(
                        parent_handle=handle,
                        parent_sha256=self._prefix(handle).sha256,
                        parent_log_weight=parent_weight,
                        factor_sha256=ranked.factor_sha256,
                        conditioned_track_roots=roots,
                        ranked_hypotheses=len(ranked.hypotheses),
                        requested_k_reached=ranked.requested_k_reached,
                        support_exhausted=ranked.support_exhausted,
                        assignment_solves=ranked.assignment_solves,
                        peak_frontier=ranked.peak_frontier,
                        peak_matrix_cells=ranked.peak_matrix_cells,
                        conditional_scan_omitted_mass_upper=ranked.omitted_mass_upper))
            selected = heapq.nsmallest(width, candidates)
            next_beam = set()
            for negative, signature, parent, choices, increments in selected:
                handle, score = parent, self._weight(parent)
                for choice, increment in zip(choices, increments):
                    if choice not in self._choices(handle):
                        raise ValueError('MHT optimizer produced an illegal identity history')
                    handle = self._child(handle, choice)
                    score = math.fsum([score, increment])
                    self.db.execute('INSERT OR REPLACE INTO weights VALUES(?,?,?)', (handle, self.revision, score))
                if self._prefix(handle).sha256 != signature or not math.isclose(score, -negative, abs_tol=1e-10):
                    raise ValueError('MHT materialized history differs from ranked candidate')
                next_beam.add(handle)
            if not next_beam or len(next_beam) != len(selected):
                raise ValueError('MHT generated duplicate or empty physical histories')
            self.scan_audits.append(
                dict(
                    source=slot[0],
                    frame=slot[1],
                    observation_indices=indices,
                    predecessor_histories=len(beam),
                    generated_candidates=len(candidates),
                    kept=len(next_beam),
                    ranked_parent_scans=parents_audit))
            beam = next_beam
        self.active, self.frontier = beam, set()
        return self.assignment_solves, False

    def _mass(self):
        active = sorted(self.active, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        weights = [self._weight(h) for h in active]
        retained = logsumexp(weights)
        # Ignored by MAP decoder; no invalid finite full-posterior bound escapes audit.
        return active, weights, retained, retained, 1.

    def _decode(self, active, weights, retained, eta, indices, fallback):
        return active[0], dict(policy='MAP_of_retained_global_histories', risk_bound=None, normalized_weights_are_complete_posterior=False)

    def _retained_digest(self):
        return digest([(h, self._prefix(h).sha256, self._weight(h)) for h in sorted(self.active)])

    def _finalize_audit(self, audit):
        if audit['restored_ancestors'] or audit['frontier_restarted_from_root']:
            raise ValueError('irreversible MHT attempted to restore a pruned predecessor')
        audit.pop('eta_upper')
        audit.pop('log_partition_upper')
        audit.update(
            kind=self.SCHEMA,
            recovery_enabled=False,
            archived_prefixes_used_for_recovery=False,
            reproduced_published_mht=False,
            global_scan_mht=True,
            pruning_policy='global_top_k_after_each_source_frame_scan',
            raw_parent_alias_weights_summed=True,
            branch_conditioned_gaussian_states=True,
            potentials_depend_on_filtered_branch_states=False,
            retained_history_sha256=self._retained_digest(),
            scans=self.scan_audits,
            assignment_solves=self.assignment_solves,
            generated_candidates=self.generated_candidates,
            conditional_scan_bounds_are_not_full_history_bounds=True)
        return audit
