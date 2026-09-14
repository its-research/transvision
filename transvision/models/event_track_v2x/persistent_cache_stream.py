"""Formal V2 first-arrival stream into durable long-sequence forest inference.

Scoring context is EXPLICITLY per-arrival candidate row, not the earlier global or component contextual-network protocol. This protocol must also be used by training and fair
baselines. Old potentials are frozen here; the lower-level backend can rescore them, but this adapter does not silently change that policy.
"""
from __future__ import annotations
import json
import math
from dataclasses import asdict

from .forest_cache_stream import CacheDelivery, CacheStreamCommit, VerifiedForestCache
from .forest_row_context import CONTEXT_RECIPE, build_row_contexts
from .forest_tracking import FEATURE_RECIPE, RawIdentityDetection, cache_detections
from .identity_forest import ForestFactors, digest
from .persistent_forest import PersistentForestCommit, PersistentForestTracker
from .recovery_task_scope import SCOPE_MODE, VerifiedEgoPoseTable


class RowContextScorer:
    """Score the new node with its exact time/distance/top-neighbor candidates.

    SQL streams old candidates instead of loading the entire historical window. Gating uses the same raw mean propagation and tie rule as append_parent_support. Only parent_limit+1
    raw feature vectors enter any neural scorer invocation.
    """
    context_recipe = CONTEXT_RECIPE

    def __init__(self, tracker, scorer):
        if not isinstance(tracker, PersistentForestTracker) or not callable(scorer):
            raise ValueError('persistent tracker and factor scorer required')
        signature = getattr(scorer, 'signature', None)
        if not isinstance(signature, str) or len(signature) != 64:
            raise ValueError('explicit frozen scorer signature required')
        self.tracker, self.scorer, self.scorer_signature = tracker, scorer, signature
        self.binding = digest([self.context_recipe, FEATURE_RECIPE, signature, asdict(tracker.config)])

    def rows(self, observations, decision_us):
        new = tuple(observations)
        if (type(decision_us) is not int or any(type(o) is not RawIdentityDetection or o.node.arrival_us > decision_us or o.sequence_id != self.tracker.sequence_id for o in new)):
            raise ValueError('unavailable or invalid raw row-context input')
        if getattr(self.scorer, 'signature', None) != self.scorer_signature:
            raise ValueError('frozen persistent scorer changed')
        t, c = self.tracker, self.tracker.config.state

        def older(lo, hi):
            for i, in t.db.execute('SELECT i FROM observations WHERE state_us BETWEEN ? AND ? ORDER BY i', (lo, hi)):
                yield i, t._observation(i)

        contexts = build_row_contexts(new, old_count=t.n, older_candidates=older, config=c, decision_us=decision_us, sequence_id=t.sequence_id)
        rows = []
        for context in contexts:
            factors = self.scorer(context.observations, context.support, decision_us)
            if (type(factors) is not ForestFactors or factors.nodes != tuple(o.node for o in context.observations)
                    or tuple(tuple(p for p, _ in row) for row in factors.rows) != context.support):
                raise ValueError('row-context scorer changed nodes or candidate support')
            rows.append(tuple((-1 if p < 0 else context.indices[p], weight) for p, weight in factors.rows[-1]))
        return tuple(rows), tuple(c.indices for c in contexts)


class PersistentForestCacheStream:
    """Durable, atomic frame receipt + factors + identity/state output
    commits."""

    def __init__(self, cache, tracker, scorer, *, origin_us, minimum_raw_score=.05, maximum_detections=64, max_receipts=100000, ego_poses=None, candidate_protocol=None):
        if (not isinstance(cache, VerifiedForestCache) or not isinstance(tracker, PersistentForestTracker) or type(origin_us) is not int or origin_us < 0
                or not math.isfinite(minimum_raw_score) or not 0 <= minimum_raw_score <= 1 or type(maximum_detections) is not int or maximum_detections < 1
                or type(max_receipts) is not int or max_receipts < 1 or tracker.sequence_id not in json.loads(cache.manifest_json)['sequences']):
            raise ValueError('invalid formal cache cohort, sequence, selection or receipt capacity')
        scoped = getattr(tracker.config, 'recovery_allocation_scope', None) == SCOPE_MODE
        if (scoped != (ego_poses is not None) or ego_poses is not None and (type(ego_poses) is not VerifiedEgoPoseTable or ego_poses.cache_sha256 != cache.manifest_sha256)):
            raise ValueError('task-scoped recovery requires its matching prediction-only pose table exclusively')
        self.ego_poses = ego_poses
        from .paper_protocol import LEGACY, PAPER
        candidate_protocol = candidate_protocol or tracker.config.state.candidate_protocol
        if candidate_protocol != tracker.config.state.candidate_protocol:
            raise ValueError('cache/tracker candidate protocol differs')
        if candidate_protocol not in {PAPER, LEGACY}:
            raise ValueError('unknown candidate protocol')
        self.candidate_protocol = candidate_protocol
        self.cache, self.tracker = cache, tracker
        self.scoring = RowContextScorer(tracker, scorer)
        self.origin_us, self.minimum_raw_score = origin_us, minimum_raw_score
        self.maximum_detections, self.max_receipts = maximum_detections, max_receipts
        self.configuration_sha256 = digest(self._configuration())
        for key, value in (('cache_binding', self.configuration_sha256), ('scorer_binding', self.scoring.binding)):
            previous = tracker.db.execute('SELECT v FROM meta WHERE k=?', (key, )).fetchone()
            if previous is None and tracker.n or previous is not None and json.loads(previous[0]) != value:
                raise ValueError('resumed cache/scoring protocol differs')

    def _configuration(self):
        base = [self.cache.manifest_sha256, self.tracker.sequence_id, self.origin_us, self.minimum_raw_score, self.maximum_detections, self.max_receipts, self.scoring.binding]
        if self.candidate_protocol != 'rbf-car-first-top64-v1':
            base.append(self.candidate_protocol)
        return base if self.ego_poses is None else base + [SCOPE_MODE, self.ego_poses.manifest_sha256]

    @staticmethod
    def _result(commit):
        return CacheStreamCommit(commit.prediction_json, commit.audit_json,
                                 json.dumps(commit.audit['cache_ingestion'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

    def step(self, deliveries, *, frame_id, reference_us, decision_us, event_id):
        deliveries = tuple(deliveries)
        if (any(type(d) is not CacheDelivery for d in deliveries) or not isinstance(event_id, str) or not event_id or type(decision_us) is not int or not isinstance(frame_id, str)
                or not frame_id or type(reference_us) is not int or not 0 <= reference_us <= decision_us or digest(self._configuration()) != self.configuration_sha256):
            raise ValueError('invalid cache event, decision or changed stream configuration')
        request = digest([frame_id, reference_us, decision_us, [asdict(d) for d in deliveries]])
        t = self.tracker
        # A duplicate cache receipt bypasses tracker.step, but must not bypass
        # the immutable learned allocation policy binding checked there.
        check_policy = getattr(t, '_check_policy', None)
        if check_policy is not None:
            check_policy()
        old = t.db.execute('SELECT prediction,audit FROM events WHERE event_id=?', (event_id, )).fetchone()
        if old:
            commit = PersistentForestCommit(*old)
            ingestion = commit.audit['cache_ingestion']
            if ingestion is None or ingestion['request_sha256'] != request:
                raise ValueError('conflicting duplicate persistent cache event')
            return self._result(commit)
        if decision_us < t.meta['decision_us']:
            raise ValueError('nonmonotonic cache decision')
        new, staged, duplicates = [], {}, []
        for d in sorted(deliveries, key=lambda d: (d.arrival_us, d.side, d.frame_id)):
            _, meta = self.cache.describe(d)
            if (d.sequence_id != t.sequence_id or d.arrival_us > decision_us or d.arrival_us < max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])):
                raise ValueError('cross-sequence or unavailable cache before payload read')
            key = d.side, d.frame_id
            prior = staged.get(key)
            if prior is None:
                prior = t.db.execute('SELECT arrival_us,frame_sha FROM cache_receipts WHERE side=? AND frame=?', key).fetchone()
            if prior is not None:
                if d.arrival_us < prior[0] or d.frame_sha256 != prior[1]:
                    raise ValueError('later event changed first receipt or cache identity')
                duplicates.append(dict(receipt=asdict(d), first_arrival_us=prior[0]))
                continue
            if d.arrival_us < t.meta['decision_us']:
                raise ValueError('new frame withheld past a committed decision')
            staged[key] = d.arrival_us, d.frame_sha256
            new.append(d)
        if t.db.execute('SELECT count(*) FROM cache_receipts').fetchone()[0] + len(new) > self.max_receipts:
            raise ValueError('receipt capacity exceeded before new payload read')
        collected = []
        for d in new:
            current = cache_detections(
                self.cache.load_arrived(d, decision_us),
                arrival_us=d.arrival_us,
                decision_us=decision_us,
                origin_us=self.origin_us,
                minimum_raw_score=self.minimum_raw_score,
                maximum_detections=self.maximum_detections,
                candidate_protocol=self.candidate_protocol)
            if len(collected) + len(current) > t.config.max_new_observations:
                raise ValueError('new-observation batch cap exceeded before scoring or committing')
            collected.extend(current)
        observations = tuple(collected)
        rows, contexts = self.scoring.rows(observations, decision_us)
        ingestion = dict(
            kind='persistent_cache_ingestion_v1',
            request_sha256=request,
            cache_manifest_sha256=self.cache.manifest_sha256,
            configuration_sha256=self.configuration_sha256,
            new_deliveries=[asdict(d) for d in new],
            duplicate_deliveries=duplicates,
            new_observations=len(observations),
            first_arrival_preserved=True,
            class_scope=['car'],
            scorer_context_recipe=self.scoring.context_recipe,
            scorer_context_indices=contexts,
            old_row_rescoring=False,
            gt_model_inputs=False)
        if self.candidate_protocol != 'rbf-car-first-top64-v1':
            ingestion.update(candidate_protocol=self.candidate_protocol, class_scope=['car', 'bicycle', 'pedestrian'], evaluation_class='car')
        if self.ego_poses is not None:
            ingestion['recovery_task_scope'] = self.ego_poses.scope(t, new, reference_us=reference_us, decision_us=decision_us)
        commit = t.step(
            observations,
            rows,
            frame_id=frame_id,
            reference_us=reference_us,
            decision_us=decision_us,
            event_id=event_id,
            cache_ingestion=ingestion,
            scorer_binding=self.scoring.binding)
        return self._result(commit)
