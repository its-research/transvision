"""Hash-pinned V2 cohort and transactional first-arrival ingestion.

Full-tree integrity checking is an offline input audit. Inference only receives
features from deliveries available at the current decision. This adapter does
not open the V2 schema to test, infer a network schedule, or manage windows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .detection_cache_v2 import DetectionCacheV2, SIDES, canonical, load_manifest
from .forest_tracking import CausalForestTracker, cache_detections
from .identity_forest import digest


@dataclass(frozen=True)
class CacheDelivery:
    sequence_id: str
    side: str
    frame_id: str
    arrival_us: int
    frame_sha256: str

    def __post_init__(self):
        if (any(not isinstance(v, str) or not v for v in (self.sequence_id, self.frame_id))
                or self.side not in SIDES or type(self.arrival_us) is not int or self.arrival_us < 0
                or not isinstance(self.frame_sha256, str) or len(self.frame_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.frame_sha256)):
            raise ValueError('invalid cache delivery identity, time or hash')

    @property
    def key(self):
        return self.sequence_id, self.side, self.frame_id


class VerifiedForestCache:
    """One exact train/val V2 cohort; no cross-cohort or feature-space mixing."""
    def __init__(self, root, expected_sha256):
        self.root = Path(root)
        manifest, entries = load_manifest(self.root, expected_sha256)
        self.manifest_sha256 = expected_sha256
        self.manifest_json = canonical(manifest)
        self.index = {}
        appearance, detectors = set(), {s: set() for s in SIDES}
        for entry in entries:
            meta = json.loads((self.root / entry['metadata']['path']).read_bytes())
            appearance.add((meta['feature_method'], meta['feature_checkpoint_sha256']))
            detectors[meta['side']].add(meta['detector_checkpoint_sha256'])
            key = meta['sequence_id'], meta['side'], meta['frame_id']
            self.index[key] = (canonical(entry), canonical(meta))
        if len(appearance) != 1 or any(len(x) > 1 for x in detectors.values()):
            raise ValueError('inconsistent frozen feature space or per-source detector checkpoint')

    def describe(self, delivery):
        if type(delivery) is not CacheDelivery:
            raise TypeError('CacheDelivery required')
        if delivery.key not in self.index:
            raise ValueError('delivery outside the sealed sequence/source/frame cohort')
        entry, meta = (json.loads(x) for x in self.index[delivery.key])
        if delivery.frame_sha256 != entry['frame_sha256']:
            raise ValueError('delivery cache hash differs from sealed frame')
        return entry, meta

    def load_arrived(self, delivery, decision_us):
        entry, meta = self.describe(delivery)
        information_us = max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
        if (type(decision_us) is not int
                or not information_us <= delivery.arrival_us <= decision_us):
            raise ValueError('unavailable or causally inconsistent cache delivery')
        # Payload identity is rechecked here; a startup audit is not a TOCTOU exemption.
        return DetectionCacheV2.load(self.root, entry)


@dataclass(frozen=True)
class CacheStreamCommit:
    prediction_json: bytes
    tracking_audit_json: bytes
    ingestion_audit_json: bytes

    @property
    def prediction(self):
        return json.loads(self.prediction_json)

    @property
    def tracking_audit(self):
        return json.loads(self.tracking_audit_json)

    @property
    def ingestion_audit(self):
        return json.loads(self.ingestion_audit_json)


class CausalForestCacheStream:
    """Deliver V2 frames to one tracker, keeping the FIRST receipt of a frame.

Different observations from one frame remain distinct; repeated transmission
does not change their original arrival features or multiply model evidence.
Ingestion commits only after the tracker's transactional step succeeds.
"""
    def __init__(self, cache, tracker, *, origin_us, minimum_raw_score=.05,
                 maximum_detections=64, max_receipts=100000, max_events=100000):
        if (type(cache) is not VerifiedForestCache or type(tracker) is not CausalForestTracker
                or type(origin_us) is not int or origin_us < 0
                or type(maximum_detections) is not int or maximum_detections < 1
                or not 0 <= minimum_raw_score <= 1
                or type(max_receipts) is not int or max_receipts < 1
                or type(max_events) is not int or max_events < 1):
            raise ValueError('invalid verified cache, tracker, selection or storage limits')
        if tracker.sequence_id not in json.loads(cache.manifest_json)['sequences']:
            raise ValueError('tracker sequence outside cache cohort')
        self.cache, self.tracker = cache, tracker
        self.origin_us, self.minimum_raw_score = origin_us, minimum_raw_score
        self.maximum_detections = maximum_detections
        self.max_receipts, self.max_events = max_receipts, max_events
        self.seen, self.events = {}, {}
        self.last_decision_us = -1
        self.last_audit_sha256 = '0' * 64
        self.configuration_sha256 = digest(self._configuration())

    def _configuration(self):
        return [self.cache.manifest_sha256, self.tracker.sequence_id, self.origin_us,
                self.minimum_raw_score, self.maximum_detections, self.max_receipts, self.max_events]

    def step(self, deliveries, *, frame_id, reference_us, decision_us, event_id):
        deliveries = tuple(deliveries)
        if any(type(d) is not CacheDelivery for d in deliveries):
            raise TypeError('CacheDelivery objects required')
        if (not isinstance(event_id, str) or not event_id or type(decision_us) is not int
                or self.configuration_sha256 != digest(self._configuration())):
            raise ValueError('invalid event/decision or changed ingestion configuration')
        request = digest([frame_id, reference_us, decision_us, [asdict(d) for d in deliveries]])
        if event_id in self.events:
            old_request, old_commit = self.events[event_id]
            if old_request != request:
                raise ValueError('conflicting duplicate ingestion event')
            return old_commit
        if decision_us < self.last_decision_us or len(self.events) >= self.max_events:
            raise ValueError('nonmonotonic decision or ingestion event storage exhausted')
        staged, new, duplicates = dict(self.seen), [], []
        # Order within a received batch cannot change the first receipt.
        for d in sorted(deliveries, key=lambda x: (x.arrival_us, x.side, x.frame_id)):
            _, meta = self.cache.describe(d)
            if (d.sequence_id != self.tracker.sequence_id or d.arrival_us > decision_us
                    or d.arrival_us < max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])):
                raise ValueError('cross-sequence or unavailable delivery before payload access')
            if d.key in staged:
                first = staged[d.key]
                if d.arrival_us < first.arrival_us:
                    raise ValueError('a later event cannot revise the first arrival time')
                duplicates.append({'receipt': asdict(d), 'first_arrival_us': first.arrival_us})
                continue
            if d.arrival_us < self.last_decision_us:
                raise ValueError('new observation was withheld past an already committed decision')
            staged[d.key] = d
            new.append(d)
        if len(staged) > self.max_receipts:
            raise ValueError('receipt storage exhausted before reading new payloads')
        observations = tuple(o for d in new for o in cache_detections(
            self.cache.load_arrived(d, decision_us), arrival_us=d.arrival_us, decision_us=decision_us,
            origin_us=self.origin_us, minimum_raw_score=self.minimum_raw_score,
            maximum_detections=self.maximum_detections))
        ingestion = {'kind': 'recoverable_forest_cache_ingestion_v1',
            'cache_manifest_sha256': self.cache.manifest_sha256,
            'configuration_sha256': self.configuration_sha256,
            'event_id': event_id, 'new_deliveries': [asdict(d) for d in new],
            'duplicate_deliveries': duplicates, 'new_observations': len(observations),
            'previous_audit_sha256': self.last_audit_sha256, 'class_scope': ['car'],
            'first_arrival_preserved': True, 'gt_model_inputs': False}
        commit = self.tracker.step(observations, frame_id=frame_id, reference_us=reference_us,
                                   decision_us=decision_us, event_id=event_id)
        ingestion['prediction_sha256'] = commit.prediction['commit_sha256']
        result = CacheStreamCommit(commit.prediction_json, commit.audit_json, canonical(ingestion))
        self.seen, self.last_decision_us = staged, decision_us
        self.last_audit_sha256 = digest(ingestion)
        self.events[event_id] = request, result
        return result
