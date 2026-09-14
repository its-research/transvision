"""Train-only V2 row-context artifacts with labels stored outside model inputs.

The low-level builder supports fixtures. Only the preparation CLI certifies the full official train cohort. Raw nodes are stored once per sequence; row contexts are indices, not
copies of embeddings. No GT boxes/IDs are exported to shards.
"""
from __future__ import annotations
import json
from bisect import bisect_left, bisect_right, insort
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .detection_cache_v2 import CLASSES, _immutable, canonical, sha_file
from .forest_cache_stream import CacheDelivery, VerifiedForestCache
from .forest_row_context import CONTEXT_RECIPE, ForestRowContext, build_row_contexts
from .forest_supervision import AnnotationIdentity, DetectionIdentityTarget, RowSupervision, make_row_supervision
from .forest_tracking import FEATURE_RECIPE, ForestTrackingConfig, RawIdentityDetection, cache_detections
from .identity_forest import IdentityNode
from .prediction_features import match_predictions

DATA_KIND = 'persistent_forest_train_rows_v1'
REASONS = ('parent', 'birth', 'out_of_support', 'uncertain_birth', 'ambiguous', 'unmatched_prediction', 'non_car')


def row_protocol(config, minimum_raw_score=.05, maximum_detections=64):
    protocol = dict(
        context_recipe=CONTEXT_RECIPE,
        feature_recipe=FEATURE_RECIPE,
        class_scope=['car'],
        parent_limit=config.parent_limit,
        max_parent_gap_us=config.max_parent_gap_us,
        gate_distance_m=config.gate_distance_m,
        process_noise=config.process_noise,
        minimum_raw_score=minimum_raw_score,
        maximum_detections=maximum_detections,
        arrival_policy='scheduled_pair_snapshot_at_reference_plus_100ms',
        old_row_rescoring=False)
    if config.candidate_protocol != 'rbf-car-first-top64-v1':
        protocol.update(candidate_protocol=config.candidate_protocol, class_scope=['car', 'bicycle', 'pedestrian'], evaluation_class='car')
    return protocol


def frozen_cache_identity(cache):
    """Train and val roots differ, but their frozen producers must agree."""
    shared, detectors = set(), {}
    for _, metadata in cache.index.values():
        meta = json.loads(metadata)
        shared.add((meta['feature_method'], meta['feature_checkpoint_sha256'], meta['calibration_sha256']))
        detectors.setdefault(meta['side'], set()).add(meta['detector_checkpoint_sha256'])
    if len(shared) != 1 or set(detectors) != {'vehicle-side', 'infrastructure-side'} or any(len(v) != 1 for v in detectors.values()):
        raise ValueError('training requires one frozen appearance/calibration and per-side detector')
    method, feature_sha, calibration_sha = next(iter(shared))
    return dict(
        feature_method=method,
        feature_checkpoint_sha256=feature_sha,
        calibration_sha256=calibration_sha,
        detector_checkpoint_sha256={side: next(iter(shas))
                                    for side, shas in sorted(detectors.items())})


@dataclass(frozen=True)
class FrameSupervision:
    sequence_id: str
    source_id: int
    frame_id: str
    timestamp_us: int
    boxes: np.ndarray
    annotations: tuple

    def __post_init__(self):
        boxes = np.asarray(self.boxes, dtype=float)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 7)
        if (boxes.shape != (len(self.annotations), 7) or not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0) or type(self.timestamp_us) is not int or self.timestamp_us < 0
                or type(self.source_id) is not int or self.source_id not in (0, 1) or any(
                    type(a) is not AnnotationIdentity or (a.sequence_id, a.source_id, a.frame_id) != (self.sequence_id, self.source_id, self.frame_id) for a in self.annotations)):
            raise ValueError('invalid converted frame supervision')
        object.__setattr__(self, 'boxes', boxes)


def frame_from_converted(info, source_id):
    tracks = np.asarray(info['gt_inds'])
    names, tokens = info['gt_names'], info['anno_tokens']
    if tracks.ndim != 1 or not (len(tracks) == len(names) == len(tokens)):
        raise ValueError('converted identity annotation lengths differ')
    if any(not np.isfinite(float(t)) or float(t) != int(t) or int(t) < 0 for t in tracks):
        raise ValueError('converted track IDs must be nonnegative integers')
    scene, frame = str(info['scene_token']), str(info['token'])
    annotations = tuple(AnnotationIdentity(scene, source_id, frame, str(int(t)).zfill(6), str(token), str(name)) for t, token, name in zip(tracks, tokens, names))
    return FrameSupervision(scene, source_id, frame, int(info['timestamp']), np.asarray(info['gt_boxes']), annotations)


def exact_cooperative_links(labels, left, right):
    lookup = [{a.reference for a in frame.annotations} for frame in (left, right)]
    used, result = set(), []
    for label in labels:
        if label['veh_frame_id'] != left.frame_id or label['inf_frame_id'] != right.frame_id:
            raise ValueError('cooperative label frame differs')
        refs = []
        for source, (prefix, frame) in enumerate((('veh', left), ('inf', right))):
            track, token = str(label[prefix + '_track_id']), str(label[prefix + '_token'])
            if track == '-1':
                if token != '-1':
                    raise ValueError('unmatched annotation carries a token')
                refs.append(None)
                continue
            ref = frame.sequence_id, source, frame.frame_id, track, token
            if ref not in lookup[source] or ref in used:
                raise ValueError('missing or repeated cooperative annotation')
            used.add(ref)
            refs.append(ref)
        if all(ref is not None for ref in refs):
            result.append(tuple(refs))
    return tuple(result)


def labels_for_frame(frame, supervision, identity_index, selected, *, distance=2.):
    metadata = frame.metadata
    source = 0 if metadata['side'] == 'vehicle-side' else 1
    if ((metadata['sequence_id'], source, metadata['frame_id'], metadata['box_reference_timestamp_us']) !=
        (supervision.sequence_id, supervision.source_id, supervision.frame_id, supervision.timestamp_us)):
        raise ValueError('prediction/converted GT frame alignment differs')
    classes = np.asarray([CLASSES.index(a.class_name) if a.class_name in CLASSES else -1 for a in supervision.annotations])
    # Full raw frame matching BEFORE the car score/top-count selection. A removed
    # high-score duplicate must not promote an unmatched lower-score prediction.
    matched = match_predictions(frame.states, frame.raw_scores, frame.class_indices, supervision.boxes, classes, distance=distance)
    labels, frame_sha = [], frame.digest()
    for raw in selected:
        if raw.source_cache_sha256 != frame_sha or raw.node.frame_id != metadata['frame_id']:
            raise ValueError('selected raw node not from matched source frame')
        gt_index = int(matched[raw.detection_index])
        labels.append(
            DetectionIdentityTarget(raw.sequence_id, source, None, status='unmatched_prediction') if gt_index < 0 else identity_index.targets[supervision.annotations[gt_index].
                                                                                                                                              reference])
    return tuple(labels)


def _new_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def write_shard(path, sequence_id, observations, contexts, targets, parent_limit):
    n, width = len(observations), parent_limit + 1
    indices = np.full((n, width), -1, dtype=np.int64)
    positive, known = np.zeros((n, width), dtype=bool), np.zeros((n, width), dtype=bool)
    for i, (context, target) in enumerate(zip(contexts, targets)):
        length = len(context.indices)
        indices[i, :length], positive[i, :length], known[i, :length] = context.indices, target.positives, target.known
    if not len(contexts) == len(targets) == n:
        raise ValueError('one row per accepted raw observation required')
    values = dict(
        features=np.asarray([o.features for o in observations], dtype=np.float64).reshape(n, 203),
        mean=np.asarray([o.mean for o in observations], dtype=np.float64).reshape(n, 9),
        covariance=np.asarray([o.covariance for o in observations], dtype=np.float64).reshape(n, 9, 9),
        score=np.asarray([o.score for o in observations], dtype=np.float64),
        source=np.asarray([o.node.source_id for o in observations], dtype=np.int64),
        node_id=np.asarray([o.node.node_id for o in observations], dtype='U64'),
        frame_id=np.asarray([o.node.frame_id for o in observations], dtype='U128'),
        cache_sha256=np.asarray([o.source_cache_sha256 for o in observations], dtype='U64'),
        information_us=np.asarray([o.node.information_us for o in observations], dtype=np.int64),
        arrival_us=np.asarray([o.node.arrival_us for o in observations], dtype=np.int64),
        state_us=np.asarray([o.state_us for o in observations], dtype=np.int64),
        detection_index=np.asarray([o.detection_index for o in observations], dtype=np.int64),
        contexts=indices,
        positive=positive,
        known=known,
        lengths=np.asarray([len(c.indices) for c in contexts], dtype=np.int64),
        decision_us=np.asarray([c.decision_us for c in contexts], dtype=np.int64),
        reason=np.asarray([t.reason for t in targets], dtype='U24'))
    if any(len(o.node.node_id) > 64 or len(o.node.frame_id) > 128 for o in observations):
        raise ValueError('raw identifiers exceed lossless shard capacity')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)
    counts = dict(Counter(t.reason for t in targets))
    return dict(
        sequence_id=sequence_id, path=Path(path).name, sha256=sha_file(path), nodes=n, supervised_rows=counts.get('birth', 0) + counts.get('parent', 0), reason_counts=counts)


class TrainingShard:

    def __init__(self, path, record, parent_limit):
        if sha_file(path) != record['sha256']:
            raise ValueError('training shard identity changed')
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        n, a = record['nodes'], self.arrays
        node_keys = ('features', 'mean', 'covariance', 'score', 'source', 'node_id', 'frame_id', 'cache_sha256', 'information_us', 'arrival_us', 'state_us', 'detection_index')
        expected = {*node_keys, 'contexts', 'positive', 'known', 'lengths', 'decision_us', 'reason'}
        if (set(a) != expected or any(len(v) != n for v in a.values()) or a['features'].shape != (n, 203) or a['mean'].shape != (n, 9) or a['covariance'].shape != (n, 9, 9)
                or any(a[k].shape != (n, parent_limit + 1) for k in ('contexts', 'positive', 'known')) or a['positive'].dtype != bool or a['known'].dtype != bool
                or any(a[k].dtype != np.int64 for k in ('contexts', 'lengths', 'decision_us', 'information_us', 'arrival_us', 'state_us', 'source', 'detection_index'))
                or np.any((a['lengths'] < 1) | (a['lengths'] > parent_limit + 1)) or dict(Counter(a['reason'].tolist())) != record['reason_counts']
                or int(np.count_nonzero(a['positive'].any(1))) != record['supervised_rows']):
            raise ValueError('invalid train shard arrays or counts')
        # Cache only validated, prediction-only observations, never neural
        # embeddings/logits or targets. A shard lives for one sequence; reuse
        # across overlapping contexts avoids repeating covariance validation.
        # Irreversible read-only backing prevents stale cached values if a
        # caller tries to modify the hash-checked arrays after construction.
        self.arrays = MappingProxyType({key: _immutable(value) for key, value in a.items()})
        self.sequence_id, self.record = record['sequence_id'], record
        self.valid_indices = _immutable(np.flatnonzero(a['positive'].any(1)))
        self._observations = {}

    def _observation(self, index):
        if index not in self._observations:
            a = self.arrays
            raw = RawIdentityDetection(
                self.sequence_id,
                IdentityNode(str(a['node_id'][index]), int(a['source'][index]), int(a['information_us'][index]), int(a['arrival_us'][index]), str(a['frame_id'][index])),
                int(a['detection_index'][index]), int(a['state_us'][index]), a['mean'][index], a['covariance'][index], float(a['score'][index]), a['features'][index],
                str(a['cache_sha256'][index]))
            self._observations[index] = raw
        return self._observations[index]

    def example(self, row):
        a = self.arrays
        length = int(a['lengths'][row])
        indices = tuple(map(int, a['contexts'][row, :length]))
        if (indices[-1] != row or indices != tuple(sorted(set(indices))) or indices[0] < 0 or np.any(a['contexts'][row, length:] != -1) or a['positive'][row, length:].any()
                or a['known'][row, length:].any()):
            raise ValueError('invalid ordered row context or padding')
        raw = tuple(self._observation(i) for i in indices)
        return (ForestRowContext(indices, raw, int(a['decision_us'][row])),
                RowSupervision(tuple(map(bool, a['positive'][row, :length])), tuple(map(bool, a['known'][row, :length])), str(a['reason'][row])))


def prepare_training_rows(cache, rows, frames, identity_index, output, *, config=None, provenance=None, finalize_check=None):
    """Offline builder; no tracker inference or state averaging is involved."""
    if not isinstance(cache, VerifiedForestCache) or json.loads(cache.manifest_json)['split'] != 'train':
        raise ValueError('sealed train-only cache required')
    config = config or ForestTrackingConfig()
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new output directory without symlink traversal required')
    cohort = json.loads(cache.manifest_json)['sequences']
    if not rows or [r['sequence_id'] for r in rows] != sorted(r['sequence_id'] for r in rows):
        raise ValueError('nonempty sequence-contiguous training schedule required')
    output.mkdir()
    protocol = row_protocol(config)
    if (config.candidate_protocol != 'rbf-car-first-top64-v1' and identity_index.class_scope != ('car', 'bicycle', 'pedestrian')):
        raise ValueError('main protocol requires all-class supervision; car filtering belongs to evaluation')
    _new_json(output / 'preparation-plan.json',
              dict(cache_sha256=cache.manifest_sha256, row_protocol=protocol, provenance=provenance or {}, schedule_frames=len(rows), paper_eligible=False))
    shards, rejected, receipt_count = [], 0, 0
    try:
        for sequence in cohort:
            scene_rows = [r for r in rows if r['sequence_id'] == sequence]
            if not scene_rows:
                raise ValueError('training schedule omits a sealed sequence')
            observations, contexts, targets, labels, times, prior, receipts = [], [], [], [], [], set(), set()
            origin = min(json.loads(meta)['box_reference_timestamp_us'] for key, (_, meta) in cache.index.items() if key[0] == sequence)
            previous = -1
            for row in scene_rows:
                reference = row['box_reference_timestamp_us']
                if type(reference) is not int or reference <= previous:
                    raise ValueError('nonmonotonic training reference schedule')
                previous, decision = reference, reference + 100_000
                deliveries = []
                for side, field in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
                    entry, meta = (json.loads(v) for v in cache.index[(sequence, side, row[field])])
                    if side == 'vehicle-side' and meta['box_reference_timestamp_us'] != reference:
                        raise ValueError('training cache/reference differs')
                    if (side, row[field]) in receipts:
                        raise ValueError('reused cooperative source frame')
                    receipts.add((side, row[field]))
                    if max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us']) > decision:
                        rejected += 1
                        continue
                    deliveries.append(CacheDelivery(sequence, side, row[field], decision, entry['frame_sha256']))
                new, new_labels = [], []
                for d in sorted(deliveries, key=lambda d: (d.arrival_us, d.side, d.frame_id)):
                    frame = cache.load_arrived(d, decision)
                    selected = cache_detections(frame, arrival_us=d.arrival_us, decision_us=decision, origin_us=origin, candidate_protocol=config.candidate_protocol)
                    new.extend(selected)
                    supervision = frames.get((sequence, d.side, d.frame_id))
                    if supervision is None:
                        supervision = frames[(d.side, d.frame_id)]
                    new_labels.extend(labels_for_frame(frame, supervision, identity_index, selected))
                    receipt_count += 1

                def older(lo, hi):
                    start, stop = bisect_left(times, (lo, -1)), bisect_right(times, (hi, len(observations)))
                    # Candidate gate has a stable index tie rule, independent of stream order.
                    for _, i in times[start:stop]:
                        yield i, observations[i]

                current = build_row_contexts(new, old_count=len(observations), older_candidates=older, config=config, decision_us=decision, sequence_id=sequence)
                labels.extend(new_labels)
                for context, label in zip(current, new_labels):
                    targets.append(make_row_supervision(context, labels, prior))
                    prior.add(label)
                for raw in new:
                    insort(times, (raw.state_us, len(observations)))
                    observations.append(raw)
                contexts.extend(current)
            record = write_shard(output / f'sequence-{len(shards):04d}.npz', sequence, observations, contexts, targets, config.parent_limit)
            shards.append(record)
        if {r['sequence_id'] for r in rows} != set(cohort):
            raise ValueError('training schedule includes another sequence')
        if finalize_check is not None:
            finalize_check()
        manifest = dict(
            kind=DATA_KIND,
            split='train',
            sequences=cohort,
            shards=shards,
            row_protocol=protocol,
            frozen_cache_identity=frozen_cache_identity(cache),
            cache_manifest_sha256=cache.manifest_sha256,
            provenance=provenance or {},
            scheduled_frames=len(rows),
            ingested_source_frames=receipt_count,
            source_unavailable=rejected,
            sealed_source_frames=len(cache.index),
            annotation_audit=identity_index.audit,
            labels_in_model_inputs=False,
            gt_boxes_or_ids_in_shards=False,
            paper_eligible=False,
            matching_rule='full_raw_score_order_same_class_xy_strict_lt_2m_before_candidate_selection',
            calibration_claim='none_local_partial_label_surrogate_not_global_forest_likelihood')
        _new_json(output / 'manifest.json', manifest)
        return manifest
    except BaseException as error:
        _new_json(output / 'failure.json',
                  dict(status='failed', error_type=type(error).__name__, error=str(error), completed_shards=len(shards), partial_outputs_not_training_ready=True))
        raise
