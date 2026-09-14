"""Prediction-only task relevance, separate from the identity decision loss.

The pose table is an offline, hash-pinned sidecar, NOT a new navigation stream.
Only poses attached to actually received vehicle cache frames can be selected.
Last received ego position is held, not extrapolated or borrowed from GT.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .detection_cache_v2 import canonical, sha_file

POSE_KIND = 'cache_bound_vehicle_ego_poses_v1'
SCOPE_KIND = 'arrived_vehicle_pose_recovery_allocation_v1'
SCOPE_MODE = 'arrived_vehicle_raw_xy50'
ROW_FIELDS = frozenset(('sequence_id', 'frame_id', 'state_us', 'information_us',
    'cache_frame_sha256', 'ego_translation_world', 'novatel_to_world_sha256', 'lidar_to_novatel_sha256'))


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


class VerifiedEgoPoseTable:
    """Immutable serialized rows; validation alone does not grant arrival access."""
    def __init__(self, path, expected_sha256, cache):
        path = Path(path)
        if (not valid_sha(expected_sha256) or any(p.is_symlink() for p in (path, *path.parents))
                or sha_file(path) != expected_sha256):
            raise ValueError('hash-pinned nonsymlink pose sidecar required')
        payload = path.read_bytes()
        data = json.loads(payload)
        if (payload != canonical(data) or set(data) != {'kind', 'cache_sha256', 'split', 'poses', 'gt_model_inputs'}
                or data['kind'] != POSE_KIND or data['cache_sha256'] != cache.manifest_sha256
                or data['split'] != json.loads(cache.manifest_json)['split'] or data['gt_model_inputs'] is not False
                or type(data['poses']) is not list):
            raise ValueError('exact prediction-only cache-bound pose table required')
        index = {}
        for row in data['poses']:
            if (type(row) is not dict or set(row) != ROW_FIELDS
                    or any(not valid_sha(row[k]) for k in ROW_FIELDS if k.endswith('_sha256'))
                    or type(row['ego_translation_world']) is not list or len(row['ego_translation_world']) != 3
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in row['ego_translation_world'])):
                raise ValueError('invalid pose row or unknown field')
            key = row['sequence_id'], 'vehicle-side', row['frame_id']
            if key not in cache.index or key in index:
                raise ValueError('pose outside vehicle cohort or duplicate')
            entry, meta = map(json.loads, cache.index[key])
            if (type(row['state_us']) is not int or type(row['information_us']) is not int
                    or row['state_us'] != meta['box_reference_timestamp_us']
                    or row['information_us'] != max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
                    or row['cache_frame_sha256'] != entry['frame_sha256']):
                raise ValueError('pose time or cache binding differs')
            index[key] = canonical(row)
        if set(index) != {k for k in cache.index if k[1] == 'vehicle-side'}:
            raise ValueError('pose table must cover the entire sealed vehicle cohort')
        if sha_file(path) != expected_sha256:
            raise ValueError('pose sidecar changed during loading')
        self.manifest_sha256, self.cache_sha256 = expected_sha256, cache.manifest_sha256
        self._index = index

    def scope(self, tracker, new_deliveries, *, reference_us, decision_us):
        # Existing first receipts are authoritative; no look-ahead through the
        # offline pose inventory. Late/out-of-order state times cannot rewind ego.
        receipts = [(frame, arrival, sha) for frame, arrival, sha in tracker.db.execute(
            "SELECT frame,arrival_us,frame_sha FROM cache_receipts WHERE side='vehicle-side'")]
        receipts += [(d.frame_id, d.arrival_us, d.frame_sha256) for d in new_deliveries if d.side == 'vehicle-side']
        choices = []
        for frame, arrival, sha in receipts:
            row = json.loads(self._index[(tracker.sequence_id, 'vehicle-side', frame)])
            if sha != row['cache_frame_sha256'] or not row['information_us'] <= arrival <= decision_us:
                raise ValueError('unavailable or changed pose receipt')
            if row['state_us'] <= reference_us:
                choices.append(dict(row, arrival_us=arrival))
        pose = max(choices, key=lambda r: (r['state_us'], r['frame_id'])) if choices else None
        stale = pose is not None and reference_us-pose['state_us'] > tracker.config.state.window_us
        return dict(kind=SCOPE_KIND, mode=SCOPE_MODE, pose_table_sha256=self.manifest_sha256,
            radius_m=50., pose=pose,
            fallback='missing_arrived_pose' if pose is None else 'stale_arrived_pose' if stale else None,
            pose_age_us=None if pose is None else reference_us-pose['state_us'],
            motion_policy='hold_last_received_ego_position', raw_point_policy='independent_constant_velocity',
            decision_loss_scope_changed=False, input_detections_filtered=False, gt_model_inputs=False)


def validate_scope(scope, tracker, ingestion, reference_us, decision_us):
    """Fail closed at the inference boundary, including persisted receipt checks."""
    fields = {'kind','mode','pose_table_sha256','radius_m','pose','fallback','pose_age_us',
        'motion_policy','raw_point_policy','decision_loss_scope_changed','input_detections_filtered','gt_model_inputs'}
    if (type(scope) is not dict or set(scope) != fields or scope['kind'] != SCOPE_KIND
            or scope['mode'] != SCOPE_MODE or not valid_sha(scope['pose_table_sha256'])
            or scope['radius_m'] != 50. or scope['motion_policy'] != 'hold_last_received_ego_position'
            or scope['raw_point_policy'] != 'independent_constant_velocity'
            or any(scope[k] is not False for k in ('decision_loss_scope_changed','input_detections_filtered','gt_model_inputs'))):
        raise ValueError('invalid prediction-only recovery allocation scope')
    pose = scope['pose']
    if pose is None:
        if scope['fallback'] != 'missing_arrived_pose' or scope['pose_age_us'] is not None:
            raise ValueError('missing pose must explicitly fall back to all recent nodes')
        return
    if (type(pose) is not dict or set(pose) != ROW_FIELDS | {'arrival_us'}
            or pose['sequence_id'] != tracker.sequence_id
            or any(type(pose[k]) is not int for k in ('state_us','information_us','arrival_us'))
            or not 0 <= pose['state_us'] <= reference_us <= decision_us
            or not pose['state_us'] <= pose['information_us'] <= pose['arrival_us'] <= decision_us
            or any(not valid_sha(pose[k]) for k in ROW_FIELDS if k.endswith('_sha256'))
            or type(pose['ego_translation_world']) is not list or len(pose['ego_translation_world']) != 3
            or any(type(v) not in (float,int) or not math.isfinite(v) for v in pose['ego_translation_world'])):
        raise ValueError('future or invalid ego pose')
    receipt = tracker.db.execute("SELECT arrival_us,frame_sha FROM cache_receipts WHERE side='vehicle-side' AND frame=?",
                                 (pose['frame_id'],)).fetchone()
    if receipt is None:
        candidates = [d for d in ingestion['new_deliveries'] if d['side']=='vehicle-side'
                      and d['frame_id']==pose['frame_id'] and d['sequence_id']==tracker.sequence_id]
        receipt = (candidates[0]['arrival_us'], candidates[0]['frame_sha256']) if len(candidates)==1 else None
    if receipt != (pose['arrival_us'], pose['cache_frame_sha256']):
        raise ValueError('ego pose has no matching arrived vehicle receipt')
    age = reference_us-pose['state_us']
    expected = 'stale_arrived_pose' if age > tracker.config.state.window_us else None
    if scope['pose_age_us'] != age or scope['fallback'] != expected:
        raise ValueError('pose age/fallback changed')


def allocation_weights(tracker, summaries, scope, reference_us):
    """These weights prioritize search, not a new Bayes action or metric bound."""
    if scope is None or scope['fallback'] is not None:
        return {c:s['weight'] for c,s in summaries.items()}, None
    ego = scope['pose']['ego_translation_world']
    counts = {}
    for component, summary in summaries.items():
        members = tracker.store.members(component)
        count = 0
        for i in summary['decision_indices']:
            raw = tracker._observation(members[i])
            dt = (reference_us-raw.state_us)/1e6
            x, y = raw.mean[0]+dt*raw.mean[7], raw.mean[1]+dt*raw.mean[8]
            count += math.hypot(x-ego[0], y-ego[1]) < 50.
        counts[component] = count
    total = sum(counts.values())
    return {c:(n/total if total else 0.) for c,n in counts.items()}, counts
