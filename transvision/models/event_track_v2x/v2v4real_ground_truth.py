"""Offline strict-Car adaptation of the native late-fusion evaluation GT path.

Real-release annotations are LOCAL to the source CAV, despite legacy helper
names saying 'world'. No detector or inference-input module imports this file.
The ID rule, first-observation deduplication and two ROI stages are explicit;
they are not claimed to repair physical identity annotation quality.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .v2v4real_inputs import V2V4RealInputError, pose_to_world, read_annotations
from .paper_evaluation_policy import VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, native_vehicle

GT_RECIPE = 'v2v4real-real-matrix-late-gt-strict-car-first-id-two-stage-roi-v1'
VEHICLE_GT_RECIPE = 'v2v4real-real-matrix-late-gt-native-vehicle-first-id-two-stage-roi-v1'
GT_RANGE = (-100., -40., -5., 100., 40., 3.)
CORNER_SIGNS = np.array([(x, y, z) for z in (-1., 1.)
                        for x, y in ((1., -1.), (1., 1.), (-1., 1.), (-1., -1.))])


def native_id(object_id, associated_id, cav_id):
    """Official integer namespace, including the deliberate CAV-0 aliasing."""
    def integer(value, *, negative_one=False):
        if type(value) not in (int, str) or not re.fullmatch(r'0|[1-9][0-9]*|-1', str(value)):
            raise V2V4RealInputError('canonical integer native identity required')
        result = int(value)
        if result < 0 and not (negative_one and result == -1):
            raise V2V4RealInputError('unsupported negative identity')
        if result >= 2**53:
            raise V2V4RealInputError('identity exceeds exact interoperable integer range')
        return result
    oid, aid, cav = integer(object_id), integer(associated_id, negative_one=True), integer(cav_id)
    if cav not in (0, 1):
        raise V2V4RealInputError('native GT protocol currently requires CAV 0 and 1')
    result = aid if aid != -1 else oid + 100*cav
    if result >= 2**53:
        raise V2V4RealInputError('mapped identity exceeds exact integer range')
    return result


def matrix_pose(value):
    if np.asarray(value).shape != (4, 4):
        raise V2V4RealInputError('real-release matrix pose required; simulation labels use a different frame')
    return pose_to_world(value)


def upright_lwh(corners):
    """Native edge-average yaw/dimensions, not a fitted minimum-area box."""
    corners = np.asarray(corners)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise V2V4RealInputError('eight finite ordered corners required')
    edges_l = corners[[0, 2, 4, 5], :2] - corners[[3, 1, 7, 6], :2]
    edges_w = corners[[0, 2, 4, 6], :2] - corners[[1, 3, 5, 7], :2]
    headings = corners[[1, 0, 5, 4], :2] - corners[[2, 3, 6, 7], :2]
    dimensions = [np.linalg.norm(edges_l, axis=1).mean(), np.linalg.norm(edges_w, axis=1).mean(),
                  abs(np.mean(corners[4:, 2] - corners[:4, 2]))]
    result = np.r_[corners[[0, 3, 5, 6]].mean(axis=0), dimensions,
                   np.arctan2(headings[:, 1], headings[:, 0]).mean()]
    if not np.isfinite(result).all() or np.any(result[3:6] <= 0):
        raise V2V4RealInputError('degenerate native upright box')
    return result


def upright_corners(boxes, *, collated=False):
    """Preserve native float32 masking versus float64-collated GT arithmetic."""
    import torch
    values = torch.as_tensor(np.asarray(boxes).reshape(-1, 7),
                             dtype=torch.float64 if collated else torch.float32)
    offsets = values[:, None, 3:6] * torch.as_tensor(CORNER_SIGNS / 2, dtype=values.dtype)
    angles = values[:, 6]
    c, s = angles.cos(), angles.sin()
    rotation = torch.zeros((len(values), 3, 3), dtype=values.dtype)
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 2, 2] = s, -s, 1.
    rotated = offsets.float() @ rotation.float()
    # Native rotate_points casts to float32 BEFORE adding collated centers.
    rotated += values[:, None, :3]
    return rotated.float().numpy()


def local_car_boxes(metadata, cav_id):
    return _local_boxes(metadata, cav_id, vehicle=False)


def _local_boxes(metadata, cav_id, *, vehicle):
    annotations = read_annotations(metadata)
    result, used = [], set()
    counts = Counter(a.raw_class for a in annotations)
    for annotation in annotations:
        if not (native_vehicle(annotation.raw_class) if vehicle else annotation.raw_class == 'Car'):
            continue
        identity = native_id(annotation.object_id, annotation.associated_id, cav_id)
        if identity in used:
            raise V2V4RealInputError('two selected annotations share one same-CAV identity')
        used.add(identity)
        # center_world is a legacy numeric field name, NOT native GT semantics.
        local_pose = pose_to_world((*annotation.center_world, *annotation.angle_degrees))
        raw_corners = CORNER_SIGNS * np.asarray(annotation.half_extents)
        raw_corners = raw_corners @ local_pose[:3, :3].T + local_pose[:3, 3]
        box = upright_lwh(raw_corners)
        mask_corners = upright_corners([box])[0]
        inside = ((mask_corners >= GT_RANGE[:3]) & (mask_corners <= GT_RANGE[3:])).all(axis=1)
        if inside.sum() >= 2:
            result.append(dict(track_id=identity, object_id=annotation.object_id,
                associated_id=annotation.associated_id, cav_id=str(cav_id), box_lwh=box))
            if vehicle:
                result[-1]['raw_class'] = annotation.raw_class
    return result, dict(counts)


def prepare_frame(metadata_by_cav, *, ego_cav):
    """Current-frame offline GT only; no history stitching or future ID repair."""
    return _prepare_frame(metadata_by_cav, ego_cav=ego_cav, vehicle=False)


def prepare_vehicle_frame(metadata_by_cav, *, ego_cav):
    """Explicit native single-class vehicle route, retaining original obj_type."""
    return _prepare_frame(metadata_by_cav, ego_cav=ego_cav, vehicle=True)


def _prepare_frame(metadata_by_cav, *, ego_cav, vehicle):
    import torch
    if set(metadata_by_cav) != {'0', '1'} or ego_cav not in metadata_by_cav:
        raise V2V4RealInputError('explicit ego and both native CAV 0/1 frames required')
    poses = {c: matrix_pose(m['lidar_pose']) for c, m in metadata_by_cav.items()}
    order = [ego_cav, *sorted(set(metadata_by_cav) - {ego_cav})]
    chosen, source_counts, local_retained, duplicates = {}, {}, 0, []
    for cav in order:
        boxes, classes = _local_boxes(metadata_by_cav[cav], cav, vehicle=vehicle)
        source_counts[cav] = classes
        local_retained += len(boxes)
        corners = upright_corners([b['box_lwh'] for b in boxes], collated=True)
        transform = torch.from_numpy((np.linalg.inv(poses[ego_cav]) @ poses[cav]).astype(np.float32))
        xyz = torch.from_numpy(corners).transpose(1, 2)
        homogeneous = torch.cat((xyz, torch.ones((len(boxes), 1, 8))), dim=1)
        projected = (transform @ homogeneous)[:, :3, :].transpose(1, 2).numpy()
        for row, points in zip(boxes, projected):
            tid = row['track_id']
            if tid in chosen:
                duplicates.append(dict(track_id=tid, kept_cav=chosen[tid]['cav_id'], suppressed_cav=cav,
                    center_distance_m=float(np.linalg.norm(chosen[tid]['corners_ego'].mean(axis=0)-points.mean(axis=0)))))
                continue
            chosen[tid] = dict(row, corners_ego=points)
    retained, outside = [], []
    for tid, row in sorted(chosen.items()):
        xy = row['corners_ego'][:, :2]
        if ((xy >= GT_RANGE[:2]) & (xy <= GT_RANGE[3:5])).all():
            retained.append(dict(track_id=tid, raw_class=row['raw_class'] if vehicle else 'Car', selected_cav=row['cav_id'],
                selected_object_id=row['object_id'], selected_associated_id=row['associated_id'],
                corners_ego=row['corners_ego'].tolist()))
            if vehicle:
                retained[-1]['evaluation_class'] = 'vehicle'
        else:
            outside.append(tid)
    return dict(objects=retained, audit=dict(raw_classes_by_source=source_counts,
        local_roi_retained=local_retained, duplicate_ids=duplicates, ego_roi_rejected_ids=outside,
        selected_cav_order=order, output_order='numeric_track_id', first_id_selected_before_ego_roi=True))


def load_train_ground_truth(root, *, expected_manifest_sha256):
    """Hash-pinned evaluator/label consumer. Never reads raw YAML or point clouds."""
    return _load_ground_truth(root, expected_manifest_sha256=expected_manifest_sha256, vehicle=False)


def load_vehicle_ground_truth(root, *, expected_manifest_sha256):
    """Separately named native vehicle GT; official test is evaluation-only."""
    return _load_ground_truth(root, expected_manifest_sha256=expected_manifest_sha256, vehicle=True)


def _load_ground_truth(root, *, expected_manifest_sha256, vehicle):
    root = Path(root).absolute()
    if (any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir()
            or re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha256 or '') is None):
        raise V2V4RealInputError('ordinary GT directory and explicit manifest hash required')
    expected_files = {'manifest.json', 'frames.jsonl', 'audit.jsonl'}
    if {p.name for p in root.iterdir()} != expected_files:
        raise V2V4RealInputError('GT projection file inventory differs')
    raw = {}
    for name in expected_files:
        p = root/name
        cap = 1024**2 if name == 'manifest.json' else 512*1024**2
        if p.is_symlink() or not p.is_file() or p.stat().st_size > cap:
            raise V2V4RealInputError('bounded ordinary GT artifact required')
        raw[name] = p.read_bytes()
    if hashlib.sha256(raw['manifest.json']).hexdigest() != expected_manifest_sha256:
        raise V2V4RealInputError('GT manifest SHA-256 differs')
    manifest = json.loads(raw['manifest.json'])
    legacy_contract = (manifest.get('kind') == 'v2v4real_native_train_gt_projection_v1' and manifest.get('recipe') == GT_RECIPE
            and manifest.get('split') == 'train' and manifest.get('class_scope') == ['Car']
            and manifest.get('test_payloads_read') is False)
    vehicle_contract = (manifest.get('kind') == 'v2v4real_native_vehicle_gt_projection_v1'
            and manifest.get('recipe') == VEHICLE_GT_RECIPE
            and manifest.get('evaluation_protocol') == VEHICLE_PROTOCOL
            and manifest.get('native_label_source') == NATIVE_VEHICLE_SELECTION
            and manifest.get('evaluation_class') == 'vehicle'
            and manifest.get('split') in ('train', 'official_test')
            and manifest.get('class_scope') == ['vehicle']
            and manifest.get('test_payloads_read') is (manifest.get('split') == 'official_test'))
    if (not (vehicle_contract if vehicle else legacy_contract)
            or manifest.get('time_basis') != 'ordinal-only-no-clock'
            or manifest.get('coordinate_frame') != 'current_ego_lidar' or manifest.get('inference_input') is not False
            or manifest.get('GT_read') is not True):
        raise V2V4RealInputError('explicit offline native GT contract required')
    for name, key in (('frames.jsonl', 'frames_sha256'), ('audit.jsonl', 'audit_sha256')):
        if hashlib.sha256(raw[name]).hexdigest() != manifest[key]:
            raise V2V4RealInputError('GT stream SHA-256 differs')
    frames = tuple(json.loads(line) for line in raw['frames.jsonl'].splitlines())
    counts, seen, objects = Counter(), set(), 0
    for frame in frames:
        scene, key = frame['sequence_id'], frame['frame_key']
        if (type(scene) is not str or not scene or type(key) is not str or not key.isascii() or not key.isdigit()
                or (scene, key) in seen or type(frame['frame_ordinal']) is not int
                or frame['frame_ordinal'] != counts[scene] or frame['ego_cav'] != manifest['ego_agents'].get(scene)):
            raise V2V4RealInputError('GT frame membership, order or ego differs')
        seen.add((scene, key)); counts[scene] += 1
        ids = []
        for row in frame['objects']:
            tid = row['track_id']
            points = np.asarray(row['corners_ego'])
            valid_class = (native_vehicle(row.get('raw_class')) and row.get('evaluation_class') == 'vehicle'
                           if vehicle else row['raw_class'] == 'Car')
            if (type(tid) is not int or not 0 <= tid < 2**53 or not valid_class
                    or points.shape != (8, 3) or points.dtype.kind not in 'fi' or not np.isfinite(points).all()
                    or not ((points[:, :2] >= GT_RANGE[:2]) & (points[:, :2] <= GT_RANGE[3:5])).all()):
                raise V2V4RealInputError('GT identity, class or geometry differs')
            ids.append(tid)
        if ids != sorted(set(ids)):
            raise V2V4RealInputError('unique numerically sorted per-frame GT IDs required')
        objects += len(ids)
    if (len(frames) != manifest['paired_frames'] or dict(counts) != manifest['sequence_frames']
            or set(counts) != set(manifest['ego_agents']) or objects != manifest['gt_objects']):
        raise V2V4RealInputError('complete declared GT frame/object coverage required')
    return manifest, frames
