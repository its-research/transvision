#!/usr/bin/env python3
"""Evaluate explicit native KITTI boxes with the source-locked DMSTrack engine.

Use --convert to convert bound RBF world states and native ego-corner GT first.
Conversion requires explicit per-frame poses and official frame mapping; this
module never infers an official sequence number, frame, ego pose or class alias.
Both JSONL inputs must contain every official-test frame, including empty frames:
{sequence_id: "0000", frame_index: 0, objects: [{track_id: 1,
 class_label: "vehicle", box_hwlxyzry: [h,w,l,x,y,z,ry],
 bbox_2d: [x1,y1,x2,y2], score: 0.9}]}
score is prediction-only; native GT truncation/occlusion/alpha are zero.
The caller must supply already converted native boxes and the actual 2D fields;
2D fields affect the unchanged backend's ignored-unmatched-prediction rule.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

COMMIT = 'd3b9949499c8e68ea33060873bd1cb95b6d4d323'
SOURCE_HASHES = {
    'AB3DMOT/scripts/KITTI/evaluate.py': 'a009b926cffbc396c9a131392b47623e32aacc268acc3d4c3c99b3b9e64d39de',
    'AB3DMOT/scripts/KITTI/munkres.py': '9662d22a7e9d148af696c0e26a4b0f349f236c16fd533243d579f0dd3dde8cbc',
    'AB3DMOT/scripts/KITTI/mailpy.py': 'f268398d772297175df09defb73e863c4f15977983c2337385b7c838c8cb5520',
    'AB3DMOT/AB3DMOT_libs/dist_metrics.py': '049adbe586811d7b158b0f8d4548178e8d52aafed7013a0917f3f2615e0786c5',
    'AB3DMOT/AB3DMOT_libs/box.py': '0c7a9d2f16743b912d7024e3756c7b847a21bcae4f7ac086afa0a09bc2ac7268',
    'AB3DMOT/AB3DMOT_libs/kitti_oxts.py': '1165ec524c593f20d916dd3a46096db0d58bfd5e333249bd602d51635a1f0867',
    'AB3DMOT/scripts/KITTI/v2v4real_val_evaluate_tracking.seqmap.val': '840783eed9ab01cdd359015b03ef9b9ba0f867a1ab78882e7dd2d8323f8d1c5f',
}
LENGTHS = (147, 114, 144, 198, 180, 310, 304, 221, 375)
SEQUENCES = {f'{i:04d}': n for i, n in enumerate(LENGTHS)}
PROTOCOL = {
    'dataset': 'v2v4real', 'split': 'official_test',
    'evaluation_class': 'vehicle', 'backend_class': 'Car',
    'backend_split': 'val', 'nominal_frequency_hz': 10,
    'box_coordinates': 'dmstrack_ab3dmot_kitti_hwlxyzry',
    'box_conversion': 'already_applied_upstream_not_inferred_by_metric_wrapper',
    'iou_type': '3D', 'iou_threshold': 0.25,
}
SEQMAP = 'AB3DMOT/scripts/KITTI/v2v4real_val_evaluate_tracking.seqmap.val'
GEOMETRY_HASHES = {
    'V2V4Real/opencood/utils/box_utils.py': '961914a1291f57e86d9a7416e7318a3e911f0de8fb98a0c4492e2a9d5785de86',
    'V2V4Real/opencood/utils/common_utils.py': '12d89510b0e8f2c29def3b2cf1963ae790e226fa056d8d7cc0b0f70cd9d5be21',
    'V2V4Real/opencood/tools/inference.py': '1a3c3b6c9afda4d66edcda9a9df790114376e39933fd960abb0551703cf623e5',
}
WORLD_LAYOUT = 'gravity_xyz_length_width_height_yaw_vxy'
GT_KIND = 'v2v4real_native_vehicle_gt_projection_v1'
GT_RECIPE = 'v2v4real-real-matrix-late-gt-native-vehicle-first-id-two-stage-roi-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(data):
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: '+value)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def regular_file(path):
    path = Path(path).absolute()
    # Parent storage symlinks are allowed: the project test link is intentional.
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular unlinked input file required: '+str(path))
    return path


def source_evidence(source):
    source = Path(source).absolute()
    files = {}
    for relative, expected in SOURCE_HASHES.items():
        path = regular_file(source/relative)
        if sha(path) != expected:
            raise ValueError('pinned DMSTrack source changed: '+relative)
        files[relative] = expected
    lines = (source/SEQMAP).read_text().splitlines()
    parsed = {}
    for line in lines:
        sequence, placeholder, start, end = line.split()
        if sequence in parsed or placeholder != 'empty' or int(start) != 0:
            raise ValueError('unexpected fixed seqmap')
        parsed[sequence] = int(end)+1
    if parsed != SEQUENCES:
        raise ValueError('fixed official-test seqmap differs')
    return {'repository': 'https://github.com/eddyhkchiu/DMSTrack',
            'commit': COMMIT, 'files': files}


def bound_file(base, binding):
    if set(binding) != {'path', 'sha256'} or not isinstance(binding['path'], str):
        raise ValueError('explicit file path and SHA-256 required')
    p = Path(binding['path'])
    if not p.is_absolute():
        if '..' in p.parts:
            raise ValueError('unsafe relative path')
        p = base/p
    p = regular_file(p)
    if sha(p) != binding['sha256']:
        raise ValueError('input checksum mismatch: '+str(p))
    return p


def numbers(values, length, label):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(label+' length differs')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError(label+' must be finite numeric values')
    return values


def geometry_evidence(source):
    files = {}
    for rel, digest in GEOMETRY_HASHES.items():
        if sha(regular_file(Path(source)/rel)) != digest:
            raise ValueError('pinned DMSTrack geometry source changed: '+rel)
        files[rel] = digest
    return {'commit': COMMIT, 'files': files}


def load_geometry(source):
    """Load unchanged fixed numerical definitions and inference export AST.

    RBF mean9 is a gravity-centred world [xyz,l,w,h,yaw,vx,vy] state.
    World corners use the original lwh template; the explicit column-vector
    world_to_ego matrix is passed to the original project_box3d. The original
    exporter then reorders xyz/hwl and swaps y,z. No heading sign is guessed.
    """
    import numpy as np
    import torch
    geometry_evidence(source)
    trees = {Path(rel).name: ast.parse((Path(source)/rel).read_bytes()) for rel in GEOMETRY_HASHES}
    def definitions(tree, names, namespace):
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('fixed geometry definitions missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-DMSTrack-geometry>', 'exec'), namespace)
    common = dict(np=np, torch=torch)
    definitions(trees['common_utils.py'], ('check_numpy_to_torch', 'rotate_points_along_z'), common)
    boxes = dict(np=np, torch=torch, sys=sys, common_utils=types.SimpleNamespace(**common))
    definitions(trees['box_utils.py'], ('boxes_to_corners_3d', 'project_box3d', 'corner_to_center'), boxes)
    exports = {}
    for prediction, function, variable in (
            (True, 'transform_and_save_detection_to_ab3dmot_format', 'v2v4real_detection'),
            (False, 'transform_and_save_tracking_label_to_ab3dmot_format', 'v2v4real_gt')):
        functions = [n for n in trees['inference.py'].body if isinstance(n, ast.FunctionDef) and n.name == function]
        if len(functions) != 1:
            raise ValueError('fixed exporter missing')
        nodes = sorted([n for n in ast.walk(functions[0]) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'boxes_3d' for t in n.targets)],
                       key=lambda n: n.lineno)
        if len(nodes) != 3:
            raise ValueError('fixed exporter array conversion differs')
        exports[prediction] = (compile(ast.Module(body=nodes, type_ignores=[]),
                                      '<pinned-DMSTrack-inference-array-export>', 'exec'), variable)
    def native(corners, prediction):
        code, variable = exports[prediction]
        namespace = dict(np=np, box_utils=types.SimpleNamespace(**boxes))
        namespace[variable] = np.asarray(corners)
        exec(code, namespace)
        return namespace['boxes_3d']
    def world(mean, transform):
        corners = boxes['boxes_to_corners_3d'](np.asarray([mean[:7]], dtype=np.float64), order='lwh')
        # check_numpy_to_torch intentionally retains the fixed source's float32
        # conversion. Record that runtime choice; do not silently replace it.
        return boxes['project_box3d'](corners, np.asarray(transform, dtype=np.float64))
    return world, native


def jsonl(path):
    return [strict_json(line) for line in Path(path).read_bytes().splitlines()]


def source_key(row):
    sid, key = row.get('sequence_id'), row.get('frame_key')
    if not isinstance(sid, str) or not sid or not isinstance(key, str) or len(key) != 6 or not key.isascii() or not key.isdigit():
        raise ValueError('explicit source scene and six-digit frame_key required')
    return sid, key


def indexed_rows(rows, key_function, label):
    result = {}
    for row in rows:
        key = key_function(row)
        if key in result:
            raise ValueError('duplicate '+label+' identity')
        result[key] = row
    if len(result) != 1993:
        raise ValueError('complete 1993 '+label+' rows required, including empty frames')
    return result


def rigid_transform(value):
    import numpy as np
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('explicit world_to_ego 4x4 required')
    matrix = np.asarray([numbers(row, 4, 'world_to_ego row') for row in value], dtype=float)
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('world_to_ego must be an explicit proper rigid column-vector transform')
    return matrix


def convert(manifest, manifest_sha256, source, output):
    """Convert a full official-test cohort; no source data or pose is inferred.

    Manifest kind=v2v4real_world_to_native_vehicle_conversion_input_v1 binds
    predictions, ground_truth_manifest, frame_mapping, world_to_ego by SHA-256.
    frame_mapping rows bind source sequence_id/frame_key/frame_ordinal/ego_cav,
    prediction_frame_id/box_reference_timestamp_us, native_sequence_id/index.
    Pose rows bind sequence_id/frame_key/ego_cav/box_reference_timestamp_us and
    world_to_ego. GT is the separate prepare_v2v4real_ground_truth vehicle output.
    prediction_class_mapping explicitly maps every admitted channel to vehicle
    or null (excluded from evaluation); it never changes upstream competition.
    """
    import numpy as np
    manifest = regular_file(manifest)
    wrapper_hash = sha(__file__)
    if sha(manifest) != manifest_sha256:
        raise ValueError('conversion manifest checksum mismatch')
    spec = strict_json(manifest.read_bytes())
    roles = ('predictions', 'ground_truth_manifest', 'frame_mapping', 'world_to_ego')
    if (set(spec) != {'kind', 'fixture', 'source_commit', 'prediction_class_mapping', *roles}
            or spec['kind'] != 'v2v4real_world_to_native_vehicle_conversion_input_v1'
            or type(spec['fixture']) is not bool or spec['source_commit'] != COMMIT):
        raise ValueError('explicit source-bound conversion input required')
    classes = spec['prediction_class_mapping']
    if (not isinstance(classes, dict) or not classes or 'vehicle' not in classes.values()
            or any(k not in ('car', 'bicycle', 'pedestrian', 'vehicle') or v not in ('vehicle', None)
                   or (k in ('bicycle', 'pedestrian') and v is not None) for k, v in classes.items())):
        raise ValueError('explicit prediction class mapping required')
    paths = {k: bound_file(manifest.parent, spec[k]) for k in roles}
    inputs = {str(manifest): manifest_sha256, **{str(p): spec[k]['sha256'] for k, p in paths.items()}}
    if len(inputs) != 5:
        raise ValueError('separate conversion artifacts required')
    gt_spec = strict_json(paths['ground_truth_manifest'].read_bytes())
    expected_gt = {'kind': GT_KIND, 'recipe': GT_RECIPE, 'split': 'official_test',
                   'evaluation_protocol': 'v2v4real-official-benchmark-vehicle-v1',
                   'native_label_source': 'params.vehicles_except_exact_obj_type_Pedestrian',
                   'coordinate_frame': 'current_ego_lidar', 'time_basis': 'ordinal-only-no-clock',
                   'box_representation': 'eight_ordered_corners_xyz_m', 'paired_frames': 1993}
    if any(gt_spec.get(k) != v for k, v in expected_gt.items()):
        raise ValueError('native vehicle GT projection contract differs')
    gt_path = bound_file(paths['ground_truth_manifest'].parent,
                         {'path': 'frames.jsonl', 'sha256': gt_spec.get('frames_sha256')})
    inputs[str(gt_path)] = gt_spec['frames_sha256']
    gt = indexed_rows(jsonl(gt_path), source_key, 'GT')
    mapping = indexed_rows(jsonl(paths['frame_mapping']), source_key, 'mapping')
    poses = indexed_rows(jsonl(paths['world_to_ego']), source_key, 'pose')
    if set(gt) != set(mapping) or set(gt) != set(poses):
        raise ValueError('GT/mapping/pose source frame coverage differs')
    def prediction_key(row):
        if (not isinstance(row.get('sequence_id'), str) or not isinstance(row.get('frame_id'), str)
                or row.get('coordinate_frame') != 'world' or row.get('state_layout') != WORLD_LAYOUT
                or type(row.get('box_reference_timestamp_us')) is not int
                or not isinstance(row.get('predictions'), list)):
            raise ValueError('explicit RBF world mean9 prediction frame required')
        return row['sequence_id'], row['frame_id'], row['box_reference_timestamp_us']
    predictions = indexed_rows(jsonl(paths['predictions']), prediction_key, 'prediction')
    targets, source_scenes, required_predictions = {}, {}, set()
    mapping_fields = {'sequence_id', 'frame_key', 'frame_ordinal', 'ego_cav', 'prediction_frame_id',
                      'box_reference_timestamp_us', 'native_sequence_id', 'native_frame_index'}
    pose_fields = {'sequence_id', 'frame_key', 'ego_cav', 'box_reference_timestamp_us', 'world_to_ego'}
    for key, row in mapping.items():
        if set(row) != mapping_fields or set(poses[key]) != pose_fields:
            raise ValueError('explicit frame mapping/pose schema differs')
        sid, index = row['native_sequence_id'], row['native_frame_index']
        if (sid not in SEQUENCES or type(index) is not int or not 0 <= index < SEQUENCES[sid]
                or (sid, index) in targets or type(row['frame_ordinal']) is not int or row['frame_ordinal'] < 0
                or index != row['frame_ordinal']
                or row['ego_cav'] not in ('0', '1') or type(row['box_reference_timestamp_us']) is not int
                or not isinstance(row['prediction_frame_id'], str)):
            raise ValueError('invalid or duplicate explicit native frame mapping')
        targets[sid, index] = key
        if source_scenes.setdefault(sid, key[0]) != key[0]:
            raise ValueError('native sequence cannot combine source scenes')
        g, pose = gt[key], poses[key]
        if (type(g.get('frame_ordinal')) is not int or g.get('frame_ordinal') != row['frame_ordinal']
                or g.get('ego_cav') != row['ego_cav']
                or gt_spec.get('ego_agents', {}).get(key[0]) != row['ego_cav']
                or pose['ego_cav'] != row['ego_cav']
                or type(pose['box_reference_timestamp_us']) is not int
                or pose['box_reference_timestamp_us'] != row['box_reference_timestamp_us']):
            raise ValueError('mapping/GT/pose ego, ordinal or time differs')
        required_predictions.add((key[0], row['prediction_frame_id'], row['box_reference_timestamp_us']))
        rigid_transform(pose['world_to_ego'])
    expected = [(s, i) for s, n in SEQUENCES.items() for i in range(n)]
    if (set(targets) != set(expected) or len(set(source_scenes.values())) != 9
            or len(required_predictions) != 1993 or required_predictions != set(predictions)):
        raise ValueError('full official sequence/prediction mapping coverage differs')
    for sid, scene in source_scenes.items():
        if {r['frame_ordinal'] for k, r in mapping.items() if k[0] == scene} != set(range(SEQUENCES[sid])):
            raise ValueError('source ordinal cohort coverage differs')
    native_source, geometry_source = source_evidence(source), geometry_evidence(source)
    world, native = load_geometry(source)
    output = Path(output).absolute()
    if output.exists():
        raise ValueError('new conversion output directory required')
    native_gt, native_predictions, identity_maps = [], [], {}
    excluded = 0
    for sid, index in expected:
        key = targets[sid, index]
        m, g, pose = mapping[key], gt[key], poses[key]
        p = predictions[key[0], m['prediction_frame_id'], m['box_reference_timestamp_us']]
        gboxes, pboxes, gids, pids = [], [], set(), set()
        if not isinstance(g.get('objects'), list):
            raise ValueError('native GT objects list required')
        for obj in g['objects']:
            ident = obj.get('track_id')
            if (type(ident) is not int or ident < 0 or ident in gids or obj.get('evaluation_class') != 'vehicle'
                    or not isinstance(obj.get('raw_class'), str) or not obj['raw_class'] or obj['raw_class'] == 'Pedestrian'):
                raise ValueError('native merged vehicle GT identity/class differs')
            gids.add(ident)
            corners = obj.get('corners_ego')
            if not isinstance(corners, list) or len(corners) != 8:
                raise ValueError('eight ordered GT corners required')
            corners = [numbers(c, 3, 'GT corner') for c in corners]
            # The fixed native GT producer exports torch float32 corners to
            # NPY; JSON serialization has no dtype, so restore that dtype.
            converted = native(np.asarray([corners], dtype=np.float32), False)[0].tolist()
            gboxes.append(dict(track_id=ident, class_label='vehicle', box_hwlxyzry=converted, bbox_2d=[0, 0, 0, 0]))
        idmap = identity_maps.setdefault(sid, {})
        for obj in p['predictions']:
            ident = obj.get('track_id')
            if (type(ident) not in (str, int) or ident == '' or (type(ident) is int and ident < 0)
                    or (type(ident).__name__, str(ident)) in pids):
                raise ValueError('invalid or duplicate prediction identity')
            identity_key = (type(ident).__name__, str(ident))
            pids.add(identity_key)
            if obj.get('class_label') not in classes:
                raise ValueError('prediction channel outside explicit class binding')
            mean = numbers(obj.get('mean'), 9, 'world mean9')
            if min(mean[3:6]) <= 0:
                raise ValueError('positive world dimensions required')
            numbers([obj.get('score')], 1, 'prediction score')
            if classes[obj['class_label']] is None:
                excluded += 1
                continue
            # Bijection over whole source sequence preserves identity continuity.
            # The mapping is recorded, not inferred from geometry or GT labels.
            native_id = idmap.setdefault(identity_key, len(idmap))
            converted = native(world(mean, pose['world_to_ego']), True)[0].tolist()
            pboxes.append(dict(track_id=native_id, class_label='vehicle', box_hwlxyzry=converted,
                               bbox_2d=[0, 0, 0, 0], score=obj['score']))
        native_gt.append(dict(sequence_id=sid, frame_index=index, objects=gboxes))
        native_predictions.append(dict(sequence_id=sid, frame_index=index, objects=pboxes))
    if sha(__file__) != wrapper_hash or any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('conversion input changed during conversion')
    if source_evidence(source) != native_source or geometry_evidence(source) != geometry_source:
        raise ValueError('conversion source changed during conversion')
    output.mkdir(parents=True)
    bindings = {}
    for name, frames in (('ground_truth', native_gt), ('predictions', native_predictions)):
        path = output/(name+'.jsonl')
        with path.open('x') as f:
            for frame in frames:
                f.write(json.dumps(frame, sort_keys=True, allow_nan=False)+'\n')
        read_frames(path, name == 'predictions')
        bindings[name] = {'path': path.name, 'sha256': sha(path)}
    write_json(output/'identity-map.json', {s: [dict(input_type=t, input_track_id=v, native_id=i)
        for (t, v), i in ids.items()] for s, ids in identity_maps.items()})
    receipt = dict(kind='v2v4real_native_box_conversion_receipt_v1', fixture=spec['fixture'],
        target_coordinates=PROTOCOL['box_coordinates'], evaluation_class='vehicle', complete_test_frames=1993,
        ground_truth_sha256=bindings['ground_truth']['sha256'], predictions_sha256=bindings['predictions']['sha256'],
        input_sha256=inputs, source=native_source, geometry_source=geometry_source, wrapper_sha256=wrapper_hash,
        prediction_class_mapping=classes, excluded_prediction_objects=excluded,
        prediction_class_mapping_semantics_independently_accepted=False,
        identity_mapping_sha256=sha(output/'identity-map.json'),
        transform_convention='explicit_column_vector_world_to_current_ego_lidar',
        export_formula='unchanged_inference.py_corner_to_center_hwl_then_xyz_reorder_then_y_z_swap',
        precision='fixed_float32_world_projection_and_restored_native_float32_GT_corner_array',
        bbox_2d='four_zeros_as_fixed_inference_export; native_ignored_unmatched_height_rule_unchanged',
        upstream_competition_modified=False, new_prediction_roi_filter_applied=False,
        poses_and_frame_mapping_independently_accepted=False, conversion_receipt_independently_accepted=False,
        world_state_or_corner_conversion_implemented=True, full_rbf_native_path_accepted=False,
        paper_performance_complete=False)
    write_json(output/'conversion-receipt.json', receipt)
    bindings['conversion_receipt'] = {'path': 'conversion-receipt.json', 'sha256': sha(output/'conversion-receipt.json')}
    write_json(output/'manifest.json', dict(kind='v2v4real_native_vehicle_input_v1', fixture=spec['fixture'],
        protocol=PROTOCOL, source_commit=COMMIT, **bindings))
    return receipt


def read_frames(path, prediction):
    frames = [strict_json(line) for line in path.read_bytes().splitlines()]
    expected = [(sid, i) for sid, count in SEQUENCES.items() for i in range(count)]
    if len(frames) != sum(LENGTHS):
        raise ValueError('complete 9 sequence / 1993 frame coverage required, including empty frames')
    objects = 0
    for frame, (sid, index) in zip(frames, expected):
        if set(frame) != {'sequence_id', 'frame_index', 'objects'}:
            raise ValueError('unexpected native frame schema')
        if (frame['sequence_id'] != sid or type(frame['frame_index']) is not int
                or frame['frame_index'] != index or not isinstance(frame['objects'], list)):
            raise ValueError('ordered official sequence/frame coverage differs')
        ids = set()
        for box in frame['objects']:
            required = {'track_id', 'class_label', 'box_hwlxyzry', 'bbox_2d'}
            if prediction:
                required.add('score')
            if set(box) != required:
                raise ValueError('unexpected native object schema; no implicit world/corner conversion')
            ident = box['track_id']
            if type(ident) is not int or ident < 0 or ident in ids:
                raise ValueError('invalid or duplicate frame track ID')
            ids.add(ident)
            if box['class_label'] != 'vehicle':
                raise ValueError('explicit merged vehicle input required; Car is backend alias only')
            values = numbers(box['box_hwlxyzry'], 7, 'native box')
            if min(values[:3]) <= 0:
                raise ValueError('positive native dimensions required')
            bbox = numbers(box['bbox_2d'], 4, 'native 2D box')
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ValueError('ordered 2D bounds required')
            if prediction:
                numbers([box['score']], 1, 'score')
            objects += 1
    if not objects:
        # The pinned backend cannot evaluate an entirely empty tracker or GT.
        raise ValueError('pinned backend requires at least one object; do not synthesize detections')
    return frames


def validate(manifest, manifest_sha256, source):
    manifest = regular_file(manifest)
    if sha(manifest) != manifest_sha256:
        raise ValueError('manifest checksum mismatch')
    spec = strict_json(manifest.read_bytes())
    required = {'kind', 'fixture', 'protocol', 'source_commit', 'predictions', 'ground_truth', 'conversion_receipt'}
    if set(spec) != required or spec['kind'] != 'v2v4real_native_vehicle_input_v1':
        raise ValueError('explicit native input manifest required')
    if type(spec['fixture']) is not bool or spec['protocol'] != PROTOCOL or spec['source_commit'] != COMMIT:
        raise ValueError('frozen native protocol/commit differs')
    paths = {k: bound_file(manifest.parent, spec[k])
             for k in ('predictions', 'ground_truth', 'conversion_receipt')}
    if paths['predictions'].samefile(paths['ground_truth']):
        raise ValueError('prediction and GT artifacts must be separate')
    conversion = strict_json(paths['conversion_receipt'].read_bytes())
    # A conversion receipt may come from --convert or an explicit upstream
    # export. Bind its scope without pretending it was independently accepted.
    if (conversion.get('kind') != 'v2v4real_native_box_conversion_receipt_v1'
            or conversion.get('fixture') is not spec['fixture']
            or conversion.get('target_coordinates') != PROTOCOL['box_coordinates']
            or conversion.get('evaluation_class') != 'vehicle'
            or conversion.get('complete_test_frames') != 1993
            or conversion.get('predictions_sha256') != spec['predictions']['sha256']
            or conversion.get('ground_truth_sha256') != spec['ground_truth']['sha256']):
        raise ValueError('explicit matching conversion receipt required')
    evidence = source_evidence(source)
    data = {k: read_frames(paths[k], k == 'predictions') for k in ('predictions', 'ground_truth')}
    checksums = {str(manifest): manifest_sha256,
                 **{str(p): spec[k]['sha256'] for k, p in paths.items()}}
    return spec, data, evidence, checksums


def write_native(frames, directory, prediction):
    directory.mkdir(parents=True)
    grouped = {s: [] for s in SEQUENCES}
    for frame in frames:
        for box in frame['objects']:
            values = [frame['frame_index'], box['track_id'], 'Car', 0, 0, 0,
                      *box['bbox_2d'], *box['box_hwlxyzry']]
            if prediction:
                values.append(box['score'])
            grouped[frame['sequence_id']].append(' '.join(map(str, values))+'\n')
    for sid, rows in grouped.items():
        with (directory/(sid+'.txt')).open('x') as f:
            f.writelines(rows)


def load_native(source):
    """Load exact source modules in a dedicated subprocess, no fallback imports.

    kitti_oxts has an unrelated xinshuo_io file loader dependency. Only its
    original roty AST is loaded, with its original numba @jit decorator intact.
    NUMBA_DISABLE_JIT=1 executes that original NumPy body: modern nopython mode
    cannot compile the upstream mixed integer/float nested-list array literal.
    No numerical function body, evaluator threshold or match rule is changed.
    """
    source_evidence(source)
    source = Path(source)
    for name in ('scripts', 'scripts.KITTI', 'AB3DMOT_libs'):
        if name in sys.modules:
            raise ValueError('native module namespace already occupied: '+name)
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    import numpy as np
    from numba import config, jit
    if config.DISABLE_JIT != 1:
        raise ValueError('native compatibility runtime requires NUMBA_DISABLE_JIT=1')
    path = source/'AB3DMOT/AB3DMOT_libs/kitti_oxts.py'
    tree = ast.parse(path.read_bytes(), filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'roty']
    if len(definitions) != 1:
        raise ValueError('pinned roty definition missing')
    rotation = types.ModuleType('AB3DMOT_libs.kitti_oxts')
    rotation.__dict__.update(np=np, jit=jit)
    rotation.__file__ = str(path)
    sys.modules[rotation.__name__] = rotation
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), rotation.__dict__)
    modules = [('AB3DMOT_libs.box', 'AB3DMOT/AB3DMOT_libs/box.py'),
               ('AB3DMOT_libs.dist_metrics', 'AB3DMOT/AB3DMOT_libs/dist_metrics.py'),
               ('scripts.KITTI.munkres', 'AB3DMOT/scripts/KITTI/munkres.py'),
               ('scripts.KITTI.mailpy', 'AB3DMOT/scripts/KITTI/mailpy.py'),
               ('rbf_dmstrack_native_metric', 'AB3DMOT/scripts/KITTI/evaluate.py')]
    for name, relative in modules:
        spec = importlib.util.spec_from_file_location(name, source/relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def backend(source, work):
    print(json.dumps({'stage': 'native_tracking_evaluation', 'status': 'starting',
                      'eta_seconds': None,
                      'eta_reason': 'unchanged_native_engine_has_no_task_progress_rate'}), flush=True)
    native = load_native(source)
    work = Path(work).absolute()
    os.chdir(work/'AB3DMOT')
    result = native.evaluate('rbf_vehicle', native.mailpy.Mail(''), 1, True, False, 0.25,
                             evaluate_v2v4real=True, seq_eval_mode='all', v2v4real_split='val')
    if (not isinstance(result, dict) or set(result) != {'AMOTA', 'AMOTP', 'sAMOTA'}
            or not all(math.isfinite(float(x)) for x in result.values())):
        raise ValueError('native evaluation did not return finite metrics')
    write_json(work/'backend-result.json', {'metrics': result,
        'runtime': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'numba', 'matplotlib')}})


def evaluate(manifest, manifest_sha256, source, output):
    spec, data, evidence, input_hashes = validate(manifest, manifest_sha256, source)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('create-once output required')
    output.mkdir(parents=True)
    native_gt = output/'AB3DMOT/scripts/KITTI/v2v4real_val_label'
    native_pred = output/'AB3DMOT/results/v2v4real/rbf_vehicle/data_0'
    write_native(data['ground_truth'], native_gt, False)
    write_native(data['predictions'], native_pred, True)
    shutil.copyfile(Path(source)/SEQMAP, output/SEQMAP)
    # Empty frames have no KITTI object rows; retain their exact complete index
    # separately, and let the unchanged fixed seqmap govern backend iteration.
    coverage = {role: {'frames': len(frames), 'empty_frames': sum(not f['objects'] for f in frames),
                      'objects': sum(len(f['objects']) for f in frames)} for role, frames in data.items()}
    write_json(output/'frame-index.json', [{'sequence_id': s, 'frame_index': i}
               for s, n in SEQUENCES.items() for i in range(n)])
    staged = {str(p): sha(p) for p in (list(native_gt.glob('*.txt'))+list(native_pred.glob('*.txt'))
                                       +[output/SEQMAP, output/'frame-index.json'])}
    wrapper_hash = sha(__file__)
    command = [sys.executable, str(Path(__file__).absolute()), '--backend', '--source', str(Path(source).absolute()), '--output', str(output)]
    write_json(output/'invocation.json', {'command': command, 'input_sha256': input_hashes,
                                        'source': evidence, 'wrapper_sha256': wrapper_hash,
                                        'coverage': coverage, 'fixture': spec['fixture'],
                                        'runtime_jit_disabled': True, 'eta_seconds': None})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(output/'matplotlib-cache'),
               NUMBA_DISABLE_JIT='1')
    with (output/'native.log').open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
    if process.returncode:
        write_json(output/'failure.json', {'status': 'native_process_failed', 'returncode': process.returncode,
                                          'fixture': spec['fixture'], 'paper_performance_complete': False})
        raise RuntimeError('native evaluator failed; preserved native.log and failure.json')
    if any(sha(p) != digest for p, digest in {**input_hashes, **staged, str(Path(__file__)): wrapper_hash}.items()):
        raise ValueError('inputs or staged native files changed during evaluation')
    if source_evidence(source) != evidence:
        raise ValueError('native source changed during evaluation')
    result = strict_json((output/'backend-result.json').read_bytes())
    if set(result.get('metrics', {})) != {'AMOTA', 'AMOTP', 'sAMOTA'}:
        raise ValueError('native metric result schema differs')
    report = {'kind': 'v2v4real_native_vehicle_metric_receipt_v1', 'status': 'native_metrics_computed',
              'fixture': spec['fixture'], 'protocol': PROTOCOL, 'coverage': coverage, 'sequences': SEQUENCES,
              'source': evidence, 'input_sha256': input_hashes, 'wrapper_sha256': wrapper_hash,
              'metrics': result['metrics'], 'runtime': result['runtime'],
              'metric_directions': {'AMOTA': 'higher', 'AMOTP': 'higher', 'sAMOTA': 'higher'},
              'amotp_definition': 'native_AB3DMOT_mean_IoU_precision_not_center_distance_m',
              'routing': 'explicit_evaluate_v2v4real_true_not_result_name',
              'backend_label_alias': {'vehicle': 'Car'}, 'native_source_modified': False,
              'runtime_jit_disabled': True,
              'runtime_adapter': 'source_AST_roty_only; NUMBA_DISABLE_JIT=1; numerical_bodies_unmodified',
              'world_state_or_corner_conversion_implemented': True,
              'conversion_receipt_independently_accepted': False,
              'full_rbf_native_path_accepted': False, 'paper_performance_complete': False,
              'artifacts': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest')
    p.add_argument('--manifest-sha256')
    p.add_argument('--convert', action='store_true', help='convert bound world mean9 and ego-corner GT input')
    p.add_argument('--evaluate-converted', action='store_true', help='with --convert, also run the fixed native metric engine')
    p.add_argument('--backend', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.evaluate_converted and not a.convert:
        p.error('--evaluate-converted requires --convert')
    if a.backend and a.convert:
        p.error('--backend cannot be combined with --convert')
    if a.backend:
        backend(a.source, a.output)
    else:
        if not a.manifest or not a.manifest_sha256:
            p.error('--manifest and --manifest-sha256 are required')
        result = convert(a.manifest, a.manifest_sha256, a.source, a.output) if a.convert else evaluate(
            a.manifest, a.manifest_sha256, a.source, a.output)
        if a.evaluate_converted:
            converted = Path(a.output)/'manifest.json'
            result = evaluate(converted, sha(converted), a.source, Path(a.output)/'native-evaluation')
        print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
