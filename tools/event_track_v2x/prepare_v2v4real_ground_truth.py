#!/usr/bin/env python3
"""Offline TRAIN-volume strict-Car GT projection and native numeric comparison.

No detector, test selection, identity learning, tracking metric or upload. This
is a separate evaluator artifact, NEVER part of the inference-only projection.
The historical default remains strict-Car/train. The separately named vehicle
protocol permits official test for evaluator-only native vehicle GT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.event_track_v2x.audit_v2v4real_native_volume import audit_volume
from tools.event_track_v2x.extract_v2v4real_archive import ordinary
from tools.event_track_v2x.prepare_v2v4real_inputs import _read
from tools.event_track_v2x.v2v4real_gt_oracle import native_oracle, SOURCE_HASHES, COMMIT
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.v2v4real_inputs import load_raw_yaml, MAX_YAML_BYTES
from transvision.models.event_track_v2x.v2v4real_ground_truth import prepare_frame, prepare_vehicle_frame, native_id, GT_RECIPE, VEHICLE_GT_RECIPE, GT_RANGE
from transvision.models.event_track_v2x.paper_evaluation_policy import VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, native_vehicle

NUMERIC_ATOL_M = 1e-4  # Fixed before real comparison: float32 compound geometry at a 100 m ROI.
SOURCES = ('tools/event_track_v2x/prepare_v2v4real_ground_truth.py',
    'tools/event_track_v2x/v2v4real_gt_oracle.py', 'tools/event_track_v2x/audit_v2v4real_native_volume.py',
    'tools/event_track_v2x/extract_v2v4real_archive.py', 'tools/event_track_v2x/prepare_v2v4real_inputs.py',
    'transvision/models/event_track_v2x/v2v4real_ground_truth.py',
    'transvision/models/event_track_v2x/paper_evaluation_policy.py',
    'transvision/models/event_track_v2x/v2v4real_inputs.py', 'transvision/models/event_track_v2x/v2v4real_numpy_yaml.py')


def prepare(volume, receipt_sha256, ego_agents, output, *, oracle_root=None, evaluation_protocol=None):
    if evaluation_protocol not in (None, VEHICLE_PROTOCOL):
        raise ValueError('unknown explicitly named evaluation protocol')
    vehicle = evaluation_protocol == VEHICLE_PROTOCOL
    volume, ego_agents = ordinary(volume, directory=True), ordinary(ego_agents)
    output = Path(output).absolute()
    ordinary(output.parent, directory=True)
    if output.exists() or output.is_symlink() or volume == output or volume in output.parents:
        raise ValueError('fresh evaluator output outside native volume required')
    source_hashes = {p: sha_file(ROOT/p) for p in SOURCES}
    ego_hash = sha_file(ego_agents)
    egos = json.loads(ego_agents.read_bytes())
    # Check the pinned split BEFORE opening any raw labels through audit_volume.
    receipt_file = ordinary(volume/'receipt.json')
    if sha_file(receipt_file) != receipt_sha256:
        raise ValueError('volume receipt SHA-256 differs')
    receipt = json.loads(receipt_file.read_bytes())
    if not vehicle and receipt.get('split') != 'train':
        raise ValueError('protocol development CLI accepts official train only; test remains unopened')
    if vehicle and receipt.get('split') not in ('train', 'test', 'official_test'):
        raise ValueError('native vehicle GT requires official train or official test volume')
    split = 'train' if receipt.get('split') == 'train' else 'official_test'
    native_audit = audit_volume(volume, receipt_sha256)
    files = {r['path']: r for r in receipt['files'] if r['path'].endswith('.yaml')}
    groups = defaultdict(dict)
    for name in sorted(files):
        scene, cav, filename = name.split('/')
        groups[(scene, Path(filename).stem)][cav] = name
    if not isinstance(egos, dict) or set(egos) != {s for s, _ in groups} or any(v not in ('0', '1') for v in egos.values()):
        raise ValueError('one explicit CAV 0/1 ego for every and only included sequence required')
    frames, audit_rows, ordinals = [], [], Counter()
    raw_classes, local_id_values = Counter(), defaultdict(set)
    error_max = 0.
    oracle_frames = 0
    with native_oracle(oracle_root, vehicle=vehicle) if oracle_root else nullcontext(None) as oracle:
        for (scene, key), paths in sorted(groups.items()):
            if set(paths) != {'0', '1'}:
                raise ValueError('both source labels required at every frame')
            metadata = {}
            for cav, name in paths.items():
                data = _read(volume/'payload'/name, MAX_YAML_BYTES)
                if hashlib.sha256(data).hexdigest() != files[name]['sha256']:
                    raise ValueError('native label changed after volume verification')
                metadata[cav] = load_raw_yaml(data)
                for oid, annotation in metadata[cav]['vehicles'].items():
                    raw_classes[annotation['obj_type']] += 1
                    if native_vehicle(annotation['obj_type']) if vehicle else annotation['obj_type'] == 'Car':
                        local_id_values[(scene, cav, str(oid))].add(native_id(oid, annotation.get('ass_id', oid), cav))
            result = (prepare_vehicle_frame if vehicle else prepare_frame)(metadata, ego_cav=egos[scene])
            if oracle:
                reference = oracle(metadata, egos[scene])
                actual = {o['track_id']: np.asarray(o['corners_ego']) for o in result['objects']}
                if actual.keys() != reference.keys():
                    raise ValueError(f'native reference IDs or ROI membership differ: {scene}/{key}')
                error = max((float(np.max(np.abs(actual[k]-reference[k]))) for k in actual), default=0.)
                if not np.isfinite(error) or error > NUMERIC_ATOL_M:
                    raise ValueError(f'native reference corners differ: {scene}/{key}, error={error}')
                error_max = max(error_max, error)
                oracle_frames += 1
            frames.append(dict(sequence_id=scene, frame_key=key, frame_ordinal=ordinals[scene],
                ego_cav=egos[scene], objects=result['objects']))
            audit_rows.append(dict(sequence_id=scene, frame_key=key, **result['audit']))
            ordinals[scene] += 1
            if len(frames) % 100 == 0:
                print(json.dumps(dict(kind='native_gt_progress', paired_frames=len(frames), scheduled=len(groups))), flush=True)
    if (len(frames) != native_audit['paired_frames'] or sha_file(ego_agents) != ego_hash
            or sha_file(receipt_file) != receipt_sha256
            or any(sha_file(ROOT/p) != h for p, h in source_hashes.items())):
        raise ValueError('coverage, input pins or implementation changed')
    raw_stream = b''.join(canonical(f)+b'\n' for f in frames)
    audit_stream = b''.join(canonical(f)+b'\n' for f in audit_rows)
    manifest = dict(kind='v2v4real_native_train_gt_projection_v1', recipe=GT_RECIPE,
        split='train', class_scope=['Car'], time_basis='ordinal-only-no-clock', coordinate_frame='current_ego_lidar',
        box_representation='eight_ordered_corners_xyz_m', gt_range=list(GT_RANGE),
        source_local_roi='at_least_two_upright_corners_inside_xyz', final_ego_roi='all_eight_corners_inside_xy',
        deduplication='first_cav_in_explicit_ego_first_order_before_final_roi',
        volume_receipt_sha256=receipt_sha256, ego_agents_sha256=ego_hash, ego_agents=egos,
        source_sha256=source_hashes, paired_frames=len(frames), source_frames=len(files),
        sequence_frames=dict(ordinals), gt_objects=sum(len(f['objects']) for f in frames),
        raw_annotation_counts=dict(raw_classes), raw_count_unit='per_source_frame_annotation',
        identity_rule='ass_id_unless_minus_one_else_object_id_plus_100_times_cav_id',
        local_tracks_with_multiple_mapped_ids=sum(len(v)>1 for v in local_id_values.values()),
        local_track_count=len(local_id_values), physical_identity_continuity_verified=False,
        reference_commit=COMMIT, reference_source_sha256=SOURCE_HASHES if oracle_root else None,
        reference_scope='unchanged_numeric_GT_definitions_with_strict_Car_prefilter_not_full_loader_or_detector',
        oracle_compared_frames=oracle_frames, oracle_all_ids_and_roi_match=oracle_frames == len(frames),
        oracle_corner_atol_m=NUMERIC_ATOL_M, oracle_max_abs_corner_error_m=error_max if oracle_root else None,
        frames_sha256=hashlib.sha256(raw_stream).hexdigest(), audit_sha256=hashlib.sha256(audit_stream).hexdigest(),
        runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform()),
        GT_read=True, inference_input=False, public_results_directly_comparable=False,
        full_official_split_verified=False, detector_executed=False, tracking_evaluation_performed=False,
        parameter_training_performed=False, test_payloads_read=False, paper_eligible=False)
    if vehicle:
        manifest.update(kind='v2v4real_native_vehicle_gt_projection_v1', recipe=VEHICLE_GT_RECIPE,
            split=split, class_scope=['vehicle'], evaluation_class='vehicle',
            evaluation_protocol=VEHICLE_PROTOCOL, native_label_source=NATIVE_VEHICLE_SELECTION,
            native_class_codes_are_not_paper_category_names=True,
            selected_native_annotation_counts={k: v for k, v in raw_classes.items() if native_vehicle(k)},
            excluded_native_annotation_counts={k: v for k, v in raw_classes.items() if not native_vehicle(k)},
            reference_scope='unchanged_native_numeric_GT_definitions_and_native_Pedestrian_exclusion',
            test_payloads_read=split == 'official_test',
            permitted_use='evaluation_only' if split == 'official_test' else 'train_only_fit_or_selection')
    output.mkdir()
    # Completion manifest is last. Incomplete output is retained for diagnosis.
    for name, data in (('frames.jsonl', raw_stream), ('audit.jsonl', audit_stream), ('manifest.json', canonical(manifest))):
        with (output/name).open('xb') as stream:
            stream.write(data)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--ego-agents', type=Path, required=True)
    parser.add_argument('--oracle-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-protocol', choices=[VEHICLE_PROTOCOL])
    args = parser.parse_args()
    prepare(args.volume, args.receipt_sha256, args.ego_agents, args.output, oracle_root=args.oracle_root,
            evaluation_protocol=args.evaluation_protocol)
