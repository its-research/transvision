"""Byte and train-role admission only; no matching or fitting mathematics."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, digest, pins):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('ordinary admitted input required')
    b = path.read_bytes()
    if hashlib.sha256(b).hexdigest() != digest:
        raise ValueError('admitted input SHA differs: '+str(path))
    pins[str(path)] = digest
    return b


def load(detector_receipt, detector_sha, target_acceptance, target_sha, partition, partition_sha):
    pins = {}
    detector = json.loads(read(detector_receipt, detector_sha, pins))
    accepted = json.loads(read(target_acceptance, target_sha, pins))
    part = json.loads(read(partition, partition_sha, pins))
    if (detector.get('kind') != 'v2v4real_motion_detector_train_input_admission_v1'
            or detector.get('official_test_used') is not False or detector.get('clock') != 'nominal_10hz'
            or accepted.get('official_test_used') is not False or accepted.get('clock') != 'nominal_10hz'
            or not all(accepted.get(k) is True for k in ('all_source_geometry_verified',
                'all_velocity_masks_and_predecessors_verified', 'all_past_identity_velocity_targets_verified'))
            or part.get('official_split') != 'train' or part.get('official_test_used_for_partition') is not False):
        raise ValueError('independently admitted train-only inputs required before arrays')
    cand_paths = [Path(p) for p in accepted['input_sha256'] if p.endswith('/candidate-receipt.json')]
    if len(cand_paths) != 1:
        raise ValueError('exact original target candidate required')
    cpath = cand_paths[0]
    cand = json.loads(read(cpath, accepted['candidate_receipt_sha256'], pins))
    if cand['official_test_used'] is not False:
        raise ValueError('train target scope required')
    def artifact(root, records, name):
        return read(root/name, records[name]['sha256'], pins)
    gt_index = [json.loads(x) for x in artifact(cpath.parent, cand['artifacts'],
                                               'candidate-frame-index.jsonl').splitlines()]
    with np.load(io.BytesIO(artifact(cpath.parent, cand['artifacts'], 'candidate-targets.npz')),
                 allow_pickle=False) as z:
        gt = {k: z[k].copy() for k in z.files}
    if set(gt) != {'states_source', 'velocity_source', 'velocity_valid', 'offsets'}:
        raise ValueError('exact source geometry and past-velocity targets required')
    groups = sorted(part['calibration_fit'])
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError('at least two original calibration groups required')
    target_rows, bykey = [], {}
    for i, row in enumerate(gt_index):
        group = part['sequence_to_name_group'][row['sequence_id']]
        if group not in groups:
            raise ValueError('selection/fitting source outside frozen calibration role')
        start, end = gt['offsets'][i:i+2]
        if start != row['offset'] or end-start != row['source_candidates']:
            raise ValueError('target row offsets differ')
        key = row['sequence_id'], row['frame_key'], row['side']
        if key in bykey:
            raise ValueError('duplicate train source target')
        prepared = dict(row, group=group, states_source=gt['states_source'][start:end],
            velocity_source=gt['velocity_source'][start:end], velocity_valid=gt['velocity_valid'][start:end])
        bykey[key] = prepared
        target_rows.append(prepared)
    if len(bykey) != accepted['source_rows'] or len(gt['states_source']) != accepted['objects']:
        raise ValueError('full accepted target coverage differs')
    seeds = {}
    if {r['seed'] for r in detector['seeds']} != {1337, 2027, 3407}:
        raise ValueError('exact three-seed training detector scope required')
    for info in detector['seeds']:
        seed = info['seed']; root = Path(detector_receipt).parent/f'seed{seed}'
        index = [json.loads(x) for x in artifact(root, info['artifacts'], 'detector-frame-index.jsonl').splitlines()]
        with np.load(io.BytesIO(artifact(root, info['artifacts'], 'detector-arrays.npz')),
                     allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        if set(data) != {'states_source_legacy7', 'raw_scores', 'class_indices', 'offsets'}:
            raise ValueError('exact admitted detector arrays required')
        if len(index) != info['source_rows'] or len(data['raw_scores']) != info['candidates']:
            raise ValueError('detector coverage differs')
        rows, seen = [], set()
        for i, row in enumerate(index):
            key = row['sequence_id'], row['frame_id'], row['side']
            if key in seen or key not in bykey:
                raise ValueError('target/detector train source scope differs')
            seen.add(key); t = bykey[key]
            if (row['frame_ordinal'] != t['frame_ordinal'] or row['source_to_world'] != t['source_to_world']
                    or row['pcd_sha256'] != t['pcd_sha256']):
                raise ValueError('original pose, clock or PCD lineage differs')
            start, end = data['offsets'][i:i+2]
            if start != row['offset'] or end-start != row['candidates']:
                raise ValueError('detector offsets differ')
            rows.append(dict(sequence_id=row['sequence_id'], side=row['side'], frame_ordinal=row['frame_ordinal'],
                source_to_world=row['source_to_world'], group=t['group'], row_sha256=row['raw_row_sha256'],
                states_source_legacy7=data['states_source_legacy7'][start:end],
                raw_scores=data['raw_scores'][start:end], class_indices=data['class_indices'][start:end],
                target_states_source=t['states_source'], target_velocity_source=t['velocity_source'],
                target_velocity_valid=t['velocity_valid'], original_row=row))
        if seen != set(bykey):
            raise ValueError('complete original train target selection required')
        seeds[seed] = rows
    return seeds, target_rows, groups, pins, detector
