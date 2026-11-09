#!/usr/bin/env python3
"""Merge newly verified vehicle GT parts into the frozen train calibration role.

This consumes offline GT only, creates no inference input, and cannot promote
historical strict-Car projections or official-test labels into calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.paper_evaluation_policy import native_vehicle
from transvision.models.event_track_v2x.v2v4real_ground_truth import load_vehicle_ground_truth
from tools.event_track_v2x.v2v4real_gt_oracle import SOURCE_HASHES, COMMIT


def pinned(path, digest):
    path = Path(path).absolute()
    if (re.fullmatch('[0-9a-f]{64}', digest or '') is None
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or sha_file(path) != digest):
        raise ValueError('ordinary SHA-256 bound input required')
    return path


def calibration_sequences(partition):
    if (partition.get('kind') != 'v2v4real_formal_nested_training_partition_v1'
            or partition.get('protocol_id') != 'v2v4real-nominal-10hz-formal-v1'
            or partition.get('official_split') != 'train'
            or partition.get('official_test_used_for_partition') is not False
            or partition.get('gt_used_for_partition') is not False):
        raise ValueError('frozen train-only role partition required')
    roles = []
    for name in ('detector_fit', 'calibration_fit', 'identity_selection'):
        values = partition.get(name)
        if (not isinstance(values, list) or not values
                or any(not isinstance(x, str) or not x for x in values)
                or len(values) != len(set(values))):
            raise ValueError('nonempty unique frozen role groups required')
        roles.append(set(values))
    if any(roles[i] & roles[j] for i in range(3) for j in range(i)):
        raise ValueError('training role groups overlap')
    mapping = partition.get('sequence_to_name_group')
    if (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(s, str) or not s or g not in set.union(*roles)
                   for s, g in mapping.items())):
        raise ValueError('explicit complete sequence-to-role mapping required')
    selected = {s for s, group in mapping.items() if group in roles[1]}
    if not selected or {mapping[s] for s in selected} != roles[1]:
        raise ValueError('calibration groups lack declared sequences')
    return selected


def merge(parts, part_sha256s, partition_path, partition_sha256, output):
    parts = tuple(Path(p).absolute() for p in parts)
    if not parts or len(parts) != len(part_sha256s) or len(parts) != len(set(parts)):
        raise ValueError('distinct GT parts with one expected manifest hash each required')
    partition_path = pinned(partition_path, partition_sha256)
    selected = calibration_sequences(json.loads(partition_path.read_bytes()))
    output = Path(output).absolute()
    if (output.exists() or any(p.is_symlink() for p in (output, *output.parents))
            or any(p == output or p in output.parents for p in parts)):
        raise ValueError('fresh output outside input projections required')
    source_hashes = {str(p): sha_file(p) for p in (Path(__file__),
        ROOT/'transvision/models/event_track_v2x/v2v4real_ground_truth.py',
        ROOT/'transvision/models/event_track_v2x/paper_evaluation_policy.py',
        ROOT/'tools/event_track_v2x/v2v4real_gt_oracle.py')}
    stable = ('kind', 'recipe', 'split', 'class_scope', 'evaluation_class', 'evaluation_protocol',
        'native_label_source', 'time_basis', 'coordinate_frame', 'box_representation', 'gt_range',
        'source_local_roi', 'final_ego_roi', 'deduplication', 'identity_rule', 'reference_commit',
        'reference_scope', 'reference_source_sha256', 'oracle_corner_atol_m')
    manifests, bindings, chosen = [], [], []
    seen = set()
    for root, expected in zip(parts, part_sha256s):
        pinned(root/'manifest.json', expected)
        manifest, frames = load_vehicle_ground_truth(root, expected_manifest_sha256=expected)
        if manifest.get('split') != 'train' or manifest.get('test_payloads_read') is not False:
            raise ValueError('official test cannot enter calibration GT')
        if (manifest.get('reference_commit') != COMMIT or manifest.get('reference_source_sha256') != SOURCE_HASHES
                or manifest.get('oracle_all_ids_and_roi_match') is not True
                or manifest.get('oracle_compared_frames') != manifest['paired_frames']
                or manifest.get('oracle_corner_atol_m') != 1e-4
                or type(manifest.get('oracle_max_abs_corner_error_m')) not in (int, float)
                or not 0 <= manifest['oracle_max_abs_corner_error_m'] <= 1e-4):
            raise ValueError('complete fixed native oracle comparison required for each part')
        if manifests and any(manifest.get(k) != manifests[0].get(k) for k in stable):
            raise ValueError('vehicle GT definitions differ across parts')
        sequences = set(manifest['ego_agents'])
        if seen & sequences:
            raise ValueError('GT sequence overlap across parts')
        seen |= sequences
        audits = [json.loads(line) for line in (root/'audit.jsonl').read_bytes().splitlines()]
        if len(audits) != len(frames):
            raise ValueError('GT audit coverage differs')
        for frame, audit in zip(frames, audits):
            if (frame['sequence_id'], frame['frame_key']) != (audit.get('sequence_id'), audit.get('frame_key')):
                raise ValueError('GT audit frame identity differs')
            if frame['sequence_id'] in selected:
                audit = dict(audit, parent_oracle_binding=dict(part_manifest_sha256=expected,
                    source_frame_sha256=hashlib.sha256(canonical(frame)).hexdigest(),
                    source_audit_sha256=hashlib.sha256(canonical(audit)).hexdigest(),
                    reference_source_sha256=SOURCE_HASHES, compared_as_member_of_complete_part=True))
                chosen.append((frame, audit))
        manifests.append(manifest)
        bindings.append(dict(path=str(root), manifest_sha256=expected,
            frames_sha256=manifest['frames_sha256'], audit_sha256=manifest['audit_sha256'],
            volume_receipt_sha256=manifest['volume_receipt_sha256'], source_sha256=manifest['source_sha256'],
            read_paired_frames=manifest['paired_frames'], selected_paired_frames=sum(f['sequence_id'] in selected for f in frames)))
    if not selected <= seen or {f['sequence_id'] for f, _ in chosen} != selected:
        raise ValueError('calibration sequence missing from GT parts')
    chosen.sort(key=lambda pair: (pair[0]['sequence_id'], pair[0]['frame_ordinal']))
    frames, audits = zip(*chosen)
    counts = Counter(f['sequence_id'] for f in frames)
    for seq in selected:
        if [f['frame_ordinal'] for f in frames if f['sequence_id'] == seq] != list(range(counts[seq])):
            raise ValueError('selected calibration ordinals incomplete')
    raw_classes = Counter()
    for audit in audits:
        if set(audit.get('raw_classes_by_source', {})) != {'0', '1'}:
            raise ValueError('per-source native class audit required')
        for values in audit['raw_classes_by_source'].values():
            if not isinstance(values, dict) or any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in values.items()):
                raise ValueError('native class audit counts invalid')
            raw_classes.update(values)
    streams = {name: b''.join(canonical(row)+b'\n' for row in rows)
               for name, rows in (('frames.jsonl', frames), ('audit.jsonl', audits))}
    manifest = {key: manifests[0][key] for key in stable}
    manifest.update(ego_agents={s: next(m['ego_agents'][s] for m in manifests if s in m['ego_agents']) for s in sorted(selected)},
        partition_sha256=partition_sha256, calibration_fit_sequences=sorted(selected),
        raw_train_sequences_read=sorted(seen), read_paired_frames=sum(m['paired_frames'] for m in manifests),
        part_bindings=bindings, paired_frames=len(frames), source_frames=2*len(frames), sequence_frames=dict(sorted(counts.items())),
        gt_objects=sum(len(f['objects']) for f in frames), raw_annotation_counts=dict(sorted(raw_classes.items())),
        selected_native_annotation_counts={k: v for k, v in sorted(raw_classes.items()) if native_vehicle(k)},
        excluded_native_annotation_counts={k: v for k, v in sorted(raw_classes.items()) if not native_vehicle(k)},
        raw_count_unit='per_source_frame_annotation_in_selected_calibration_role',
        volume_receipt_sha256=hashlib.sha256(canonical(sorted(m['volume_receipt_sha256'] for m in manifests))).hexdigest(),
        source_sha256=source_hashes, oracle_compared_frames=len(frames), oracle_all_ids_and_roi_match=True,
        oracle_result_inherited_from_complete_part_comparisons=True,
        oracle_max_abs_corner_error_m=max(m['oracle_max_abs_corner_error_m'] for m in manifests),
        frames_sha256=hashlib.sha256(streams['frames.jsonl']).hexdigest(), audit_sha256=hashlib.sha256(streams['audit.jsonl']).hexdigest(),
        permitted_use='train_only_fit_or_selection', GT_read=True, inference_input=False, test_payloads_read=False,
        physical_identity_continuity_verified=False, paper_eligible=False, public_results_directly_comparable=False,
        full_official_split_verified=False, detector_executed=False, tracking_evaluation_performed=False,
        parameter_training_performed=False, local_tracks_with_multiple_mapped_ids=None, local_track_count=None)
    # Re-read after processing so a part changed while merging is never accepted.
    for root, expected in zip(parts, part_sha256s):
        load_vehicle_ground_truth(root, expected_manifest_sha256=expected)
    pinned(partition_path, partition_sha256)
    if any(sha_file(p) != digest for p, digest in source_hashes.items()):
        raise ValueError('merger source changed during execution')
    output.mkdir(parents=True)
    for name, data in (*streams.items(), ('manifest.json', canonical(manifest))):
        with (output/name).open('xb') as stream:
            stream.write(data)
    loaded, reread = load_vehicle_ground_truth(output, expected_manifest_sha256=sha_file(output/'manifest.json'))
    if loaded != manifest or tuple(reread) != tuple(frames):
        raise ValueError('merged vehicle GT readback differs')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts', nargs='+', required=True)
    p.add_argument('--part-sha256s', nargs='+', required=True)
    p.add_argument('--partition', required=True)
    p.add_argument('--partition-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(merge(a.parts, a.part_sha256s, a.partition, a.partition_sha256, a.output), sort_keys=True))


if __name__ == '__main__':
    main()
