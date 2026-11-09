#!/usr/bin/env python3
"""Bind all native author GT frames to independently admitted original YAML GT."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['auditor_sha256']
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    try:
        for ref in plan['references'].values():
            assert sha(ref['path']) == ref['sha256']
        gt = json.loads(Path(plan['references']['qualified_original_GT_manifest']['path']).read_bytes())
        features = json.loads(Path(plan['references']['qualified_native_features']['path']).read_bytes())
        assert gt['oracle_all_ids_and_roi_match'] and gt['oracle_compared_frames'] == 1993
        assert gt['full_official_split_verified'] and gt['split'] == 'official_test'
        assert gt['frames_sha256'] == plan['references']['qualified_original_GT_frames']['sha256']
        assert gt['oracle_corner_atol_m'] == plan['corner_atol_m']
        assert gt['gt_range'] == plan['gt_range'] and gt['final_ego_roi'] == 'all_eight_corners_inside_xy'
        assert gt['native_label_source'] == 'params.vehicles_except_exact_obj_type_Pedestrian'
        root = Path(features['extracted_author_files'])
        rows = [json.loads(line) for line in Path(plan['references']['qualified_original_GT_frames']['path']).read_text().splitlines()]
        by_seq = {}
        for row in rows:
            by_seq.setdefault(row['sequence_id'], []).append(row)
        assert list(sorted(by_seq)) == plan['sequence_names']
        index = 0
        count = 0
        maximum = 0.0
        consumed = {}
        mapping = []
        for ordinal, name in enumerate(plan['sequence_names']):
            frames = sorted(by_seq[name], key=lambda r: r['frame_ordinal'])
            assert len(frames) == plan['sequence_lengths'][ordinal]
            first = index
            for frame, row in enumerate(frames):
                assert row['frame_ordinal'] == frame
                arrays = {}
                for suffix in ('gt', 'gt_object_id'):
                    key = f'no_fusion_keep_all/npy/{index:04d}_{suffix}.npy'
                    p = root / key
                    binding = features['files'][key]
                    assert not p.is_symlink() and sha(p) == binding['sha256']
                    arrays[suffix] = np.load(p, allow_pickle=False)
                    consumed[key] = binding
                ids, corners = arrays['gt_object_id'], arrays['gt']
                expected = {o['track_id']: o for o in row['objects']}
                assert ids.dtype.kind in 'iu' and len(set(map(int, ids))) == len(ids)
                assert set(map(int, ids)) == set(expected), ('GT identities', name, frame)
                assert corners.shape == (len(ids), 8, 3)
                for ident, corner in zip(ids, corners):
                    obj = expected[int(ident)]
                    assert obj['evaluation_class'] == 'vehicle' and obj['raw_class'] != 'Pedestrian'
                    error = float(np.max(np.abs(corner.astype(np.float64) - np.asarray(obj['corners_ego'], dtype=np.float64))))
                    assert error <= plan['corner_atol_m'], ('GT corner mapping', name, frame, int(ident), error)
                    maximum = max(maximum, error)
                    count += 1
                index += 1
            mapping.append(dict(native_sequence=f'{ordinal:04d}', official_test_sequence=name,
                                first_global_index=first, frames=len(frames)))
            print(json.dumps(mapping[-1]), flush=True)
        assert index == 1993 and count == gt['gt_objects'] == 31419
        for name, binding in consumed.items():
            assert sha(root / name) == binding['sha256']
        result = dict(kind='DMSTrack_complete_native_GT_to_qualified_official_test_original_YAML_mapping_v1',
            references=plan['references'], plan_sha256=sha(args.plan), auditor_sha256=sha(__file__),
            author_commit=features['author_commit'], mapping=mapping, frames=index, GT_objects=count,
            maximum_corner_error_m=maximum, fixed_corner_atol_m=plan['corner_atol_m'],
            official_test_original_source_mapping_independently_verified=True,
            original_GT_identity_geometry_category_ROI_match=True,
            categories='merged vehicle; native author Car alias; excludes exact Pedestrian',
            GT_range=gt['gt_range'], source_local_roi=gt['source_local_roi'], final_ego_roi=gt['final_ego_roi'],
            nominal_frequency_Hz=10, nominal_frequency_source='frozen experiment contract; not measured sensor timing',
            native_split_alias='val', official_split='official_test',
            train_test_overlap_disclosed=True, statistical_train_test_independence_claim=False,
            native_predictions_produced=False, native_method_accepted=False,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out / 'receipt.json').open('x').write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    except BaseException as error:
        (out / 'failure.json').open('x').write(json.dumps(dict(error_type=type(error).__name__, error=str(error),
            failed_evidence_preserved=True, tolerance_unchanged=True, native_method_accepted=False)) + '\n')
        raise


if __name__ == '__main__':
    main()
