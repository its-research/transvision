#!/usr/bin/env python3
"""Independently compare native CSV/KITTI inputs to qualified author corners."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def new(path, value):
    path.open('x').write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def independent_box(corners):
    """Scalar float64 geometry, without importing the author's producer."""
    c = np.asarray(corners, dtype=np.float64)
    assert c.shape == (8, 3) and np.isfinite(c).all()
    center = [math.fsum(float(c[i, axis]) for i in (0, 3, 5, 6)) / 4 for axis in range(3)]
    height = abs(math.fsum(float(c[i+4, 2] - c[i, 2]) for i in range(4)) / 4)

    def edge_average(edges):
        return math.fsum(math.hypot(float(c[a, 0] - c[b, 0]), float(c[a, 1] - c[b, 1]))
                         for a, b in edges) / 4

    length = edge_average(((0, 3), (2, 1), (4, 7), (5, 6)))
    width = edge_average(((0, 1), (2, 3), (4, 5), (6, 7)))
    yaw = math.fsum(math.atan2(float(c[a, 1] - c[b, 1]), float(c[a, 0] - c[b, 0]))
                    for a, b in ((1, 2), (0, 3), (5, 6), (4, 7))) / 4
    return np.array([height, width, length, center[0], center[2], center[1], yaw])


def audit(plan, plan_path):
    assert sha(__file__) == plan['auditor_sha256']
    output = Path(plan['output'])
    output.mkdir(exist_ok=False)
    try:
        for binding in plan['references'].values():
            assert sha(binding['path']) == binding['sha256']
        source_proof = json.loads(Path(plan['references']['source']['path']).read_bytes())
        features = json.loads(Path(plan['references']['feature_inputs']['path']).read_bytes())
        assert source_proof['author_commit'] == features['author_commit'] == plan['author_commit']
        source = Path(source_proof['source'])
        feature_root = Path(features['extracted_author_files'])
        loaded = {}
        texts = {}

        def load(name):
            key = 'no_fusion_keep_all/npy/' + name
            path = feature_root / key
            binding = features['files'][key]
            assert not path.is_symlink() and path.stat().st_size == binding['bytes']
            assert sha(path) == binding['sha256']
            value = np.load(path, allow_pickle=False)
            assert value.dtype.kind in 'fiu' and np.isfinite(value).all()
            loaded[key] = binding
            return value

        def text(name):
            path = source / name
            assert not path.is_symlink() and sha(path) == source_proof['files'][name]['sha256']
            texts[name] = source_proof['files'][name]
            return path.read_text().splitlines()

        maximum = [0.0] * 7
        normalized_maximum = 0.0
        score_maximum = 0.0
        sequences = []
        start = 0
        for seq_index, length in enumerate(plan['sequence_lengths']):
            seq = f'{seq_index:04d}'
            rows_per_cav = {}
            for cav in ('ego', '1'):
                name = f'AB3DMOT/data/v2v4real/detection/multi_sensor_differentiable_kalman_filter_Car_val/{cav}/{seq}.txt'
                rows = np.array([[float(x) for x in line.split(',')] for line in text(name)])
                assert rows.ndim == 2 and rows.shape[1] == 15 and np.isfinite(rows).all()
                assert (rows[:, 1] == 2).all() and (rows[:, 2:6] == 0).all() and (rows[:, 14] == 0).all()
                expected_frames = []
                count = 0
                for frame in range(length):
                    prefix = f'{cav}/{start+frame:04d}'
                    corners = load(prefix + '_pred.npy')
                    scores = load(prefix + '_pred_score.npy')
                    assert corners.shape == (len(scores), 8, 3) and scores.shape == (len(corners),)
                    actual = rows[count:count+len(corners)]
                    assert len(actual) == len(corners) and (actual[:, 0] == frame).all()
                    for i, corner in enumerate(corners):
                        expected = independent_box(corner)
                        bound = np.full(7, plan['float32_roundoff_multiplier'] * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(corner)))))
                        bound[-1] = plan['float32_roundoff_multiplier'] * np.finfo(np.float32).eps * math.pi
                        error = np.abs(actual[i, 7:14] - expected)
                        maximum = np.maximum(maximum, error).tolist()
                        normalized_maximum = max(normalized_maximum, float(np.max(error / bound)))
                        assert (error <= bound).all(), ('detection geometry', seq, cav, frame, i, error.tolist(), bound.tolist())
                        score_error = abs(float(actual[i, 6]) - float(scores[i]))
                        score_maximum = max(score_maximum, score_error)
                        assert score_error <= plan['score_text_absolute_tolerance']
                    count += len(corners)
                    expected_frames.extend([frame] * len(corners))
                assert count == len(rows) and np.array_equal(rows[:, 0], expected_frames)
                rows_per_cav[cav] = count

            gt_name = f'AB3DMOT/scripts/KITTI/v2v4real_val_label/{seq}.txt'
            gt_lines = text(gt_name)
            gt_count = 0
            empty_gt_frames = 0
            for frame in range(length):
                corners = load(f'{start+frame:04d}_gt.npy')
                ids = load(f'{start+frame:04d}_gt_object_id.npy')
                assert corners.shape == (len(ids), 8, 3) and ids.shape == (len(corners),)
                assert ids.dtype.kind in 'iu' and len(set(map(int, ids))) == len(ids)
                if not len(ids):
                    empty_gt_frames += 1
                for i, corner in enumerate(corners):
                    cells = gt_lines[gt_count].split()
                    assert len(cells) == 17 and cells[2] == 'Car'
                    assert int(cells[0]) == frame and int(cells[1]) == int(ids[i])
                    assert all(float(x) == 0 for x in cells[3:10])
                    expected = independent_box(corner)
                    actual = np.array([float(x) for x in cells[10:17]])
                    bound = np.full(7, plan['float32_roundoff_multiplier'] * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(corner)))))
                    bound[-1] = plan['float32_roundoff_multiplier'] * np.finfo(np.float32).eps * math.pi
                    error = np.abs(actual - expected)
                    maximum = np.maximum(maximum, error).tolist()
                    normalized_maximum = max(normalized_maximum, float(np.max(error / bound)))
                    assert (error <= bound).all(), ('GT geometry', seq, frame, i, error.tolist(), bound.tolist())
                    gt_count += 1
            assert gt_count == len(gt_lines)
            row = dict(sequence=seq, frames=length, first_global_index=start, detection_rows=rows_per_cav,
                       GT_rows=gt_count, empty_GT_frames=empty_gt_frames)
            sequences.append(row)
            start += length
            print(json.dumps(row), flush=True)
        assert start == 1993 and sum(sum(s['detection_rows'].values()) for s in sequences) == 60046
        # The exact source text and NPY files remain stable after the geometry audit.
        for name, binding in loaded.items():
            assert sha(feature_root / name) == binding['sha256']
        for name, binding in texts.items():
            assert sha(source / name) == binding['sha256']
        new(output / 'receipt.json', dict(kind='DMSTrack_complete_native_detection_GT_to_author_corner_parity_v1',
            author_commit=plan['author_commit'], plan_sha256=sha(plan_path), auditor_sha256=sha(__file__),
            references=plan['references'], sequences=sequences, frames=1993, CAVs=2, detection_rows=60046,
            source_NPY_files=loaded, native_text_files=texts, maximum_geometry_errors=maximum,
            maximum_error_divided_by_fixed_float32_bound=normalized_maximum, maximum_score_text_error=score_maximum,
            all_native_frames_and_row_order_preserved=True, all_published_GT_IDs_and_Car_aliases_preserved=True,
            original_yaml_GT_categories_and_ROI_independently_verified=False,
            official_test_original_source_mapping_independently_verified=False,
            native_predictions_produced=False, native_metrics_independently_verified=False,
            native_method_accepted=False, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException as error:
        new(output / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            fixed_tolerances_preserved=True, failed_rows_preserved=True, native_method_accepted=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    audit(json.loads(args.plan.read_bytes()), args.plan)


if __name__ == '__main__':
    main()
