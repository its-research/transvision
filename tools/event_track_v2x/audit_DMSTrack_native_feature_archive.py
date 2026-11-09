#!/usr/bin/env python3
"""Hash/extract every author ZIP member and audit native per-frame input shapes."""
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import stat
import types
import zipfile
from pathlib import Path, PurePosixPath

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
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['auditor_sha256']
    for ref in plan['references'].values():
        assert sha(ref['path']) == ref['sha256']
    acquisition = json.loads(Path(plan['references']['acquisition_plan']['path']).read_bytes())
    spec = next(r for r in acquisition['files'] if r['name'] == args.name)
    archive = Path(acquisition['output']) / spec['name']
    byte_receipt = archive.with_name(archive.name + '.byte-receipt.json')
    byte_proof = json.loads(byte_receipt.read_bytes())
    assert byte_proof['source'] == spec and byte_proof['saved_file_independently_reread']
    assert byte_proof['plan_sha256'] == plan['references']['acquisition_plan']['sha256']
    assert byte_proof['producer_sha256'] == acquisition['producer_sha256']
    assert byte_proof['bytes'] == archive.stat().st_size == spec['bytes']
    assert sha(archive) == byte_proof['sha256']
    source_proof = json.loads(Path(plan['references']['author_source_readback']['path']).read_bytes())
    assert source_proof['author_commit'] == acquisition['author_source_commit']
    source = Path(source_proof['source'])
    main_file = source / 'DMSTrack/main_dkf.py'
    assert sha(main_file) == source_proof['files']['DMSTrack/main_dkf.py']['sha256']
    nodes = {n.name: n for n in ast.parse(main_file.read_bytes()).body if isinstance(n, ast.FunctionDef)}
    namespace = dict(os=os)
    exec(compile(ast.Module(body=[nodes['get_config'], nodes['get_global_timestamp_index']],
                            type_ignores=[]), '<pinned-author-config-and-index>', 'exec'), namespace)
    cfg = types.SimpleNamespace(det_name='multi_sensor_differentiable_kalman_filter', dataset='v2v4real')
    config = namespace['get_config'](cfg, 'Car', 'train', 'val', 'all', False)
    split = 'train' if spec['name'].startswith('train_') else 'val'
    settings = config[0] if split == 'train' else config[1]
    expected = {}
    detections = []
    detection_count = 0
    root_name = spec['name'].removesuffix('.zip')
    for seq in settings['seq_eval']:
        seq_index = int(seq)
        previous = settings['len_record'][seq_index - 1] if seq_index else 0
        length = settings['len_record'][seq_index] - previous
        for cav in config[5]:
            relative = f"AB3DMOT/data/v2v4real/detection/{settings['result_sha']}/{cav}/{seq}.txt"
            csv = source / relative
            assert sha(csv) == source_proof['files'][relative]['sha256']
            values = np.loadtxt(csv, delimiter=',', ndmin=2)
            assert values.ndim == 2 and values.shape[1] == 15 and np.isfinite(values).all()
            assert (values[:, 0] == np.floor(values[:, 0])).all()
            assert (values[:, 0] >= 0).all() and (values[:, 0] < length).all()
            assert (values[:, 1] == 2).all() and (values[:, 7:10] > 0).all()
            detection_count += len(values)
            detections.append(dict(path=relative, sha256=sha(csv), rows=len(values), sequence=seq, cav=cav))
            for frame in range(length):
                index = namespace['get_global_timestamp_index'](seq, frame, settings['len_record'])
                assert index == previous + frame
                count = int(np.count_nonzero(values[:, 0] == frame))
                for suffix, shape in [('feature', [count, 256, 5, 5]),
                                      ('early_feature', [count, 64, 20, 20]),
                                      ('transformation_matrix', [4, 4])]:
                    name = f'{root_name}/npy/{cav}/{index:04d}_{suffix}.npy'
                    assert name not in expected
                    expected[name] = dict(shape=shape, sequence=seq, frame=frame,
                                          global_timestamp_index=index, cav=cav, detection_count=count, suffix=suffix)
    output = Path(plan['output']) / root_name
    output.mkdir(parents=True, exist_ok=False)
    extraction = output / 'author-files'
    extraction.mkdir()
    records = {}
    seen = set()
    total = 0
    feature_records = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            pure = PurePosixPath(info.filename)
            assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in info.filename
            assert pure.parts[0] == root_name and info.filename not in seen
            assert not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1
            seen.add(info.filename)
            target = extraction / pure
            assert not any(p.is_symlink() for p in (target, *target.parents))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            count = 0
            with zf.open(info) as member, target.open('xb') as stream:
                for block in iter(lambda: member.read(8 * 1024**2), b''):
                    stream.write(block)
                    h.update(block)
                    count += len(block)
            assert count == info.file_size and sha(target) == h.hexdigest()
            records[info.filename] = dict(bytes=count, sha256=h.hexdigest(), zip_crc32=info.CRC)
            total += count
            if info.filename in expected:
                value = np.load(target, mmap_mode='r', allow_pickle=False)
                assert value.dtype.kind == 'f' and list(value.shape) == expected[info.filename]['shape']
                assert np.isfinite(value).all()
                if expected[info.filename]['suffix'] == 'transformation_matrix':
                    assert np.linalg.slogdet(value)[0] != 0
                feature_records[info.filename] = dict(expected[info.filename], dtype=str(value.dtype),
                                                       numel=int(value.size), all_values_finite=True)
                del value
            if len(records) % 1000 == 0:
                print(json.dumps(dict(name=spec['name'], members_hashed=len(records),
                                      uncompressed_bytes=total, feature_arrays_verified=len(feature_records))), flush=True)
    assert set(feature_records) == set(expected)
    result = dict(kind='rbf_DMSTrack_author_native_all_ZIP_member_and_detection_feature_input_audit',
                  name=spec['name'], split_alias=split, archive_sha256=byte_proof['sha256'],
                  byte_receipt_sha256=sha(byte_receipt), plan_sha256=sha(args.plan),
                  auditor_sha256=sha(__file__), author_commit=source_proof['author_commit'],
                  author_config=settings, detection_files=detections,
                  source_frames_per_CAV=settings['len_record'][-1], cav_ids=config[5],
                  sequences=len(settings['seq_eval']), detection_rows=detection_count,
                  regular_members=len(records), uncompressed_bytes=total, files=records,
                  required_feature_arrays=len(expected), feature_arrays=feature_records,
                  all_members_independently_CRC_size_SHA_verified=True,
                  all_native_feature_shapes_counts_and_values_verified=True,
                  extracted_author_files=str(extraction),
                  feature_assembly_contract='repeat256x5x5 by4 on both spatial axes +64x20x20 =>320x20x20',
                  official_test_original_source_mapping_independently_verified=False,
                  GT_geometry_categories_frequency_ROI_verified=False,
                  native_feature_neural_forward_run=False, native_predictions_produced=False,
                  native_metrics_independently_verified=False, native_method_accepted=False,
                  same_resource_complete=False, paper_complete=False,
                  checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = output / 'native-feature-input-receipt.json'
    path.open('x').write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(receipt=str(path), native_feature_input_shapes_and_byte_audit_passed=True,
                         native_method_accepted=False)), flush=True)


if __name__ == '__main__':
    main()
