#!/usr/bin/env python3
"""Bridge existing native raw-top64 exports to a separately named 9D cache.

No detector, GT, velocity estimator, or covariance fitter is executed here.
An explicit source-local causal motion/covariance contract supplies the two
missing velocity columns and the full 9x9 covariance for every raw candidate.
The contract also binds its method/model/train-fit receipt and independent
admission receipt. Missing contracts are errors, never zero/default values.

The historical train envelope is preserved. The separate official-test kind
additionally requires the original test projection, official split admission,
and a new producer receipt bound to the raw manifest/archive. A renamed train
envelope or an input-directory label cannot establish test provenance.
An input admission receipt is imported evidence, not independently reproduced
by this adapter. Output readback establishes persistence/coverage only.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.detection_cache_v2 import BOX_LAYOUT, SIDES, canonical, contained_file, sha_file
from transvision.models.event_track_v2x.paper_calibration import ExistenceCalibration
from transvision.models.event_track_v2x.paper_evaluation_policy import (
    VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, require_vehicle_binding, vehicle_binding,
)
from transvision.models.event_track_v2x.paper_native_cache import (
    NATIVE_FEATURE, NO_IMAGE, NativeDetectionFrame, NativePaperCache, write_native_cache,
)
from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress
from tools.event_track_v2x.fit_v2v4real_vehicle_calibration import pinned

RAW_LAYOUT = 'source_lidar_legacy_xyz_width_length_height_neg_yaw_minus_pi_over_2'
RAW_RULE = 'raw_score_ge_0.05_stable_top64_before_nms'
RAW_FIELDS = {'sequence_id', 'frame_id', 'frame_ordinal', 'side', 'source_to_world',
              'pcd_sha256', 'payload', 'payload_sha256', 'candidates'}
RAW_ARRAYS = {'states_source_legacy7', 'raw_scores', 'class_indices', 'appearance', 'appearance_valid'}
RAW_MANIFEST_FIELDS = {'anchor_array_sha256', 'anchor_file_sha256', 'anchor_generation', 'appearance_valid_candidates',
    'candidate_protocol', 'checkpoint_seed', 'checkpoint_sha256', 'cohort', 'covariance_exported',
    'downstream_covariance_velocity_and_identity_admission_pending', 'export_groups', 'feature_method', 'gt_read',
    'kind', 'matching_candidate_rule', 'official_test_read', 'paper_metric', 'partition_sha256',
    'prepared_projection_sha256', 'protocol_id', 'rows', 'rows_sha256', 'sequence_ids', 'split',
    'split_admission_sha256', 'state_layout', 'total_candidates', 'velocity_exported'}
PROJECTION_FIELDS = {'cav_id', 'frame_key', 'frame_ordinal', 'is_ego', 'lidar_pose', 'pcd_path',
                     'pcd_sha256', 'pcd_size_bytes', 'sequence_id', 'source_to_world'}
MOTION_KIND = 'v2v4real_source_local_causal_motion_covariance_v1'
# Official config used by this historical export lineage (also pinned by the
# existing run_v2v4real_pointpillar_smoke.py admission).
DETECTOR_CONFIG_SHA256 = '138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1'


def check(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(row):
    return row['sequence_id'], row['frame_id'], row['side']


def _loaded_source_paths():
    """Bind actual loaded local implementations, following imported symbols.

    Package initializers are recorded without expanding all of their unrelated
    registry imports. Transitive function/class/module globals of the concrete
    cache/calibration helpers are followed. No guessed source-root filenames or
    root-level snapshot can substitute for each module's actual ``__file__``.
    """
    pending = {__name__, 'transvision.models.event_track_v2x.detection_cache_v2',
        'transvision.models.event_track_v2x.paper_native_cache', 'transvision.models.event_track_v2x.paper_calibration',
        'transvision.models.event_track_v2x.paper_evaluation_policy', 'transvision.models.event_track_v2x.experiment_progress',
        'tools.event_track_v2x.fit_v2v4real_vehicle_calibration'}
    found = {}
    def location(name):
        module = sys.modules.get(name)
        file = getattr(module, '__file__', None)
        check(file is not None, 'loaded source module has no file: '+name)
        path = Path(file).absolute()
        check(path.suffix == '.py' and path.is_file() and path.is_relative_to(ROOT)
              and not any(p.is_symlink() for p in (path, *path.parents)), 'loaded local source path differs: '+name)
        return path
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found[name] = location(name)
        for value in tuple(vars(sys.modules[name]).values()):
            imported = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, '__module__', None)
            if isinstance(imported, str) and (imported.startswith('transvision.models.event_track_v2x.')
                                             or imported.startswith('tools.event_track_v2x.')):
                pending.add(imported)
    # Import-time package code is part of the local execution identity too.
    for name in tuple(found):
        parts = name.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if getattr(sys.modules.get(parent), '__file__', None):
                found.setdefault(parent, location(parent))
    return dict(sorted(found.items()))


def _verify_loaded_sources(paths, hashes):
    check(_loaded_source_paths() == paths and all(sha_file(path) == value for path, value in hashes.items()),
          'loaded bridge/helper source changed during conversion')


def _contract(path, expected_sha, *, manifest, raw_sha, archive_sha, projection_sha, clock_sha, inputs):
    path = pinned(path, expected_sha)
    inputs[path] = expected_sha
    contract = json.loads(path.read_bytes())
    required = {'kind', 'dataset', 'split', 'raw_manifest_sha256', 'raw_index_sha256', 'raw_archive_sha256',
        'projection_frames_sha256', 'clock_sha256', 'state_layout', 'coordinate_system', 'velocity_units',
        'reference_time', 'gt_read_online', 'cross_source_inputs', 'detector_config', 'method', 'model', 'fit_receipt',
        'independent_admission', 'sequence_origin_us', 'frames'}
    check(set(contract) == required, 'exact motion/covariance contract required; unknown fields forbidden')
    check(contract['kind'] == MOTION_KIND and contract['dataset'] == 'v2v4real' and contract['split'] == manifest['split'],
          'explicit source-local motion/covariance contract for the same split required')
    check((contract['raw_manifest_sha256'], contract['raw_index_sha256'], contract['raw_archive_sha256'],
           contract['projection_frames_sha256'], contract['clock_sha256']) ==
          (raw_sha, manifest['rows_sha256'], archive_sha, projection_sha, clock_sha), 'motion input byte bindings differ')
    check((contract['state_layout'], contract['coordinate_system'], contract['velocity_units'], contract['reference_time']) ==
          (BOX_LAYOUT, 'source_lidar', 'metres_per_second', 'source_capture'), 'motion units/layout/reference differ')
    check(contract['gt_read_online'] is False and contract['cross_source_inputs'] is False,
          'online GT or cross-source motion requires a different audited contract')
    assets = {}
    for name in ('detector_config', 'method', 'model', 'fit_receipt', 'independent_admission'):
        record = contract[name]
        check(set(record) == {'path', 'sha256'}, 'explicit motion method/model/receipt bindings required')
        asset = pinned(contained_file(path.parent, record['path']), record['sha256'])
        inputs[asset] = record['sha256']
        assets[name] = asset
    method = json.loads(assets['method'].read_bytes())
    check(set(method) == {'kind', 'velocity_method', 'initial_observation_policy', 'full_covariance_method', 'implementation'},
          'explicit velocity, initial observation and covariance methods required')
    check(method['kind'] == 'v2v4real_causal_motion_covariance_method_v1' and
          all(isinstance(method[k], str) and method[k].strip() for k in
              ('velocity_method', 'initial_observation_policy', 'full_covariance_method')),
          'missing motion/covariance policy; no defaults are provided')
    implementation = method['implementation']
    check(set(implementation) == {'path', 'sha256'}, 'method implementation byte binding required')
    source = pinned(contained_file(assets['method'].parent, implementation['path']), implementation['sha256'])
    inputs[source] = implementation['sha256']
    fit = json.loads(assets['fit_receipt'].read_bytes())
    check(fit.get('kind') == 'v2v4real_motion_covariance_train_fit_v1' and fit.get('fit_split') == 'train'
          and fit.get('official_test_used') is False and fit.get('model_sha256') == contract['model']['sha256']
          and fit.get('method_sha256') == contract['method']['sha256']
          and isinstance(fit.get('fit_groups'), list) and fit['fit_groups']
          and all(isinstance(x, str) and x for x in fit['fit_groups'])
          and len(set(fit['fit_groups'])) == len(fit['fit_groups']), 'separate train-only motion fit receipt required')
    admission = json.loads(assets['independent_admission'].read_bytes())
    check(admission.get('kind') == 'v2v4real_motion_covariance_independent_admission_v1'
          and all(admission.get(k) is True for k in ('causal_source_only_verified', 'initial_observation_policy_verified',
                  'full_nine_state_covariance_verified', 'independent_acceptance_verified'))
          and all(admission.get(k+'_sha256') == contract[k]['sha256'] for k in ('method', 'model', 'fit_receipt')),
          'explicit independently admitted motion/covariance method and fit are required')
    return path, contract


def build(raw_root, raw_manifest_sha256, raw_archive_sha256, projection_frames, projection_frames_sha256,
          motion_contract, motion_contract_sha256, clock, clock_sha256, calibration, calibration_sha256,
          calibration_receipt, calibration_receipt_sha256, output, *, fixture=False, progress=None,
          raw_export_receipt=None, raw_export_receipt_sha256=None):
    """Consume immutable raw candidates and already supplied motion contracts.

    Motion ``frames`` entries are keyed by SHA-256(canonical raw-index row),
    and contain a hash-bound NPZ (``velocity_source`` N x 2, ``covariances``
    N x 9 x 9) plus the exact raw-row hashes used by its causal estimator.
    Dependencies must include the current frame and may include only earlier
    frames from the same sequence and source. Empty source frames are required.
    This interface validates declared support; it does not rerun the estimator.
    """
    source_paths = _loaded_source_paths()
    source_sha256 = {str(path): sha_file(path) for path in sorted(set(source_paths.values()))}
    check(type(fixture) is bool, 'explicit fixture flag required')
    output = Path(output).absolute()
    receipt_path = output.parent/(output.name+'-raw-bridge-receipt.json')
    check(not output.exists() and not receipt_path.exists() and
          not any(p.is_symlink() for p in (output, receipt_path, *output.parents)), 'fresh separate cache output required')
    root = Path(raw_root).absolute()
    check(root not in output.parents and Path(motion_contract).absolute().parent not in output.parents,
          'cache must be separate from immutable raw and motion assets')
    inputs = {}
    def bind(path, expected):
        path = pinned(path, expected)
        inputs[path] = expected
        return path
    manifest_path = bind(root/'native-feature-manifest', raw_manifest_sha256)
    manifest_raw = manifest_path.read_bytes()
    m = json.loads(manifest_raw)
    is_test = m.get('kind') == 'v2v4real_official_test_raw_native_features_v1'
    split = 'official_test' if is_test else 'train'
    check(set(m) == RAW_MANIFEST_FIELDS | ({'projection_manifest', 'split_admission'} if is_test else set()),
          'exact split-specific GT-free raw manifest fields required')
    check(m.get('kind') == ('v2v4real_official_test_raw_native_features_v1' if is_test else 'v2v4real_train_raw_native_features_v1')
          and m.get('split') == split and m.get('cohort') == ('official_test_raw_native_features' if is_test else 'official_train_raw_native_features')
          and m.get('protocol_id') == 'v2v4real-nominal-10hz-formal-v1', 'explicit split-specific raw envelope required')
    check(m.get('candidate_protocol') == 'rbf-all-class-top64-v1' and m.get('matching_candidate_rule') == RAW_RULE
          and m.get('state_layout') == RAW_LAYOUT and m.get('feature_method') == NATIVE_FEATURE,
          'native raw-top64 layout/feature contract differs; NMS caches are not accepted')
    check(m.get('official_test_read') is is_test and all(m.get(k) is False for k in ('gt_read', 'paper_metric', 'velocity_exported', 'covariance_exported')),
          'raw envelope contains unsupported supervision/state claims')
    index_path = bind(root/'native-feature-index', m['rows_sha256'])
    index_raw = index_path.read_bytes()
    rows = [json.loads(line) for line in index_raw.splitlines()]
    check(len(rows) == m['rows'] and rows, 'complete source row index required')
    projection_path = bind(projection_frames, projection_frames_sha256)
    if is_test:
        documents = {}
        for name in ('projection_manifest', 'split_admission'):
            record = m[name]
            check(set(record) == {'path', 'sha256'}, 'explicit test projection/split records required')
            documents[name] = json.loads(bind(contained_file(root, record['path']), record['sha256']).read_bytes())
        projection, admission = documents['projection_manifest'], documents['split_admission']
        check(m['projection_manifest']['sha256'] == m['prepared_projection_sha256']
              and projection.get('kind') == 'v2v4real_pose_lidar_projection_v2'
              and projection.get('dataset_split') == 'test' and projection.get('gt_in_projection') is False
              and projection.get('frames_sha256') == projection_frames_sha256
              and projection.get('source_frame_count') == m['rows'], 'official test projection lineage differs')
        check(m['split_admission']['sha256'] == m['split_admission_sha256']
              and admission.get('kind') == 'v2v4real_official_split_admission_v1' and admission.get('split') == 'official_test'
              and admission.get('projection_manifest_sha256') == m['prepared_projection_sha256']
              and admission.get('official_original_split_preserved') is True
              and admission.get('official_split_membership_verified') is True and admission.get('official_test_unchanged') is True,
              'official test split admission required; historical train envelope cannot be renamed')
        check(raw_export_receipt is not None and raw_export_receipt_sha256 is not None, 'new official-test raw export receipt required')
        export = json.loads(bind(raw_export_receipt, raw_export_receipt_sha256).read_bytes())
        check(export.get('kind') == 'v2v4real_official_test_raw_export_v1' and export.get('split') == 'official_test'
              and export.get('gt_read') is False and export.get('official_test_read') is True and export.get('paper_metric') is False
              and export.get('raw_manifest_sha256') == raw_manifest_sha256 and export.get('raw_archive_sha256') == raw_archive_sha256
              and export.get('checkpoint_sha256') == m['checkpoint_sha256'] and export.get('rows_sha256') == m['rows_sha256']
              and export.get('projection_manifest_sha256') == m['prepared_projection_sha256']
              and export.get('projection_frames_sha256') == projection_frames_sha256
              and export.get('split_admission_sha256') == m['split_admission_sha256'], 'new test producer byte/split binding differs')
    else:
        check(raw_export_receipt is None and raw_export_receipt_sha256 is None, 'test export receipt cannot be attached to a train envelope')
    poses = {}
    for line in projection_path.read_bytes().splitlines():
        p = json.loads(line)
        check(set(p) == PROJECTION_FIELDS, 'exact GT-free projection metadata fields required')
        check(type(p['is_ego']) is bool, 'explicit source role required')
        key = p['sequence_id'], p['frame_key'], 'vehicle-side' if p['is_ego'] else 'infrastructure-side'
        check(key not in poses, 'duplicate projection source frame')
        poses[key] = p
    payloads, by_hash, by_key = {}, {}, {}
    for row in rows:
        check(set(row) == RAW_FIELDS, 'raw row fields differ; GT/identity fields forbidden')
        key = identity(row)
        check(key in poses and key not in by_key and row['payload'] not in payloads, 'duplicate or unknown raw source frame/payload')
        pose = poses[key]
        check(row['source_to_world'] == pose['source_to_world'] and row['pcd_sha256'] == pose['pcd_sha256']
              and row['frame_ordinal'] == pose['frame_ordinal'], 'raw pose/source projection binding differs')
        check(type(row['frame_ordinal']) is int and row['frame_ordinal'] >= 0 and type(row['candidates']) is int
              and 0 <= row['candidates'] <= 64, 'invalid ordinal or raw top64 count')
        rel = PurePosixPath(row['payload'])
        check(rel.as_posix() == row['payload'] and len(rel.parts) == 2 and rel.parts[0] == 'payloads'
              and '..' not in rel.parts and rel.suffix == '.npz', 'unsafe raw payload path')
        payloads[row['payload']], by_key[key], by_hash[digest(canonical(row))] = row, row, row
    check(set(by_key) == set(poses) and sorted(m['sequence_ids']) == sorted({k[0] for k in by_key}),
          'all projected source frames, including empty frames, are required')
    for sequence in m['sequence_ids']:
        left = {r['frame_ordinal']: r['frame_id'] for r in rows if r['sequence_id'] == sequence and r['side'] == 'vehicle-side'}
        right = {r['frame_ordinal']: r['frame_id'] for r in rows if r['sequence_id'] == sequence and r['side'] == 'infrastructure-side'}
        count = sum(r['sequence_id'] == sequence for r in rows)
        check(left == right and sorted(left) == list(range(len(left))) and count == 2*len(left),
              'paired complete ordinal schedule required; empty frames cannot be inferred or omitted')
    clock_path = bind(clock, clock_sha256)
    clock_config = json.loads(clock_path.read_bytes())
    check(clock_config.get('protocol_id') == m['protocol_id'] and clock_config.get('dataset') == 'v2v4real'
          and clock_config.get('nominal_rate_hz') == 10 and clock_config.get('period_us') == 100000
          and clock_config.get('measured') is False, 'frozen nominal 10 Hz clock required; measured time must not be inferred')
    contract_path, contract = _contract(motion_contract, motion_contract_sha256, manifest=m, raw_sha=raw_manifest_sha256,
        archive_sha=raw_archive_sha256, projection_sha=projection_frames_sha256, clock_sha=clock_sha256, inputs=inputs)
    check(fixture or contract['detector_config']['sha256'] == DETECTOR_CONFIG_SHA256,
          'historical production detector config differs from frozen native export')
    origins = contract['sequence_origin_us']
    check(set(origins) == set(m['sequence_ids']) and all(type(v) is int and v >= 0 for v in origins.values()),
          'explicit nominal clock origin for every sequence required')
    motion_rows, motion_files = {}, set()
    for item in contract['frames']:
        check(set(item) == {'raw_row_sha256', 'arrays', 'dependencies'}, 'exact motion frame fields required')
        row_sha = item['raw_row_sha256']
        check(row_sha in by_hash and row_sha not in motion_rows, 'duplicate or foreign motion row')
        row, dependencies = by_hash[row_sha], item['dependencies']
        check(isinstance(dependencies, list) and row_sha in dependencies and len(set(dependencies)) == len(dependencies),
              'explicit unique causal support including the current source frame required')
        for dep in dependencies:
            check(dep in by_hash, 'unbound motion input')
            prior = by_hash[dep]
            check(prior['sequence_id'] == row['sequence_id'] and prior['side'] == row['side']
                  and prior['frame_ordinal'] <= row['frame_ordinal'], 'future/cross-sequence/cross-source motion dependency forbidden')
        record = item['arrays']
        check(set(record) == {'path', 'sha256'}, 'motion payload byte binding required')
        path = bind(contained_file(contract_path.parent, record['path']), record['sha256'])
        check(path not in motion_files and path.stat().st_size <= 256*1024, 'reused or oversized motion payload')
        motion_files.add(path)
        motion_rows[row_sha] = path
    check(set(motion_rows) == set(by_hash), 'motion payloads must cover every source frame including zero candidates')
    model_path = bind(calibration, calibration_sha256)
    proof_path = bind(calibration_receipt, calibration_receipt_sha256)
    proof = json.loads(proof_path.read_bytes())
    check(proof.get('kind') == 'v2v4real_native_vehicle_existence_calibration_v1'
          and proof.get('protocol_id') == VEHICLE_PROTOCOL and proof.get('fit_split') == 'train'
          and proof.get('official_test_used') is False and proof.get('gt_written_to_calibration_artifact') is False
          and proof.get('calibration_sha256') == calibration_sha256 and proof.get('checkpoint_sha256') == m['checkpoint_sha256'],
          'separate vehicle train-only calibration for this detector required')
    require_vehicle_binding(proof.get('evaluation_class_binding'))
    model = ExistenceCalibration(**json.loads(model_path.read_bytes()))
    check(list(model.fit_groups) == proof.get('fit_groups'), 'calibration group binding differs')
    # The original archive is streamed without extraction. Input bytes survive unchanged.
    archive_path = bind(root/'native-features', raw_archive_sha256)
    expected = set(payloads) | {'manifest.json', 'predictions.jsonl', 'anchors.npy'}
    tracker = ExperimentProgress('native_raw_to_9d_contract_bridge', len(rows),
        eta_scope='local transformation/persistence only; scientific acceptance excluded') if progress is None else None
    def report(n):
        tracker.update(n) if tracker else progress(dict(completed_rows=n, total_rows=len(rows)))
    def frames():
        seen, candidates, valid_candidates, completed = set(), 0, 0, 0
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                check(member.isfile() and member.name in expected and member.name not in seen, 'unsafe, unknown or duplicate archive member')
                seen.add(member.name)
                limit = 64*1024**2 if member.name == 'anchors.npy' else max(len(index_raw), len(manifest_raw)) if member.name in {'manifest.json', 'predictions.jsonl'} else 256*1024
                check(0 <= member.size <= limit, 'oversized archive member')
                raw = archive.extractfile(member).read(limit+1)
                check(len(raw) == member.size, 'truncated archive member')
                if member.name in {'manifest.json', 'predictions.jsonl'}:
                    check(raw == (manifest_raw if member.name == 'manifest.json' else index_raw), 'separate and archived metadata differ')
                    continue
                if member.name == 'anchors.npy':
                    check(digest(raw) == m['anchor_file_sha256'], 'anchor bytes differ')
                    continue
                row = payloads[member.name]
                row_sha = digest(canonical(row))
                check(digest(raw) == row['payload_sha256'], 'raw candidate bytes differ')
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    check(set(z.files) == RAW_ARRAYS, 'unknown raw candidate array fields')
                    arrays = {k: z[k] for k in z.files}
                state7 = arrays.pop('states_source_legacy7')
                n = row['candidates']
                check(state7.shape == (n, 7) and state7.dtype.kind == 'f' and arrays['raw_scores'].shape == (n,)
                      and np.all(arrays['raw_scores'] >= .05) and np.all(np.diff(arrays['raw_scores']) <= 0)
                      and np.all(arrays['class_indices'] == 0), 'raw geometry/order/native class differs')
                with np.load(motion_rows[row_sha], allow_pickle=False) as z:
                    check(set(z.files) == {'velocity_source', 'covariances'}, 'unknown motion payload fields; online GT forbidden')
                    velocity, covariance = z['velocity_source'], z['covariances']
                check(velocity.shape == (n, 2) and velocity.dtype == state7.dtype,
                      'explicit matching source-local velocity dtype/shape required; no implicit casting')
                states = np.concatenate((state7, velocity), axis=1)
                pose = np.asarray(row['source_to_world'], dtype=float)
                check(pose.shape == (4, 4) and np.array_equal(pose[3], [0., 0., 0., 1.]), 'invalid homogeneous source pose')
                stamp = origins[row['sequence_id']] + row['frame_ordinal']*clock_config['period_us']
                meta = dict(kind='detection_cache_v2', schema_version=2, sequence_id=row['sequence_id'], frame_id=row['frame_id'],
                    side=row['side'], agent_mask=SIDES[row['side']], dataset_split=split, dataset_sha256=m['prepared_projection_sha256'],
                    box_reference_timestamp_us=stamp, source_image_timestamp_us=stamp, coordinate_system='source_lidar',
                    lidar_to_world_row_rotation=pose[:3, :3].T.tolist(), lidar_to_world_translation=pose[:3, 3].tolist(),
                    image_sha256=NO_IMAGE, box_layout=BOX_LAYOUT, detector_config_sha256=contract['detector_config']['sha256'],
                    detector_checkpoint_sha256=m['checkpoint_sha256'], feature_checkpoint_sha256=m['checkpoint_sha256'],
                    feature_method=NATIVE_FEATURE, calibration_sha256=calibration_sha256, calibration_fit_split='train',
                    raw_manifest_sha256=raw_manifest_sha256, raw_arrays_sha256=row['payload_sha256'], raw_metadata_sha256=row_sha,
                    arrays_sha256='0'*64)
                frame = NativeDetectionFrame(canonical(meta), states=states, covariances=covariance,
                    scores=model.apply(arrays['raw_scores']), **arrays)
                # Validate actual construction as well as later archive checks; never alter the seven observed columns.
                check(frame.states[:, :7].tobytes() == state7.tobytes(), 'raw seven-state bytes changed')
                candidates += n
                valid_candidates += int(frame.appearance_valid.sum())
                yield frame
                # write_native_cache persists the yielded frame before advancing
                # this iterator. Check actual decoded bytes, including empty rows.
                with np.load(output/f'{completed:08d}.npz', allow_pickle=False) as stored:
                    check(stored['states'].dtype == state7.dtype and stored['states'][:, :7].tobytes() == state7.tobytes()
                          and stored['states'][:, 7:].tobytes() == velocity.tobytes(), 'persisted raw states/velocity changed')
                    for name, value in dict(arrays, covariances=covariance).items():
                        check(stored[name].dtype == value.dtype and stored[name].shape == value.shape
                              and stored[name].tobytes() == value.tobytes(), 'persisted raw/motion array bytes changed: '+name)
                completed += 1
                report(completed)
        check(seen == expected and candidates == m['total_candidates'] and valid_candidates == m['appearance_valid_candidates'],
              'full raw archive coverage/counts differ')
        check(all(sha_file(path) == value for path, value in inputs.items()), 'immutable source/contract bytes changed during conversion')
        _verify_loaded_sources(source_paths, source_sha256)
    producer = dict(fit_split='train', candidate_protocol='rbf-all-class-top64-v1', postprocessing=RAW_RULE,
        detector_checkpoint_sha256=m['checkpoint_sha256'],
        raw_export_manifest_sha256=raw_manifest_sha256, raw_export_archive_sha256=raw_archive_sha256,
        motion_contract_sha256=motion_contract_sha256, motion_independent_admission_sha256=contract['independent_admission']['sha256'],
        clock_sha256=clock_sha256, timestamp_basis='explicit_origin_plus_nominal_frame_ordinal_10hz_not_measured',
        calibration_receipt_sha256=calibration_receipt_sha256, projection_frames_sha256=projection_frames_sha256,
        source_sha256=source_sha256, source_modules={name: str(path) for name, path in source_paths.items()},
        **vehicle_binding())
    cache_sha = write_native_cache(output, frames(), split=split, producer=producer, fixture=fixture)
    cache = NativePaperCache(output, cache_sha)
    check(len(cache.index) == len(rows), 'persisted source frame coverage differs')
    _verify_loaded_sources(source_paths, source_sha256)
    receipt = dict(kind='v2v4real_raw_top64_explicit_motion_contract_bridge_v1', cache_sha256=cache_sha,
        split=split, frames=len(rows), detections=m['total_candidates'], raw_manifest_sha256=raw_manifest_sha256,
        raw_archive_sha256=raw_archive_sha256, motion_contract_sha256=motion_contract_sha256,
        input_sha256={str(path): value for path, value in inputs.items()}, evaluation_class_binding=vehicle_binding(),
        source_sha256=source_sha256, source_modules={name: str(path) for name, path in source_paths.items()},
        original_raw_bytes_preserved=True, complete_source_frame_coverage_verified=True,
        gt_payload_opened=False, detector_executed=False, velocity_estimator_executed=False, covariance_fitter_executed=False,
        motion_independent_admission_imported=True, motion_semantics_independently_reverified=False,
        independent_acceptance_verified=False, paper_results_verified=False, fixture=fixture)
    with receipt_path.open('xb') as stream:
        stream.write(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root', 'raw-manifest-sha256', 'raw-archive-sha256', 'projection-frames', 'projection-frames-sha256',
                 'motion-contract', 'motion-contract-sha256', 'clock', 'clock-sha256', 'calibration', 'calibration-sha256',
                 'calibration-receipt', 'calibration-receipt-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--raw-export-receipt')
    parser.add_argument('--raw-export-receipt-sha256')
    print(json.dumps(build(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
    main()
