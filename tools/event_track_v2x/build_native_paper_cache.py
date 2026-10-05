#!/usr/bin/env python3
"""Frozen official PointPillar + prediction-only native PCD manifest to V2
cache.

No automatic downloads, training, GT YAML loading, timestamp inference or fabricated detections. Supply a hash-bound anchor array from the same detector.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def recalibrate_native_cache(raw_cache, raw_cache_sha256, calibration_path, calibration_sha256,
                             calibration_receipt, calibration_receipt_sha256, output):
    """Reuse native raw candidates; change only scores and their explicit binding.

    This does not admit a missing velocity/covariance or raw-export conversion
    contract. In particular it refuses the historical NMS builder below.
    """
    import numpy as np
    from transvision.models.event_track_v2x.detection_cache_v2 import ARRAYS, canonical, sha_file
    from transvision.models.event_track_v2x.paper_calibration import ExistenceCalibration
    from transvision.models.event_track_v2x.paper_evaluation_policy import (
        VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, require_vehicle_binding, vehicle_binding,
    )
    from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, NativeDetectionFrame, write_native_cache
    from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress
    from tools.event_track_v2x.fit_v2v4real_vehicle_calibration import pinned
    output = Path(output).absolute()
    receipt_output = output.parent/(output.name+'-recalibration-receipt.json')
    if (output.exists() or receipt_output.exists() or any(p.is_symlink() for p in (output, receipt_output, *output.parents))
            or Path(raw_cache).absolute() in output.parents):
        raise ValueError('fresh vehicle cache and separate receipt outputs required')
    model_path = pinned(calibration_path, calibration_sha256)
    proof_path = pinned(calibration_receipt, calibration_receipt_sha256)
    proof = json.loads(proof_path.read_bytes())
    if (proof.get('kind') != 'v2v4real_native_vehicle_existence_calibration_v1'
            or proof.get('protocol_id') != VEHICLE_PROTOCOL or proof.get('fit_split') != 'train'
            or proof.get('official_test_used') is not False or proof.get('gt_written_to_calibration_artifact') is not False
            or proof.get('calibration_sha256') != calibration_sha256):
        raise ValueError('separate vehicle train-only calibration receipt required')
    require_vehicle_binding(proof.get('evaluation_class_binding'))
    calibration = ExistenceCalibration(**json.loads(model_path.read_bytes()))
    if list(calibration.fit_groups) != proof.get('fit_groups'):
        raise ValueError('calibration parameter training groups differ from receipt')
    original = NativePaperCache(raw_cache, raw_cache_sha256)
    manifest = json.loads(original.manifest_json)
    producer = manifest['producer']
    if (producer.get('candidate_protocol') != 'rbf-all-class-top64-v1'
            or producer.get('postprocessing') != 'raw_score_ge_0.05_stable_top64_before_nms'
            or producer.get('native_label_source') != NATIVE_VEHICLE_SELECTION
            or producer.get('detector_checkpoint_sha256') != proof.get('checkpoint_sha256')):
        raise ValueError('native raw-top64 vehicle detector lineage required; legacy NMS cache is not compatible')
    retained_names = sorted(ARRAYS-{'scores'})
    progress = ExperimentProgress('vehicle_raw_cache_recalibration', manifest['frame_count'],
                                 eta_scope='score transformation and persistence only')
    def frames():
        for completed, entry in enumerate(manifest['frames'], 1):
            frame = NativeDetectionFrame.load(raw_cache, entry)
            if (frame.metadata['detector_checkpoint_sha256'] != proof['checkpoint_sha256']
                    or frame.count > 64 or np.any(frame.raw_scores < .05) or np.any(frame.class_indices != 0)):
                raise ValueError('raw native vehicle candidate or detector binding differs')
            meta = dict(frame.metadata, calibration_sha256=calibration_sha256)
            arrays = {key: getattr(frame, key) for key in retained_names}
            yield NativeDetectionFrame(canonical(meta), scores=calibration.apply(frame.raw_scores), **arrays)
            progress.update(completed)
    new_producer = dict(producer, **vehicle_binding())
    new_producer.update(raw_cache_sha256=raw_cache_sha256, calibration_receipt_sha256=calibration_receipt_sha256,
        calibration_sha256=calibration_sha256, original_raw_candidate_arrays_reused=True,
        raw_candidate_postprocessing_unchanged=True,
        calibration_independent_acceptance_verified=proof.get('independent_acceptance_verified') is True)
    digest = write_native_cache(output, frames(), split=manifest['split'], producer=new_producer, fixture=manifest['fixture'])
    created = NativePaperCache(output, digest)
    new_manifest = json.loads(created.manifest_json)
    verification = ExperimentProgress('vehicle_cache_raw_byte_preservation', manifest['frame_count'],
                                     eta_scope='local array preservation readback only; independent acceptance excluded')
    if new_manifest['frame_count'] != manifest['frame_count'] or new_manifest['detection_count'] != manifest['detection_count']:
        raise ValueError('candidate coverage changed during recalibration')
    for completed, (old, new) in enumerate(zip(manifest['frames'], new_manifest['frames']), 1):
        a, b = NativeDetectionFrame.load(raw_cache, old), NativeDetectionFrame.load(output, new)
        for key in retained_names:
            x, y = getattr(a, key), getattr(b, key)
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise ValueError('raw candidate bytes changed during recalibration: '+key)
        for key in set(a.metadata)-{'arrays_sha256', 'calibration_sha256'}:
            if a.metadata[key] != b.metadata[key]:
                raise ValueError('raw frame metadata changed during recalibration: '+key)
        if not np.array_equal(b.scores, calibration.apply(a.raw_scores)):
            raise ValueError('persisted vehicle calibration scores differ')
        verification.update(completed)
    NativePaperCache(raw_cache, raw_cache_sha256)
    pinned(model_path, calibration_sha256); pinned(proof_path, calibration_receipt_sha256)
    receipt = dict(kind='v2v4real_native_vehicle_cache_recalibration_v1', cache_sha256=digest,
        raw_cache_sha256=raw_cache_sha256, calibration_sha256=calibration_sha256,
        calibration_receipt_sha256=calibration_receipt_sha256, evaluation_class_binding=vehicle_binding(),
        preserved_array_fields=retained_names, raw_array_bytes_preserved=True,
        metadata_changes=['arrays_sha256', 'calibration_sha256'], frames=manifest['frame_count'],
        detections=manifest['detection_count'], fixture=manifest['fixture'], gt_payload_opened=False,
        detector_executed=False, independent_acceptance_verified=False,
        new_velocity_covariance_contract_admitted=False, paper_results_verified=False)
    with receipt_output.open('xb') as stream:
        stream.write(canonical(receipt))
    return receipt


def build(source, manifest_path, manifest_sha256, output, *, device='cpu'):
    import numpy as np
    import yaml

    from tools.event_track_v2x.run_v2v4real_pointpillar_smoke import COMMIT, SOURCE_PINS, verify_sources
    from transvision.models.event_track_v2x.detection_cache_v2 import BOX_LAYOUT, SIDES, canonical, contained_file, sha_file
    from transvision.models.event_track_v2x.paper_calibration import ExistenceCalibration
    from transvision.models.event_track_v2x.paper_native_cache import NATIVE_FEATURE, NO_IMAGE, NativeDetectionFrame, write_native_cache
    from transvision.models.event_track_v2x.paper_pointpillar import FrozenPointPillar
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    from transvision.models.event_track_v2x.v2v4real_pcd import read_native_pcd
    path = Path(manifest_path).absolute()
    if sha_file(path) != manifest_sha256:
        raise ValueError('native input manifest changed')
    m = json.loads(path.read_bytes())
    if set(m) != {'kind', 'dataset', 'split', 'dataset_sha256', 'checkpoint', 'anchors', 'calibration', 'covariance_diagonal', 'frames', 'fixture'}:
        raise ValueError('unknown native inputs; GT fields forbidden')
    if m['kind'] != 'rbf_native_prediction_inputs_v1' or m['dataset'] != 'v2v4real' or type(m['fixture']) is not bool:
        raise ValueError('invalid native prediction manifest')
    PaperProtocol(m['dataset'], m['split'])
    assets = {}
    for name in ('checkpoint', 'anchors', 'calibration'):
        assets[name] = contained_file(path.parent, m[name]['path'])
        if sha_file(assets[name]) != m[name]['sha256']:
            raise ValueError(name + ' identity changed')
    calibration = ExistenceCalibration(**json.loads(assets['calibration'].read_bytes()))
    for f in m['frames']:
        if set(f) != {'sequence_id', 'frame_id', 'side', 'timestamp_us', 'lidar_to_world_row_rotation', 'lidar_to_world_translation', 'pcd'}:
            raise ValueError('unknown native frame fields; GT forbidden')
        if type(f['timestamp_us']) is not int or f['timestamp_us'] < 0:
            raise ValueError('explicit measured microsecond timestamp required')
        if sha_file(contained_file(path.parent, f['pcd']['path'])) != f['pcd']['sha256']:
            raise ValueError('PCD identity changed')
    source = Path(source).resolve()
    verify_sources(source)
    if any(k == 'opencood' or k.startswith('opencood.') for k in sys.modules):
        raise ValueError('refuse unverified previously imported detector')
    sys.path.insert(0, str(source))
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    from opencood.models.point_pillar import PointPillar
    from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range
    config_name = 'opencood/hypes_yaml/point_pillar_late_fusion.yaml'
    config = load_point_pillar_params(yaml.safe_load((source / config_name).read_text()))
    with assets['anchors'].open('rb') as stream:
        anchors = np.load(stream, allow_pickle=False)
    detector = FrozenPointPillar(
        PointPillar(config['model']['args']),
        SpVoxelPreprocessor(config['preprocess'], train=False),
        anchors,
        checkpoint=assets['checkpoint'],
        checkpoint_sha256=m['checkpoint']['sha256'],
        device=device)
    lidar_range = config['preprocess']['cav_lidar_range']

    def frames():
        for f in m['frames']:
            cloud = read_native_pcd(contained_file(path.parent, f['pcd']['path']), expected_sha256=f['pcd']['sha256'])
            points = mask_points_by_range(mask_ego_points(cloud.xyzi), lidar_range)
            arrays = detector.predict(points, lidar_range=lidar_range, covariance_diagonal=m['covariance_diagonal'], calibration=calibration)
            meta = dict(
                kind='detection_cache_v2',
                schema_version=2,
                sequence_id=f['sequence_id'],
                frame_id=f['frame_id'],
                side=f['side'],
                agent_mask=SIDES[f['side']],
                dataset_split=m['split'],
                dataset_sha256=m['dataset_sha256'],
                box_reference_timestamp_us=f['timestamp_us'],
                source_image_timestamp_us=f['timestamp_us'],
                coordinate_system='source_lidar',
                lidar_to_world_row_rotation=f['lidar_to_world_row_rotation'],
                lidar_to_world_translation=f['lidar_to_world_translation'],
                image_sha256=NO_IMAGE,
                box_layout=BOX_LAYOUT,
                detector_config_sha256=SOURCE_PINS[config_name],
                detector_checkpoint_sha256=m['checkpoint']['sha256'],
                feature_checkpoint_sha256=m['checkpoint']['sha256'],
                feature_method=NATIVE_FEATURE,
                calibration_sha256=m['calibration']['sha256'],
                calibration_fit_split='train',
                raw_manifest_sha256=manifest_sha256,
                raw_arrays_sha256=f['pcd']['sha256'],
                raw_metadata_sha256=__import__('hashlib').sha256(canonical(f)).hexdigest(),
                arrays_sha256='0' * 64)
            yield NativeDetectionFrame(canonical(meta), **arrays)
        verify_sources(source)
        if sha_file(path) != manifest_sha256 or any(sha_file(v) != m[k]['sha256'] for k, v in assets.items()):
            raise ValueError('native assets changed during export')

    return write_native_cache(
        output,
        frames(),
        split=m['split'],
        fixture=m['fixture'],
        producer=dict(
            fit_split='train',
            official_commit=COMMIT,
            source_pins=SOURCE_PINS,
            input_sha256=manifest_sha256,
            anchor_sha256=m['anchors']['sha256'],
            detector_checkpoint_sha256=m['checkpoint']['sha256'],
            native_detector_label_scope='vehicle_mapped_to_car',
            postprocessing='raw_score_0.05_rotated_nms_0.15_no_evaluation_roi',
            covariance_diagonal=m['covariance_diagonal'],
            native_protocol_reproduction=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'manifest', 'manifest-sha256', 'raw-cache', 'raw-cache-sha256',
                 'calibration', 'calibration-sha256', 'calibration-receipt', 'calibration-receipt-sha256'):
        p.add_argument('--' + name)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.raw_cache:
        names = ('raw_cache_sha256', 'calibration', 'calibration_sha256', 'calibration_receipt', 'calibration_receipt_sha256')
        if any(not getattr(a, name) for name in names) or any((a.source, a.manifest, a.manifest_sha256)):
            p.error('raw-cache mode requires calibration parameters and receipt hashes; no detector execution arguments')
        print(json.dumps(recalibrate_native_cache(a.raw_cache, a.raw_cache_sha256, a.calibration, a.calibration_sha256,
             a.calibration_receipt, a.calibration_receipt_sha256, a.output), sort_keys=True))
        return
    if not all((a.source, a.manifest, a.manifest_sha256)) or any((a.raw_cache_sha256, a.calibration, a.calibration_sha256,
                                                                          a.calibration_receipt, a.calibration_receipt_sha256)):
        p.error('historical detector mode requires source, manifest and manifest-sha256 only')
    print(json.dumps(dict(cache_sha256=build(a.source, a.manifest, a.manifest_sha256, a.output, device=a.device))))


if __name__ == '__main__':
    main()
