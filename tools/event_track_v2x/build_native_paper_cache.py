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
    for name in ('source', 'manifest', 'manifest-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    print(json.dumps(dict(cache_sha256=build(a.source, a.manifest, a.manifest_sha256, a.output, device=a.device))))


if __name__ == '__main__':
    main()
