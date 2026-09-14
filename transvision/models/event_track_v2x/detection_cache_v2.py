"""Versioned prediction-only cache with calibrated uncertainty and appearance.

V2 deliberately does not subclass V1: legacy MMDet box axes are not V1's
physical length/width/yaw axes, and image information time is not box time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ARRAYS = frozenset({'states', 'raw_scores', 'scores', 'class_indices',
                    'covariances', 'appearance', 'appearance_valid'})
META_FIELDS = frozenset({'kind', 'schema_version', 'sequence_id', 'frame_id', 'side',
    'box_reference_timestamp_us', 'source_image_timestamp_us', 'coordinate_system',
    'lidar_to_world_row_rotation', 'lidar_to_world_translation', 'image_sha256',
    'box_layout', 'agent_mask', 'dataset_split', 'dataset_sha256',
    'detector_config_sha256', 'detector_checkpoint_sha256', 'feature_checkpoint_sha256',
    'feature_method', 'calibration_sha256', 'calibration_fit_split',
    'raw_manifest_sha256', 'raw_arrays_sha256', 'raw_metadata_sha256', 'arrays_sha256'})
FEATURE_METHOD = 'imagenet-r50-c5-roialign3-mean16x128-l2-v1'
BOX_LAYOUT = 'mmdet3d-legacy-gravity-dimxyz-yaw-vxy'
SIDES = {'vehicle-side': 1, 'infrastructure-side': 2}
CLASSES = ('car', 'bicycle', 'pedestrian')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def contained_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('empty/non-string cache path')
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != relative:
        raise ValueError('noncanonical cache path')
    root = Path(root)
    target = root / p
    if not target.is_file() or any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts)+1)):
        raise ValueError('cache payload must be a regular file without symlink ancestors')
    target.resolve().relative_to(root.resolve())
    return target


def _immutable(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True, eq=False)
class DetectionCacheV2:
    """One immutable source frame; no GT, association, or object identity fields."""
    metadata_json: bytes
    states: np.ndarray
    raw_scores: np.ndarray
    scores: np.ndarray
    class_indices: np.ndarray
    covariances: np.ndarray
    appearance: np.ndarray
    appearance_valid: np.ndarray

    def __post_init__(self):
        meta = json.loads(self.metadata_json)
        if set(meta) != META_FIELDS or self.metadata_json != canonical(meta):
            raise ValueError('unknown/noncanonical V2 metadata (possible supervision leakage)')
        if meta['kind'] != 'detection_cache_v2' or meta['schema_version'] != 2:
            raise ValueError('incorrect V2 version')
        if meta['box_layout'] != BOX_LAYOUT or meta['coordinate_system'] != 'source_lidar':
            raise ValueError('unsupported box/coordinate convention')
        if meta['side'] not in SIDES or type(meta['agent_mask']) is not int or meta['agent_mask'] != SIDES[meta['side']]:
            raise ValueError('agent mask does not identify the original source')
        if meta['dataset_split'] not in {'train', 'val'} or meta['calibration_fit_split'] != 'train':
            raise ValueError('test input or non-train calibration forbidden')
        if meta['feature_method'] != FEATURE_METHOD:
            raise ValueError('unknown feature recipe')
        for key, value in meta.items():
            if key.endswith('_sha256') and (not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)):
                raise ValueError('invalid provenance hash: ' + key)
        for key in ['sequence_id', 'frame_id']:
            if not isinstance(meta[key], str) or not re.fullmatch('[A-Za-z0-9_-]+', meta[key]):
                raise ValueError('invalid source identity')
        for key in ['box_reference_timestamp_us', 'source_image_timestamp_us']:
            if type(meta[key]) is not int or meta[key] < 0:
                raise ValueError('source timestamps must be nonnegative integer microseconds')
        rotation = np.asarray(meta['lidar_to_world_row_rotation'], dtype=float)
        translation = np.asarray(meta['lidar_to_world_translation'], dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all()
                or not np.isfinite(translation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5)):
            raise ValueError('invalid source pose')
        n = len(self.states)
        shapes = {'states': (n, 9), 'raw_scores': (n,), 'scores': (n,), 'class_indices': (n,),
                  'covariances': (n, 9, 9), 'appearance': (n, 128), 'appearance_valid': (n,)}
        for key, shape in shapes.items():
            a = np.asarray(getattr(self, key))
            if a.shape != shape or a.dtype.hasobject or not np.isfinite(a).all():
                raise ValueError('invalid V2 array: ' + key)
            object.__setattr__(self, key, _immutable(a))
        if self.class_indices.dtype.kind not in 'iu' or np.any((self.class_indices < 0) | (self.class_indices > 2)):
            raise ValueError('invalid coarse class')
        if self.appearance_valid.dtype.kind != 'b':
            raise ValueError('appearance validity must be boolean')
        if np.any(self.states[:, 3:6] <= 0) or np.any((self.states[:, 6] < -np.pi) | (self.states[:, 6] >= np.pi)):
            raise ValueError('invalid box dimensions or yaw')
        for score in [self.scores, self.raw_scores]:
            if np.any((score < 0) | (score > 1)):
                raise ValueError('invalid confidence')
        if not np.allclose(self.covariances, self.covariances.transpose(0, 2, 1), atol=1e-10):
            raise ValueError('nonsymmetric covariance')
        np.linalg.cholesky(self.covariances)
        if not np.allclose(np.linalg.norm(self.appearance[self.appearance_valid], axis=1), 1, atol=1e-5):
            raise ValueError('appearance must be L2 normalized')
        if np.any(self.appearance[~self.appearance_valid] != 0):
            raise ValueError('missing appearance must be zero with validity false')

    @property
    def metadata(self):
        return json.loads(self.metadata_json)  # fresh copy; cannot mutate the contract.

    @property
    def information_timestamp_us(self):
        m = self.metadata
        return max(m['source_image_timestamp_us'], m['box_reference_timestamp_us'])

    @property
    def count(self):
        return len(self.states)

    def available_at(self, decision_timestamp_us):
        return self.information_timestamp_us <= decision_timestamp_us

    def digest(self):
        h = hashlib.sha256(self.metadata_json)
        for key in sorted(ARRAYS):
            a = getattr(self, key)
            h.update(key.encode()); h.update(a.dtype.str.encode())
            h.update(canonical(list(a.shape))); h.update(a.tobytes())
        return h.hexdigest()

    @classmethod
    def load(cls, root, entry):
        if set(entry) != {'metadata', 'arrays', 'detections', 'frame_sha256'}:
            raise ValueError('unknown V2 frame inventory fields')
        paths = {}
        for key in ['metadata', 'arrays']:
            record = entry[key]
            if set(record) != {'path', 'bytes', 'sha256'}:
                raise ValueError('unknown payload record fields')
            if (type(record['bytes']) is not int or record['bytes'] <= 0
                    or not isinstance(record['sha256'], str)
                    or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
                raise ValueError('invalid payload record identity')
            path = contained_file(root, record['path'])
            if path.stat().st_size != record['bytes'] or sha_file(path) != record['sha256']:
                raise ValueError('V2 payload identity changed')
            paths[key] = path
        with np.load(paths['arrays'], allow_pickle=False) as z:
            if set(z.files) != ARRAYS:
                raise ValueError('unknown V2 array fields')
            frame = cls(paths['metadata'].read_bytes(), **{key: z[key] for key in z.files})
        if (frame.metadata['arrays_sha256'] != entry['arrays']['sha256'] or frame.count != entry['detections']
                or frame.digest() != entry['frame_sha256']):
            raise ValueError('V2 metadata/array/frame binding differs')
        return frame


def load_manifest(root, expected_sha256, *, agent_mask=3):
    """A source mask is a view of the same root, never a new detector run."""
    root = Path(root)
    if root.is_symlink() or type(agent_mask) is not int or agent_mask not in {1, 2, 3}:
        raise ValueError('invalid cache root or source mask')
    path = contained_file(root, 'manifest.json')
    if sha_file(path) != expected_sha256:
        raise ValueError('V2 root identity differs')
    raw_manifest = path.read_bytes()
    m = json.loads(raw_manifest)
    expected = {'kind', 'schema_version', 'split', 'sequences', 'frames', 'frame_count', 'detection_count',
                'source_manifests', 'calibration_sha256', 'dataset_sha256', 'gt_in_cache', 'test_payloads_read'}
    if (set(m) != expected or m['kind'] != 'detection_cache_v2_manifest' or m['schema_version'] != 2
            or m['split'] not in {'train', 'val'} or m['gt_in_cache'] is not False or m['test_payloads_read'] is not False):
        raise ValueError('invalid V2 root contract')
    if raw_manifest != canonical(m):
        raise ValueError('noncanonical V2 root manifest')
    if (not isinstance(m['sequences'], list) or not m['sequences']
            or any(not isinstance(x, str) or not re.fullmatch('[A-Za-z0-9_-]+', x) for x in m['sequences'])
            or m['sequences'] != sorted(set(m['sequences']))
            or not isinstance(m['frames'], list)
            or any(type(m[k]) is not int or m[k] < 0 for k in ['frame_count', 'detection_count'])
            or not isinstance(m['source_manifests'], list) or not m['source_manifests']
            or len(set(m['source_manifests'])) != len(m['source_manifests'])):
        raise ValueError('invalid V2 cohort inventory')
    for value in [m['calibration_sha256'], m['dataset_sha256'], *m['source_manifests']]:
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('invalid V2 root provenance hash')
    expected_files = {'manifest.json'}
    identities, counts, source_manifests = set(), 0, set()
    selected = []
    for entry in m['frames']:
        f = DetectionCacheV2.load(root, entry)
        v = f.metadata
        identity = (v['side'], v['sequence_id'], v['frame_id'])
        if (identity in identities or v['sequence_id'] not in m['sequences'] or v['dataset_split'] != m['split']
                or v['calibration_sha256'] != m['calibration_sha256'] or v['dataset_sha256'] != m['dataset_sha256']
                or v['raw_manifest_sha256'] not in m['source_manifests']):
            raise ValueError('V2 cohort/provenance mismatch')
        identities.add(identity); counts += f.count
        source_manifests.add(v['raw_manifest_sha256'])
        for key in ['metadata', 'arrays']:
            relative = entry[key]['path']
            if relative in expected_files:
                raise ValueError('V2 payload path reused')
            expected_files.add(relative)
        if v['agent_mask'] & agent_mask:
            selected.append(entry)
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if (actual_files != expected_files or any(p.is_symlink() for p in root.rglob('*'))
            or len(identities) != m['frame_count'] or counts != m['detection_count']
            or source_manifests != set(m['source_manifests'])
            or {x[1] for x in identities} != set(m['sequences'])):
        raise ValueError('V2 full-tree coverage mismatch')
    return m, tuple(selected)
