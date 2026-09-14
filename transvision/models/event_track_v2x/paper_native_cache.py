"""Dataset-aware V2 numerical payloads with a NEW V2V4Real outer manifest.

The original SPD cache loader still rejects test. Native LiDAR-only appearance has its own feature identity; it is never represented as an ImageNet embedding. V2's image timestamp
slot denotes the same LiDAR capture for this envelope; its image hash is the canonical no-image marker below.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .detection_cache_v2 import ARRAYS, DetectionCacheV2, canonical, contained_file, sha_file
from .forest_cache_stream import VerifiedForestCache
from .paper_protocol import PaperProtocol

NATIVE_FEATURE = 'rbf-pointpillar-bev128-bilinear-l2-v1'
NO_IMAGE = hashlib.sha256(b'v2v4real-lidar-only-no-image-v1').hexdigest()


class NativeDetectionFrame(DetectionCacheV2):

    def _validate_dataset_feature(self, meta):
        PaperProtocol('v2v4real', meta['dataset_split'])
        if (meta['calibration_fit_split'] != 'train' or meta['feature_method'] != NATIVE_FEATURE or meta['image_sha256'] != NO_IMAGE
                or meta['source_image_timestamp_us'] != meta['box_reference_timestamp_us']):
            raise ValueError('native LiDAR feature/calibration contract differs')


class NativePaperCache(VerifiedForestCache):

    def __init__(self, root, expected_sha256):
        self.root = Path(root)
        path = contained_file(root, 'manifest.json')
        if sha_file(path) != expected_sha256:
            raise ValueError('native cache manifest changed')
        manifest = json.loads(path.read_bytes())
        required = {'kind', 'dataset', 'split', 'sequences', 'frames', 'frame_count', 'detection_count', 'gt_in_cache', 'fixture', 'producer'}
        if set(manifest) != required or manifest['kind'] != 'rbf_native_v2_cache_v1' or manifest['dataset'] != 'v2v4real':
            raise ValueError('invalid dataset-specific cache envelope')
        PaperProtocol(manifest['dataset'], manifest['split'])
        if manifest['gt_in_cache'] is not False or type(manifest['fixture']) is not bool:
            raise ValueError('GT or unknown evidence status in native cache')
        self.manifest_sha256, self.manifest_json = expected_sha256, canonical(manifest)
        self.index, files, count, producers = {}, {'manifest.json'}, 0, set()
        for entry in manifest['frames']:
            frame = NativeDetectionFrame.load(root, entry)
            meta = frame.metadata
            key = meta['sequence_id'], meta['side'], meta['frame_id']
            if key in self.index or meta['dataset_split'] != manifest['split']:
                raise ValueError('duplicate native source frame or mixed split')
            self.index[key] = canonical(entry), canonical(meta)
            producers.add((meta['feature_checkpoint_sha256'], meta['detector_checkpoint_sha256'], meta['calibration_sha256'], meta['dataset_sha256']))
            count += frame.count
            for kind in ('metadata', 'arrays'):
                name = entry[kind]['path']
                if name in files:
                    raise ValueError('reused native payload path')
                files.add(name)
        if (not self.index or len(producers) != 1 or len(self.index) != manifest['frame_count'] or count != manifest['detection_count']
                or sorted({k[0]
                           for k in self.index}) != manifest['sequences'] or files != {p.relative_to(self.root).as_posix()
                                                                                       for p in self.root.rglob('*') if p.is_file()}
                or any(p.is_symlink() for p in self.root.rglob('*'))):
            raise ValueError('native cache coverage or frozen producer mismatch')

    def load_arrived(self, delivery, decision_us):
        entry, meta = self.describe(delivery)
        if type(decision_us) is not int or not meta['box_reference_timestamp_us'] <= delivery.arrival_us <= decision_us:
            raise ValueError('native frame not yet available')
        return NativeDetectionFrame.load(self.root, entry)


def write_native_cache(root, frames, *, split, producer, fixture=False):
    PaperProtocol('v2v4real', split)
    if not isinstance(producer, dict) or producer.get('fit_split') != 'train':
        raise ValueError('native frozen producer requires train provenance')
    root = Path(root)
    if root.exists() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('create-once native cache required')
    root.mkdir()
    entries, sequences, count = [], set(), 0
    for i, frame in enumerate(frames):
        if type(frame) is not NativeDetectionFrame or frame.metadata['dataset_split'] != split:
            raise ValueError('native frame type/split differs')
        array_path = root / f'{i:08d}.npz'
        with array_path.open('xb') as stream:
            np.savez_compressed(stream, **{k: getattr(frame, k) for k in ARRAYS})
        meta = dict(frame.metadata, arrays_sha256=sha_file(array_path))
        # Revalidate and rebind digest after persistence.
        frame = NativeDetectionFrame(canonical(meta), **{k: getattr(frame, k) for k in ARRAYS})
        meta_path = root / f'{i:08d}.json'
        with meta_path.open('xb') as stream:
            stream.write(frame.metadata_json)

        def record(path):
            return dict(path=path.name, bytes=path.stat().st_size, sha256=sha_file(path))

        entries.append(dict(metadata=record(meta_path), arrays=record(array_path), detections=frame.count, frame_sha256=frame.digest()))
        sequences.add(meta['sequence_id'])
        count += frame.count
    manifest = dict(
        kind='rbf_native_v2_cache_v1',
        dataset='v2v4real',
        split=split,
        sequences=sorted(sequences),
        frames=entries,
        frame_count=len(entries),
        detection_count=count,
        gt_in_cache=False,
        fixture=fixture,
        producer=producer)
    with (root / 'manifest.json').open('xb') as stream:
        stream.write(canonical(manifest))
    sha = sha_file(root / 'manifest.json')
    NativePaperCache(root, sha)
    return sha
