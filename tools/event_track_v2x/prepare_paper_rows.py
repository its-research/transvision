#!/usr/bin/env python3
"""Join frozen caches with separately sealed TRAIN supervision, never feed GT
to models.

The supervision envelope uses source-local legacy V2 boxes and exact annotation references. Native release converters must explicitly resolve native IDs before writing it;
uncertain source correspondences must not be invented.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare(cache_root, cache_sha256, supervision_path, supervision_sha256, output, *, dataset, fixture=False):
    import numpy as np

    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
    from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentity, AnnotationIdentityIndex
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
    from transvision.models.event_track_v2x.forest_training_data import FrameSupervision, prepare_training_rows
    from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    PaperProtocol(dataset, 'train').require_train()
    if sha_file(supervision_path) != supervision_sha256:
        raise ValueError('offline supervision changed')
    m = json.loads(Path(supervision_path).read_bytes())
    if set(m) != {'kind', 'dataset', 'split', 'fixture', 'frames', 'links', 'schedule', 'provenance'}:
        raise ValueError('unknown offline supervision envelope')
    if (m['kind'] != 'rbf_train_supervision_v1' or m['dataset'] != dataset or m['split'] != 'train' or type(fixture) is not bool or m['fixture'] != fixture):
        raise ValueError('train-only supervision/evidence mismatch')
    cache = (VerifiedForestCache if dataset == 'spd' else NativePaperCache)(cache_root, cache_sha256)
    frames, annotations = {}, []
    for f in m['frames']:
        if set(f) != {'sequence_id', 'source_id', 'frame_id', 'timestamp_us', 'boxes', 'annotations'}:
            raise ValueError('unknown supervision frame')
        ann = tuple(AnnotationIdentity(**a) for a in f['annotations'])
        frame = FrameSupervision(f['sequence_id'], f['source_id'], f['frame_id'], f['timestamp_us'], np.asarray(f['boxes'], float).reshape(-1, 7), ann)
        side = ('vehicle-side', 'infrastructure-side')[f['source_id']]
        key = (f['sequence_id'], side, f['frame_id'])
        if key in frames:
            raise ValueError('duplicate sequence-qualified supervision frame')
        frames[key] = frame
        annotations.extend(ann)
    identity = AnnotationIdentityIndex(annotations, m['links'], class_scope=('car', 'bicycle', 'pedestrian'))

    def check():
        if sha_file(supervision_path) != supervision_sha256:
            raise ValueError('supervision changed during preparation')

    return prepare_training_rows(
        cache,
        m['schedule'],
        frames,
        identity,
        output,
        config=PaperForestTrackingConfig(),
        provenance=dict(m['provenance'], dataset=dataset, fixture_only=fixture, supervision_sha256=supervision_sha256),
        finalize_check=check)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'cache-sha256', 'supervision', 'supervision-sha256', 'output', 'dataset'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.cache, a.cache_sha256, a.supervision, a.supervision_sha256, a.output, dataset=a.dataset, fixture=a.fixture), sort_keys=True))


if __name__ == '__main__':
    main()
