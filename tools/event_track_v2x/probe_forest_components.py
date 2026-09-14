#!/usr/bin/env python3
"""Full TRAIN resource probe of nonoverlapping raw-observation windows.

All car observations selected by the declared frozen rule are counted. No GT,
accuracy metric, model fitting or test/val payload is accepted. Cutting window
boundaries here is ONLY a sizing probe, not an identity-preserving handoff.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from transvision.models.event_track_v2x.detection_cache_v2 import (
    DetectionCacheV2, canonical, load_manifest, sha_file,
)
from transvision.models.event_track_v2x.forest_components import split_forest
from transvision.models.event_track_v2x.forest_tracking import (
    ForestTrackingConfig, append_parent_support, cache_detections,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors


def probe(cache, expected_sha256, *, window_us=2_000_000, max_component_nodes=128,
          minimum_raw_score=.05, maximum_detections=64):
    path = Path(cache)/'manifest.json'
    if path.is_symlink() or sha_file(path) != expected_sha256:
        raise ValueError('cache identity differs')
    header = json.loads(path.read_bytes())
    if header.get('split') != 'train':
        raise ValueError('component resource design uses train only')
    config = ForestTrackingConfig(component_mode=True, potential_context='component',
        window_us=window_us, max_nodes=10000, max_component_nodes=max_component_nodes)
    # Validate selection even for an empty input before the full payload audit.
    if (not isinstance(minimum_raw_score, (int, float)) or not 0 <= minimum_raw_score <= 1
            or type(maximum_detections) is not int or maximum_detections < 1):
        raise ValueError('invalid fixed candidate selection')
    started = time.monotonic()
    manifest, entries = load_manifest(cache, expected_sha256)
    scenes = defaultdict(list)
    for entry in entries:
        meta = json.loads((Path(cache)/entry['metadata']['path']).read_bytes())
        scenes[meta['sequence_id']].append((meta, entry))
    results, frames, observations_count = [], 0, 0
    for sequence, values in sorted(scenes.items()):
        origin = min(m['box_reference_timestamp_us'] for m, _ in values)
        groups = defaultdict(list)
        for meta, entry in values:
            information = max(meta['source_image_timestamp_us'], meta['box_reference_timestamp_us'])
            groups[(information-origin)//window_us].append((information, meta, entry))
        for number, group in sorted(groups.items()):
            decision = max(t for t, _, _ in group)
            observations = []
            for information, meta, entry in sorted(group, key=lambda x: (x[0], x[1]['side'], x[1]['frame_id'])):
                frame = DetectionCacheV2.load(cache, entry)
                observations.extend(cache_detections(frame, arrival_us=information, decision_us=decision,
                    origin_us=origin, minimum_raw_score=minimum_raw_score, maximum_detections=maximum_detections))
                frames += 1
            observations.sort(key=lambda o: (o.node.arrival_us, o.node.source_id, o.node.frame_id, o.detection_index))
            observations = tuple(observations)
            # No over-capacity observation or component is silently discarded.
            # max_nodes is not used as a filter in this non-inference resource probe.
            before = time.monotonic()
            support = append_parent_support(observations, (), config)
            factors = ForestFactors(tuple(o.node for o in observations),
                                     tuple(tuple((p, 0.) for p in row) for row in support))
            components = split_forest(factors)
            sizes = sorted(len(c.indices) for c in components)
            large = [size for size in sizes if size > max_component_nodes]
            observations_count += len(observations)
            results.append({'sequence_id': sequence, 'window_index': number,
                'frames': len(group), 'observations': len(observations),
                'first_information_us': min(t for t, _, _ in group), 'last_information_us': decision,
                'component_count': len(sizes), 'component_sizes': sizes,
                'largest_component': max(sizes, default=0), 'over_capacity_components': len(large),
                'over_capacity_observations': sum(large),
                'support_construction_seconds': time.monotonic()-before})
        print('FOREST_COMPONENT_SEQUENCE '+json.dumps({'sequence_id': sequence,
            'complete_frames': frames, 'complete_windows': len(results)}), flush=True)
    if frames != manifest['frame_count'] or set(scenes) != set(manifest['sequences']):
        raise ValueError('resource probe did not cover the exact full train cohort')
    all_sizes = [n for r in results for n in r['component_sizes']]
    sources = [Path(__file__)] + [ROOT/'transvision/models/event_track_v2x'/name for name in (
        'forest_components.py', 'forest_tracking.py', 'identity_forest.py', 'hypothesis_bank.py',
        'detection_cache_v2.py', 'tracking_v2.py', 'prediction_features.py', 'fusion.py', 'arrays.py')]
    return {'kind': 'recoverable_forest_fulltrain_component_resource_probe_v1',
        'cache_manifest_sha256': expected_sha256, 'class_scope': ['car'], 'split': 'train',
        'frames': frames, 'sequences': len(scenes), 'windows': results, 'window_count': len(results),
        'observations': observations_count, 'component_count': len(all_sizes),
        'component_size_quantiles_0_50_95_100': np.quantile(all_sizes, [0, .5, .95, 1]).tolist() if all_sizes else [],
        'largest_component': max(all_sizes, default=0),
        'over_capacity_components': sum(r['over_capacity_components'] for r in results),
        'over_capacity_observations': sum(r['over_capacity_observations'] for r in results),
        'config': asdict(config), 'minimum_raw_score': minimum_raw_score,
        'maximum_detections_per_source_frame': maximum_detections,
        'window_policy': 'nonoverlapping_information_time_blocks_per_sequence',
        'arrival_policy': 'information_time_zero_transport_delay_proxy',
        'window_boundary_edges_omitted_for_sizing_only': True,
        'paired_output_schedule_applied': False, 'identity_preserving_handoff_executed': False,
        'component_inference_executed': False, 'source_sha256': {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sources},
        'elapsed_seconds_including_full_cache_audit': time.monotonic()-started,
        'resource_probe_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        'inference_memory_or_latency_measurement': False, 'model_fitted': False,
        'gt_payloads_read': False, 'validation_or_test_payloads_read': False,
        'paper_performance_evidence': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--cache-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window-us', type=int, default=2_000_000)
    p.add_argument('--max-component-nodes', type=int, default=128)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        p.error('output must be a new file')
    result = probe(args.cache, args.cache_sha256, window_us=args.window_us,
                   max_component_nodes=args.max_component_nodes)
    with args.output.open('xb') as stream:
        stream.write(canonical(result))
    print('FOREST_COMPONENT_COMPLETE '+json.dumps({k: v for k, v in result.items()
        if k not in {'windows', 'source_sha256', 'config'}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
