#!/usr/bin/env python3
"""Frozen SPD-val scan-MHT adapter; no GT, training, test or public upload.

The reusable helper permits explicitly labelled development subsets. The CLI
requires the complete official SPD validation schedule and sealed V2 cache.
No production replay source is edited to register this independently tested
backend while other full-sequence experiments bind those sources.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig, PersistentMHTTracker
from tools.event_track_v2x.run_persistent_forest_v2 import _new
from tools.event_track_v2x.run_probabilistic_tracking_v2 import input_files, runtime_evidence, validation_sources
from tools.event_track_v2x.run_tracking_v2 import schedule_rows, SPLIT_SHA
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer, LearnedForestScorer
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream


def mht_sources():
    paths = [Path(__file__), ROOT/'tools/event_track_v2x/persistent_mht_tracking.py',
             ROOT/'tools/event_track_v2x/ranked_partial_assignment.py']
    return dict(validation_sources(), **{p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths})


def mht_runtime_evidence(device):
    import scipy
    from scipy.optimize import _lsap
    return dict(runtime_evidence(device), scipy=scipy.__version__,
                assignment_binary_sha256=sha_file(_lsap.__file__),
                sqlite_version=sqlite3.sqlite_version)


def replay_mht_rows(cache, rows, output, config, *, scorer=None, plan=None):
    if type(cache) is not VerifiedForestCache or type(config) is not PersistentMHTConfig:
        raise TypeError('verified V2 cache and MHT configuration required')
    split = json.loads(cache.manifest_json)['split']
    if split not in ('train', 'val'):
        raise ValueError('MHT adapter excludes all test payloads')
    if scorer is not None and (type(scorer) is not LearnedForestScorer
                               or scorer.options['process_noise'] != config.state.process_noise):
        raise ValueError('frozen learned scorer with matching state protocol required')
    previous, order, origins = {}, [], {}
    for row in rows:
        if set(row) != {'sequence_id', 'vehicle_frame', 'infrastructure_frame', 'box_reference_timestamp_us'}:
            raise ValueError('prediction-only schedule fields required')
        scene, reference = row['sequence_id'], row['box_reference_timestamp_us']
        if not isinstance(scene, str) or not scene or type(reference) is not int or reference <= previous.get(scene, -1):
            raise ValueError('invalid MHT sequence/reference schedule')
        previous[scene] = reference
        order.append(scene)
        for side, key in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
            _, meta = (json.loads(v) for v in cache.index[(scene, side, row[key])])
            if side == 'vehicle-side' and meta['box_reference_timestamp_us'] != reference:
                raise ValueError('cache/schedule reference differs')
    if not rows or order != sorted(order):
        raise ValueError('nonempty sequence-contiguous schedule required')
    for (scene, _, _), (_, meta) in cache.index.items():
        reference = json.loads(meta)['box_reference_timestamp_us']
        origins[scene] = min(origins.get(scene, reference), reference)
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new ordinary output directory required')
    sources = mht_sources()
    plan = dict(plan or {}, kind='scan_mht_replay_plan_v1', cache_sha256=cache.manifest_sha256,
                cache_split=split, configuration=asdict(config), source_sha256=sources,
                scheduled_frames=len(rows), schedule_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
                scorer_signature=None if scorer is None else scorer.signature,
                class_scope=['car'], geometry_baseline=scorer is None,
                paper_eligible=False, parameter_training=False, gt_model_inputs=False,
                same_latency_or_memory_verified=False, reproduced_public_method=False)
    output.mkdir()
    _new(output/'plan.json', plan)
    tracker, scene, frames = None, None, 0
    heads, timings, frame_timings = {}, [], []
    rejected = {'vehicle-side': 0, 'infrastructure-side': 0}
    started = time.monotonic()
    try:
        with (output/'predictions.jsonl').open('xb') as predictions, (output/'tracking.jsonl').open('xb') as audits, \
                (output/'frame-timings.jsonl').open('xb') as times:
            for row in rows:
                frame_started = time.monotonic()
                if scene != row['sequence_id']:
                    if tracker is not None:
                        heads[scene]['database_sha256'] = tracker.close()
                        tracker = None
                    scene = row['sequence_id']
                    database = output/f'sequence-{len(heads):04d}.sqlite'
                    tracker = PersistentMHTTracker(database, sequence_id=scene, config=config)
                    stream = PersistentForestCacheStream(cache, tracker,
                        scorer or GeometryForestScorer(birth_logit=-4., process_noise=config.state.process_noise),
                        origin_us=origins[scene])
                    heads[scene] = dict(database=database.name, frames=0)
                reference = row['box_reference_timestamp_us']
                decision = reference+100000
                deliveries, skipped = [], []
                for side, key in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
                    entry, meta = (json.loads(v) for v in cache.index[(scene, side, row[key])])
                    information = max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
                    if information <= decision:
                        deliveries.append(CacheDelivery(scene, side, row[key], decision, entry['frame_sha256']))
                    else:
                        rejected[side] += 1
                        skipped.append(dict(side=side, frame_id=row[key], information_us=information))
                before = time.monotonic()
                commit = stream.step(deliveries, frame_id=row['vehicle_frame'], event_id=row['vehicle_frame'],
                                     reference_us=reference, decision_us=decision)
                timings.append(time.monotonic()-before)
                predictions.write(commit.prediction_json+b'\n')
                audits.write(canonical(dict(tracking=commit.tracking_audit,
                    source_unavailable_before_payload_read=skipped))+b'\n')
                frames += 1
                heads[scene].update(frames=heads[scene]['frames']+1,
                                    prediction_sha256=commit.prediction['commit_sha256'])
                frame_timings.append(time.monotonic()-frame_started)
                times.write(canonical(dict(sequence_id=scene, frame_id=row['vehicle_frame'],
                    box_reference_timestamp_us=reference, decision_timestamp_us=decision,
                    step_seconds=timings[-1], frame_seconds=frame_timings[-1]))+b'\n')
                if frames % 100 == 0:
                    print(json.dumps(dict(kind='scan_mht_progress', frames=frames, scheduled=len(rows),
                                          elapsed_seconds=time.monotonic()-started)), flush=True)
            heads[scene]['database_sha256'] = tracker.close()
            tracker = None
        if mht_sources() != sources or sha_file(cache.root/'manifest.json') != cache.manifest_sha256:
            raise ValueError('MHT sources or cache manifest changed during replay')
        result = dict(kind='scan_mht_scheduled_replay_v1', status='complete', scheduled_frames=len(rows),
            completed_frames=frames, sequence_heads=heads, plan_sha256=sha_file(output/'plan.json'),
            predictions_sha256=sha_file(output/'predictions.jsonl'), tracking_sha256=sha_file(output/'tracking.jsonl'),
            frame_timings_sha256=sha_file(output/'frame-timings.jsonl'), source_unavailable=rejected,
            elapsed_seconds=time.monotonic()-started,
            latency_seconds_p50_p95_p99_max=np.quantile(timings, [.5, .95, .99, 1.]).tolist(),
            frame_latency_seconds_p50_p95_p99_max=np.quantile(frame_timings, [.5, .95, .99, 1.]).tolist(),
            latency_scope='cache_loading_scoring_MHT_state_commit_excludes_startup_and_output_file_io',
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            database_bytes=sum((output/r['database']).stat().st_size for r in heads.values()),
            exclusive_host=False, same_latency_or_memory_verified=False, global_scan_mht=True,
            geometry_development_baseline=scorer is None, learned_identity_enabled=scorer is not None,
            cache_split=split, cache_sha256=cache.manifest_sha256,
            source_arrival_policy='scheduled_pair_snapshot_at_reference_plus_100ms',
            gt_model_inputs=False, parameter_training=False, test_payloads_read=False,
            raw_row_context_shared_with_recoverable=True, reproduced_public_method=False, paper_eligible=False)
        _new(output/'receipt.json', result)
        return result
    except BaseException as error:
        if tracker is not None:
            tracker.close()
        _new(output/'failure.json', dict(status='failed', completed_frames=frames, scheduled_frames=len(rows),
            error_type=type(error).__name__, error=str(error), partial_outputs_not_final_results=True))
        raise


def run(args):
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        raise ValueError('checkpoint path and hash must be supplied together')
    config = PersistentMHTConfig(state=ForestTrackingConfig(active_limit=getattr(args, 'width', 4)),
        **{k: getattr(args, k, field.default) for k, field in PersistentMHTConfig.__dataclass_fields__.items()
           if k.startswith('max_assignment_') or k == 'max_generated_candidates'})
    rows = schedule_rows(args.schedule, args.schedule_sha256)
    cache = VerifiedForestCache(args.cache, args.cache_sha256)
    manifest = json.loads(cache.manifest_json)
    if (manifest['split'] != 'val' or manifest['frame_count'] != 7189
            or set(manifest['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full sealed SPD validation cache required')
    scorer, checkpoint = None, None
    if args.checkpoint is not None:
        scorer, checkpoint = load_identity_checkpoint(args.checkpoint, args.checkpoint_sha256,
                                                      config=config.state, device=args.device)
        if checkpoint.get('full_official_train') is not True or frozen_cache_identity(cache) != checkpoint['frozen_cache_identity']:
            raise ValueError('full train checkpoint and matching frozen producers required')
    runtime = mht_runtime_evidence(args.device)
    inputs = input_files(args, checkpoint)
    if (inputs[str((Path(args.cache)/'manifest.json').absolute())] != args.cache_sha256
            or inputs[str(Path(args.schedule).absolute())] != args.schedule_sha256
            or checkpoint is not None and (
                inputs[str((Path(args.checkpoint)/'checkpoint.json').absolute())] != args.checkpoint_sha256
                or inputs[str((Path(args.checkpoint)/checkpoint['weights']['path']).absolute())] != checkpoint['weights']['sha256'])):
        raise ValueError('MHT inputs changed during preflight')
    result = replay_mht_rows(cache, rows, args.output, config, scorer=scorer, plan=dict(
        split_sha256=SPLIT_SHA, schedule_sha256=args.schedule_sha256, input_file_sha256=inputs, runtime=runtime,
        checkpoint_sha256=args.checkpoint_sha256, checkpoint_seed=None if checkpoint is None else checkpoint['seed'],
        full_official_schedule_verified=True,
        val_seen_during_research=True, validation_parameter_search=False, validation_checkpoint_selection=False))
    if any(sha_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError('MHT validation input changed during replay')
    if result['completed_frames'] != 3316 or len(result['sequence_heads']) != 21:
        raise ValueError('complete official validation required for final receipt')
    _new(args.output/'full-validation-receipt.json', dict(result,
        full_official_validation_schedule_completed=True, checkpoint_sha256=args.checkpoint_sha256))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'schedule', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('cache-sha256', 'schedule-sha256'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--width', type=int, default=4)
    for key in ('max_assignment_solves', 'max_assignment_frontier', 'max_assignment_matrix_cells', 'max_generated_candidates'):
        p.add_argument('--'+key.replace('_', '-'), type=int, default=getattr(PersistentMHTConfig(), key))
    print(json.dumps(run(p.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
