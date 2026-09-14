#!/usr/bin/env python3
"""Profile one no-new-input teacher event on a COPY of a sealed database.

This is a debugging experiment, not a scheduled tracking replay or a latency
benchmark. No factors, GT, future frames or trained models are downloaded.
Both cache modes use identical input state, operation and inference budgets.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.allocation_training import allocation_sources, _directory
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker


def profile(database, database_sha256, prediction_sha256, output, *, use_probe_cache=True):
    database = Path(database).absolute()
    if type(use_probe_cache) is not bool or not database.is_file() or database.is_symlink():
        raise ValueError('sealed ordinary database and explicit cache mode required')
    if any(Path(str(database)+suffix).exists() for suffix in ('-wal','-shm','-journal')):
        raise ValueError('close/checkpoint the source database before profiling a copy')
    if sha_file(database) != database_sha256:
        raise ValueError('source database identity differs')
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as connection:
        meta = {k:json.loads(v) for k,v in connection.execute('SELECT k,v FROM meta')}
    if (meta['schema'] != AllocationTeacherTracker.SCHEMA or not meta['state']['events']
            or meta['state']['prediction_sha256'] != prediction_sha256):
        raise ValueError('committed teacher source and expected output head required')
    sources = allocation_sources() | {Path(__file__).relative_to(ROOT).as_posix():sha_file(__file__)}
    output = _directory(output)
    copied = output/'state.sqlite'
    shutil.copyfile(database,copied)
    if sha_file(copied) != database_sha256 or sha_file(database) != database_sha256:
        raise ValueError('source changed while copying')
    tracker = AllocationTeacherTracker.open(copied, expected_prediction_sha256=prediction_sha256,
                                            expected_database_sha256=database_sha256)
    profiler = cProfile.Profile()
    try:
        if not use_probe_cache:
            tracker.MAX_PROBE_CACHE_HANDLES = 0
        cache_ingestion = (None if meta.get('cache_binding') is None else dict(
            kind='persistent_cache_ingestion_v1', configuration_sha256=meta['cache_binding'], new_deliveries=[]))
        started = time.monotonic()
        profiler.enable()
        try:
            result = tracker.step((),(),frame_id='teacher-profile-no-new-input',event_id='teacher-profile-no-new-input',
                reference_us=meta['state']['reference_us']+1,decision_us=meta['state']['decision_us']+1,
                cache_ingestion=cache_ingestion,scorer_binding=meta.get('scorer_binding'))
        finally:
            profiler.disable()
        elapsed = time.monotonic()-started
        _new_json(output/'prediction.json', result.prediction)
        _new_json(output/'audit.json', result.audit)
        after_sha = tracker.close(); tracker = None
        if sha_file(database) != database_sha256 or allocation_sources() != {k:v for k,v in sources.items() if k != Path(__file__).relative_to(ROOT).as_posix()}:
            raise ValueError('source database or solver changed during profiling')
        profiler.dump_stats(str(output/'event.pstats'))
        report = dict(kind='copied_teacher_no_new_input_event_profile_v1',status='complete',
            original_database_sha256=database_sha256, original_prediction_sha256=prediction_sha256,
            original_event_count=meta['state']['events'], copied_database_sha256=after_sha,
            source_sha256=sources,use_probe_cache=use_probe_cache,profiled_step_seconds=elapsed,
            observation_count=result.audit['observation_count'],new_observations=result.audit['new_observations'],
            prediction_sha256=sha_file(output/'prediction.json'),audit_sha256=sha_file(output/'audit.json'),
            model_regret_upper=result.audit['model_regret_upper'],search_steps=result.audit['search_steps'],
            teacher_probe_executions=result.audit['teacher_probe_executions'],
            teacher_probe_cache_hits=result.audit['teacher_probe_cache_hits'],
            teacher_probe_cache_peak_handles=result.audit['teacher_probe_cache_peak_handles'],
            teacher_requested_search_steps=result.audit['teacher_requested_search_steps'],
            scheduled_replay=False,latency_benchmark=False,tracking_validation=False,paper_eligible=False)
        _new_json(output/'receipt.json',report)
        print(json.dumps(report,sort_keys=True),flush=True)
        return report
    except BaseException as error:
        _new_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--database-sha256',required=True)
    parser.add_argument('--prediction-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--disable-probe-cache',action='store_true')
    args=parser.parse_args()
    profile(args.database,args.database_sha256,args.prediction_sha256,args.output,
            use_probe_cache=not args.disable_probe_cache)
