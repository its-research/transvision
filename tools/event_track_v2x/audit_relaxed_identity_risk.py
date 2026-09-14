#!/usr/bin/env python3
"""Compare an action-local model bound on a sealed tracker state copy.

Does not change action selection, ingest new inputs, fit a model, or publish.
Oversized components are reported uncomputed with bound one, never omitted.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.train_forest_identity import training_sources
from transvision.models.event_track_v2x.allocation_training import _directory, allocation_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker
from transvision.models.event_track_v2x.completion_component_tracking import CompletionTeacherTracker
from transvision.models.event_track_v2x.relaxed_identity_risk import action_risk_certificate


def run(database, database_sha256, prediction_sha256, output, *, maximum_nodes=1024):
    if type(maximum_nodes) is not int or not 1 <= maximum_nodes <= 4096:
        raise ValueError('explicit message capacity from 1 to 4096 nodes required')
    database = Path(database).absolute()
    if (not database.is_file() or database.is_symlink() or sha_file(database) != database_sha256
            or any(Path(str(database)+suffix).exists() for suffix in ('-wal', '-shm', '-journal'))):
        raise ValueError('closed sealed database required')
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as connection:
        metadata = {key: json.loads(value) for key, value in connection.execute('SELECT k,v FROM meta')}
    supported = {cls.SCHEMA: cls for cls in (AllocationTeacherTracker, CompletionTeacherTracker)}
    if metadata['schema'] not in supported or metadata['state']['prediction_sha256'] != prediction_sha256:
        raise ValueError('supported teacher schema and expected output head required')
    sources = dict(training_sources(), **allocation_sources())
    for name in ('tools/event_track_v2x/audit_relaxed_identity_risk.py',
                 'transvision/models/event_track_v2x/relaxed_identity_risk.py'):
        sources[name] = sha_file(ROOT/name)
    output = _directory(output)
    copy = output/'state.sqlite'
    shutil.copyfile(database, copy)
    tracker = supported[metadata['schema']].open(copy, expected_prediction_sha256=prediction_sha256,
                                                expected_database_sha256=database_sha256)
    started = time.monotonic()
    try:
        components = []
        for component in tracker.store.live():
            kernel = tracker._kernel(component)
            summary = json.loads(tracker.db.execute('SELECT payload FROM component_summaries WHERE component=?',
                                                     (component,)).fetchone()[0])
            row = dict(component=component, nodes=kernel.n, loss_nodes=len(summary['decision_indices']),
                stored_output_model_bound=summary['decision']['risk_bound'], stored_eta_upper=summary['eta_upper'])
            if kernel.n > maximum_nodes:
                components.append(dict(row, computed=False, reason='message_node_capacity', model_regret_upper=1.))
                continue
            factors = ForestFactors(tuple(kernel._observation(i).node for i in range(kernel.n)),
                                    tuple(kernel._row(i) for i in range(kernel.n)))
            actions = []
            for handle in sorted(kernel.active | {kernel.meta['output']}):
                report = action_risk_certificate(factors, kernel.parents(handle, maximum=maximum_nodes),
                    scope=summary['decision_indices'], maximum_nodes=maximum_nodes)
                actions.append(dict(handle=handle, is_stored_output=handle == kernel.meta['output'], **report))
            components.append(dict(row, computed=True, actions=actions))
        if tracker.meta != metadata['state'] or sha_file(database) != database_sha256:
            raise ValueError('original database or committed output changed')
        if any(sha_file(ROOT/path) != digest for path, digest in sources.items()):
            raise ValueError('risk diagnostic sources changed during execution')
        result = dict(kind='relaxed_identity_risk_state_comparison_v1', status='complete',
            source_database_sha256=database_sha256, source_prediction_sha256=prediction_sha256,
            source_schema=metadata['schema'], source_scorer_binding=metadata.get('scorer_binding'),
            source_sha256=sources, components=components, observations=tracker.n,
            committed_events=tracker.meta['events'], elapsed_seconds=time.monotonic()-started,
            maximum_message_nodes=maximum_nodes, action_selection_changed=False,
            new_inputs_or_gt_read=False, tracking_validation=False, paper_eligible=False)
        _new_json(output/'comparison.json', result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('components', 'source_sha256')}, sort_keys=True))
        return result
    except BaseException as error:
        _new_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        tracker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--database-sha256', required=True)
    parser.add_argument('--prediction-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-nodes', type=int, default=1024)
    args = parser.parse_args()
    run(args.database, args.database_sha256, args.prediction_sha256, args.output, maximum_nodes=args.maximum_nodes)
