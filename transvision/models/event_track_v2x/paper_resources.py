"""Auditable heterogeneous costs, not an invented scalar 'equal K' budget."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from .detection_cache_v2 import sha_file


def database_inventory(path, expected_sha256):
    path = Path(path)
    if sha_file(path) != expected_sha256:
        raise ValueError('state database changed')
    db = sqlite3.connect(path.absolute().as_uri() + '?mode=ro', uri=True)
    try:
        page_size = db.execute('PRAGMA page_size').fetchone()[0]
        pages = db.execute('PRAGMA page_count').fetchone()[0]
        free = db.execute('PRAGMA freelist_count').fetchone()[0]
        try:
            tables = [dict(name=name, allocated_bytes=size) for name, size in db.execute('SELECT name,sum(pgsize) FROM dbstat GROUP BY name ORDER BY name')]
            detail = True
        except sqlite3.OperationalError:
            tables = []
            detail = False
    finally:
        db.close()
    if sha_file(path) != expected_sha256:
        raise ValueError('state database changed during inventory')
    return dict(
        file_bytes=path.stat().st_size,
        page_size=page_size,
        allocated_pages=pages,
        free_pages=free,
        table_page_inventory=tables,
        detailed_storage_available=detail,
        resident_memory_estimate=False,
        includes_indexes_and_event_audit=True)


def summarize(output, databases, timings):
    output = Path(output)
    totals = dict(
        scorer_forward_calls=0,
        model_forward_calls=0,
        scorer_node_tokens=0,
        scorer_pair_tokens=0,
        stored_factor_edges=0,
        posterior_search_steps=0,
        action_search_steps=0,
        action_materialization_steps=0,
        state_updates=0,
        assignment_solves=0,
        generated_candidates=0,
        priority_feature_rows=0,
        teacher_requested_search_steps=0,
        teacher_probe_executions=0,
        teacher_probe_cache_hits=0,
        jpda_iterations=0,
        jpda_transitions=0,
        jpda_message_updates=0)
    with (output / 'audit.jsonl').open('rb') as stream:
        for line in stream:
            audit = json.loads(line)
            contexts = audit['cache_ingestion']['scorer_context_indices']
            totals['scorer_forward_calls'] += len(contexts)
            totals['scorer_node_tokens'] += sum(len(c) for c in contexts)
            totals['scorer_pair_tokens'] += sum(len(c)**2 for c in contexts)
            totals['stored_factor_edges'] += sum(len(c) for c in contexts)
            totals['posterior_search_steps'] += audit.get('search_steps', audit.get('expansions', 0))
            for key in ('action_search_steps', 'action_materialization_steps', 'state_updates', 'assignment_solves', 'generated_candidates', 'priority_feature_rows',
                        'teacher_requested_search_steps', 'teacher_probe_executions', 'teacher_probe_cache_hits'):
                totals[key] += audit.get(key, 0)
            for scan in audit.get('conditional_scans', []):
                for key in ('iterations', 'transitions', 'message_updates'):
                    totals['jpda_' + key] += scan['marginals'][key]
    plan = json.loads((output / 'plan.json').read_bytes())
    if plan['configuration']['method'] != 'geometry':
        totals['model_forward_calls'] = totals['scorer_forward_calls']
    seconds = np.asarray([r['seconds'] for r in timings], float)
    return dict(
        kind='rbf_resource_vector_v1',
        costs=totals,
        databases={s: database_inventory(output / b['path'], b['sha256'])
                   for s, b in databases.items()},
        latency=dict(
            events=len(seconds),
            p50_seconds=float(np.quantile(seconds, .5)),
            p95_seconds=float(np.quantile(seconds, .95)),
            maximum_seconds=float(seconds.max()),
            scope='complete_stream_step_including_scoring_replay_and_sql_commit',
            warmup_removed=False),
        equal_resources_verified=False,
        measured_gpu_forward_time=False,
        heterogeneous_counters_are_not_additive=True,
        note='Pair tokens are allocated solver/model shape counts, not FLOPs; disk bytes are not resident memory.')
