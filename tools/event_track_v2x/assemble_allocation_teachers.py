#!/usr/bin/env python3
"""Assemble COMPLETE independent train-sequence teachers, never partial prefixes.

Leaves retain their own immutable outputs and process measurements. The joined
directory is a derived artifact, NOT one execution or a campaign resource peak.
All 46 official train sequences are required by the CLI. No training/upload.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from transvision.models.event_track_v2x.allocation_training import _directory, allocation_sources
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json

PAIR_SHA256 = '787b7e4dfa3fb9d97ba0a6a4a216a2990af990dc9226ceec01053e7d2e7da984'
KIND = 'component_allocation_teacher_assembled_train_plan_v1'
STREAMS = dict(predictions_sha256='predictions.jsonl', tracking_sha256='tracking.jsonl',
               frame_timings_sha256='frame-timings.jsonl')
COMMON = ('configuration', 'source_sha256', 'allocation_training_binding', 'identity_checkpoint_sha256',
    'ego_pose_table_sha256', 'class_scope', 'cache_sha256', 'cooperative_metadata_sha256',
    'cache_split', 'allocation_teacher', 'allocation_policy_signature')
VARIABLE = frozenset(('scheduled_frames', 'completed_frames', 'sequence_heads', 'plan_sha256',
    *STREAMS, 'source_unavailable', 'elapsed_seconds', 'latency_seconds_p50_p95_p99_max',
    'frame_latency_seconds_p50_p95_p99_max', 'process_peak_rss_bytes', 'database_bytes'))


def _pairs(path, allow_fixture):
    if type(allow_fixture) is not bool:
        raise TypeError('explicit fixture mode required')
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('ordinary cooperative metadata required')
    checksum = sha_file(path)
    if not allow_fixture and checksum != PAIR_SHA256:
        raise ValueError('official full train cooperative metadata required')
    pairs = json.loads(path.read_bytes())
    expected = {}
    infrastructure = set()
    for pair in pairs:
        scene, frame = pair['vehicle_sequence'], pair['vehicle_frame']
        if (pair['infrastructure_sequence'] != scene or frame in expected.setdefault(scene, set())
                or pair['infrastructure_frame'] in infrastructure):
            raise ValueError('distinct within-sequence cooperative frames required')
        expected[scene].add(frame); infrastructure.add(pair['infrastructure_frame'])
    if (len(expected) < 2 or not allow_fixture and (len(expected) != 46 or len(pairs) != 7445)
            or sha_file(path) != checksum):
        raise ValueError('complete distinct train sequence cohort required')
    return expected, checksum


def _common(plan):
    return {key: plan.get(key) for key in COMMON}


def _sources(plan):
    expected = allocation_sources()
    for name in ('collect_allocation_training.py', 'run_persistent_forest_v2.py',
                 'prepare_forest_training.py', 'train_forest_identity.py'):
        path = 'tools/event_track_v2x/'+name
        expected[path] = sha_file(ROOT/path)
    if any(plan['source_sha256'].get(p) != h for p, h in expected.items()):
        raise ValueError('current teacher solver and producer sources required')
    if any(sha_file(contained_file(ROOT, p)) != h for p, h in plan['source_sha256'].items()):
        raise ValueError('teacher source binding changed')


def _leaf(path, digest, expected, pair_sha, allow_fixture):
    path = Path(path).absolute()
    report = inspect(path, digest, teacher=True, allow_fixture=allow_fixture)
    receipt = json.loads((path/'receipt.json').read_bytes())
    plan = report['plan']; scene = plan.get('development_sequence')
    final_path = contained_file(path, 'development-teacher-receipt.json')
    final = json.loads(final_path.read_bytes())
    if (plan.get('kind') != 'component_allocation_teacher_full_train_plan_v1'
            or plan.get('full_official_train_verified') is not False
            or plan.get('input_full_official_train_verified') is not (not allow_fixture)
            or plan.get('geometry_development') is not False or scene not in expected
            or set(receipt['sequence_heads']) != {scene}
            or final != dict(receipt, full_official_train_trace_completed=False)
            or plan.get('cooperative_metadata_sha256') != pair_sha
            or not allow_fixture and plan.get('cache_sha256') != TRAIN_CACHE_SHA256
            or plan['allocation_training_binding']['configuration'] != plan['configuration']):
        raise ValueError('complete single-sequence teacher provenance required')
    keys = [(e['sequence_id'], e['frame_id']) for e in report['events']]
    if (len(keys) != len(expected[scene]) or len(set(keys)) != len(keys)
            or set(keys) != {(scene, f) for f in expected[scene]}
            or plan.get('scheduled_frames') != len(keys)):
        raise ValueError('teacher leaf differs from complete sequence schedule')
    if receipt['sequence_heads'][scene]['frames'] != len(keys):
        raise ValueError('sequence head frame count differs')
    frame_seconds=[]
    with contained_file(path,'frame-timings.jsonl').open('rb') as stream:
        for line in stream:
            row=json.loads(line); step_seconds,total=row['step_seconds'],row['frame_seconds']
            if (type(total) not in (int,float) or not math.isfinite(total)
                    or not 0 <= step_seconds <= total):
                raise ValueError('invalid teacher frame timing')
            frame_seconds.append(total)
    if receipt['frame_latency_seconds_p50_p95_p99_max'] != np.quantile(frame_seconds,[.5,.95,.99,1.]).tolist():
        raise ValueError('teacher frame latency summary differs')
    _sources(plan)
    artifacts = {path/'receipt.json': digest, path/'plan.json': receipt['plan_sha256'],
                 final_path: sha_file(final_path)}
    artifacts.update({contained_file(path, name): receipt[key] for key, name in STREAMS.items()})
    head = receipt['sequence_heads'][scene]
    artifacts[contained_file(path, head['database'])] = head['database_sha256']
    return dict(path=path, scene=scene, report=report, receipt=receipt, plan=plan, artifacts=artifacts,
        input=dict(sequence_id=scene, path=str(path), receipt_sha256=digest,
                   final_sha256=artifacts[final_path]))


def _leaves(replays, expected, pair_sha, allow_fixture):
    if len(replays) != len(expected):
        raise ValueError('exactly one complete replay per train sequence required')
    leaves = [_leaf(path, digest, expected, pair_sha, allow_fixture) for path, digest in replays]
    if {r['scene'] for r in leaves} != set(expected):
        raise ValueError('duplicate or omitted teacher sequence')
    leaves.sort(key=lambda r: r['scene'])
    first = leaves[0]
    modes = {k: v for k, v in first['receipt'].items() if k not in VARIABLE}
    if any(_common(r['plan']) != _common(first['plan']) or
            {k: v for k, v in r['receipt'].items() if k not in VARIABLE} != modes for r in leaves[1:]):
        raise ValueError('teacher model/configuration/sources/execution modes differ')
    return leaves


def _unchanged(leaves):
    for leaf in leaves:
        if any(sha_file(p) != h for p, h in leaf['artifacts'].items()):
            raise ValueError('teacher leaf changed during assembly')
    _sources(leaves[0]['plan'])


def _joined_digest(leaves, name):
    h = hashlib.sha256()
    for leaf in leaves:
        with contained_file(leaf['path'], name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
    return h.hexdigest()


def verify_assembly(directory, plan, receipt, cooperative_metadata, *, allow_fixture=False):
    """Reopen every leaf; flags alone cannot certify the derived full cohort."""
    directory = Path(directory)
    expected, pair_sha = _pairs(cooperative_metadata, allow_fixture)
    if (plan.get('kind') != KIND or plan.get('full_official_train_verified') is not (not allow_fixture)
            or plan.get('input_full_official_train_verified') is not (not allow_fixture)
            or plan.get('development_sequence') is not None
            or plan.get('assembly_tool_sha256') != sha_file(__file__)
            or receipt.get('execution_topology') != 'independent_complete_sequence_teachers'):
        raise ValueError('explicit current-source teacher assembly required')
    leaves = _leaves([(r['path'], r['receipt_sha256']) for r in plan['assembly_inputs']],
                      expected, pair_sha, allow_fixture)
    if any(receipt.get(k) != v for k,v in leaves[0]['receipt'].items() if k not in VARIABLE):
        raise ValueError('assembled execution modes differ from sequence teachers')
    expected_unavailable = Counter()
    timings, frame_timings = [], []
    for leaf in leaves:
        expected_unavailable.update(leaf['receipt']['source_unavailable'])
        with contained_file(leaf['path'],'frame-timings.jsonl').open('rb') as stream:
            for line in stream:
                row=json.loads(line);timings.append(row['step_seconds']);frame_timings.append(row['frame_seconds'])
    if (receipt.get('source_unavailable') != dict(expected_unavailable)
            or receipt.get('elapsed_seconds') != sum(r['receipt']['elapsed_seconds'] for r in leaves)
            or receipt.get('elapsed_scope') != 'sum_of_sequence_worker_elapsed_not_campaign_wall'
            or receipt.get('process_peak_rss_bytes') != max(r['receipt']['process_peak_rss_bytes'] for r in leaves)
            or receipt.get('rss_scope') != 'maximum_sequence_worker_high_water_not_concurrent_campaign_peak'
            or receipt.get('latency_seconds_p50_p95_p99_max') != np.quantile(timings,[.5,.95,.99,1.]).tolist()
            or receipt.get('frame_latency_seconds_p50_p95_p99_max') != np.quantile(frame_timings,[.5,.95,.99,1.]).tolist()
            or receipt.get('source_sequence_run_count') != len(leaves)
            or receipt.get('teacher_latency_is_not_deployment_latency') is not True):
        raise ValueError('assembled resource scope or derived measurements differ')
    if (plan['assembly_inputs'] != [r['input'] for r in leaves]
            or _common(plan) != _common(leaves[0]['plan'])
            or receipt['scheduled_frames'] != sum(map(len, expected.values()))
            or receipt['completed_frames'] != receipt['scheduled_frames']
            or plan['scheduled_frames'] != receipt['scheduled_frames']
            or receipt['plan_sha256'] != sha_file(directory/'plan.json')):
        raise ValueError('assembled source membership or cohort differs')
    for key, name in STREAMS.items():
        if receipt[key] != _joined_digest(leaves, name) or sha_file(contained_file(directory,name)) != receipt[key]:
            raise ValueError('assembled stream differs from immutable sequence teachers')
    heads = {}
    for i, leaf in enumerate(leaves):
        head = dict(leaf['receipt']['sequence_heads'][leaf['scene']], database=f'sequence-{i:04d}.sqlite')
        if sha_file(contained_file(directory,head['database'])) != head['database_sha256']:
            raise ValueError('assembled database differs from sequence teacher')
        heads[leaf['scene']] = head
    if (receipt['sequence_heads'] != heads or receipt.get('database_bytes') !=
            sum(contained_file(directory,h['database']).stat().st_size for h in heads.values())):
        raise ValueError('assembled sequence heads differ')
    _unchanged(leaves)
    return leaves


def assemble(replays, cooperative_metadata, output, *, allow_fixture=False):
    expected, pair_sha = _pairs(cooperative_metadata, allow_fixture)
    leaves = _leaves(replays, expected, pair_sha, allow_fixture)
    output = _directory(output)
    plan = dict(leaves[0]['plan'], kind=KIND, development_sequence=None,
        full_official_train_verified=not allow_fixture, input_full_official_train_verified=not allow_fixture,
        scheduled_frames=sum(map(len,expected.values())), assembly_inputs=[r['input'] for r in leaves],
        assembly_tool_sha256=sha_file(__file__), assembly_is_inference_or_training=False,
        paper_eligible=False)
    _new_json(output/'plan.json',plan)
    try:
        for name in STREAMS.values():
            with (output/name).open('xb') as target:
                for leaf in leaves:
                    with contained_file(leaf['path'],name).open('rb') as source:
                        shutil.copyfileobj(source,target,1024*1024)
        heads = {}; unavailable = Counter(); timings = []; frames = []
        for i, leaf in enumerate(leaves):
            r = leaf['receipt']; head = dict(r['sequence_heads'][leaf['scene']])
            with contained_file(leaf['path'],head['database']).open('rb') as source, \
                    (output/f'sequence-{i:04d}.sqlite').open('xb') as target:
                shutil.copyfileobj(source,target,1024*1024)
            head['database']=f'sequence-{i:04d}.sqlite'; heads[leaf['scene']]=head
            unavailable.update(r['source_unavailable'])
            with contained_file(leaf['path'],'frame-timings.jsonl').open('rb') as stream:
                for line in stream:
                    timing=json.loads(line); timings.append(timing['step_seconds']); frames.append(timing['frame_seconds'])
        result=dict(leaves[0]['receipt'], scheduled_frames=plan['scheduled_frames'],
            completed_frames=plan['scheduled_frames'], sequence_heads=heads,plan_sha256=sha_file(output/'plan.json'),
            source_unavailable=dict(unavailable), elapsed_seconds=sum(r['receipt']['elapsed_seconds'] for r in leaves),
            elapsed_scope='sum_of_sequence_worker_elapsed_not_campaign_wall',
            process_peak_rss_bytes=max(r['receipt']['process_peak_rss_bytes'] for r in leaves),
            rss_scope='maximum_sequence_worker_high_water_not_concurrent_campaign_peak',
            database_bytes=sum((output/h['database']).stat().st_size for h in heads.values()),
            latency_seconds_p50_p95_p99_max=np.quantile(timings,[.5,.95,.99,1.]).tolist(),
            frame_latency_seconds_p50_p95_p99_max=np.quantile(frames,[.5,.95,.99,1.]).tolist(),
            execution_topology='independent_complete_sequence_teachers',
            source_sequence_run_count=len(leaves), teacher_latency_is_not_deployment_latency=True,
            paper_eligible=False, **{key:sha_file(output/name) for key,name in STREAMS.items()})
        verify_assembly(output,plan,result,cooperative_metadata,allow_fixture=allow_fixture)
        _new_json(output/'receipt.json',result)
        name='development-teacher-receipt.json' if allow_fixture else 'full-train-teacher-receipt.json'
        _new_json(output/name,dict(result,full_official_train_trace_completed=not allow_fixture))
        return result
    except BaseException as error:
        _new_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            partial_outputs_not_final_results=True,original_sequence_teachers_unchanged=True))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='append',nargs=2,required=True,metavar=('DIRECTORY','RECEIPT_SHA256'))
    parser.add_argument('--cooperative-metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=assemble(args.replay,args.cooperative_metadata,args.output)
    print(json.dumps(dict(status=result['status'],completed_frames=result['completed_frames'],
        sequences=len(result['sequence_heads']),receipt_sha256=sha_file(args.output/'receipt.json'),
        training_submitted=False),sort_keys=True))
