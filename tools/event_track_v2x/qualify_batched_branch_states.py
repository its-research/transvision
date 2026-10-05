"""Paired real-factor forest replay, profiling and bounded state candidate check.

Uses immutable original per-event factors and observations from an accepted
database, not final potentials or reconstructed/nonempty pseudo schedules.
This validates an execution candidate, not detector/NN/metric acceptance.
"""
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
import shutil
import sqlite3
import sys
import time
import types


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--events', type=int, default=12)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--no-profile', action='store_true', help='separate throughput measurement after hotspot profiling')
    parser.add_argument('--recorded-reference-acceptance', type=Path,
                        help='reuse an independently accepted complete sequence instead of rerunning serial work')
    parser.add_argument('--reference-acceptance-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.events < 1 or args.output.exists():
        raise ValueError('positive event count and new output required')
    receipt_path = args.replay/'receipt.json'
    if sha(receipt_path) != args.receipt_sha256:
        raise ValueError('input receipt hash differs')
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get('status') != 'software_replay_completed':
        raise ValueError('completed input replay required')
    dbinfo = receipt['databases'][args.sequence]
    path = (args.replay/dbinfo['path']).resolve()
    if path.parent != args.replay.resolve() or sha(path) != dbinfo['sha256']:
        raise ValueError('input database hash/path differs')
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    reference_binding = None
    if args.recorded_reference_acceptance is not None:
        proof_path = args.recorded_reference_acceptance
        if sha(proof_path) != args.reference_acceptance_sha256:
            raise ValueError('reference acceptance changed')
        proof = json.loads(proof_path.read_bytes())
        if (proof.get('kind') != 'rbf_serial_batched_single_complete_sequence_independent_scope_v1'
                or proof.get('sequence') != args.sequence or proof.get('events') != args.events
                or proof.get('all_discrete_outputs_identical') is not True
                or proof.get('all_continuous_outputs_match') is not True
                or db.execute('SELECT count(*) FROM events').fetchone()[0] != args.events):
            raise ValueError('complete independently checked reference sequence required')
        for role, checksum in proof['reports'].items():
            if role not in ('serial', 'batched') or sha(proof_path.parent/(role+'-independent.json')) != checksum:
                raise ValueError('reference independent report changed')
        report = json.loads((proof_path.parent/'serial-independent.json').read_bytes())
        if (report['structure']['database_sha256'] != dbinfo['sha256']
                or report['fresh_state']['database_sha256'] != dbinfo['sha256']
                or report['fresh_state'].get('stored_states_and_chosen_outputs_checked') is not True
                or report['fresh_state']['events'] != args.events
                or report['fresh_state']['atol'] != 1e-8 or report['fresh_state']['rtol'] != 1e-8):
            raise ValueError('reference raw-state acceptance does not bind this database')
        reference_binding = dict(acceptance=str(proof_path), sha256=sha(proof_path),
                                 database=str(path), database_sha256=dbinfo['sha256'])
    elif args.reference_acceptance_sha256 is not None:
        raise ValueError('reference path required with hash')
    args.output.mkdir(parents=True)
    # Snapshot the tested implementation before import; subsequent workspace
    # edits cannot change an active candidate process.
    root = Path(__file__).resolve().parents[2]
    package = 'transvision/models/event_track_v2x'
    target = args.output/'source'/package
    target.mkdir(parents=True)
    for file in (root/package).glob('*.py'):
        shutil.copyfile(file, target/file.name)
    shutil.copyfile(__file__, args.output/Path(__file__).name)
    sources = {str(p.relative_to(args.output/'source')): sha(p) for p in target.glob('*.py')}
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        mod = types.ModuleType(name)
        mod.__path__ = [str(args.output/'source'/name.replace('.', '/'))]
        sys.modules[name] = mod
    import numpy as np
    from transvision.models.event_track_v2x.persistent_forest import _raw
    from transvision.models.event_track_v2x.persistent_forest import PersistentForestCommit
    from transvision.models.event_track_v2x.forest_tracking import load_tracking_config
    from transvision.models.event_track_v2x.exclusive_completion_tracking import ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig
    from transvision.models.event_track_v2x.batched_branch_states import BatchedStateExclusiveTracker
    from transvision.models.event_track_v2x.detection_cache_v2 import canonical
    cfg = json.loads(db.execute("SELECT v FROM meta WHERE k='config'").fetchone()[0])
    config = PersistentExclusiveCompletionConfig(state=load_tracking_config(cfg.pop('state')), **cfg)
    events = []
    old_n = 0
    for prediction, audit in db.execute('SELECT prediction,audit FROM events ORDER BY ordinal LIMIT ?', (args.events,)):
        p, a = json.loads(prediction), json.loads(audit)
        n = a['observation_count']
        raw = tuple(_raw(r[0]) for r in db.execute('SELECT raw FROM observations WHERE i>=? AND i<? ORDER BY i', (old_n, n)))
        if len(raw) != a['new_observations'] or len(raw) != len(a['appended_rows']):
            raise ValueError('original event coverage differs')
        events.append((raw, a, p))
        old_n = n
    if len(events) != args.events:
        raise ValueError('requested actual event prefix unavailable')
    device_info = dict(device=args.device, GPU_executed=False)
    if args.device != 'cpu':
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        props = torch.cuda.get_device_properties(args.device)
        device_info.update(GPU_executed=True, name=props.name, uuid=str(getattr(props, 'uuid', 'unknown')),
                           torch=torch.__version__, cuda=torch.version.cuda)
    def write(name, value):
        with (args.output/name).open('xb') as f:
            f.write(canonical(value)+b'\n')
    write('input-binding.json', dict(source_database_sha256=dbinfo['sha256'],
          receipt_sha256=args.receipt_sha256, source_files=sources,
          configuration=asdict(config), events=args.events, sequence=args.sequence,
          device=device_info, max_batch=args.max_batch, profiling=not args.no_profile,
          recorded_reference=reference_binding,
          command=sys.argv, qualifier_sha256=sha(__file__)))
    results, metrics = {}, {}
    runs = [('serial', ExclusiveCompletionTracker), ('candidate', BatchedStateExclusiveTracker)]
    if reference_binding is not None:
        results['serial'] = [PersistentForestCommit(canonical(p), canonical(a)) for _, a, p in events]
        runs = runs[1:]
    for role, cls in runs:
        kw = dict(device=args.device, max_batch=args.max_batch) if role == 'candidate' else {}
        tracker = cls(args.output/(role+'.sqlite'), sequence_id=args.sequence, config=config, **kw)
        profiler = cProfile.Profile()
        started = time.monotonic()
        collected = []
        try:
            for i, (raw, a, p) in enumerate(events):
                if not args.no_profile:
                    profiler.enable()
                result = tracker.step(raw, a['appended_rows'], frame_id=p['frame_id'], event_id=a['event_id'],
                    reference_us=p['box_reference_timestamp_us'], decision_us=p['decision_timestamp_us'],
                    rescored_rows=a['rescored_rows'], decision_indices=a['decision_indices'],
                    cache_ingestion=a['cache_ingestion'], scorer_binding=a['scorer_binding'])
                if not args.no_profile:
                    profiler.disable()
                collected.append(result)
                elapsed = time.monotonic()-started
                print(json.dumps(dict(role=role, completed_events=i+1, total_events=len(events),
                    elapsed_seconds=elapsed, ETA='unknown: heterogeneous branch search')), flush=True)
            elapsed = time.monotonic()-started
            if not args.no_profile:
                profiler.dump_stats(str(args.output/(role+'.prof')))
                with (args.output/(role+'-profile.txt')).open('x') as f:
                    pstats.Stats(profiler, stream=f).sort_stats('cumulative').print_stats(35)
            metrics[role] = dict(elapsed_seconds=elapsed, events_per_second=len(events)/elapsed)
            if role == 'candidate':
                metrics[role].update(batch_calls=tracker.state_batch.calls, batch_rows=tracker.state_batch.rows,
                    maximum_actual_batch=tracker.state_batch.maximum_batch)
            results[role] = collected
        finally:
            tracker.close()
    max_error = 0.
    for original, candidate in zip(results['serial'], results['candidate'], strict=True):
        a, b = original.audit, candidate.audit
        for key in ('state_updates', 'search_steps', 'total_prefix_nodes', 'total_frontier',
                    'model_regret_upper', 'log_partition_upper', 'factor_rows_sha256', 'allocation_trace'):
            if a[key] != b[key]:
                raise ValueError('search/work/factor divergence: '+key)
        for x, y in zip(a['components'], b['components'], strict=True):
            for key in ('component', 'output_handle', 'active', 'frontier'):
                if x[key] != y[key]:
                    raise ValueError('component search diverged: '+key)
        for x, y in zip(original.prediction['predictions'], candidate.prediction['predictions'], strict=True):
            for key in ('track_id', 'class_label', 'score'):
                if x[key] != y[key]:
                    raise ValueError('discrete prediction differs: '+key)
            for key in ('mean', 'covariance'):
                v, w = np.asarray(x[key]), np.asarray(y[key])
                if not np.allclose(v, w, atol=1e-8, rtol=1e-8):
                    raise ValueError('fixed float64 state tolerance failed')
                max_error = max(max_error, float(np.max(np.abs(v-w))))
    # Check every materialized branch, including branches not selected for
    # output. Output-only parity could miss sibling-state contamination.
    state_count = 0
    reference_database = path if reference_binding is not None else args.output/'serial.sqlite'
    left = sqlite3.connect(reference_database.resolve().as_uri()+'?mode=ro', uri=True)
    right = sqlite3.connect((args.output/'candidate.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        import re
        names = [r[0] for r in left.execute("SELECT name FROM sqlite_master WHERE type='table'")
                 if re.fullmatch(r'pc[0-9]+_states', r[0])]
        right_names = [r[0] for r in right.execute("SELECT name FROM sqlite_master WHERE type='table'")
                       if re.fullmatch(r'pc[0-9]+_states', r[0])]
        if set(names) != set(right_names):
            raise ValueError('materialized component state tables differ')
        for table in names:
            for (h, v), (g, w) in zip(left.execute('SELECT h,payload FROM '+table+' ORDER BY h'),
                                    right.execute('SELECT h,payload FROM '+table+' ORDER BY h'), strict=True):
                if h != g:
                    raise ValueError('materialized state handle differs')
                v, w = json.loads(v), json.loads(w)
                for key in ('first_us', 'last_us', 'last_order', 'max_score'):
                    if v[key] != w[key]:
                        raise ValueError('branch provenance differs: '+key)
                for key in ('mean', 'covariance'):
                    a, b = np.asarray(v[key]), np.asarray(w[key])
                    if not np.allclose(a, b, atol=1e-8, rtol=1e-8):
                        raise ValueError('nonselected branch state mismatch')
                    max_error = max(max_error, float(np.max(np.abs(a-b))))
                state_count += 1
    finally:
        left.close()
        right.close()
    if sha(path) != dbinfo['sha256']:
        raise ValueError('original database changed during read')
    write('candidate-check.json', dict(kind='real_factor_prefix_batched_state_candidate_v1',
        input_binding_sha256=sha(args.output/'input-binding.json'), events=len(events),
        numerical_atol=1e-8, numerical_rtol=1e-8, maximum_state_error=max_error,
        materialized_branch_states_checked=state_count,
        discrete_actions_factors_work_identical=True, metrics=metrics,
        observed_wall_time_ratio=(metrics['serial']['elapsed_seconds']/metrics['candidate']['elapsed_seconds']
                                  if reference_binding is None else None),
        profiling_enabled=not args.no_profile,
        measurement_order=[role for role, _ in runs], isolated_performance_acceptance=False,
        recorded_reference=reference_binding,
        complete_reference_sequence_checked=reference_binding is not None,
        full_cohort_acceptance=False, production_promotion_allowed=False,
        device=device_info, database_sha256={r:sha(args.output/(r+'.sqlite')) for r in metrics}))
    print(json.dumps(dict(receipt=str(args.output/'candidate-check.json'), metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
