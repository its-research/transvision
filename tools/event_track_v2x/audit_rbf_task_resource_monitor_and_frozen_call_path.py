"""Audit reported telemetry and the source-bound call path without running replay."""
import argparse
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-receipt', required=True, type=Path)
    args = parser.parse_args()
    monitor_path = args.monitor_receipt.resolve()
    assert monitor_path.is_relative_to(R / 'receipts')
    monitor = json.loads(monitor_path.read_bytes())
    assert monitor['kind'] == 'rbf_allowlisted_Task_resource_monitor_cpu_io_scoped_observation_v1'
    assert sha(monitor['binding_receipt']) == monitor['binding_receipt_sha256']
    assert sha(monitor['source_freeze']) == monitor['source_freeze_sha256']
    binding = json.loads(Path(monitor['binding_receipt']).read_bytes())
    ledger = json.loads((R / 'receipts/20260928-execution-ledger.json').read_bytes())
    assert any(e.get('receipt') == str(monitor_path) and e.get('receipt_sha256') == sha(monitor_path)
               for e in ledger['entries'])
    jobs = []
    for entry in binding['dispatch_receipts']:
        assert sha(entry['path']) == entry['sha256']
        jobs += json.loads(Path(entry['path']).read_bytes())['jobs']
    summaries = []
    for row in monitor['tasks']:
        job = next(j for j in jobs if j['task_id'] == row['task_id'])
        assert row['source_sha256'] == job['plan']['bootstrap_sha256']
        assert row['world_size'] == job['plan']['world_size']
        metrics = {}
        for series in row['monitor_series']:
            dates, values = series['dates_epoch_milliseconds'], series['recent_values']
            assert len(dates) == len(values) == series['sample_count'] > 0
            assert dates == sorted(dates)
            assert all(row['observation_window_from_epoch_seconds'] <= d/1000 <=
                       row['observation_window_to_epoch_seconds'] for d in dates)
            assert statistics.mean(values) == series['mean']
            assert min(values) == series['minimum'] and max(values) == series['maximum']
            metrics[series['series']] = series['mean']
        gpu_means = {k: v for k, v in metrics.items() if k.endswith('_utilization')}
        assert len(gpu_means) == row['world_size']
        summaries.append(dict(task_id=row['task_id'], seed=row['seed'], method=row['method'],
            baseline_K=row['baseline_K'], task_status_at_observation=row['task_status'],
            worker=row['last_worker'], visible_logical_GPU_mean_series=gpu_means,
            host_global_CPU_mean=metrics['cpu_usage'], host_global_read_rate_mean=metrics['io_read_mbs'],
            host_global_write_rate_mean=metrics['io_write_mbs'],
            CPU_and_IO_not_individual_forest_process_attribution=True,
            reported_units_remote_source_not_independently_verified=True))

    original = R / 'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    archive_sha = sha(original)
    assert archive_sha == '038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
    assert all(j['plan']['source']['sha256'] == archive_sha for j in jobs)
    main_source = R / 'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004/bootstrap.py'
    assert all(sha(main_source) == j['plan']['bootstrap_sha256'] for j in jobs if j['plan']['method'] == 'rbf')
    patches = {}
    for statement in ast.parse(main_source.read_text()).body:
        if isinstance(statement, ast.Assign) and any(isinstance(n, ast.Name) and n.id in
                ('PATCHES', 'REPLACEMENTS') for n in statement.targets):
            patches.update(ast.literal_eval(statement.value))
    names = ('exclusive_paper_runtime.py', 'persistent_cache_stream.py', 'forest_potentials.py',
             'recoverable_identity.py', 'persistent_component_tracking.py')
    snapshots = {}
    with tarfile.open(original) as archive:
        for name in names:
            member = 'transvision/models/event_track_v2x/' + name
            if member in patches:
                spec = patches[member]
                raw = zlib.decompress(base64.b64decode(spec['data']))
                assert hashlib.sha256(raw).hexdigest() == spec['sha256']
                if 'before_sha256' in spec:
                    assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == spec['before_sha256']
            else:
                raw = archive.extractfile(member).read()
            snapshots[name] = raw
    evidence = [
        ('exclusive_paper_runtime.py', 'for index, event in enumerate(events):', 'Each rank processes its assigned sequence events serially.'),
        ('exclusive_paper_runtime.py', "streams[sequence].step(deliveries", 'Replay event timing includes cache/scoring/tracking in one step.'),
        ('persistent_cache_stream.py', 'for context in contexts:', 'The scorer runs separately for each legal causal row context.'),
        ('persistent_cache_stream.py', 'factors = self.scorer(', 'A row invokes the original learned scorer, not a batch of queries.'),
        ('persistent_cache_stream.py', 'commit = t.step(', 'After neural rows, the same event waits for persistent tracking.'),
        ('forest_potentials.py', 'model_digest(self.model) != self.model_sha256', 'The frozen-model guard hashes the model on every scorer invocation.'),
        ('recoverable_identity.py', 'value = tensor.detach().cpu().contiguous()', 'The hash guard copies each model tensor to CPU before hashing.'),
        ('forest_potentials.py', "F.log_softmax(row, dim=0).to(device='cpu', dtype=torch.float64).tolist()", 'Normalized neural factors are transferred to CPU for forest processing.'),
        ('persistent_component_tracking.py', 'while remaining:', 'Component allocation/search executes a serial Python loop.'),
        ('persistent_component_tracking.py', "self.db.execute('COMMIT')", 'The persistent event transaction commits after inference.'),
    ]
    call_path = []
    for filename, needle, interpretation in evidence:
        lines = snapshots[filename].decode().splitlines()
        matches = [(i+1, line.strip()) for i, line in enumerate(lines) if needle in line]
        if filename == 'persistent_component_tracking.py' and needle == "self.db.execute('COMMIT')":
            functions = [n for n in ast.walk(ast.parse(snapshots[filename]))
                         if isinstance(n, ast.FunctionDef) and n.name == 'step']
            assert len(functions) == 1
            matches = [(i, line) for i, line in matches
                       if functions[0].lineno <= i <= functions[0].end_lineno]
        assert len(matches) == 1, (filename, needle, matches)
        line_number, text = matches[0]
        call_path.append(dict(source_file=filename, source_sha256=hashlib.sha256(snapshots[filename]).hexdigest(),
            line_number=line_number, exact_line=text, interpretation=interpretation))

    root = R / 'source-freezes/rbf-Task-monitor-and-frozen-call-path-audit-v1-20261004'
    root.mkdir(exist_ok=True)
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name('rbf_nested_seen_val_v2_common.py')):
        target = root / path.name
        if target.exists():
            assert sha(target) == sha(path)
        else:
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
    source_root = root / 'effective-main-source-snapshots'
    source_root.mkdir(exist_ok=True)
    for filename, raw in snapshots.items():
        target = source_root / filename
        if target.exists():
            assert target.read_bytes() == raw
        else:
            with target.open('xb') as stream:
                stream.write(raw)
    freeze = root / 'source-freeze.json'
    if not freeze.exists():
        new(freeze, dict(kind='rbf_Task_resource_monitor_and_frozen_call_path_audit_source_v1',
            original_source_archive=str(original), original_source_archive_sha256=archive_sha,
            current_main_bootstrap=str(main_source), current_main_bootstrap_sha256=sha(main_source),
            sources={str(p.relative_to(root)): dict(sha256=sha(p), bytes=p.stat().st_size)
                     for p in root.rglob('*.py')}, experiment_code_modified=False))
        register(freeze, 'rbf-Task-monitor-and-frozen-call-path-audit-source')
    now = datetime.datetime.now(datetime.timezone.utc)
    output = R / 'receipts' / ('rbf-Task-CPU-IO-and-frozen-call-path-scoped-audit-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(output, dict(kind='rbf_Task_CPU_IO_and_source_bound_frozen_call_path_scoped_audit_v1',
        checked_at_utc=now.isoformat(), monitor_receipt=str(monitor_path), monitor_receipt_sha256=sha(monitor_path),
        source_freeze=str(freeze), source_freeze_sha256=sha(freeze), summaries=summaries,
        reported_timestamp_sample_mean_range_audit_passed=True,
        host_CPU_measurements_must_not_be_summed_for_two_A100_tasks_on_same_host=True,
        frozen_current_main_call_path=call_path,
        inference='Low logical GPU load is consistent with repeated per-row model hashing/copying plus serial CPU forest work; the cause and time share require a process-level or separately qualified profiler.',
        causal_per_process_profile_obtained=False, independent_physical_GPU_utilization_admission=False,
        performance_or_Stage2_result=False, running_source_or_checkpoints_modified=False,
        tasks_restarted=False, new_GPU_experiment_dispatched=False, whole_experiment_ETA='unknown',
        next_action='Obtain a configured read-only remote process entry if available; prepare a distinct measured optimization contract that preserves causal row support, frozen model identity, forest transactions and fixed numeric tolerances before any changed experiment.',
        optional_remote_entry_question_pending=True, goal_status='active'))
    register(output, 'rbf-Task-monitor-and-frozen-call-path-scoped-audit')
    print(json.dumps(dict(receipt=str(output), sha256=sha(output), audited_tasks=len(summaries),
                         source_call_path_evidence=len(call_path), causal_profile=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
