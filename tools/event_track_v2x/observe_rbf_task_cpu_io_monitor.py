"""Bounded read-only Task monitor telemetry, with explicit host/process scope."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-binding-receipt', type=Path, required=True)
    args = parser.parse_args()
    binding_path = args.live_binding_receipt.resolve()
    assert binding_path.is_relative_to(R / 'receipts')
    binding = json.loads(binding_path.read_bytes())
    assert binding['kind'] == 'rbf_final_refit_main_K4_Top1_one_bounded_live_v1'
    jobs = [job for item in binding['dispatch_receipts'] for job in json.loads(Path(item['path']).read_bytes())['jobs']]
    for item in binding['dispatch_receipts']:
        assert sha(item['path']) == item['sha256'], 'dispatch journal changed: obtain a new bounded observation'
    from clearml import Task
    directory = R / 'source-freezes/rbf-allowlisted-Task-CPU-IO-monitor-v1-20261004'
    directory.mkdir(exist_ok=True)
    for original in (Path(__file__).resolve(), Path(__file__).resolve().with_name('rbf_nested_seen_val_v2_common.py')):
        destination = directory / original.name
        if destination.exists():
            assert sha(destination) == sha(original)
        else:
            with destination.open('xb') as stream:
                stream.write(original.read_bytes())
    freeze = directory / 'source-freeze.json'
    local_monitor = Path(__import__('clearml.utilities.resource_monitor', fromlist=['x']).__file__)
    if not freeze.exists():
        new(freeze, dict(kind='rbf_allowlisted_Task_CPU_IO_monitor_read_only_source_v1',
            sources={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in directory.glob('*.py')},
            local_clearml_monitor_reference=str(local_monitor), local_clearml_monitor_reference_sha256=sha(local_monitor),
            remote_monitor_implementation_byte_identity_verified=False,
            experiment_started=False, raw_Agent_configuration_read=False))
        register(freeze, 'rbf-Task-CPU-IO-monitor-source')
    rows = []
    for observation in binding['tasks']:
        job = next(job for job in jobs if job['task_id'] == observation['task_id'])
        task = Task.get_task(task_id=job['task_id'])
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['bootstrap_sha256']
        data = task.get_reported_scalars(max_samples=480, x_axis='timestamp')
        now = datetime.datetime.now(datetime.timezone.utc)
        earliest = now.timestamp()-600
        series = []
        for title, variants in data.items():
            if title not in (':monitor:machine', ':monitor:gpu'):
                continue
            for name, metric in variants.items():
                if title==':monitor:machine':
                    if name not in ('cpu_usage','memory_used_gb','memory_free_gb','io_read_mbs','io_write_mbs','network_rx_mbs','network_tx_mbs'):
                        continue
                elif re.fullmatch(r'gpu_[0-7]_(utilization|mem_used_gb|mem_free_gb)', name) is None:
                    continue
                x, y = metric['x'], metric['y']
                assert len(x)==len(y)
                assert all(type(v) in (int,float) and math.isfinite(v) for v in x+y)
                assert x==sorted(x)
                recent = [(timestamp/1000, value) for timestamp,value in zip(x,y)
                          if earliest<=timestamp/1000<=now.timestamp()]
                values = [value for _,value in recent]
                series.append(dict(title=title,series=name,returned_sample_count=len(x),
                    dates_epoch_milliseconds=[timestamp for timestamp,value in zip(x,y) if earliest<=timestamp/1000<=now.timestamp()],
                    recent_values=values,sample_count=len(values),mean=statistics.mean(values) if values else None,
                    minimum=min(values) if values else None,maximum=max(values) if values else None,
                    newest_observed_timestamp=max(x)/1000 if x else None,
                    histogram_may_downsample_and_average=True))
        runtime = task.data.runtime or {}
        spec = {key:runtime[key] for key in ('platform','python_version','OS','processor','cpu_cores','memory_gb','gpu_count','gpu_type') if key in runtime}
        rows.append(dict(task_id=task.id,seed=job['seed'],method=job['plan']['method'],
            baseline_K=job['plan']['configuration']['state']['active_limit'] if job['plan']['method']=='topk' else None,
            task_status=str(task.status),last_worker=task.data.last_worker,
            source_sha256=job['plan']['bootstrap_sha256'],world_size=job['plan']['world_size'],
            observation_window_from_epoch_seconds=earliest,observation_window_to_epoch_seconds=now.timestamp(),
            runtime_machine_spec=spec,monitor_series=series,
            experiment_artifacts={key:dict(bytes=value.size,sha256=value.hash) for key,value in task.artifacts.items()},
            process_level_CPU_IO_profile_obtained=False,whole_experiment_ETA='unknown'))
    now = datetime.datetime.now(datetime.timezone.utc)
    output = R / 'receipts' / ('rbf-allowlisted-Task-CPU-IO-monitor-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(output, dict(kind='rbf_allowlisted_Task_resource_monitor_cpu_io_scoped_observation_v1',
        checked_at_utc=now.isoformat(),command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--live-binding-receipt',str(binding_path)],
        source_freeze=str(freeze),source_freeze_sha256=sha(freeze),binding_receipt=str(binding_path),binding_receipt_sha256=sha(binding_path),
        max_histogram_samples_per_series=480,x_axis='timestamp: epoch milliseconds',window_seconds=600,tasks=rows,
        scope='Task-reported ResourceMonitor series, not an independent OS process profile; local SDK uses global psutil CPU/disk/network counters, process-tree RSS when available, and visible logical CUDA device telemetry',
        io_rate_semantics_reference='local ResourceMonitor differentiates cumulative _mbs counters before reporting; do not differentiate already reported series again',
        local_monitor_implementation_sha256=sha(local_monitor),remote_monitor_implementation_byte_identity_verified=False,
        physical_GPU_UUID_identity_independently_verified=False,worker_authentication_configuration_exists=(Path.home()/'.ssh/config').exists(),
        remote_process_credentials_or_identity_guessed=False,raw_Agent_configuration_read=False,credentials_written_or_printed=False,
        process_level_CPU_IO_profile_obtained=False,experiment_restarted=False,existing_checkpoint_or_numerical_limits_changed=False,
        paper_cost_or_Stage2_acceptance=False,goal_status='active'))
    register(output, 'rbf-Task-resource-monitor-CPU-IO-scoped-observation')
    print(json.dumps(dict(receipt=str(output),sha256=sha(output),
        tasks=[{key:value for key,value in row.items() if key not in ('monitor_series','experiment_artifacts')} |
               dict(monitor_summaries=[{key:value for key,value in metric.items() if key not in ('dates_epoch_milliseconds','recent_values')}
                                       for metric in row['monitor_series']]) for row in rows]),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
