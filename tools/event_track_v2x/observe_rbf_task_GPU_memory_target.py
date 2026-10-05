"""Bounded allowlisted Task GPU-memory percentage observation; no mutation.

Do not divide Task mem_used_gb by mem_used_gb+mem_free_gb: the SDK may
report process-tree used memory alongside device-global free memory.
"""
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


def series_samples(metric, earliest, now):
    x, y = metric['x'], metric['y']
    assert len(x) == len(y) and x == sorted(x) and len(x) == len(set(x))
    assert all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in x+y)
    assert all(1e12 <= date < 1e13 for date in x), 'expected timestamp epoch milliseconds'
    return [(date/1000,value) for date,value in zip(x,y) if earliest <= date/1000 <= now]


def summarize_percentage(samples, now):
    assert all(0 <= value <= 100 for _,value in samples), 'invalid memory percentage'
    if not samples:
        return dict(sample_count=0,status='unknown',reason='no recent reported device-memory percentage')
    age = now-max(date for date,_ in samples)
    values = [value for _,value in samples]
    latest = values[-1]
    status = 'unknown_stale' if age > 240 else 'below_75_percent' if latest < 75 else 'above_80_percent' if latest > 80 else 'within_75_to_80_percent'
    return dict(sample_count=len(values),mean_percent=statistics.mean(values),minimum_percent=min(values),
        maximum_percent=max(values),latest_percent=latest,latest_age_seconds=age,status=status)


def bound_logical_count(workers):
    counts = []
    for worker in workers:
        match = re.fullmatch(r'[^:]+:gpu([0-9]+(?:,[0-9]+)*)',worker['id'])
        assert match, 'unknown Worker GPU binding; do not infer a physical count'
        cards = list(map(int,match[1].split(',')))
        assert len(cards) == len(set(cards))
        counts.append(len(cards))
    assert len(counts) == 1, 'one Task on multiple Worker bindings requires independent mapping'
    return counts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-binding-receipt',type=Path,required=True)
    args = parser.parse_args()
    binding_path = args.live_binding_receipt.resolve()
    assert binding_path.is_relative_to(R/'receipts')
    binding = json.loads(binding_path.read_bytes())
    assert binding['kind'] == 'rbf_final_refit_main_K4_Top1_one_bounded_live_v1'
    assert binding['Agent_configuration_and_nonallowlisted_logs_excluded'] is True
    observed = datetime.datetime.fromisoformat(binding['checked_at_utc'])
    now = datetime.datetime.now(datetime.timezone.utc)
    assert 0 <= (now-observed).total_seconds() < 300
    known = {}
    for dispatch in binding['dispatch_receipts']:
        assert sha(dispatch['path']) == dispatch['sha256']
        for job in json.loads(Path(dispatch['path']).read_bytes())['jobs']:
            known[job['task_id']] = job
    workers = [w for w in binding['fleet']['workers'] if w.get('task_id') and 'L40' not in w['id']]
    tasks = sorted({w['task_id'] for w in workers})
    assert tasks
    own = Path(__file__).resolve().parent
    frozen = json.loads((own/'source-freeze.json').read_bytes())
    assert frozen['kind'] == 'rbf_allowlisted_Task_GPU_memory_target_observer_source_v2_binding_cardinality'
    for name, record in frozen['sources'].items():
        assert sha(own/name) == record['sha256']
    for record in frozen['references']:
        assert sha(record['path']) == record['sha256']
    from clearml import Task
    rows = []
    for task_id in tasks:
        task = Task.get_task(task_id=task_id)
        script_sha = hashlib.sha256(task.data.script.diff.encode()).hexdigest() if task.data.script.diff else None
        if task_id in known:
            assert script_sha == known[task_id]['plan']['bootstrap_sha256']
        data = task.get_reported_scalars(max_samples=480,x_axis='timestamp')
        now = datetime.datetime.now(datetime.timezone.utc)
        series = {}
        for name,metric in data.get(':monitor:gpu',{}).items():
            match = re.fullmatch(r'gpu_([0-7])_(mem_usage|mem_used_gb|mem_free_gb)',name)
            if match:
                samples = series_samples(metric,now.timestamp()-600,now.timestamp())
                series[name] = dict(samples_epoch_seconds=samples,returned_sample_count=len(metric['x']))
        task_workers = [w for w in workers if w['task_id']==task_id]
        bound_count = bound_logical_count(task_workers)
        reported_ids = sorted({int(name.split('_')[1]) for name in series})
        logical_ids = list(range(bound_count))
        if task_id in known:
            assert known[task_id]['plan']['world_size'] == bound_count
        summaries = []
        for gpu in logical_ids:
            samples = series.get(f'gpu_{gpu}_mem_usage',{}).get('samples_epoch_seconds',[])
            summaries.append(dict(logical_GPU=gpu,reported_device_memory_percentage=summarize_percentage(samples,now.timestamp()),
                used_and_free_GB_not_combined_into_a_false_device_ratio=True))
        rows.append(dict(task_id=task_id,status=str(task.status),reported_last_worker=task.data.last_worker,
            binding_snapshot_workers=[w['id'] for w in workers if w['task_id']==task_id],
            declared_current_binding_cardinality=bound_count,
            excluded_reported_logical_ids_outside_current_binding=[i for i in reported_ids if i not in logical_ids],
            older_or_out_of_binding_monitor_series_do_not_increase_physical_card_count=True,
            original_RBF_source_identity_verified=(task_id in known),observed_manual_script_sha256=script_sha,
            logical_GPUs=summaries,allowlisted_series=series,
            physical_GPU_UUID_identity_independently_verified=False,task_or_worker_configuration_not_read=True))
    checked = datetime.datetime.now(datetime.timezone.utc)
    output = R/'receipts'/('rbf-Task-reported-GPU-memory-target-'+checked.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(output,dict(kind='rbf_allowlisted_Task_reported_GPU_memory_75_to_80_target_observation_v2_binding_cardinality',
        checked_at_utc=checked.isoformat(),binding_receipt=str(binding_path),binding_receipt_sha256=sha(binding_path),
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--live-binding-receipt',str(binding_path)],
        source_freeze_sha256=sha(own/'source-freeze.json'),GPU_memory_target_percent=[75,80],
        physical_non_L40_GPUs_bound_at_snapshot=binding['fleet']['physical_non_L40_GPUs']-binding['fleet']['physical_unbound_GPUs'],
        tasks=rows,metric='Task-reported gpu_N_mem_usage; device used/total percentage in referenced SDK',
        local_SDK_monitor_reference=frozen['monitor_reference'],remote_monitor_source_bytes_independently_verified=False,
        histogram_may_downsample_and_average=True,task_reported_percentage_is_not_independent_NVML_readback=True,
        task_mem_used_GB_may_be_process_tree_while_free_GB_is_device_global=True,
        no_artificial_memory_padding_permitted=True,real_batching_or_independent_sequence_concurrency_only=True,
        existing_running_experiments_or_frozen_batch_sizes_changed=False,
        GPU_compute_utilization_not_equivalent_to_memory_occupancy=True,
        actual_task_launched_or_restarted=False,credentials_or_Agent_configuration_printed=False,
        full_experiment_ETA='unknown',paper_acceptance=False,goal_status='active'))
    register(output,'rbf-Task-reported-GPU-memory-target-observation')
    print(json.dumps(dict(receipt=str(output),sha256=sha(output),
        cards=[dict(task_id=row['task_id'],**card) for row in rows for card in row['logical_GPUs']],
        target_percent=[75,80],physical_GPU_UUID_independently_verified=False)),flush=True)


if __name__ == '__main__':
    main()
