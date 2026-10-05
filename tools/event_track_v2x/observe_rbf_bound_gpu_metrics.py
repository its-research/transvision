"""Read only documented Worker metrics; never persist Agent configuration."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-binding-receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = args.live_binding_receipt.resolve()
    assert receipt.is_relative_to(R / 'receipts')
    live = json.loads(receipt.read_bytes())
    assert live['Agent_configuration_and_nonallowlisted_logs_excluded'] is True
    assert live['kind'] == 'rbf_final_refit_main_K4_Top1_one_bounded_live_v1'
    now = datetime.datetime.now(datetime.timezone.utc)
    observed = datetime.datetime.fromisoformat(live['checked_at_utc'])
    assert 0 <= (now - observed).total_seconds() < 300, 'refresh this same live observation first'
    workers = [entry for entry in live['fleet']['workers'] if entry.get('task_id') and 'L40' not in entry['id']]
    assert workers and len({entry['id'] for entry in workers}) == len(workers)
    identities = {entry['id']: entry['task_id'] for entry in workers}
    worker_ids = sorted(identities)
    root = R / 'source-freezes/rbf-allowlisted-bound-Worker-metrics-v3-epoch-ms-20261004'
    root.mkdir(exist_ok=True)
    for original in (Path(__file__).resolve(), Path(__file__).resolve().with_name('rbf_nested_seen_val_v2_common.py')):
        path = root / original.name
        if path.exists():
            assert sha(path) == sha(original)
        else:
            with path.open('xb') as stream:
                stream.write(original.read_bytes())
    freeze = root / 'source-freeze.json'
    if not freeze.exists():
        new(freeze, dict(kind='rbf_allowlisted_bound_Worker_metrics_read_only_source_v3_epoch_ms',
            sources={path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in root.glob('*.py')},
            raw_Agent_configuration_read=False, experiment_launched=False))
        register(freeze, 'rbf-bound-Worker-metrics-read-only-source')
    allowed = ('gpu_usage', 'gpu_memory_used', 'cpu_usage', 'memory_used')
    query = dict(worker_ids=worker_ids, from_date=now.timestamp()-300, to_date=now.timestamp(),
                 interval=30, items=[dict(key=key, category='avg') for key in allowed], split_by_variant=True)
    from clearml.backend_api.session import Session
    response = Session().send_request('workers', 'get_stats', json=query)
    assert response.status_code == 200, ('metrics HTTP status', response.status_code)
    data = response.json()['data']
    readings = []
    for entry in data.get('workers', []):
        worker = entry['worker']
        assert worker in identities
        for metric in entry.get('metrics', []):
            key = metric['metric']
            if key not in allowed:
                continue
            for aggregate in metric.get('stats', []):
                if aggregate['aggregation'] != 'avg':
                    continue
                # This server returns dates inside each aggregation. Older
                # schemas place the same axis on the metric itself.
                if aggregate.get('dates') is not None and metric.get('dates') is not None:
                    assert aggregate['dates'] == metric['dates']
                dates = aggregate.get('dates')
                date_axis_location = 'aggregation.dates'
                if dates is None:
                    dates = metric.get('dates') or []
                    date_axis_location = 'metric.dates'
                assert all(type(value) in (int, float) for value in dates)
                assert dates == sorted(dates) and len(set(dates)) == len(dates)
                values = aggregate.get('values') or []
                assert len(values) == len(dates)
                assert all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values)
                if dates and all(1e12 <= date < 1e13 for date in dates):
                    date_unit = 'epoch_milliseconds'
                    seconds = [date / 1000 for date in dates]
                else:
                    assert all(1e9 <= date < 1e10 for date in dates)
                    date_unit = 'epoch_seconds'
                    seconds = list(dates)
                samples = [(date, value) for date, value in zip(seconds, values)
                           if query['from_date'] <= date <= query['to_date']]
                sampled_values = [value for _, value in samples]
                readings.append(dict(worker=worker, task_id_at_binding_snapshot=identities[worker],
                    metric=key, hardware_variant=metric.get('variant'), aggregation='avg',
                    date_axis_location=date_axis_location,
                    wire_date_unit=date_unit, wire_dates=dates, wire_values=values,
                    wire_date_intervals_seconds=sorted(set(b-a for a,b in zip(seconds,seconds[1:]))),
                    dates_epoch_seconds=[date for date, _ in samples], values=sampled_values,
                    sample_count=len(sampled_values), mean=statistics.mean(sampled_values) if sampled_values else None,
                    minimum=min(sampled_values) if sampled_values else None, maximum=max(sampled_values) if sampled_values else None))
    stamp = datetime.datetime.now(datetime.timezone.utc)
    path = R / 'receipts' / ('rbf-bound-Worker-reported-hardware-variant-metrics-' + stamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    value = dict(kind='rbf_allowlisted_recent_bound_Worker_hardware_variant_metrics_v3_epoch_ms',
        checked_at_utc=stamp.isoformat(), live_binding_receipt=str(receipt), live_binding_receipt_sha256=sha(receipt),
        command=[sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--live-binding-receipt', str(receipt)],
        source_freeze=str(freeze), source_freeze_sha256=sha(freeze), metrics_API='ClearML workers.get_stats',
        request=query, readings=readings, physical_binding_count_is_not_GPU_utilization=True,
        hardware_variant_physical_GPU_identity_independently_verified=False, direct_NVML_sample_taken=False,
        raw_Agent_configuration_read=False, raw_API_response_persisted=False, credentials_written_or_printed=False,
        sampled_utilization_is_not_experiment_acceptance=True, experiment_restarted=False,
        whole_experiment_ETA='unknown', goal_status='active')
    new(path, value)
    register(path, value['kind'])
    print(json.dumps(dict(receipt=str(path), receipt_sha256=sha(path), requested_workers=len(worker_ids),
        GPU_metrics=[{key: reading[key] for key in ('worker', 'hardware_variant', 'sample_count', 'mean', 'minimum', 'maximum')}
                     for reading in readings if reading['metric']=='gpu_usage'],
        no_physical_card_identity_or_idle_permission_inferred=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
