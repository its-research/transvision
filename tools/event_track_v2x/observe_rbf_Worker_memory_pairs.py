"""Read paired device-memory MB aggregates; never claim per-card identity."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import sys

from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live-binding-receipt',type=Path,required=True)
    args=parser.parse_args();live=json.loads(args.live_binding_receipt.read_bytes())
    assert live['kind']=='rbf_final_refit_main_K4_Top1_one_bounded_live_v1'
    assert live['Agent_configuration_and_nonallowlisted_logs_excluded'] is True
    now=datetime.datetime.now(datetime.timezone.utc)
    assert 0<=(now-datetime.datetime.fromisoformat(live['checked_at_utc'])).total_seconds()<300
    workers={w['id']:w['task_id'] for w in live['fleet']['workers'] if w.get('task_id') and 'L40' not in w['id']}
    assert workers
    from clearml.backend_api.services.v2_20 import workers as schema
    schema_path=Path(schema.__file__)
    assert 'GPU free memory MBs' in schema.MachineStats.__doc__ and 'GPU used memory MBs' in schema.MachineStats.__doc__
    root=R/'source-freezes/rbf-allowlisted-Worker-device-memory-pairs-v1-20261004'
    root.mkdir(exist_ok=True)
    for original in (Path(__file__).resolve(),Path(__file__).resolve().with_name('rbf_nested_seen_val_v2_common.py')):
        dest=root/original.name
        if dest.exists():assert sha(dest)==sha(original)
        else:
            with dest.open('xb') as stream:stream.write(original.read_bytes())
    freeze=root/'source-freeze.json'
    if not freeze.exists():
        new(freeze,dict(kind='rbf_Worker_device_memory_pair_observer_source_v1',
            sources={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.glob('*.py')},
            API_schema_reference=dict(path=str(schema_path),sha256=sha(schema_path)),
            memory_used_and_free_unit='MB per documented Worker MachineStats schema',
            raw_Agent_configuration_read_or_saved=False,actual_experiment_launched=False))
        register(freeze,'rbf-Worker-device-memory-pair-observer-source')
    reference=json.loads(freeze.read_bytes())
    assert reference['API_schema_reference']['sha256']==sha(schema_path)
    query=dict(worker_ids=sorted(workers),from_date=now.timestamp()-300,to_date=now.timestamp(),interval=30,
        items=[dict(key=k,category='avg') for k in ('gpu_memory_used','gpu_memory_free')],split_by_variant=True)
    from clearml.backend_api.session import Session
    response=Session().send_request('workers','get_stats',json=query)
    assert response.status_code==200
    data=response.json()['data'];series={}
    for worker in data.get('workers',[]):
        wid=worker['worker'];assert wid in workers
        for metric in worker.get('metrics',[]):
            key=metric['metric'];assert key in ('gpu_memory_used','gpu_memory_free')
            identity=(wid,metric.get('variant'));target=series.setdefault(identity,{})
            for aggregate in metric.get('stats',[]):
                if aggregate['aggregation']!='avg':continue
                dates=aggregate.get('dates',metric.get('dates',[]));values=aggregate.get('values',[])
                assert len(dates)==len(values) and dates==sorted(set(dates))
                assert all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in dates+values)
                if dates and all(1e12<=d<1e13 for d in dates):seconds=[d/1000 for d in dates]
                else:
                    assert all(1e9<=d<1e10 for d in dates);seconds=dates
                assert key not in target
                target[key]={d:v for d,v in zip(seconds,values) if query['from_date']<=d<=query['to_date']}
    records=[]
    for (wid,variant),metrics in series.items():
        used=metrics.get('gpu_memory_used',{});free=metrics.get('gpu_memory_free',{})
        common=sorted(set(used)&set(free));samples=[]
        for date in common:
            total=used[date]+free[date]
            if total<=0:continue
            samples.append(dict(timestamp=date,mean_used_MB=used[date],mean_free_MB=free[date],
                ratio_of_paired_means_percent=100*used[date]/total))
        values=[s['ratio_of_paired_means_percent'] for s in samples]
        records.append(dict(worker=wid,task_id_at_binding_snapshot=workers[wid],hardware_variant=variant,
            sample_count=len(samples),samples=samples,latest_percent=values[-1] if values else None,
            mean_percent=statistics.mean(values) if values else None,
            group_mean_only=variant is None,independent_physical_card_or_UUID_mapping=False))
    stamp=datetime.datetime.now(datetime.timezone.utc)
    output=R/'receipts'/('rbf-Worker-device-memory-paired-aggregate-'+stamp.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(output,dict(kind='rbf_Worker_device_memory_paired_aggregate_observation_v1',checked_at_utc=stamp.isoformat(),
        source_freeze_sha256=sha(freeze),binding_receipt=str(args.live_binding_receipt),binding_receipt_sha256=sha(args.live_binding_receipt),
        command=[sys.executable,str(Path(__file__).resolve()),'--live-binding-receipt',str(args.live_binding_receipt)],
        request=query,records=records,units='Worker API documented device MB; not Task process-tree memory',
        percentage_is_ratio_of_same_timestamp_means=True,per_card_target_achieved=False,
        physical_UUID_independently_verified=False,remote_Agent_sampler_source_independently_verified=False,
        GPU_memory_target_percent=[75,80],no_idle_or_colocation_permission_inferred=True,
        actual_experiment_restarted=False,raw_API_response_or_credentials_saved=False,paper_acceptance=False))
    register(output,'rbf-Worker-device-memory-paired-aggregate')
    print(json.dumps(dict(receipt=str(output),sha256=sha(output),
        groups=[{k:v for k,v in record.items() if k!='samples'} for record in records])),flush=True)


if __name__=='__main__':main()
