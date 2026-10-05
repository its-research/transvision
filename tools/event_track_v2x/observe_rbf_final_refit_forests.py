"""One bounded observation; never persist Agent config or arbitrary log lines."""
import datetime
import json

from rbf_nested_seen_val_v2_common import R,new,register,sha
from submit_rbf_seen_val_joint_identity import snapshot


def main():
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    paths=[R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',
           R/'receipts/rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json']
    jobs=[j for p in paths if p.exists() for j in json.loads(p.read_bytes())['jobs']]
    _,fleet=snapshot(APIClient(),jobs)
    rows=[]
    for job in jobs:
        task=Task.get_task(task_id=job['task_id'])
        progress=[]
        for chunk in task.get_reported_console_output(number_of_reports=25):
            for line in chunk.splitlines():
                if not line.startswith('{'):
                    continue
                try:
                    v=json.loads(line)
                except ValueError:
                    continue
                if not isinstance(v,dict):
                    continue
                if v.get('phase') in ('coupled_forest_sequence_progress','coupled_failure_before_receipt_upload'):
                    fields=('phase','seed','method','rank','sequence_id','completed_sequences','total_sequences',
                        'sequence_completed','committed_events','ETA_seconds','ETA_scope','exception_type','experiment_accepted')
                elif v.get('kind')=='rbf_experiment_progress_v1' and v.get('stage')=='paper_replay_events':
                    fields=('kind','stage','completed','total','elapsed_seconds','eta_seconds','eta_scope','eta_status')
                else:
                    continue
                progress.append({key:v[key] for key in fields if key in v})
        row=dict(seed=job['seed'],method=job['plan']['method'],task_id=task.id,status=str(task.status),
            last_worker=task.data.last_worker,world_size=job['plan']['world_size'],queue=job['queue'],
            progress=progress[-16:],artifacts={k:dict(bytes=a.size,sha256=a.hash) for k,a in task.artifacts.items()},
            ETA='unknown' if not progress else 'see task-level current-stage progress; whole experiment unknown',
            full_forest_independently_accepted=False,paper_performance_complete=False)
        rows.append(row)
        print(json.dumps(row),flush=True)
    now=datetime.datetime.now(datetime.timezone.utc)
    path=R/'receipts'/('rbf-final-refit-forest-and-topK-allowlisted-live-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    value=dict(kind='rbf_final_refit_main_and_topK_one_bounded_live_v1',checked_at_utc=now.isoformat(),
        tasks=rows,fleet=fleet,dispatch_receipts=[dict(path=str(p),sha256=sha(p)) for p in paths if p.exists()],
        pending_uncreated_topK_seed3407=not any(j['seed']==3407 and j['plan']['method']=='topk' for j in jobs),
        Agent_configuration_and_nonallowlisted_logs_excluded=True)
    new(path,value);register(path,value['kind'])
    print(json.dumps(dict(receipt=str(path),physical_non_L40_GPUs=fleet['physical_non_L40_GPUs'],
        physical_worker_task_bound_GPUs=fleet['physical_worker_task_bound_GPUs'],
        physical_unbound_GPUs=fleet['physical_unbound_GPUs'],worker_binding_is_not_utilization_measurement=True)),flush=True)


if __name__=='__main__':
    main()
