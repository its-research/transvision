"""One bounded live observation, retaining only explicitly allowed progress."""
import datetime
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R, new, register
from submit_rbf_seen_val_joint_identity import JOURNAL, snapshot


def main():
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    jobs = json.loads(JOURNAL.read_bytes())['jobs']
    _,fleet = snapshot(APIClient(),jobs)
    rows = []
    for job in jobs:
        task = Task.get_task(task_id=job['task_id'])
        progress, errors = [],set()
        for chunk in task.get_reported_console_output(number_of_reports=30):
            for line in chunk.splitlines():
                match = re.fullmatch(r'(AssertionError|RuntimeError|ValueError|ImportError|ModuleNotFoundError|OSError)(?::.*)?',line)
                if match:
                    errors.add(match[1])
                if not line.startswith('{'):
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(value,dict) or value.get('stage') != 'target_free_seen_val_joint_identity_GPU_forward':
                    continue
                progress.append({key:value[key] for key in ('stage','rank','completed_rows','total_rows','ETA_seconds','ETA_scope') if key in value})
        output = R/f'artifacts/rbf-matching-seen-val-joint-identity-full-independent-output-admission-v1-20261004/seed{job["seed"]}/full-numeric/completion.json'
        row = dict(seed=job['seed'],task_id=task.id,status=str(task.status),last_worker=task.data.last_worker,
            world_size=job['plan']['world_size'],queue=job['queue'],progress=progress[-8:],error_types=sorted(errors),
            artifacts={k:dict(bytes=a.size,sha256=a.hash) for k,a in task.artifacts.items()},
            independent_full_NN_numeric_receipt_exists=output.exists(),raw_agent_configuration_excluded=True)
        rows.append(row)
        print(json.dumps(row),flush=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    path = R/'receipts'/('rbf-matching-seen-val-three-seed-allowlisted-live-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    new(path,dict(kind='rbf_matching_seen_val_three_seed_one_bounded_allowlisted_live_status_v1',
        checked_at_utc=now.isoformat(),tasks=rows,fleet=fleet,full_online_RBF_accepted=False,paper_performance_complete=False))
    register(path,'rbf_matching_seen_val_three_seed_one_bounded_allowlisted_live_status_v1')
    print(json.dumps(dict(receipt=str(path),physical_non_L40_GPUs=fleet['physical_non_L40_GPUs'],
        physical_worker_task_bound_GPUs=fleet['physical_worker_task_bound_GPUs'],physical_unbound_GPUs=fleet['physical_unbound_GPUs'],
        worker_binding_is_not_utilization_measurement=True)),flush=True)


if __name__ == '__main__':
    main()
