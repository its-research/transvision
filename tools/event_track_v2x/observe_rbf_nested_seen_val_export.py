"""One bounded ClearML snapshot with experiment-only, allowlisted log fields."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from clearml import Task
from clearml.backend_api.session.client import APIClient

R=Path('/Volumes/Data/test/recover-before-fuse')
KEYS=('stage','key','completed_bytes','total_bytes','ETA_seconds','ETA_scope','seed','side',
      'actual_devices','total_frames','shard','frames','total','elapsed_seconds','detections',
      'appearance_valid','weights_unchanged','actual_cuda_device','TF32_enabled')


def safe_progress(chunks):
    records=[]
    errors=[]
    apt=False
    for chunk in chunks:
        for line in chunk.splitlines():
            apt=apt or '[Connecting to archive.ubuntu.com]' in line
            for kind in ('RuntimeError','ValueError','AssertionError','ImportError','ModuleNotFoundError','OSError'):
                if line.startswith(kind+':') or line==kind:
                    errors.append(kind)
            match=re.match(r'^(?:shard\d+ )?(?:EVENTTRACK_CACHE_[A-Z_]+ )?(\{.*\})$',line)
            if not match:
                continue
            try:
                value=json.loads(match[1])
            except ValueError:
                continue
            if not isinstance(value,dict):
                continue
            allowed={k:value[k] for k in KEYS if k in value}
            if allowed:
                records.append(allowed)
    return records,sorted(set(errors)),apt


def main():
    journal=json.loads((R/'receipts/rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-20261004.json').read_bytes())
    rows=[]
    for job in journal['jobs']:
        task=Task.get_task(task_id=job['task_id'])
        progress,errors,apt=safe_progress(task.get_reported_console_output(number_of_reports=35))
        row=dict(seed=job['seed'],side=job['side'],task_id=task.id,status=str(task.status),
            last_worker=task.data.last_worker,sanitized_progress=progress,error_types=errors,
            startup_apt_wait_seen=apt,raw_agent_configuration_excluded=True,
            artifacts={k:dict(sha256=a.hash,bytes=a.size) for k,a in task.artifacts.items()},
            independent_acceptance=False)
        rows.append(row)
        print(json.dumps(dict(seed=row['seed'],side=row['side'],task_id=task.id,status=row['status'],
            worker=row['last_worker'],artifact_keys=list(row['artifacts']),progress=progress[-2:],error_types=errors)),flush=True)
    api=APIClient()
    workers=[w.to_dict() for w in api.workers.get_all()]
    queues=[q.to_dict() for q in api.queues.get_all()]
    now=datetime.datetime.now(datetime.timezone.utc)
    value=dict(kind='rbf_nested_seen_val_six_GPU4_allowlisted_bounded_live_status_v1',
        checked_at_utc=now.isoformat(),tasks=rows,
        workers=[dict(id=w['id'],ip=w.get('ip'),task_id=w.get('task',{}).get('id'),
            queues=[q['id'] for q in w.get('queues',[])],last_activity_time=str(w['last_activity_time'])) for w in workers],
        queues=[dict(id=q['id'],name=q['name'],pending=[e['task'] for e in q.get('entries',[])]) for q in queues],
        full_online_RBF_accepted=False,paper_performance_complete=False)
    path=R/'receipts'/('rbf-nested-seen-val-six-GPU4-allowlisted-live-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    ledger_path=R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_bytes())
        ledger['entries'].append(dict(kind=value['kind'],receipt=str(path),
            receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),goal_status='active'))
        temporary=ledger_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
        os.replace(temporary,ledger_path)
    print('ALLOWLISTED_STATUS_RECEIPT',str(path),flush=True)


if __name__=='__main__':
    main()
