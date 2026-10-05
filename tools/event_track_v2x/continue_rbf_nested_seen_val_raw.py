"""Launch one independent CPU readback per completed val producer; never retry."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from clearml import Task

R=Path('/Volumes/Data/test/recover-before-fuse')
S=R/'source-freezes/rbf-nested-detector-seen-val-raw-GPU4-v2-20261004'
ROOT=R/'artifacts/rbf-nested-detector-seen-val-independent-raw-readback-v1-20261004'


def register(path,kind):
    ledger_path=R/'receipts/20260928-execution-ledger.json'
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_bytes())
        if any(e.get('receipt')==str(path) for e in ledger['entries']):
            return
        ledger['entries'].append(dict(kind=kind,receipt=str(path),
            receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),goal_status='active'))
        temporary=ledger_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
        os.replace(temporary,ledger_path)


def main():
    preparation=json.loads((S/'raw-readback-source-freeze.json').read_bytes())
    for name,digest in preparation['sources'].items():
        assert hashlib.sha256((S/name).read_bytes()).hexdigest()==digest
    jobs=json.loads((R/'receipts/rbf-nested-detector-matching-seen-val-raw-GPU-dispatch-20261004.json').read_bytes())['jobs']
    done=set()
    while len(done)<len(jobs):
        for job in jobs:
            if job['task_id'] in done:
                continue
            task=Task.get_task(task_id=job['task_id'])
            output=ROOT/('seed'+str(job['seed']))/job['side']
            launch=R/'receipts'/('rbf-nested-seen-val-independent-raw-seed%d-%s-launch-20261004.json'%(job['seed'],job['side']))
            if launch.exists() or output.exists():
                done.add(job['task_id'])
                continue
            if str(task.status) in ('failed','stopped','aborted','closed'):
                print(json.dumps(dict(task_id=task.id,status=str(task.status),failure_preserved=True,no_readback_launched=True)),flush=True)
                done.add(task.id)
                continue
            if str(task.status)!='completed':
                continue
            logdir=ROOT/'launch-logs'
            logdir.mkdir(parents=True,exist_ok=True)
            log=logdir/('seed%d-%s.log'%(job['seed'],job['side']))
            command=[sys.executable,'-B',str(S/'readback.py'),'--task-id',task.id]
            with log.open('x') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            value=dict(kind='rbf_nested_seen_val_completed_task_independent_raw_CPU_readback_launch_v1',
                task_id=task.id,seed=job['seed'],side=job['side'],pid=process.pid,command=command,log=str(log),
                reader_source_sha256=preparation['sources']['readback.py'],
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                independent_acceptance=False,ETA='unknown until real byte progress',no_remote_task_created=True)
            with launch.open('x') as stream:
                json.dump(value,stream,indent=2)
                stream.write('\n')
            register(launch,value['kind'])
            print(json.dumps(dict(task_id=task.id,pid=process.pid,independent_CPU_readback_started=True)),flush=True)
            done.add(task.id)
        if len(done)<len(jobs):
            time.sleep(30)
    print(json.dumps(dict(all_six_producer_states_handled=True,independent_acceptance_must_be_checked_separately=True)),flush=True)


if __name__=='__main__':
    main()
