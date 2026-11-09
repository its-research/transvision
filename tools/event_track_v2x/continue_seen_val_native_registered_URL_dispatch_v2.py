#!/usr/bin/env python3
"""Continue specifically bound complete-member admission into guarded dispatch.

Never rebuild inputs, retry a failed GPU task, or override native-v2 training
priority or physical GPU reservations. Polling is only for missing prerequisites
or disjoint idle capacity. Created tasks and failures are retained.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(1337,3407),required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_bytes())
    spec=importlib.util.spec_from_file_location('frozen_native_seen_val_dispatch',plan['dispatcher']['path'])
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    freeze=dispatch.source_gate();sha=dispatch.sha
    assert sha(__file__)==plan['continuation_sha256']
    assert sha(plan['dispatcher']['path'])==plan['dispatcher']['sha256']
    assert sha(plan['dispatch_source_freeze']['path'])==plan['dispatch_source_freeze']['sha256']
    job=next(j for j in plan['jobs'] if j['seed']==args.seed)
    for record in job['existing_input_receipts']:
        assert sha(record['path'])==record['sha256']
    out=Path(job['output']);out.mkdir(exist_ok=False)
    member=Path(job['member_admission']);started=time.monotonic()
    def progress(stage,**values):
        value=dict(seed=args.seed,stage=stage,full_seen_val_replay_accepted=False,
                   full_G_or_paper_accepted=False,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
        temporary=out/'progress.new.json';temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,out/'progress.json')
    try:
        while not member.exists():
            assert time.monotonic()-started<plan['prerequisite_wait_seconds']
            assert not (member.parent/'failure.json').exists(),'specific member audit failed; preserve failure'
            os.kill(job['member_auditor_pid'],0)
            command=subprocess.check_output(['ps','-p',str(job['member_auditor_pid']),'-o','args='],text=True)
            assert job['member_continuation_source']['path'] in command and f'--seed {args.seed}' in command
            assert sha(job['member_continuation_source']['path'])==job['member_continuation_source']['sha256']
            progress('waiting_for_specific_complete_registered_member_admission',actual_member_auditor_pid=job['member_auditor_pid'],GPU_task_created=False)
            time.sleep(30)
        admitted=dispatch.member_gate(member,Path(job['publication']),args.seed,freeze)
        dispatch.new(out/'complete-member-admission-bound.json',admitted)
        dispatch.register(out/'complete-member-admission-bound.json','rbf_seen_val_native_dispatch_complete_member_binding_v2')
        waiting={'qualified_native_v2_training_has_first_idle_GPU_priority','remaining_main_and_fixed_baselines_have_dispatch_priority','waiting_for_disjoint_idle_GPU'}
        attempts=0
        while True:
            assert time.monotonic()-started<plan['total_wait_seconds']
            dispatch.source_gate()
            command=[sys.executable,plan['dispatcher']['path'],'--seed',str(args.seed)]
            for name in ('main-admission','main-byte-admission','publication','member-admission'):
                command.extend(['--'+name,job[name.replace('-','_')]])
            command.append('--execute');attempts+=1
            result=subprocess.run(command,capture_output=True,text=True)
            attempt=out/f'dispatch-check-{attempts:05d}.json'
            dispatch.new(attempt,dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,automatic_failed_task_retry=False))
            assert result.returncode==0,'guarded dispatcher failed; preserve exact output and any created task'
            lines=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            assert len(lines)==1;value=lines[0]
            status=value.get('status')
            progress('guarded_dispatcher_result',dispatcher_result=value,attempt=attempts)
            print(json.dumps(dict(seed=args.seed,dispatcher_result=value)),flush=True)
            if status in waiting:
                time.sleep(plan['capacity_poll_seconds']);continue
            assert 'task_id' in value or 'existing_task_id' in value
            receipt=dict(kind='rbf_seen_val_native_registered_URL_specific_input_guarded_dispatch_continuation_v2',seed=args.seed,dispatcher_result=value,member_admission=admitted,attempt_receipt=str(attempt),attempt_receipt_sha256=sha(attempt),failed_tasks_retried=False,full_seen_val_replay_accepted=False,full_G_or_paper_accepted=False)
            dispatch.new(out/'continuation-exit.json',receipt);dispatch.register(out/'continuation-exit.json',receipt['kind']);return
    except BaseException as error:
        receipt=dict(kind='rbf_seen_val_native_registered_URL_dispatch_continuation_failure_v2',seed=args.seed,error_type=type(error).__name__,error=str(error),automatic_retry=False,failed_or_partial_assets_and_tasks_preserved=True,full_G_or_paper_accepted=False)
        dispatch.new(out/'failure.json',receipt);dispatch.register(out/'failure.json',receipt['kind']);raise


if __name__=='__main__':main()
