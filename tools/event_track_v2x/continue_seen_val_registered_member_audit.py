#!/usr/bin/env python3
"""Continue a specific live publication into a read-only complete member audit."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rbf_nested_seen_val_v2_common import register, sha, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['continuation_sha256']
    job = next(r for r in plan['jobs'] if r['seed'] == args.seed)
    assert sha(plan['auditor']['path']) == plan['auditor']['sha256']
    assert sha(job['created_receipt']['path']) == job['created_receipt']['sha256']
    created = json.loads(Path(job['created_receipt']['path']).read_bytes())
    assert created['task_id'] == job['task_id'] and created['identity'] == job['identity']
    assert sha(job['publisher_source']['path']) == job['publisher_source']['sha256']
    directory = Path(job['output'])
    directory.mkdir(exist_ok=False)
    publication = Path(job['publication'])
    started = time.monotonic()
    try:
        while not publication.exists():
            assert time.monotonic() - started < plan['prerequisite_wait_seconds']
            os.kill(job['publisher_pid'], 0)
            command = subprocess.check_output(['ps', '-p', str(job['publisher_pid']), '-o', 'args='], text=True)
            assert job['publisher_source']['path'] in command and f"--seed {job['seed']}" in command
            assert str(publication.parent) in command
            assert sha(plan['auditor']['path']) == plan['auditor']['sha256']
            progress = dict(stage='specific_live_publication_not_yet_accepted',
                            seed=job['seed'], task_id=job['task_id'],
                            actual_publisher_pid=job['publisher_pid'],
                            publisher_process_command_verified=True,
                            member_audit_complete=False,
                            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            temporary = directory / 'progress.new.json'
            temporary.write_text(json.dumps(progress, indent=2) + '\n')
            os.replace(temporary, directory / 'progress.json')
            time.sleep(30)
        from clearml import Task
        task = Task.get_task(task_id=job['task_id'])
        value = json.loads(publication.read_bytes())
        assert value['task_id'] == task.id and value['identity'] == job['identity']
        assert value['manifest'] == created['manifest'] and value['seed'] == job['seed']
        assert str(task.status) == 'completed'
        assert task.get_parameters()['General/publication_identity'] == job['identity']
        assert sha(plan['auditor']['path']) == plan['auditor']['sha256']
        output = directory / 'all-members-independent-readback.json'
        command = [sys.executable, plan['auditor']['path'], '--publication', str(publication),
                   '--output', str(output)]
        with (directory / 'member-audit.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0
        proof = json.loads(output.read_bytes())
        assert proof['task_id'] == task.id and proof['all_registered_members_independently_streamed_and_hashed']
        register(output, proof['kind'])
        new(directory / 'process-exit.json', dict(returncode=0, seed=job['seed'], task_id=task.id,
            independent_member_receipt=str(output), independent_member_receipt_sha256=sha(output),
            new_GPU_task_created=False, input_or_NN_rebuilt=False, replay_or_paper_accepted=False))
        print(json.dumps(dict(seed=job['seed'], task_id=task.id, full_member_readback_passed=True)), flush=True)
    except BaseException as error:
        new(directory / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            seed=job['seed'], task_id=job['task_id'], accepted=False,
            failed_or_partial_assets_preserved=True, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
