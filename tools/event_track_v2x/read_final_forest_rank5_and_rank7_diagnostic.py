"""One isolated read of the failed rank5 and previously unregistered rank7.

Preserve the corrupt full-size partial, all six matched archives and the failed
prefetch/controller. This downloads bytes only; it never restarts an experiment,
promotes a candidate, extracts an archive, or starts CPU acceptance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

R=Path('/Volumes/Data/test/recover-before-fuse')
NAME='rbf-final-forest-seed2027-rank5-rank7-isolated-byte-diagnostic-v1-20261005'
READER=R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004'
FAILURE=R/'receipts/rbf-final-refit-rank-byte-prefetch-seed2027-20261004T174532644652Z-finished.json'
CONTROLLER_FAILURE=R/'artifacts/rbf-final-forest-seed2027-after-prefetch-continuation-v1-20261005/controller-failure.json'
CACHE=R/'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed2027'
LOCAL_BAD_SHA='d030ea8dfa5f5f52dbe45d9ff8c8c195ae253747d57892a5165b049eb48f937f'
TASK_ID='4e9019afec67453ba5a83e8401949991'
PINNED={'read_rbf_final_refit_forest_outputs.py':'53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61',
        'rbf_nested_seen_val_v2_common.py':'58e616f116155b9bd2a9f0ed52b6b0ebc47895c6e9957271a6d88db51b6502ec'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'source-freeze.json').read_bytes())
    assert freeze['kind']==NAME
    for name,checksum in freeze['sources'].items():assert sha(own/name)==checksum
    for item in freeze['references']:assert sha(item['path'])==item['sha256']
    for name,checksum in PINNED.items():assert sha(READER/name)==checksum
    sys.path.insert(0,str(READER))
    spec=importlib.util.spec_from_file_location('unchanged_rank_byte_reader',READER/'read_rbf_final_refit_forest_outputs.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    failed=json.loads(FAILURE.read_bytes())
    assert failed['task_id']==TASK_ID and failed['seed']==2027
    assert failed['failures']=={'replay-rank5':'AssertionError'} and failed['all_snapshot_bytes_match'] is False
    assert failed['registered_snapshot_unchanged_at_end'] is True
    assert set(failed['independently_read_artifacts'])=={f'replay-rank{i}' for i in (0,1,2,3,4,6)}
    partial=CACHE/'replay-rank5.tar.gz.partial'
    assert partial.stat().st_size==failed['registered_snapshot']['replay-rank5']['bytes']==1918350389
    assert sha(partial)==LOCAL_BAD_SHA!=failed['registered_snapshot']['replay-rank5']['sha256']
    assert not (CACHE/'replay-rank5.tar.gz').exists() and not (CACHE/'replay-rank7.tar.gz').exists()
    assert not (CACHE/'independent-byte-coverage-factor-admission.json').exists()
    from clearml import Task
    journal=R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
    job=next(j for j in json.loads(journal.read_bytes())['jobs'] if j['seed']==2027)
    plan=job['plan'];assert job['task_id']==TASK_ID and plan['method']=='rbf' and plan['world_size']==8
    assert hashlib.sha256(reader.canonical(plan)).hexdigest()==job['recipe_sha256']==failed['recipe_sha256']
    task=Task.get_task(task_id=TASK_ID)
    assert str(task.status)=='completed'
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']
    assert json.loads(task.get_parameters()['General/plan'])==plan
    assert task.get_parameters()['General/recipe_sha256']==job['recipe_sha256']
    assert set(task.artifacts)=={'receipt','exclusive-source-manifest'}|{f'replay-rank{i}' for i in range(8)}
    inventory={k:dict(sha256=a.hash,bytes=a.size) for k,a in task.artifacts.items()}
    assert all(inventory[k]==v for k,v in failed['registered_snapshot'].items())
    for key,item in failed['independently_read_artifacts'].items():
        assert (CACHE/(key+'.tar.gz')).stat().st_size==item['bytes']
    selection={key:inventory[key] for key in ('replay-rank5','replay-rank7')}
    assert selection['replay-rank7']==dict(bytes=1800343413,sha256='90e5b2e68de53832e010733b1d34533e0a0746a070b3abe8086f7ecfb9d3b2da')
    out=R/'artifacts'/NAME
    if not args.execute:
        print(json.dumps(dict(selected=selection,local_bad_sha256=LOCAL_BAD_SHA,remote_rank5_sha256=selection['replay-rank5']['sha256'],
            execute=False,output=str(out),no_experiment_restart=True)),flush=True);return 0
    assert not out.exists(), 'existing attempt must be inspected, never replaced or automatically retried'
    with (CACHE/'rank-byte-prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir()
        started=dict(kind=NAME,task_id=TASK_ID,seed=2027,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            command=sys.argv,pid=os.getpid(),source_freeze_sha256=sha(own/'source-freeze.json'),
            recipe_sha256=job['recipe_sha256'],failure_receipt=dict(path=str(FAILURE),sha256=sha(FAILURE)),
            controller_failure=dict(path=str(CONTROLLER_FAILURE),sha256=sha(CONTROLLER_FAILURE)),
            original_partial=dict(path=str(partial),bytes=partial.stat().st_size,sha256=LOCAL_BAD_SHA),
            independently_read_prior_archives_preserved=failed['independently_read_artifacts'],selected=selection,
            all_terminal_registered_artifacts=inventory,automatic_retry=False,uploads_performed=False,
            experiment_tasks_created=0,extraction_started=False,experiment_accepted=False,whole_experiment_ETA='unknown')
        reader.new(out/'started.json',started);reader.register(out/'started.json',NAME+'_started')
        print(json.dumps(dict(stage='isolated_rank_byte_diagnostic_started',pid=os.getpid(),receipt=str(out/'started.json'),selected=selection)),flush=True)
        results={};errors={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(reader.read_artifact,task,key,out/(key+'.tar.gz')):key for key in selection}
            for future in as_completed(futures):
                key=futures[future]
                try:
                    result=future.result();assert result==selection[key];results[key]=result
                    print(json.dumps(dict(stage='isolated_rank_bytes_hash_matched',artifact=key,**result)),flush=True)
                except Exception as error:
                    errors[key]=type(error).__name__
                    print(json.dumps(dict(stage='isolated_rank_bytes_failed',artifact=key,error_type=type(error).__name__,automatic_retry=False)),flush=True)
        try:
            task.reload();assert str(task.status)=='completed'
            assert {k:dict(sha256=a.hash,bytes=a.size) for k,a in task.artifacts.items()}==inventory
            assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']
            assert json.loads(task.get_parameters()['General/plan'])==plan
            assert sha(partial)==LOCAL_BAD_SHA
        except Exception as error:errors['terminal_identity_recheck']=type(error).__name__
        finished=dict(started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            independently_read_artifacts=results,failures=errors,all_selected_bytes_match=not errors and results==selection,
            original_partial_preserved=True,canonical_cache_promoted=False,full_byte_event_factor_admission=False,
            full_forest_CPU_admission=False,paper_performance_complete=False)
        reader.new(out/'finished.json',finished);reader.register(out/'finished.json',NAME+'_finished')
        print(json.dumps(dict(stage='isolated_rank_byte_diagnostic_finished',receipt=str(out/'finished.json'),
            all_selected_bytes_match=finished['all_selected_bytes_match'],experiment_accepted=False)),flush=True)
        return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
