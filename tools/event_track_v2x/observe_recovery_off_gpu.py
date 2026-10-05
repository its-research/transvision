"""One allowlisted live check of the separately named recovery-off tasks."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

R = Path('/Volumes/Data/test/recover-before-fuse')
ROOT = R/'source-freezes/rbf-final-refit-recovery-off-bound-GPU-v1-20261005'
JOURNAL = R/'receipts/rbf-final-refit-recovery-off-bound-GPU-dispatch-20261005.json'


def main():
    sys.path.insert(0, str(ROOT))
    from rbf_nested_seen_val_v2_common import new, register, sha
    from recovery_off_dispatch_gate import require_qualified_source
    prepared = json.loads((ROOT/'preparation.json').read_bytes())
    require_qualified_source(ROOT, prepared)
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    from submit_rbf_seen_val_joint_identity import snapshot
    jobs = json.loads(JOURNAL.read_bytes())['jobs']
    _, fleet = snapshot(APIClient(), jobs)
    rows = []
    for job in jobs:
        task = Task.get_task(task_id=job['task_id'])
        params = task.get_parameters()
        assert params['General/semantic_forest_identity'] == job['semantic_forest_identity']
        assert params['General/recipe_sha256'] == job['recipe_sha256']
        assert json.loads(params['General/plan']) == job['plan']
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == prepared['bootstrap_sha256']
        progress = []
        for chunk in task.get_reported_console_output(number_of_reports=25):
            for line in chunk.splitlines():
                if not line.startswith('{'): continue
                try: value = json.loads(line)
                except ValueError: continue
                if not isinstance(value, dict): continue
                if value.get('phase') in ('registered_input_transfer','coupled_forest_sequence_progress',
                                          'coupled_failure_before_receipt_upload'):
                    keys = ('phase','role','completed_bytes','total_bytes','seed','method','rank',
                            'sequence_id','completed_sequences','total_sequences','sequence_completed',
                            'committed_events','ETA_seconds','ETA_scope','exception_type')
                elif value.get('kind') == 'rbf_experiment_progress_v1':
                    keys = ('kind','stage','completed','total','elapsed_seconds','eta_seconds','eta_scope','eta_status')
                else: continue
                progress.append({k:value[k] for k in keys if k in value})
        row = dict(seed=job['seed'], task_id=task.id, status=str(task.status),
                   actual_last_worker=task.data.last_worker, world_size=job['plan']['world_size'],
                   source_and_configuration_live_match=True, progress=progress[-10:],
                   artifacts={k:dict(bytes=a.size,sha256=a.hash) for k,a in task.artifacts.items()},
                   full_forest_independently_accepted=False, whole_experiment_ETA='unknown')
        rows.append(row)
        print(json.dumps(row), flush=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = R/'receipts'/('rbf-recovery-off-GPU-live-'+stamp+'.json')
    value = dict(kind='rbf_recovery_off_GPU_allowlisted_live_v1', tasks=rows, fleet=fleet,
                 observer_sha256=sha(__file__), journal_sha256=sha(JOURNAL),
                 preparation_sha256=sha(ROOT/'preparation.json'),
                 qualification_sha256=sha(ROOT/'source-qualification.json'),
                 full_forest_independently_accepted=False, paper_performance_complete=False)
    new(path,value); register(path,value['kind'])
    print(json.dumps(dict(receipt=str(path),sha256=sha(path))), flush=True)


if __name__ == '__main__': main()
