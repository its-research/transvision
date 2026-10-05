"""Distinct final-refit full-train forest candidates on idle GPU4/GPU8 workers.

Frozen nested development replays are preserved. The new weights and their
independently admitted row outputs define a different experiment, with the
same real cache, event schedule, exclusive kernel, limits and bound allocator.
Producing a candidate does not admit the whole forest or teacher training.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R, new, register, sha
from submit_rbf_final_identity import atomic, IMAGE
from submit_rbf_seen_val_joint_identity import snapshot

ROOT = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
JOURNAL = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
PROJECT = 'Thesis/Recover-Before-Fuse/Inference'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    prepared = json.loads((ROOT/'preparation.json').read_bytes())
    for name,record in prepared['execution_sources'].items():
        assert sha(ROOT/name) == record['sha256']
    for proof in prepared['prerequisite_receipts']:
        assert sha(proof['path']) == proof['sha256']
    assert prepared['all_three_distinct_final_refit_weights_and_full_NN_rows_admitted'] is True
    assert prepared['exclusive_kernel_and_configuration_unchanged'] is True
    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    api = APIClient()
    dry_reserved = []

    def save():
        now = datetime.datetime.now(datetime.timezone.utc)
        value = dict(kind='rbf_final_refit_full_train_exclusive_forest_GPU_dispatch_v1',
            checked_at_utc=now.isoformat(),jobs=jobs,
            preparation_sha256=sha(ROOT/'preparation.json'),GPU_model_restriction=False,
            L40S_CPU_only=True,TF32_enabled=False,full_forest_independently_accepted=False,
            learned_Stage2_complete=False,paper_performance_complete=False)
        atomic(JOURNAL,value)
        version = R/'receipts'/('rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-'
            +now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(version,value)
        register(version,value['kind'])

    for seed in (2027,1337,3407):
        item = next(v for v in prepared['seeds'] if v['seed'] == seed)
        semantic = hashlib.sha256(canonical(item)).hexdigest()
        name = f'RBF final-refit full-train exclusive forest seed{seed} '+semantic[:16]
        matches = Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
        assert len(matches) <= 1
        prior = next((v for v in jobs if v['seed'] == seed),None)
        if matches:
            task = matches[0]
            assert task.get_parameters()['General/semantic_forest_identity'] == semantic
            assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == prepared['bootstrap_sha256']
            assert prior is not None and prior['task_id'] == task.id, 'existing task missing journal; preserve and inspect'
            print(json.dumps(dict(seed=seed,existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
            continue
        assert prior is None, 'journal task must be inspected, never replaced'
        plan = item['plan']
        for spec in [plan[k] for k in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest')]+plan['forward_outputs']:
            asset = Task.get_task(task_id=spec['task'])
            assert str(asset.status) == 'completed'
            artifact = asset.artifacts[spec['key']]
            assert artifact.hash == spec['sha256'] and artifact.size == spec['bytes']
        choices,fleet = snapshot(api,jobs,dry_reserved)
        if not choices:
            print(json.dumps(dict(seed=seed,status='waiting_for_disjoint_idle_GPU',ETA='unknown')),flush=True)
            continue
        choice = choices[0]
        print(json.dumps(dict(seed=seed,choice=choice,execute=args.execute,
            physical_non_L40_GPUs=fleet['physical_non_L40_GPUs'],physical_unbound_GPUs=fleet['physical_unbound_GPUs'])),flush=True)
        if not args.execute:
            dry_reserved.extend((host,set(cards)) for host,cards in choice['bindings'])
            continue
        plan = dict(plan,world_size=choice['world_size'])
        digest = hashlib.sha256(canonical(plan)).hexdigest()
        task = Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
        task.output_uri = 'http://10.100.34.118:8081'
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',
            diff=(ROOT/'bootstrap.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=digest,
            semantic_forest_identity=semantic,seed=seed,world_size=choice['world_size'],
            GPU_model_restriction=False,TF32_enabled=False,
            ETA='unknown_until_actual_transfer_or_committed_event_progress',paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','all-class-top64','final-refit-full-train',
            'exclusive-recoverable-forest','bound-allocation','independent-full-forest-admission-pending'])
        task.reload()
        assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == prepared['bootstrap_sha256']
        job = dict(seed=seed,task_id=task.id,semantic_forest_identity=semantic,recipe_sha256=digest,
            plan=plan,queue=choice['queue_name'],bindings=choice['bindings'],eligible_workers=choice['eligible_workers'],
            fleet=fleet,status='created_before_enqueue',ETA='unknown_until_actual_event_progress')
        jobs.append(job)
        save()
        fresh,_ = snapshot(api,jobs[:-1],dry_reserved)
        assert any(v['queue_id'] == choice['queue_id'] for v in fresh), 'capacity changed; created task retained'
        Task.enqueue(task,queue_id=choice['queue_id'])
        task.reload()
        job['status'] = str(task.status)
        save()
        print(json.dumps(dict(seed=seed,task_id=task.id,queue=choice['queue_name'],
            world_size=choice['world_size'],status=str(task.status),full_forest_accepted=False)),flush=True)


if __name__ == '__main__':
    main()
