"""Dispatch distinct, fully admitted learned-priority replays on disjoint GPUs.

No model-specific GPU restriction; frozen execution supports four or eight
cards. Existing/failed attempts are retained. Worker bindings never certify
75-80 percent memory use, acceptance, or same-resource paper performance.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha
from publish_final_refit_priority_checkpoint import canonical, qualify, verify_publication, PROJECT, DESTINATION
from submit_rbf_final_identity import atomic, IMAGE
from submit_rbf_seen_val_joint_identity import snapshot

PRODUCER = R/'source-freezes/rbf-final-refit-learned-priority-full-train-GPU-v1-20261005'
PRODUCER_FREEZE = 'de49412b52eba785d1e7f414998168f563d0dd7a84def27eece5b21e86825973'
PUBLICATION = R/'source-freezes/rbf-final-refit-priority-checkpoint-publication-v2-20261005'
PUBLICATION_FREEZE = '6b449f227a232bb579875c23a58de7110a65d4b500cb40af8f63dcfec893a25a'
VARIANT = 'final_refit_learned_priority_full_train_v1'
JOURNAL = R/'receipts/rbf-final-refit-learned-priority-full-train-GPU-dispatch-20261005.json'
CORE_JOURNALS = [R/'receipts'/name for name in (
    'rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-topK-GPU-dispatch-20261004.json',
    'rbf-final-refit-full-train-fixed-Top1-GPU-dispatch-20261004.json')]
OTHER_JOURNALS = [R/'receipts'/name for name in (
    'rbf-final-refit-capacity-witness-teacher-GPU-dispatch-20261005.json',
    'rbf-batched-complete-sequence-GPU-dispatch-20261004.json',
    'rbf-branch-state-CUDA-collision-safe-dispatch-20261005.json')]


def source_gate():
    own = Path(__file__).resolve().parent
    for directory, expected in ((own, None), (PRODUCER, PRODUCER_FREEZE), (PUBLICATION, PUBLICATION_FREEZE)):
        path = directory/'source-freeze.json'
        if expected is not None: assert sha(path) == expected
        freeze = json.loads(path.read_bytes())
        for name, item in freeze['sources'].items(): assert sha(directory/name) == item['sha256']
        for item in freeze.get('references', ()): assert sha(item['path']) == item['sha256']
    # Qualifier imports must be the qualified v2 implementation, not a mutable checkout copy.
    assert sha(own/'publish_final_refit_priority_checkpoint.py') == sha(PUBLICATION/'publish_final_refit_priority_checkpoint.py')
    return json.loads((PRODUCER/'source-control.json').read_bytes())


def expected_plan(main, publication, control, producer_sha, dispatcher_sha):
    assert main['seed'] == publication['seed'] in (1337, 2027, 3407)
    assert main['configuration']['allocation'] == 'bound' and main['method'] == 'rbf'
    assert main['configuration']['backend'] == 'exclusive_root_partition_regions_v1'
    assert main['configuration']['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert publication['kind'] == 'rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1'
    assert publication['independent_cloud_bytes_verified'] is True
    assert publication['learned_replay_accepted'] is publication['paper_performance_complete'] is False
    assert set(publication['artifacts']) == {'audit', 'checkpoint', 'weights'}
    patches = control['patches']
    assert set(main['exclusive_patches']) < set(patches)
    assert all(patches[k] == v for k,v in main['exclusive_patches'].items())
    assert set(patches)-set(main['exclusive_patches']) == {
        'transvision/models/event_track_v2x/'+name for name in (
            'exclusive_teacher_probe_witness.py', 'exclusive_witness_paper_runtime.py',
            'exclusive_priority_admission.py', 'exclusive_allocation_training.py')}
    return dict(copy.deepcopy(main), world_size=4, bootstrap_sha256=producer_sha,
        dispatcher_sha256=dispatcher_sha, exclusive_source_freeze_sha256=PRODUCER_FREEZE,
        exclusive_patches=copy.deepcopy(patches), configuration=dict(copy.deepcopy(main['configuration']), allocation='learned'),
        priority_inputs=copy.deepcopy(publication['artifacts']), priority_publication=copy.deepcopy(publication),
        priority_policy_signature=publication['policy_signature'],
        scope='full original paired train final-model learned priority; independent forest, ordering and cost verification pending')


def semantic(plan):
    return hashlib.sha256(canonical({k:v for k,v in plan.items() if k != 'world_size'})).hexdigest()


def reservations():
    jobs = [];missing = []
    for path in CORE_JOURNALS:
        entries = json.loads(path.read_bytes())['jobs'] if path.exists() else []
        assert len({j['seed'] for j in entries}) == len(entries) and {j['seed'] for j in entries} <= {1337,2027,3407}
        missing.extend(dict(journal=str(path), seed=s) for s in (1337,2027,3407) if s not in {j['seed'] for j in entries})
        jobs.extend(entries)
    for path in OTHER_JOURNALS:
        if path.exists(): jobs.extend(json.loads(path.read_bytes())['jobs'])
    assert len({j['task_id'] for j in jobs}) == len(jobs)
    return jobs, missing


def verify_existing(task, job, base, identity):
    assert job is not None and job['task_id'] == task.id, 'unjournaled task requires recovery; never create a duplicate'
    assert job['allocation_variant'] == VARIANT and job['semantic_learned_identity'] == identity
    assert job['plan'] == dict(base, world_size=job['plan']['world_size'])
    assert job['plan']['world_size'] in (4,8)
    assert job['recipe_sha256'] == hashlib.sha256(canonical(job['plan'])).hexdigest()
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == base['bootstrap_sha256']
    params = task.get_parameters()
    assert params['General/semantic_learned_identity'] == identity
    assert params['General/recipe_sha256'] == job['recipe_sha256'] and json.loads(params['General/plan']) == job['plan']


def dispatch(args, Task, api, base):
    assert args.seed == base['seed']
    jobs = json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    assert len({j['seed'] for j in jobs}) == len(jobs)
    prior = next((j for j in jobs if j['seed'] == args.seed), None)
    identity = semantic(base)
    name = f'RBF final-refit learned-priority full-train seed{args.seed} '+identity[:16]
    matches = Task.get_tasks(project_name=PROJECT, task_name='^'+re.escape(name)+'$')
    assert len(matches) <= 1
    if matches:
        verify_existing(matches[0], prior, base, identity)
        print(json.dumps(dict(existing_task_id=matches[0].id, status=str(matches[0].status), duplicate_not_created=True)))
        return
    assert prior is None, 'existing failed or missing attempt cannot be replaced automatically'
    others, missing = reservations()
    if missing:
        print(json.dumps(dict(status='remaining_main_and_fixed_baselines_have_priority', missing=missing, task_created=False, ETA='unknown')))
        return
    def choices():
        candidates, fleet = snapshot(api, others+jobs)
        workers = {w['id']:w for w in fleet['workers']}
        return [c for c in candidates if c['world_size'] in (4,8)
                and all('L40' not in workers[w]['id'].upper() for w in c['eligible_workers'])], fleet
    candidates, fleet = choices()
    if not candidates:
        print(json.dumps(dict(status='waiting_for_disjoint_idle_GPU', task_created=False, ETA='unknown')))
        return
    choice = candidates[0];plan = dict(base, world_size=choice['world_size'])
    recipe = hashlib.sha256(canonical(plan)).hexdigest()
    if not args.execute:
        print(json.dumps(dict(seed=args.seed, choice=choice, recipe_sha256=recipe, execute=False, task_created=False)))
        return
    intent = R/'receipts'/f'rbf-final-learned-seed{args.seed}-{identity[:16]}-create-intent.json'
    new(intent, dict(semantic_learned_identity=identity, plan=plan, recipe_sha256=recipe,
                     queue=choice['queue_name'], command=sys.argv, source_sha256=sha(__file__),
                     task_id_unknown_until_create_returns=True, automatic_retry=False))
    register(intent, 'rbf-final-learned-task-create-intent')
    task = Task.create(project_name=PROJECT, task_name=name, task_type=Task.TaskTypes.inference, binary='python3.12')
    job = dict(seed=args.seed, task_id=task.id, allocation_variant=VARIANT, semantic_learned_identity=identity,
        recipe_sha256=recipe, plan=plan, queue=choice['queue_name'], bindings=choice['bindings'],
        eligible_workers=choice['eligible_workers'], fleet=fleet, priority_publication=str(args.priority_publication),
        priority_publication_sha256=sha(args.priority_publication), status='created_before_configuration',
        ETA='unknown_until_actual_task_progress')
    def save():
        now = datetime.datetime.now(datetime.timezone.utc)
        value = dict(kind='rbf_final_refit_learned_priority_GPU_dispatch_v1', jobs=jobs+[job], checked_at_utc=now.isoformat(),
            source_freeze_sha256=sha(Path(__file__).resolve().parent/'source-freeze.json'), GPU_model_restriction=False,
            L40S_CPU_only=True, TF32_enabled=False, actual_memory_utilization_admitted=False,
            learned_replay_accepted=False, same_resource_performance_accepted=False, paper_performance_complete=False)
        atomic(JOURNAL, value)
        path = R/'receipts'/('rbf-final-learned-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(path, value);register(path, value['kind'])
    save()
    try:
        task.output_uri = DESTINATION
        task.set_script(repository='', branch='', commit='', working_dir='.', entry_point='bootstrap.py',
                        diff=(PRODUCER/'bootstrap.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE, docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(plan=canonical(plan).decode(), recipe_sha256=recipe, semantic_learned_identity=identity,
            seed=args.seed, world_size=choice['world_size'], GPU_model_restriction=False, TF32_enabled=False,
            memory_target_percent='75-80', artificial_memory_padding=False,
            learned_replay_accepted=False, ETA='unknown_until_actual_task_progress', paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse', 'final-refit-full-train', 'all-class-top64',
                       'learned-priority', 'priority-checkpoint-independently-admitted', 'full-replay-admission-pending'])
        task.reload();verify_existing(task, job, base, identity)
        current, _ = choices()
        assert choice in current, 'physical GPU binding changed; created task preserved'
        Task.enqueue(task, queue_id=choice['queue_id']);task.reload();job['status'] = str(task.status);save()
        print(json.dumps(dict(task_id=task.id, seed=args.seed, status=str(task.status), queue=choice['queue_name'],
                             world_size=choice['world_size'], learned_replay_accepted=False)))
    except BaseException as error:
        job['dispatch_failure_type'] = type(error).__name__;job['automatic_retry'] = False;save();raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(1337,2027,3407), required=True)
    for name in ('run', 'numeric-audit', 'teacher-journal', 'published-prerequisite', 'runtime-admission', 'priority-publication'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true');args = parser.parse_args();control = source_gate()
    if not args.priority_publication.is_file() or not args.numeric_audit.is_file() or not (args.run/'completion.json').is_file():
        print(json.dumps(dict(status='waiting_for_real_priority_fit_audit_and_independent_publication', task_created=False, ETA='unknown')))
        return
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    paths, context = qualify(args, Task)
    publication = verify_publication(args.priority_publication, paths, context, Task)
    main_task = Task.get_task(task_id=context['upstream_tasks']['main']['task_id'])
    main_plan = json.loads(main_task.get_parameters()['General/plan'])
    assert hashlib.sha256(canonical(main_plan)).hexdigest() == context['upstream_tasks']['main']['recipe_sha256']
    base = expected_plan(main_plan, publication, control, sha(PRODUCER/'bootstrap.py'), sha(__file__))
    if not args.execute:
        dispatch(args, Task, APIClient(), base);return
    with (R/'receipts/rbf-final-learned-priority-dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args, Task, APIClient(), base)


if __name__ == '__main__':main()
