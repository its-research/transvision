"""Create-once K1/K4 seen-val dispatch after exact train and input admission."""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile

from rbf_nested_seen_val_v2_common import R,new,register,sha
import prepare_seen_val_fixed_baselines as producer
import publish_rbf_seen_val_forest_inputs as publication
import submit_rbf_seen_val_bound_forest as bound_dispatch
from submit_rbf_final_identity import atomic,IMAGE
from submit_rbf_seen_val_joint_identity import snapshot

PRODUCERS = R/'source-freezes/rbf-seen-val-fixed-K1-K4-GPU-producers-v1-20261005'
PRODUCERS_SHA = '6b805c219c6ea53782741634b11691d8e56c67fcd1799c16cafa4954917e4329'
PUB = R/'source-freezes/rbf-seen-val-forest-publication-dispatch-v1-20261005'
PUB_SHA = 'a2645f27e1bf516aa54552fd35e817901004bc836107f1f3dc29d0eeb0916651'
JOURNAL = R/'receipts/rbf-seen-val-fixed-K1-K4-GPU-dispatch-20261005.json'
BOUND_JOURNAL = R/'receipts/rbf-seen-val-bound-forest-GPU-dispatch-20261005.json'
PATH_ARGUMENTS = ('main_admission','main_byte_admission','publication','baseline_admission','baseline_byte_admission')
canonical = producer.canonical


def source_gate():
    directory=Path(__file__).resolve().parent
    own=json.loads((directory/'source-freeze.json').read_bytes())
    assert own['kind']=='rbf_seen_val_fixed_baseline_GPU_dispatch_source_v1'
    for name,item in own['sources'].items():assert sha(directory/name)==item['sha256']
    for item in own['references']:assert sha(item['path'])==item['sha256']
    assert sha(PRODUCERS/'source-freeze.json')==PRODUCERS_SHA
    produced=json.loads((PRODUCERS/'source-freeze.json').read_bytes())
    for name,item in produced['sources'].items():assert sha(PRODUCERS/name)==item['sha256']
    for item in produced['references']:assert sha(item['path'])==item['sha256']
    assert sha(PUB/'source-freeze.json')==PUB_SHA
    published=json.loads((PUB/'source-freeze.json').read_bytes())
    for module,reference in ((producer,produced),(publication,published),(bound_dispatch,published)):
        path=Path(module.__file__).resolve()
        assert path.parent==directory and sha(path)==reference['sources'][path.name]['sha256']
    publication.source_gate()
    return sha(directory/'source-freeze.json')


def remote_gate(plan, publication_path, Task):
    path=PRODUCERS/f"K{plan['baseline_K']}"/'bootstrap.py'
    freeze=json.loads((PRODUCERS/'source-freeze.json').read_bytes())
    assert sha(PRODUCERS/'source-freeze.json')==PRODUCERS_SHA
    assert sha(path)==freeze['sources'][str(path.relative_to(PRODUCERS))]['sha256']==plan['bootstrap_sha256']
    tree=ast.parse(path.read_text())
    functions={'seen_val_admission','cpu_contract','fixed_seen_val_admission'}
    constants={'PARENT_SHA','MAIN_FREEZE_SHA','INPUT_INDEX_SHA','WIDTH','CPU_FREEZE_SHA','BASELINE_SOURCE_SHA','BASELINE_TEMPLATES'}
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in functions) or
        (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in constants)]
    assert len(nodes)==len(functions)+len(constants)
    def fetch(spec,destination):
        local=publication_path.parent/'independent-cloud-bytes'/spec['key']
        raw=local.read_bytes()
        assert not local.is_symlink() and len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
        destination.write_bytes(raw)
    namespace=dict(Task=Task,json=json,hashlib=hashlib,canonical=canonical,fetch=fetch,write=new)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    with tempfile.TemporaryDirectory(prefix='rbf-fixed-seen-val-admission-') as directory:
        namespace['fixed_seen_val_admission'](plan,Path(directory))


def qualify(args, Task):
    source_gate()
    payload=producer.local_cpu_prerequisite(args.K,args.seed,args.baseline_admission,args.baseline_byte_admission)
    proof,main,_,_=publication.prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    records,binding=publication.local_inputs(args.seed,args.main_admission,proof,main)
    manifest=publication.manifest(args.seed,records,proof,main,Task)
    published=publication.publication(args.publication,manifest,Task)
    main_plan=publication.expected_plan(main,binding,published,4)
    base=producer.make_plan(args.K,main_plan,payload,sha(PRODUCERS/f'K{args.K}'/'bootstrap.py'),4)
    remote_gate(base,args.publication,Task)
    return base


def semantic(plan):
    return hashlib.sha256(canonical({k:v for k,v in plan.items() if k!='world_size'})).hexdigest()


def verify_existing(task, job, base):
    assert job is not None and job['task_id']==task.id
    assert job['seed']==base['seed'] and job['K']==base['baseline_K']
    assert job['allocation_variant']==f"final_refit_seen_val_fixed_K{job['K']}_v1"
    assert job['semantic_baseline_identity']==semantic(base)==semantic(job['plan'])
    assert job['recipe_sha256']==hashlib.sha256(canonical(job['plan'])).hexdigest()
    assert job['plan']==dict(base,world_size=job['plan']['world_size']) and job['plan']['world_size'] in (4,8)
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==base['bootstrap_sha256']
    params=task.get_parameters()
    assert params['General/semantic_baseline_identity']==job['semantic_baseline_identity']
    assert params['General/recipe_sha256']==job['recipe_sha256']
    assert json.loads(params['General/plan'])==job['plan']


def other_jobs():
    jobs,missing=bound_dispatch.core_jobs()
    if BOUND_JOURNAL.exists():jobs+=json.loads(BOUND_JOURNAL.read_bytes())['jobs']
    assert len({j['task_id'] for j in jobs})==len(jobs)
    return jobs,missing


def choices(api,jobs):
    options,fleet=snapshot(api,jobs)
    workers={w['id']:w for w in fleet['workers']}
    options=[o for o in options if o['world_size'] in (4,8)
        and o['eligible_workers'] and all(w in workers and 'L40' not in w.upper() for w in o['eligible_workers'])]
    return options,fleet


def dispatch(args, Task, api, base):
    assert args.seed==base['seed'] and args.K==base['baseline_K']
    jobs=json.loads(JOURNAL.read_bytes())['jobs'] if JOURNAL.exists() else []
    assert len({(j['K'],j['seed']) for j in jobs})==len(jobs)
    assert all(j['K'] in (1,4) and j['seed'] in (1337,2027,3407) for j in jobs)
    prior=next((j for j in jobs if (j['K'],j['seed'])==(args.K,args.seed)),None)
    identity=semantic(base)
    name=f'RBF final-refit complete seen-val fixed K{args.K} seed{args.seed} '+identity[:16]
    matches=Task.get_tasks(project_name=publication.PROJECT,task_name='^'+re.escape(name)+'$');assert len(matches)<=1
    if matches:
        task=matches[0];verify_existing(task,prior,base)
        print(json.dumps(dict(K=args.K,seed=args.seed,existing_task_id=task.id,status=str(task.status),duplicate_not_created=True)),flush=True)
        return
    assert prior is None, 'preserve existing created/queued/failed attempt; no automatic replacement'
    intent=R/'receipts'/f'rbf-seen-val-fixed-K{args.K}-seed{args.seed}-dispatch-intent-20261005.json'
    assert not intent.exists(), 'unknown previous creation outcome must be reconciled, never retried automatically'
    others,missing=other_jobs()
    if missing:
        print(json.dumps(dict(K=args.K,seed=args.seed,status='waiting_for_prior_core_dispatch',missing=missing,GPU_task_created=False,ETA='unknown')),flush=True);return
    options,fleet=choices(api,others+jobs)
    if not options:
        print(json.dumps(dict(K=args.K,seed=args.seed,status='waiting_for_disjoint_idle_GPU',GPU_task_created=False,ETA='unknown')),flush=True);return
    choice=options[0];plan=dict(base,world_size=choice['world_size'])
    remote_gate(plan,args.publication,Task)
    if not args.execute:
        print(json.dumps(dict(K=args.K,seed=args.seed,choice=choice,semantic_baseline_identity=identity,execute=False,GPU_task_created=False)),flush=True);return
    command=['python',str(Path(__file__).resolve()),'--K',str(args.K),'--seed',str(args.seed)]
    for key in PATH_ARGUMENTS:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    command+=['--execute']
    input_hashes={k:sha(getattr(args,k)) for k in PATH_ARGUMENTS}
    freeze_sha=sha(Path(__file__).with_name('source-freeze.json'))
    new(intent,dict(kind='rbf_seen_val_fixed_baseline_dispatch_intent_v1',seed=args.seed,K=args.K,command=command,
        semantic_baseline_identity=identity,recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
        source_freeze_sha256=freeze_sha,input_receipt_sha256=input_hashes,choice=choice,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
    register(intent,'rbf_seen_val_fixed_baseline_dispatch_intent_v1')
    task=Task.create(project_name=publication.PROJECT,task_name=name,task_type=Task.TaskTypes.inference,binary='python3.12')
    job=dict(seed=args.seed,K=args.K,task_id=task.id,allocation_variant=f'final_refit_seen_val_fixed_K{args.K}_v1',
        semantic_baseline_identity=identity,recipe_sha256=hashlib.sha256(canonical(plan)).hexdigest(),plan=plan,
        bindings=choice['bindings'],eligible_workers=choice['eligible_workers'],queue=choice['queue_name'],fleet=fleet,
        input_paths={k:str(getattr(args,k)) for k in PATH_ARGUMENTS},
        input_receipt_sha256=input_hashes,
        status='created_before_configuration',ETA='unknown_until_task_level_progress',command=command)
    def save():
        now=datetime.datetime.now(datetime.timezone.utc)
        value=dict(kind='rbf_seen_val_fixed_K1_K4_GPU_dispatch_v1',checked_at_utc=now.isoformat(),jobs=jobs+[job],
            source_freeze_sha256=freeze_sha,GPU_model_restriction=False,
            L40S_CPU_only=True,TF32_enabled=False,actual_memory_utilization_admitted=False,
            full_seen_val_baseline_admission=False,same_resource_performance_accepted=False,paper_performance_complete=False)
        atomic(JOURNAL,value)
        path=R/'receipts'/('rbf-seen-val-fixed-baseline-dispatch-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        new(path,value);register(path,value['kind'])
    save()
    try:
        task.output_uri=publication.DESTINATION
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',
            diff=(PRODUCERS/f'K{args.K}'/'bootstrap.py').read_text())
        task.set_packages(['clearml==2.1.2'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 16g -e CLEARML_AGENT_FORCE_TASK_INIT=0 '
            '-e CLEARML_FILES_HOST=http://10.100.34.118:8081 -e NCCL_P2P_DISABLE=1 '
            '-e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_TF32_OVERRIDE=0 '
            '-e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1')
        task.set_parameters(dict(plan=canonical(plan).decode(),recipe_sha256=job['recipe_sha256'],semantic_baseline_identity=identity,
            seed=args.seed,K=args.K,world_size=plan['world_size'],GPU_model_restriction=False,TF32_enabled=False,
            memory_target_percent='75-80',artificial_memory_padding=False,full_seen_val_baseline_admission=False,
            ETA='unknown_until_actual_transfer_or_forest_event_progress',paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','SPD-seen-val-exploratory','all-class-top64',f'fixed-K{args.K}',
            'full-train-interface-admitted','full-val-admission-pending'])
        task.reload();verify_existing(task,job,base)
        assert qualify(args,Task)==dict(base,world_size=4), 'local or remote prerequisite changed before enqueue'
        source_gate()
        fresh_others,fresh_missing=other_jobs();assert not fresh_missing
        fresh,_=choices(api,fresh_others+jobs)
        assert choice in fresh, 'physical GPU bindings changed; created task retained without retry'
        Task.enqueue(task,queue_id=choice['queue_id']);task.reload();job['status']=str(task.status);save()
        print(json.dumps(dict(K=args.K,seed=args.seed,task_id=task.id,queue=choice['queue_name'],status=str(task.status),full_seen_val_baseline_admission=False)),flush=True)
    except BaseException as error:
        job.update(dispatch_failure_type=type(error).__name__,automatic_retry=False);save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--K',type=int,choices=(1,4),required=True)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for key in PATH_ARGUMENTS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();source_gate()
    missing=[dict(role=k,path=str(getattr(args,k))) for k in PATH_ARGUMENTS if not getattr(args,k).is_file()]
    if missing:
        print(json.dumps(dict(K=args.K,seed=args.seed,status='waiting_for_full_train_main_baseline_and_published_val_inputs',
            missing=missing,GPU_task_created=False,ETA='unknown')),flush=True);return
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    if args.execute:
        with (R/'receipts/rbf-seen-val-fixed-baseline-dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            dispatch(args,Task,APIClient(),qualify(args,Task))
    else:dispatch(args,Task,APIClient(),qualify(args,Task))


if __name__=='__main__':main()
