"""Portable final-model CPU export/fit after complete independent teacher proof.

No tasks are created. The original optimizer, model and fit entry stay frozen.
The distinct source consumer needs its own Linux runtime admission before use.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

PREPARATION_SHA='0579fffe9cd61217d238fe8e0dc87aa222ae533944be349a7383f895875c09ea'
TEACHER_BOOTSTRAP='c847e3bef79402d7f8cf755530091f3a25601e07e7295c910f8aaefab8c01359'
MAIN_BOOTSTRAP='e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
COLLECTOR_SHA='6186a90746ee0439a7a4815fa0747fc7b322b607b8bc9bd990045fb542f0ec98'
RUNTIME_KIND='rbf_final_refit_priority_Linux_CPU_runtime_independent_byte_admission_v1'
EXPORTER='tools/event_track_v2x/train_exclusive_paper_priority.py'
GATE='transvision/models/event_track_v2x/exclusive_priority_admission.py'
TRAINER='transvision/models/event_track_v2x/exclusive_allocation_training.py'
TRAINER_SHA='e61f2803d3e50e85c409659821e3395a38f2c7aadae8d831efb377b335610adf'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical(value)+b'\n')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_gate(consumer):
    consumer=Path(consumer)
    require(sha(consumer/'preparation.json')==PREPARATION_SHA,'final consumer preparation changed')
    proof=json.loads((consumer/'preparation.json').read_bytes())
    source=consumer/'source'
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    require(actual==set(proof['sources']),'complete frozen consumer files required')
    for name,entry in proof['sources'].items():
        p=source/name
        require(not any(v.is_symlink() for v in (p,*p.parents)),'symlink source refused')
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'consumer source differs')
    require(sha(source/TRAINER)==TRAINER_SHA,'original optimizer/model/fit/checkpoint source required')
    return source,proof


def registered(task,entries):
    require(set(task.artifacts)==set(entries),'complete registered artifact set required')
    for name,item in entries.items():
        actual=task.artifacts[name]
        require(actual.hash==item['sha256'] and actual.size==item['bytes'],'registered bytes changed')


def completed(get_task,task_id):
    task=get_task(task_id);task.reload()
    require(str(task.status)=='completed','completed upstream task required: '+task_id)
    return task


def live_provenance(get_task,job,main,byte,target,binding,published):
    """Read-only task checks; no collector execution or numerical replay."""
    plan=job['plan']
    require(job['allocation_variant']=='final_refit_exclusive_capacity_raw_witness_teacher_v1','final teacher variant required')
    require(job['task_id']==target['task_id']==binding['task_id'],'teacher task differs')
    require(plan==binding['original_plan'] and hashlib.sha256(canonical(plan)).hexdigest()==job['recipe_sha256']==target['recipe_sha256'],
        'teacher original registered plan differs')
    teacher=completed(get_task,job['task_id'])
    require(hashlib.sha256(teacher.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']==TEACHER_BOOTSTRAP,'teacher runtime changed')
    params=teacher.get_parameters()
    require(params['General/recipe_sha256']==job['recipe_sha256'] and json.loads(params['General/plan'])==plan,'registered teacher plan differs')
    registered(teacher,target['registered_artifacts'])
    task=completed(get_task,main['task_id']);params=task.get_parameters()
    original=json.loads(params['General/plan'])
    require(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==original['bootstrap_sha256']==MAIN_BOOTSTRAP,'main runtime changed')
    require(params['General/recipe_sha256']==byte['recipe_sha256']==hashlib.sha256(canonical(original)).hexdigest(),'main plan differs')
    registered(task,byte['artifacts'])
    for name in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest','forward_outputs',
                 'source_replacements','final_refit_model_sha256','original_nested_model_sha256'):
        require(plan[name]==original[name],'main/teacher input role differs: '+name)
    require(plan['configuration']==dict(original['configuration'],allocation='teacher'),'teacher changes main search contract')
    require(plan['world_size'] in (4,8) and original['world_size'] in (4,8),'qualified GPU cardinality required')
    reference=plan['final_main_prerequisite_artifact']
    require(reference['sha256']==sha(published) and reference['bytes']==Path(published).stat().st_size,'published prerequisite differs')
    for item in [reference,*[plan[k] for k in ('source','events','checkpoint','weights_archive','cache_archive','cache_manifest')],*plan['forward_outputs']]:
        upstream=completed(get_task,item['task']);artifact=upstream.artifacts[item['key']]
        require(artifact.hash==item['sha256'] and artifact.size==item['bytes'],'registered input changed')
    return teacher


def runtime_gate(path,get_task):
    receipt=json.loads(Path(path).read_bytes())
    require(receipt['kind']==RUNTIME_KIND and receipt['source_preparation_sha256']==PREPARATION_SHA
        and receipt['bridge_sha256']==sha(__file__),'new final-consumer runtime admission required')
    for key in ('all_source_bytes_verified','entrypoints_imported','Linux_CPU_only','independent_cloud_bytes_verified'):
        require(receipt[key] is True,'missing actual runtime scope: '+key)
    require(receipt['dataset_read'] is receipt['priority_fit'] is False,'runtime qualification must be data-free')
    require(receipt['torch_version']=='2.6.0+cu124' and receipt['numpy_version']=='1.26.4','qualified numerical environment required')
    task=completed(get_task,receipt['task_id'])
    require(hashlib.sha256(task.data.script.diff.encode()).hexdigest()==receipt['bootstrap_sha256'],'runtime probe code changed')
    registered(task,receipt['registered_artifacts'])
    return receipt


def monitored(command,*,cwd,log,stage,epoch_path=None):
    started=time.monotonic();previous=None;samples=[]
    with Path(log).open('xb') as stream:
        child=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        while True:
            status=child.poll();epochs=None
            if epoch_path is not None and epoch_path.exists():
                try:
                    epochs=0
                    for line in epoch_path.read_bytes().splitlines():
                        try:row=json.loads(line)
                        except ValueError:continue
                        require(row.get('epoch')==epochs+1,'noncontiguous epoch progress')
                        epochs+=1
                except (OSError,ValueError,TypeError):epochs=None
            elapsed=time.monotonic()-started;eta=None
            if epochs is not None and epochs>0:
                if not samples or samples[-1][0]!=epochs:samples.append((epochs,elapsed))
                rate=elapsed/epochs
                if len(samples)>=2:
                    first,last=samples[max(0,len(samples)-4)],samples[-1]
                    rate=max(rate,(last[1]-first[1])/(last[0]-first[0]))
                eta=max(0.,(10-epochs)*rate)
            marker=(status,epochs,int(elapsed//30))
            if marker!=previous:
                print(json.dumps(dict(stage=stage,completed_epochs=epochs,total_epochs=10 if epoch_path else None,
                    elapsed_seconds=elapsed,ETA_seconds=eta,ETA_scope='current seed fit only; export and independent admission excluded',
                    ETA_reason='actual completed epochs' if eta is not None else 'task-level progress unavailable')),flush=True)
                previous=marker
            if status is not None:
                if status!=0:raise subprocess.CalledProcessError(status,command)
                return elapsed
            time.sleep(1)


def main():
    sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('consumer','cohort','main-admission','main-byte-admission','teacher-target','teacher-journal',
                 'published-prerequisite','official-schedule','cache-root','original-event-envelope','runtime-admission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name,value in vars(args).items():setattr(args,name,value.absolute())
    source,preparation=source_gate(args.consumer)
    gate=load_module('final_priority_consumer_gate',source/GATE)
    root=args.cohort;receipt_path=root/'receipt.json';receipt=json.loads(receipt_path.read_bytes())
    plan=json.loads((root/'plan.json').read_bytes());binding_path=root/'collection-binding.json'
    require(receipt['files']['collection-binding.json']==sha(binding_path),'collection binding changed')
    gate.verify(plan,sha(receipt_path),main=args.main_admission,main_sha256=sha(args.main_admission),
        byte=args.main_byte_admission,byte_sha256=sha(args.main_byte_admission),target=args.teacher_target,
        target_sha256=sha(args.teacher_target),collection_binding=binding_path,collection_binding_sha256=sha(binding_path))
    target=json.loads(args.teacher_target.read_bytes());binding=json.loads(binding_path.read_bytes())
    byte=json.loads(args.main_byte_admission.read_bytes());main_proof=json.loads(args.main_admission.read_bytes())
    collected_path=root/'independent-byte-cohort-collection.json';collected=json.loads(collected_path.read_bytes())
    require(sha(collected_path)==target['byte_cohort_sha256'] and collected['collector_sha256']==COLLECTOR_SHA
        and collected['collection_receipt_sha256']==sha(receipt_path)
        and collected['task_id']==target['task_id'] and collected['registered_artifacts']==target['registered_artifacts'],
        'qualified complete teacher collection required')
    require(sha(args.published_prerequisite)==target['published_prerequisite_sha256']==binding['published_prerequisite_sha256']
        and json.loads(args.published_prerequisite.read_bytes())==target['main_prerequisite'],'original main prerequisite differs')
    jobs=[j for j in json.loads(args.teacher_journal.read_bytes())['jobs'] if j['task_id']==target['task_id']]
    require(len(jobs)==1,'one admitted teacher attempt required')
    require(sha(args.original_event_envelope)==jobs[0]['plan']['events']['sha256'],'original event envelope differs')
    from clearml import Task
    get_task=lambda task_id:Task.get_task(task_id=task_id)
    live_provenance(get_task,jobs[0],main_proof,byte,target,binding,args.published_prerequisite)
    runtime_gate(args.runtime_admission,get_task)
    require(platform.system()=='Linux','qualified Linux CPU runtime required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np
    import torch
    require(torch.__version__=='2.6.0+cu124' and np.__version__=='1.26.4' and not torch.cuda.is_initialized(),'qualified CPU numerical runtime required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=args.output
    require(not output.exists() and not any(p.is_symlink() for p in (output,*output.parents)),'new nonsymlink output required')
    output.mkdir(parents=True,exist_ok=False);local=output/'source';shutil.copytree(source,local)
    for name,entry in preparation['sources'].items():require(sha(local/name)==entry['sha256'],'copied consumer bytes differ')
    driver=local/EXPORTER;data=output/'data';fit=output/'fit';seed=target['seed']
    export=[sys.executable,str(driver),'export','--replay',str(root),'--receipt-sha256',sha(receipt_path),
        '--output',str(data),'--official-schedule',str(args.official_schedule),'--cache-root',str(args.cache_root),
        '--original-event-envelope',str(args.original_event_envelope)]
    for name,path in (('main-admission',args.main_admission),('main-byte-admission',args.main_byte_admission),('target-admission',args.teacher_target)):
        export+=['--'+name,str(path),'--'+name+'-sha256',sha(path)]
    record=dict(kind='rbf_final_refit_original_paired_priority_CPU_export_fit_v1',seed=seed,
        source_preparation_sha256=PREPARATION_SHA,bridge_sha256=sha(__file__),teacher_task_id=target['task_id'],
        main_task_id=main_proof['task_id'],teacher_target_sha256=sha(args.teacher_target),teacher_receipt_sha256=sha(receipt_path),
        runtime_admission_sha256=sha(args.runtime_admission),CPU_threads=1,CPU_only=True,epochs=10,
        model_optimizer_fit_checkpoint_source_sha256=TRAINER_SHA,strict_pipeline_isolated_selection=False,
        capacity_targets_are_model_decision_bounds=True,exact_identity_loss_claimed=False,paper_eligible=False,
        export_command=export,actual_checkpoint_independently_admitted=False)
    write(output/'plan.json',record)
    try:
        export_seconds=monitored(export,cwd=local,log=output/'export.log',stage='full_teacher_export')
        command=[sys.executable,str(driver),'fit','--data',str(data),'--manifest-sha256',sha(data/'manifest.json'),
            '--output',str(fit),'--epochs','10','--seed',str(seed)]
        write(output/'fit-command.json',command)
        fit_seconds=monitored(command,cwd=local,log=output/'fit.log',stage='priority_fit',epoch_path=fit/str(seed)/'epochs.jsonl')
        source_gate(args.consumer)
        for name,entry in preparation['sources'].items():require(sha(local/name)==entry['sha256'],'fit source changed')
        write(output/'completion.json',dict(record,status='completed_pending_independent_checkpoint_admission',
            export_seconds=export_seconds,fit_seconds=fit_seconds,training_manifest_sha256=sha(data/'manifest.json'),
            checkpoint_sha256=sha(fit/str(seed)/'checkpoint.json'),weights_sha256=sha(fit/str(seed)/'weights.npz'),
            full_Stage2_complete=False,paper_performance_complete=False))
    except BaseException as error:
        write(output/'failure.json',dict(error_type=type(error).__name__,automatic_retry=False,
            actual_checkpoint_independently_admitted=False,full_Stage2_complete=False))
        raise


if __name__=='__main__':main()
