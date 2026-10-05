"""Publish only a completed final-main prerequisite; never raw experiment data.

The default is read-only. Publication is create-once, byte-read back from the
registered ClearML artifact and reused only after exact identity checks.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R,new,register,sha
from rbf_final_refit_teacher_binding import canonical,prerequisites,PREREQUISITE_KIND

PROJECT='Thesis/Recover-Before-Fuse/Inference'
DESTINATION='http://10.100.34.118:8081'
KIND='rbf_final_refit_teacher_main_prerequisite_independent_cloud_bytes_v1'
KEY='final-main-prerequisite'
READER=R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py'
READER_SHA='d8a9542fddc0f661c723d3f8661dbc317d106d7c34f0b66c11d64c7d65a672e7'
PROOF_KEYS={'kind','seed','main_task_id','main_acceptance_sha256','byte_admission_sha256',
    'final_model_sha256','sequence_proof_sha256','main_prerequisite_verified',
    'teacher_runtime_or_targets_admitted','teacher_task_created','learned_Stage2_complete','paper_performance_complete'}


def proof_bytes(proof):
    assert set(proof)==PROOF_KEYS, 'publication contains only the declared proof fields'
    assert proof['kind']==PREREQUISITE_KIND and proof['seed'] in (1337,2027,3407)
    assert proof['main_prerequisite_verified'] is True
    for key in ('teacher_runtime_or_targets_admitted','teacher_task_created','learned_Stage2_complete','paper_performance_complete'):
        assert proof[key] is False
    assert re.fullmatch('[a-f0-9]{32}',proof['main_task_id'])
    assert len(proof['sequence_proof_sha256'])==46
    assert set(proof['sequence_proof_sha256'])=={f'sequence-{i:02d}.json' for i in range(46)}
    for value in [proof[k] for k in ('main_acceptance_sha256','byte_admission_sha256','final_model_sha256')]+list(proof['sequence_proof_sha256'].values()):
        assert re.fullmatch('[a-f0-9]{64}',value)
    return canonical(proof)+b'\n'


def source_gate():
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'source-freeze.json').read_bytes())
    for name,spec in freeze['sources'].items(): assert sha(own/name)==spec['sha256']
    for spec in freeze['references']: assert sha(spec['path'])==spec['sha256']
    assert sha(READER)==READER_SHA
    return own


def read_artifact(task,destination):
    assert sha(READER)==READER_SHA
    spec=importlib.util.spec_from_file_location('teacher_prerequisite_exact_byte_reader',READER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.read_artifact(task,KEY,destination)


def publication(path,proof,Task):
    """Check an existing registered publication without performing a new upload."""
    expected=proof_bytes(proof);checksum=hashlib.sha256(expected).hexdigest()
    value=json.loads(path.read_bytes())
    assert value['kind']==KIND and value['main_prerequisite']==proof
    assert value['full_cloud_bytes_independently_read'] is True
    assert value['identity']==checksum
    assert value['destination']==DESTINATION and value['project']==PROJECT
    assert value['raw_source_or_database_uploaded'] is False
    spec=value['artifact']
    assert spec==dict(task=value['task_id'],key=KEY,sha256=checksum,bytes=len(expected))
    task=Task.get_task(task_id=value['task_id'])
    assert str(task.status)=='completed' and set(task.artifacts)=={KEY}
    assert task.get_parameters()['General/publication_identity']==checksum
    assert task.artifacts[KEY].hash==checksum and task.artifacts[KEY].size==len(expected)
    local=path.parent/'independent-cloud-bytes'/KEY
    assert not local.is_symlink() and local.read_bytes()==expected
    return spec,local


def run(args,Task,proof):
    assert args.seed==proof['seed']
    payload=proof_bytes(proof);identity=hashlib.sha256(payload).hexdigest()
    if not args.execute:
        print(json.dumps(dict(seed=args.seed,publication_requested=False,GPU_task_created=False,
            artifact_key=KEY,bytes=len(payload),sha256=identity,destination=DESTINATION,project=PROJECT)),flush=True)
        return
    name=f'RBF final-refit teacher prerequisite seed{args.seed} '+identity[:16]
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches)<=1
    if matches or args.output.exists():
        accepted=args.output/'independent-publication.json'
        assert len(matches)==1 and accepted.is_file(), 'preserve incomplete publication for explicit recovery'
        spec,_=publication(accepted,proof,Task)
        assert spec['task']==matches[0].id
        print(json.dumps(dict(task_id=spec['task'],existing_receipt=str(accepted),not_repeated=True)),flush=True)
        return
    assert args.output.resolve().is_relative_to(R/'artifacts')
    assert not any(p.is_symlink() for p in (args.output,*args.output.parents))
    args.output.mkdir(parents=True,exist_ok=False)
    local=args.output/(KEY+'.json');local.write_bytes(payload)
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing)
    started=args.output/'task-created-before-upload.json'
    new(started,dict(task_id=task.id,identity=identity,seed=args.seed,destination=DESTINATION,project=PROJECT,
        artifacts={KEY:dict(sha256=identity,bytes=len(payload))},source_sha256=sha(__file__),
        command=sys_command(args),raw_source_or_database_uploaded=False,teacher_task_created=False))
    register(started,'rbf-final-teacher-prerequisite-publication-started')
    try:
        task.output_uri=DESTINATION
        task.set_parameters(dict(publication_identity=identity,seed=args.seed,main_task_id=proof['main_task_id'],
            source_sha256=sha(__file__),raw_source_or_database_uploaded=False,teacher_targets_admitted=False))
        task.add_tags(['Recover-Before-Fuse','final-refit-teacher-prerequisite','metadata-proof-only'])
        assert task.upload_artifact(KEY,artifact_object=local,wait_on_upload=True)
        task.reload();assert set(task.artifacts)=={KEY}
        assert task.artifacts[KEY].hash==identity and task.artifacts[KEY].size==len(payload)
        dest=args.output/'independent-cloud-bytes';dest.mkdir(exist_ok=False)
        assert read_artifact(task,dest/KEY)==dict(sha256=identity,bytes=len(payload))
        assert (dest/KEY).read_bytes()==payload
        task.mark_completed(force=True);task.reload();assert str(task.status)=='completed'
        path=args.output/'independent-publication.json'
        new(path,dict(kind=KIND,task_id=task.id,identity=identity,seed=args.seed,
            main_prerequisite=proof,destination=DESTINATION,project=PROJECT,
            artifact=dict(task=task.id,key=KEY,sha256=identity,bytes=len(payload)),
            source_freeze_sha256=sha(Path(__file__).resolve().parent/'source-freeze.json'),
            full_cloud_bytes_independently_read=True,raw_source_or_database_uploaded=False,
            teacher_task_created=False,full_teacher_target_admission=False,paper_performance_complete=False))
        register(path,'rbf-final-teacher-prerequisite-independent-publication')
        publication(path,proof,Task)
        print(json.dumps(dict(task_id=task.id,receipt=str(path),teacher_targets_admitted=False)),flush=True)
    except BaseException as error:
        path=args.output/'failure.json'
        new(path,dict(task_id=task.id,identity=identity,exception_type=type(error).__name__,
            original_task_and_partials_preserved=True,automatic_retry=False,publication_accepted=False))
        register(path,'rbf-final-teacher-prerequisite-publication-failure');raise


def sys_command(args):
    return ['python',str(Path(__file__).resolve()),'--seed',str(args.seed),'--main-admission',str(args.main_admission),
        '--main-byte-admission',str(args.main_byte_admission),'--output',str(args.output),'--execute']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for name in ('main-admission','main-byte-admission','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    source_gate()
    if not args.main_admission.is_file() or not args.main_byte_admission.is_file():
        print(json.dumps(dict(seed=args.seed,status='waiting_for_full_independent_final_main_admission',ETA='unknown',task_created=False)),flush=True);return
    proof,_,_,_=prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    if args.execute:
        os.environ['CLEARML_FILES_HOST']=DESTINATION
        from clearml.storage.helper import _HttpDriver
        container=_HttpDriver._Container(name=DESTINATION)
        assert container._should_attach_auth_header() and 'Authorization' in container.get_headers(None)
    from clearml import Task
    if args.execute:
        with (R/'receipts'/f'rbf-final-teacher-prerequisite-seed{args.seed}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args,Task,proof)
    else:run(args,Task,proof)


if __name__=='__main__':main()
