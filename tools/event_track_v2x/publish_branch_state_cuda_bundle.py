"""Publish one admitted CUDA input bundle and independently read its bytes.

This creates a data task, never a GPU experiment. An interrupted or failed
publication is preserved for explicit recovery; no duplicate task is created.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R, new, register, sha
from read_rbf_final_refit_top1_outputs import read_artifact

PROJECT = 'Thesis/Recover-Before-Fuse/Inference'


def specification(package):
    assert package.resolve().is_relative_to(R/'artifacts') and not package.is_symlink()
    proof = json.loads((package/'package-readback.json').read_bytes())
    assert proof['kind'] == 'rbf_branch_state_CUDA_input_package_independent_stream_readback_v1'
    assert proof['full_archive_stream_hashes_match'] is True
    assert sha(package/'inputs.tar.gz') == proof['archive_sha256']
    assert (package/'inputs.tar.gz').stat().st_size == proof['archive_bytes']
    assert sha(package/'manifest.json') == proof['manifest_sha256']
    manifest = json.loads((package/'manifest.json').read_bytes())
    assert manifest['kind'] == 'rbf_branch_state_CUDA_exact_source_and_admitted_reference_bundle_v1'
    assert manifest['CPU_acceptance_sha256'] == proof['CPU_acceptance_sha256']
    assert manifest['GPU_devices_required'] == 1 and manifest['actual_GPU_execution'] is False
    assert proof['members'] == len(manifest['files'])+1
    files = {key:package/name for key,name in [('inputs','inputs.tar.gz'),('manifest','manifest.json'),('local-package-readback','package-readback.json')]}
    specs = {key:dict(sha256=sha(path),bytes=path.stat().st_size) for key,path in files.items()}
    identity = hashlib.sha256(json.dumps(specs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return files, specs, identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    frozen=Path(__file__).resolve().parent
    freeze=json.loads((frozen/'source-freeze.json').read_bytes())
    for name,record in freeze['sources'].items():
        assert sha(frozen/name)==record['sha256'] and (frozen/name).stat().st_size==record['bytes']
    files,specs,identity=specification(args.package)
    if not args.execute:
        print(json.dumps(dict(identity=identity,artifacts=specs,publication_requested=False,GPU_task_created=False)),flush=True)
        return
    os.environ['CLEARML_FILES_HOST']='http://10.100.34.118:8081'
    from clearml import Task
    from clearml.storage.helper import _HttpDriver
    container=_HttpDriver._Container(name='http://10.100.34.118:8081')
    assert container._should_attach_auth_header() and 'Authorization' in container.get_headers(None)
    name='RBF admitted branch-state CUDA candidate inputs '+identity[:16]
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches)<=1
    if matches or args.output.exists():
        accepted=args.output/'independent-publication.json'
        assert len(matches)==1 and accepted.is_file(), 'preserve existing publication; inspect before recovery'
        proof=json.loads(accepted.read_bytes());task=matches[0]
        assert proof['task_id']==task.id and proof['identity']==identity
        assert str(task.status)=='completed' and proof['full_cloud_bytes_independently_read'] is True
        assert set(task.artifacts)==set(specs)
        for key,spec in specs.items():
            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
            assert sha(args.output/'independent-cloud-bytes'/key)==spec['sha256']
        print(json.dumps(dict(task_id=task.id,existing_receipt=str(accepted),not_repeated=True)),flush=True)
        return
    assert args.output.resolve().is_relative_to(R/'artifacts')
    args.output.mkdir(parents=True,exist_ok=False)
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.data_processing)
    created=args.output/'task-created-before-upload.json'
    new(created,dict(task_id=task.id,identity=identity,artifacts=specs,package=str(args.package),
        command=['python',str(Path(__file__).resolve()),'--package',str(args.package),'--output',str(args.output),'--execute'],
        source_sha256=sha(__file__),source_freeze_sha256=sha(frozen/'source-freeze.json'),
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_task_created=False))
    register(created,'rbf-branch-state-CUDA-input-publication-created')
    try:
        task.output_uri='http://10.100.34.118:8081'
        task.set_parameters(dict(input_publication_identity=identity,source_sha256=sha(__file__),
            GPU_devices_required=1,actual_GPU_execution=False,paper_performance_complete=False))
        task.add_tags(['Recover-Before-Fuse','branch-state-CUDA-measurement-input','independently-admitted-CPU-reference','diagnostic-input-only'])
        for key,path in files.items():assert task.upload_artifact(key,artifact_object=path,wait_on_upload=True)
        task.reload();assert set(task.artifacts)==set(specs)
        readback=args.output/'independent-cloud-bytes';readback.mkdir(exist_ok=False)
        for key,spec in specs.items():
            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
            assert read_artifact(task,key,readback/key)==spec
        task.mark_completed(force=True);task.reload();assert str(task.status)=='completed'
        proof=args.output/'independent-publication.json'
        new(proof,dict(kind='rbf_branch_state_CUDA_input_full_independent_cloud_byte_readback_v1',
            task_id=task.id,identity=identity,artifacts={k:dict(task=task.id,key=k,**v) for k,v in specs.items()},
            package_readback_sha256=sha(args.package/'package-readback.json'),
            CPU_acceptance_sha256=json.loads((args.package/'manifest.json').read_bytes())['CPU_acceptance_sha256'],
            source_freeze_sha256=sha(frozen/'source-freeze.json'),full_cloud_bytes_independently_read=True,
            actual_GPU_execution=False,GPU_task_created=False,production_promotion_allowed=False))
        register(proof,'rbf-branch-state-CUDA-input-independent-publication')
        print(json.dumps(dict(task_id=task.id,receipt=str(proof),GPU_task_created=False)),flush=True)
    except BaseException as error:
        failure=args.output/'failure.json'
        new(failure,dict(task_id=task.id,identity=identity,exception_type=type(error).__name__,
            original_task_and_partials_preserved=True,automatic_retry=False,publication_accepted=False))
        register(failure,'rbf-branch-state-CUDA-input-publication-failure');raise


if __name__=='__main__':main()
