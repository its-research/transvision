"""ClearML producer for one exact-input, single-device CUDA execution candidate.

Consumes an independently byte-read published bundle. No GPU model restriction,
CPU fallback, synthetic allocation, retry, or paper acceptance is permitted.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlsplit


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')


def unpack_verified(archive_path,manifest_path,destination):
    manifest=json.loads(manifest_path.read_bytes())
    assert manifest['kind']=='rbf_branch_state_CUDA_exact_source_and_admitted_reference_bundle_v1'
    assert manifest['GPU_devices_required']==1 and manifest['actual_GPU_execution'] is False
    expected=dict(manifest['files'],**{'manifest.json':dict(bytes=manifest_path.stat().st_size,sha256=sha(manifest_path))})
    for name,spec in expected.items():
        p=Path(name)
        assert name and p.as_posix()==name and not p.is_absolute() and '..' not in p.parts
        assert type(spec['bytes']) is int and spec['bytes']>=0
    assert not destination.exists(), 'preserve any previous extraction'
    destination.mkdir(parents=True,exist_ok=False)
    seen=set()
    with tarfile.open(archive_path,'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            assert member.size==expected[member.name]['bytes']
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();size=0
            with archive.extractfile(member) as stream,target.open('xb') as output:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    size+=len(block);assert size<=member.size;digest.update(block);output.write(block)
            assert size==member.size and digest.hexdigest()==expected[member.name]['sha256']
            seen.add(member.name)
    assert seen==set(expected)
    return manifest


def fetch(spec,destination):
    import requests
    from clearml import Task
    from clearml.backend_api.session import Session
    source=Task.get_task(task_id=spec['task']);assert str(source.status)=='completed'
    artifact=source.artifacts[spec['key']]
    assert artifact.hash==spec['sha256'] and artifact.size==spec['bytes']
    url=urlsplit(artifact.url)
    assert url.scheme=='http' and url.netloc in ('10.100.34.118:8081','10.100.35.118:8081')
    assert not url.username and not url.password
    started=last=time.monotonic();total=0;digest=hashlib.sha256()
    with requests.get(url._replace(netloc='10.100.34.118:8081').geturl(),headers={'Authorization':'Bearer '+Session().token},
                      stream=True,timeout=(10,120),allow_redirects=False) as response:
        assert response.status_code==200
        with destination.open('xb') as output:
            for block in response.iter_content(1024**2):
                total+=len(block);assert total<=spec['bytes'];digest.update(block);output.write(block)
                now=time.monotonic()
                if now-last>=30:
                    print(json.dumps(dict(stage='CUDA_candidate_input_transfer',key=spec['key'],completed_bytes=total,
                        total_bytes=spec['bytes'],ETA_seconds=(now-started)*(spec['bytes']-total)/total,ETA_scope='current input transfer only')),flush=True)
                    last=now
    assert total==spec['bytes'] and digest.hexdigest()==spec['sha256']


def command(manifest):
    expected=['python','execution/tools/event_track_v2x/qualify_batched_branch_states_gpu.py',
        '--replay','reference','--receipt-sha256','d30d1879c65ee84cfbbefd04eabbf1acd22d7954e30f5811779530955cd307de',
        '--sequence','0000','--events','195','--recorded-reference-acceptance','reference-admission/independent-sequence-receipt.json',
        '--reference-acceptance-sha256','d104416c40b65e9e6082afc41fa74ad6f7f0d21f8a7e411ebd1420ac5bc0adbc',
        '--device','cuda:0','--max-batch','64','--output','CUDA-candidate']
    assert manifest['command_relative_to_bundle_root']==expected, 'undeclared replay command'
    return [sys.executable,*expected[1:]]


def main():
    os.environ['CLEARML_FILES_HOST']='http://10.100.34.118:8081'
    os.environ['NVIDIA_TF32_OVERRIDE']='0';os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    from clearml import Task
    task=Task.init(project_name='Thesis/Recover-Before-Fuse/Inference',task_name='RBF measured branch-state CUDA candidate',
        reuse_last_task_id=True,auto_connect_arg_parser=False,auto_connect_frameworks=False,auto_resource_monitoring=True)
    parameters=task.get_parameters();plan=json.loads(parameters['General/plan'])
    assert hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()==parameters['General/recipe_sha256']
    assert plan['recipe']=='rbf_one_complete_sequence_independent_root_state_CUDA_candidate_v1'
    assert plan['bootstrap_sha256']==sha(__file__) and plan['required_compute_GPUs']==1
    root=Path('branch-state-CUDA-work');root.mkdir(exist_ok=False)
    failure=None;result=None;device=None
    try:
        for key in ('inputs','manifest'):fetch(plan[key],root/key)
        manifest=unpack_verified(root/'inputs',root/'manifest',root/'bundle')
        assert manifest['CPU_acceptance_sha256']==plan['CPU_acceptance_sha256']
        proof=root/'bundle/CPU-candidate-admission/acceptance.json'
        assert sha(proof)==plan['CPU_acceptance_sha256']
        admitted=json.loads(proof.read_bytes())
        assert admitted['kind']=='rbf_optimized_branch_state_single_sequence_independent_correspondence_and_fresh_state_v3'
        assert admitted['events']==195 and admitted['states']==56389 and admitted['atol']==admitted['rtol']==1e-8
        assert admitted['source_sha256']==sha(root/'bundle/execution/tools/event_track_v2x/accept_optimized_branch_state_sequence.py')
        assert sha(root/'bundle/execution/witness-source-freeze.json')==manifest['portable_branch_commitment_witness_source_freeze_sha256']
        import torch
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.device_count()>=1, 'CUDA required; no CPU fallback'
        props=torch.cuda.get_device_properties(0)
        device=dict(used_device='cuda:0',used_GPU_count=1,visible_GPU_count=torch.cuda.device_count(),
            name=props.name,uuid=str(getattr(props,'uuid','unknown')),torch=torch.__version__,CUDA=torch.version.cuda)
        assert device['uuid'] not in ('unknown','None','')
        write(root/'device-before-execution.json',device)
        argv=command(manifest)
        write(root/'command.json',dict(argv=argv,working_directory='bundle',shell=False,TF32_enabled=False))
        with subprocess.Popen(argv,cwd=root/'bundle',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
            for line in process.stdout:print(line.rstrip('\n'),flush=True)
            returncode=process.wait()
        assert returncode==0, 'CUDA candidate process failed; partials preserved'
        output=root/'bundle/CUDA-candidate';result=json.loads((output/'candidate-check.json').read_bytes())
        measurement=json.loads((output/'GPU-runtime-measurement.json').read_bytes())
        assert result['events']==195 and result['device']['GPU_executed'] is True
        assert result['complete_reference_sequence_checked'] is True
        assert result['metrics']['candidate']['batch_calls']>0 and result['metrics']['candidate']['batch_rows']>0
        assert result['materialized_branch_states_checked']==56389 and result['discrete_actions_factors_work_identical'] is True
        assert result['numerical_atol']==result['numerical_rtol']==1e-8
        assert measurement['completed_events']==195 and measurement['replay_failure'] is None
        assert measurement['TF32_matmul'] is measurement['TF32_cudnn'] is False
        assert measurement['GPU_uuid']==device['uuid'] and measurement['sampler_finished'] is True
        assert not measurement['sampler_errors'] and measurement['peak_process_tensor_allocated_bytes']>0
        # Preserve exact branch projections for cross-platform verification;
        # NumPy/BLAS variants can differ in bytes while agreeing numerically.
        witness=[sys.executable,'execution/tools/event_track_v2x/export_branch_state_commitment_witness.py',
            '--database','CUDA-candidate/candidate.sqlite','--database-sha256',result['database_sha256']['candidate'],
            '--output','CUDA-candidate/branch-commitment-witness.jsonl']
        with subprocess.Popen(witness,cwd=root/'bundle',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
            for line in process.stdout:print(line.rstrip('\n'),flush=True)
            witness_returncode=process.wait()
        assert witness_returncode==0, 'branch commitment export failed; candidate retained'
        witness_receipt=json.loads((output/'branch-commitment-witness.receipt.json').read_bytes())
        assert witness_receipt['events']==195 and witness_receipt['all_branch_commitments_reproduced'] is True
        assert witness_receipt['database_sha256']==result['database_sha256']['candidate']
        assert sha(output/'branch-commitment-witness.jsonl')==witness_receipt['witness_sha256']
    except BaseException as error:
        failure=dict(exception_type=type(error).__name__,automatic_retry=False,partials_preserved=True)
        raise
    finally:
        output=root/'bundle/CUDA-candidate'
        files={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()} if output.exists() else {}
        receipt=root/'receipt.json'
        write(receipt,dict(kind='rbf_measured_branch_state_CUDA_complete_sequence_candidate_v1',task_id=task.id,plan=plan,
            device=device,failure=failure,candidate_checks_completed=result is not None and failure is None,output_files=files,
            full_forest_on_GPU=False,independent_GPU_acceptance=False,full_cohort_accepted=False,
            memory_target_percent=[75,80],memory_target_accepted=False,paper_performance_complete=False))
        archive_path=root/'candidate-output.tar.gz'
        with tarfile.open(archive_path,'x:gz') as archive:
            for name in ('device-before-execution.json','command.json','receipt.json'):
                path=root/name
                if path.exists():archive.add(path,arcname=name)
            if output.exists():archive.add(output,arcname='CUDA-candidate')
        assert task.upload_artifact('candidate-output',artifact_object=archive_path,wait_on_upload=True)
        assert task.upload_artifact('receipt',artifact_object=receipt,wait_on_upload=True)
        task.flush(wait_for_uploads=True)
    task.close()


if __name__=='__main__':main()
