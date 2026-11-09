#!/usr/bin/env python3
"""Run the pinned original nine-sequence native method after exact CPU inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from clearml import Task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path, value):
    path.open('x').write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def ssh(code, payload=None, python='python3'):
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10', 'lbin@10.100.35.112',
            shlex.quote(python) + ' -c ' + shlex.quote(code)]
    return subprocess.run(argv, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# This bootstrap contains only the public native adapter and exact-input guards.
# The denied RBF CUDA branch-state and Linux priority packages are absent.
REMOTE = '''import sys,json,hashlib,os,subprocess
from pathlib import Path
p=json.loads(sys.stdin.buffer.read())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def new(path,value):path.open('x').write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\\n')
root=Path(p['CPU_root']);inputs=root/'native-inference-inputs-v1'
receipt=inputs/'input-preparation-receipt.json'
assert sha(receipt)==p['CPU_input_receipt_sha256']
v=json.loads(receipt.read_bytes())
assert v['preparation_source_sha256']==p['CPU_preparation_source_sha256']
assert v['author_commit']==p['author_commit'] and v['source_files']==9696
assert v['native_feature_members']==41895
assert not v['author_tracked_source_changed'] and not v['RBF_priority_or_CUDA_branch_state_source_transported']
assert v['qualified_Mac_val_feature_receipt_sha256']==p['qualified_Mac_val_feature_receipt_sha256']
assert v['native_feature_files']==p['qualified_val_files']
assert sha(root/'environment-lock-v3.txt')==p['environment_lock_sha256']
assert sha(root/'native-env/bin/python')==p['python_sha256']
assert v['assets']==p['expected_CPU_assets']
for a in v['assets'].values():
 assert Path(a['path']).stat().st_size==a['bytes'] and sha(a['path'])==a['sha256']
for i,(name,binding) in enumerate(v['native_feature_files'].items(),1):
 path=inputs/'native-feature-bytes'/name
 assert not path.is_symlink() and path.stat().st_size==binding['bytes'] and sha(path)==binding['sha256']
 if i%4000==0:print(json.dumps(dict(stage='independent_CPU_consumed_feature_disk_readback',members=i)),flush=True)
source=Path(v['source']);assert source==inputs/'author-source/DMSTrack'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],universal_newlines=True).strip()==p['author_commit']
assert not subprocess.check_output(['git','-C',str(source),'diff','HEAD','--'])
adapter=root/'portable-author-native-adapter-v2'
assert not adapter.exists() and not adapter.is_symlink()
adapter.mkdir()
for name,binding in p['adapter_files'].items():
 raw=binding['source'].encode();assert hashlib.sha256(raw).hexdigest()==binding['sha256']
 path=adapter/name;path.parent.mkdir(parents=True,exist_ok=True);path.open('xb').write(raw)
bindings=root/'native-inference-run-v1-bindings';bindings.mkdir(exist_ok=False)
dataset=bindings/'dataset-manifest.json';split=bindings/'split-mapping.json'
new(dataset,dict(kind='DMSTrack_actual_CPU_native_inference_input_binding_v1',task_id=p['task_id'],CPU_input_receipt_sha256=p['CPU_input_receipt_sha256'],qualified_Mac_val_feature_receipt_sha256=p['qualified_Mac_val_feature_receipt_sha256'],protocol_references=p['protocol_references'],actual_input_receipt=v,independent_CPU_consumed_feature_disk_readback=True))
split.open('x').write(p['split_mapping_source']);assert sha(split)==p['split_mapping_sha256']
def bind(path):return dict(path=str(path),sha256=sha(path))
plan=dict(method='DMSTrack',source=str(source),python=str(root/'native-env/bin/python'),table=3,original_protocol=True,
 checkpoint=bind(inputs/'model_ego_epoch_3.pth'),remote_checkpoint=bind(inputs/'model_1_epoch_3.pth'),dataset_manifest=bind(dataset),split_mapping=bind(split),environment_lock=bind(root/'environment-lock-v3.txt'),
 clearml_task_id=p['task_id'],author_commit=p['author_commit'],nominal_frequency_Hz=10,official_split='official_test',native_split_alias='val',train_test_overlap_disclosed=True,CPU_only=True,same_resource_measurement=False,native_metrics_independently_verified=False)
new(bindings/'plan.json',plan)
sys.path.insert(0,str(adapter))
from tools.event_track_v2x.run_public_baseline import run
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
out=root/'native-inference-output-v1'
preflight=run(plan,out,execute=False);new(bindings/'actual-preflight.json',preflight)
print(json.dumps(dict(stage='actual_full_native_CPU_process_start',task_id=p['task_id'],author_commit=p['author_commit'],argv=preflight['argv'],native_method_accepted=False)),flush=True)
result=run(plan,out,execute=True)
new(bindings/'native-inference-result.json',dict(task_id=p['task_id'],result=result,CPU_input_receipt_sha256=p['CPU_input_receipt_sha256'],native_metrics_independently_verified=False,native_method_accepted=False))
print(json.dumps(dict(stage='actual_full_native_CPU_process_exit',task_id=p['task_id'],status=result['status'],native_method_accepted=False)),flush=True)
assert result['status']=='native_process_completed'
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha(__file__) == plan['watcher_sha256']
    for ref in plan['references'].values():
        assert sha(ref['path']) == ref['sha256']
    output = Path(plan['output'])
    output.mkdir(exist_ok=False)
    task = None
    try:
        deadline = time.monotonic() + plan['maximum_input_wait_seconds']
        probe = "from pathlib import Path;import json;root=Path(" + repr(plan['CPU_root']) + ");p=root/'native-inference-inputs-v1/input-preparation-receipt.json';f=root/'native-inference-inputs-v1/continuation-failure-v2.json';print(json.dumps(dict(ready=p.is_file(),failed=f.is_file())))"
        while True:
            observed = ssh(probe)
            assert observed.returncode == 0, observed.stderr.decode()
            state = json.loads(observed.stdout)
            assert not state['failed'], 'Preserve the exact native input preparation failure'
            if state['ready']:
                break
            command = subprocess.check_output(['ps', '-p', str(plan['CPU_input_preparation_PID']), '-o', 'command='], text=True)
            assert plan['CPU_input_preparation_command'] in command
            assert time.monotonic() < deadline, 'Fixed input dependency wait exceeded; no cap relaxation'
            (output / 'progress.json').write_text(json.dumps(dict(stage='waiting_for_exact_qualified_CPU_input_receipt',
                actual_preparation_PID=plan['CPU_input_preparation_PID'],native_Task_created=False,native_method_accepted=False,
                observed_at_unix=time.time()), sort_keys=True) + '\n')
            time.sleep(30)
        read = ssh("from pathlib import Path;import sys;sys.stdout.buffer.write((Path(" + repr(plan['CPU_root']) + ")/'native-inference-inputs-v1/input-preparation-receipt.json').read_bytes())")
        assert read.returncode == 0, read.stderr.decode()
        CPU_input = json.loads(read.stdout)
        assert CPU_input['preparation_source_sha256'] == plan['CPU_preparation_source_sha256']
        assert CPU_input['author_commit'] == plan['author_commit']
        features = json.loads(Path(plan['references']['val_feature_admission']['path']).read_bytes())
        assert CPU_input['native_feature_files'] == {k:dict(bytes=v['bytes'],sha256=v['sha256']) for k,v in features['files'].items()}
        assert CPU_input['assets'] == plan['expected_CPU_assets']
        saved = output / 'independent-CPU-input-receipt.json'
        saved.open('xb').write(read.stdout)
        # No empty native inference Task is created before complete CPU inputs.
        name = 'DMSTrack original nine-sequence CPU inference ' + plan['execution_identity'][:16]
        assert not Task.get_tasks(project_name=plan['project'], task_name='^' + name + '$')
        task = Task.create(project_name=plan['project'], task_name=name, task_type=Task.TaskTypes.inference,
                           detect_repository=False, packages=False, add_task_init_call=False)
        task.set_script(repository='https://github.com/eddyhkchiu/DMSTrack.git', branch='', commit=plan['author_commit'],
                        working_dir='DMSTrack', entry_point='main_dkf.py', diff='')
        task.set_parameters(dict(execution_identity=plan['execution_identity'],author_commit=plan['author_commit'],
            wrapper_sha256=plan['watcher_sha256'],CPU_input_receipt_sha256=sha(saved),native_original_protocol=True,
            table=3,CPU_only=True,nominal_frequency_Hz=10,native_split_alias='val',official_split='official_test',
            train_test_overlap_disclosed=True,native_metrics_independently_verified=False,native_method_accepted=False))
        task.add_tags(['Recover-Before-Fuse','public-native','DMSTrack','CPU-only','known-overlap'])
        new(output / 'Task-created-after-exact-CPU-inputs.json',dict(task_id=task.id,execution_identity=plan['execution_identity'],
            CPU_input_receipt_sha256=sha(saved),watcher_sha256=plan['watcher_sha256'],GPU_Task_created=False,native_method_accepted=False))
        payload = {k:plan[k] for k in ['CPU_root','CPU_preparation_source_sha256','author_commit','qualified_Mac_val_feature_receipt_sha256',
            'environment_lock_sha256','python_sha256','expected_CPU_assets','protocol_references']}
        payload.update(task_id=task.id,CPU_input_receipt_sha256=sha(saved),qualified_val_files=CPU_input['native_feature_files'])
        payload['adapter_files'] = {}
        for name,binding in plan['adapter_files'].items():
            raw=Path(binding['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==binding['sha256']
            payload['adapter_files'][name]=dict(source=raw.decode(),sha256=binding['sha256'])
        split=Path(plan['references']['original_GT_native_mapping']['path']).read_bytes()
        payload.update(split_mapping_source=split.decode(),split_mapping_sha256=hashlib.sha256(split).hexdigest())
        task.mark_started(force=True)
        argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','lbin@10.100.35.112',
              shlex.quote(plan['CPU_python'])+' -u -c '+shlex.quote(REMOTE)]
        with (output/'native-process-control.log').open('xb') as log:
            result=subprocess.run(argv,input=json.dumps(payload).encode(),stdout=log,stderr=subprocess.STDOUT)
        new(output/'native-process-control-exit.json',dict(task_id=task.id,returncode=result.returncode,
            wrapper_sha256=sha(__file__),control_log_sha256=sha(output/'native-process-control.log'),
            native_metrics_independently_verified=False,native_method_accepted=False))
        fetch=ssh("from pathlib import Path;import sys;sys.stdout.buffer.write((Path("+repr(plan['CPU_root'])+")/'native-inference-run-v1-bindings/native-inference-result.json').read_bytes())")
        if fetch.returncode==0:
            (output/'native-inference-result.json').open('xb').write(fetch.stdout)
        assert result.returncode==0 and fetch.returncode==0, 'Retain original native process failure and outputs'
        value=json.loads(fetch.stdout)
        assert value['task_id']==task.id and value['result']['status']=='native_process_completed'
        assert task.upload_artifact('native-process-receipt',artifact_object=output/'native-inference-result.json',wait_on_upload=True,retries=0)
        assert task.upload_artifact('native-wrapper-control-log',artifact_object=output/'native-process-control.log',wait_on_upload=True,retries=0)
        assert task.upload_artifact('native-input-protocol-plan',artifact_object=args.plan,wait_on_upload=True,retries=0)
        task.mark_completed(force=True)
        new(output/'native-prediction-process-completion.json',dict(task_id=task.id,native_process_completed=True,
            full_native_outputs_CPU_root=plan['CPU_root']+'/native-inference-output-v1',native_output_archive_and_cloud_bytes_pending=True,
            native_metrics_independently_verified=False,native_method_accepted=False))
    except BaseException as error:
        new(output/'failure.json',dict(task_id=task.id if task else None,error_type=type(error).__name__,error=str(error),
            original_process_logs_partial_assets_and_failed_outputs_preserved=True,automatic_retry=False,native_method_accepted=False))
        if task:
            task.mark_failed(force=True)
        raise


if __name__ == '__main__':
    main()
