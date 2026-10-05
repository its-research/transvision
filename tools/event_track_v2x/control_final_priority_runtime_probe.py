"""Deduplicate, dispatch and independently read the final consumer CPU probe.

The only outgoing source payload is the frozen import/CLI consumer archive;
no observations, database, predictions, checkpoint or dataset is included.
"""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re

from rbf_nested_seen_val_v2_common import R,new,register,sha

D=Path(__file__).resolve().parent
J=R/'receipts/rbf-final-refit-priority-Linux-CPU-runtime-probe-v1-dispatch-20261005.json'
OUT=R/'artifacts/rbf-final-refit-priority-Linux-CPU-runtime-probe-v1-20261005'
Q='8d0f8b54037249eeb0f1cc70cbfe73ab'
FILES='http://10.100.34.118:8081'
IMAGE='gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb'
PROJECT='Thesis/Recover-Before-Fuse/Verification'
ORIGINAL=dict(task='49176387d324412a9ce9aa7799334f50',key='source',bytes=445539,
    sha256='038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03')
WHEELS_RECEIPT=R/'artifacts/rbf-priority-consumer-Linux-CPU-runtime-wheels-v1-20261002/independent-cloud-readback/acceptance.json'
WHEELS_RECEIPT_SHA='269309822f4f720b3e4b4f715c9ea73a091c464a250482864032822d21152106'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def source_gate():
    freeze=json.loads((D/'source-freeze.json').read_bytes())
    for name,digest in freeze['sources'].items():assert sha(D/name)==digest
    preparation=json.loads((D/'preparation.json').read_bytes())
    assert preparation['bootstrap_sha256']==sha(D/'bootstrap.py')
    ast.parse((D/'bootstrap.py').read_text())
    return preparation


def check_asset(get_task,asset):
    task=get_task(asset['task']);task.reload()
    assert str(task.status)=='completed'
    actual=task.artifacts[asset['key']]
    assert actual.hash==asset['sha256'] and actual.size==asset['bytes']
    return actual


def make_plan(preparation,get_task):
    assert sha(WHEELS_RECEIPT)==WHEELS_RECEIPT_SHA
    wheels=json.loads(WHEELS_RECEIPT.read_bytes())
    assert wheels['all_registered_bytes_independently_verified'] is True and wheels['exact_source_members_verified']==4
    assets={k:dict(task=wheels['task_id'],key=k,**v) for k,v in wheels['artifacts'].items()}
    assert set(assets)=={'wheels','manifest'}
    for asset in (ORIGINAL,*assets.values()):check_asset(get_task,asset)
    keys=('source_preparation_sha256','bridge_sha256','source_role_audit_sha256','full_source_member_count',
          'embedded_source_sha256','embedded_manifest_sha256','consumer_member_count','bootstrap_sha256')
    return dict({k:preparation[k] for k in keys},original_source=ORIGINAL,runtime_wheels=assets['wheels'],
        runtime_wheel_manifest=assets['manifest'],runtime_wheel_byte_admission_sha256=WHEELS_RECEIPT_SHA,
        offline_runtime_versions={'cryptography':'46.0.3','cffi':'2.0.0','pycparser':'2.23','pip':'24.3.1'},
        cryptography_version='46.0.3',runtime_packages=['clearml==2.1.2'],CPU_only=True,dataset_read=False,
        priority_training_executed=False,full_Stage2_complete=False,paper_performance_complete=False)


def idle_workers(queue,workers,now):
    assert queue['name']=='GPU3-L40S'
    if queue.get('entries'):return []
    return [w for w in workers if 'L40S' in w['id'] and not (w.get('task') or {}).get('id')
        and any(q['id']==Q for q in w.get('queues',[]))
        and 0 <= (now-datetime.datetime.fromisoformat(str(w['last_activity_time']).replace('Z','+00:00'))).total_seconds()<90]


def same_task(task,plan,recipe):
    task.reload()
    assert task.get_parameters()['General/recipe_sha256']==recipe
    assert json.loads(task.get_parameters()['General/plan'])==plan
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']


def dispatch(execute):
    from clearml import Task
    from clearml.backend_api.session.client import APIClient
    lock=J.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prep=source_gate();get_task=lambda task_id:Task.get_task(task_id=task_id)
    plan=make_plan(prep,get_task);encoded=canonical(plan);recipe=hashlib.sha256(encoded.encode()).hexdigest()
    name='RBF final-refit priority consumer Linux CPU no-data imports '+recipe[:16]
    matches=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    assert len(matches)<=1
    if J.exists():
        row=json.loads(J.read_bytes());assert row['recipe_sha256']==recipe and row['plan']==plan
        task=get_task(row['task_id']);task.reload()
        assert not matches or matches[0].id==task.id
        if row.get('configuration_verified'):same_task(task,plan,recipe)
        print(json.dumps(dict(task_id=task.id,status=str(task.status),existing_attempt_preserved=True,automatic_retry=False)))
        return
    assert not matches,'an unjournaled matching task must be recovered, never duplicated'
    api=APIClient();queue=api.queues.get_by_id(queue=Q).to_dict();workers=[w.to_dict() for w in api.workers.get_all()]
    idle=idle_workers(queue,workers,datetime.datetime.now(datetime.timezone.utc))
    if not execute or not idle:
        print(json.dumps(dict(ready_for_explicit_execution=bool(idle),no_Task_created=True,execute=execute,
            idle_L40S_CPU_workers=[w['id'] for w in idle],source_payload_bytes=(D/'consumer-source.tar.gz').stat().st_size,
            dataset_read=False,ETA='unknown')));return
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.testing,binary='python3.12')
    row=dict(kind='rbf_final_priority_runtime_probe_dispatch_v1',task_id=task.id,plan=plan,recipe_sha256=recipe,
        dispatcher_sha256=sha(__file__),source_freeze_sha256=sha(D/'source-freeze.json'),queue=Q,image=IMAGE,
        configuration_verified=False,status='created_before_configuration',CPU_only=True,dataset_read=False,
        worker_snapshot=[dict(id=w['id'],task_id=(w.get('task') or {}).get('id')) for w in workers if 'L40S' in w['id']],
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    new(J,row)
    try:
        task.output_uri=FILES
        task.set_script(repository='',branch='',commit='',working_dir='.',entry_point='bootstrap.py',diff=(D/'bootstrap.py').read_text())
        task.set_packages(plan['runtime_packages'])
        task.set_base_docker(IMAGE,docker_arguments='--shm-size 4g -e CLEARML_AGENT_FORCE_TASK_INIT=0 -e CUDA_VISIBLE_DEVICES= -e CLEARML_FILES_HOST='+FILES+' -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e NVIDIA_TF32_OVERRIDE=0')
        task.set_parameters(dict(plan=encoded,recipe_sha256=recipe,device='cpu',dataset_read=False,training_started=False,ETA='unknown until actual import stages'))
        task.add_tags(['Recover-Before-Fuse','final-refit-priority-runtime','Linux-CPU-no-data','no-paper-result'])
        same_task(task,plan,recipe);row['configuration_verified']=True
        fresh=idle_workers(api.queues.get_by_id(queue=Q).to_dict(),[w.to_dict() for w in api.workers.get_all()],datetime.datetime.now(datetime.timezone.utc))
        assert fresh,'capacity changed; keep created task and do not retry'
        Task.enqueue(task,queue_id=Q);task.reload();row['status']=str(task.status)
    except BaseException as error:
        row.update(status='preserved_dispatch_failure',error_type=type(error).__name__,automatic_retry=False)
        raise
    finally:
        temporary=J.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(J)
        register(J,'rbf-final-priority-linux-runtime-dispatch')
    print(json.dumps(dict(task_id=task.id,status=str(task.status),CPU_only=True,training_started=False,ETA='unknown')))


def validate_report(report,plan,task_id,worker):
    assert report['kind']=='rbf_final_refit_priority_Linux_CPU_import_CLI_probe_v1'
    assert report['task_id']==task_id and report['plan']==plan and 'L40S' in worker
    assert report['platform']=='Linux' and report['python_version'].startswith('3.12.')
    assert report['Torch_version']=='2.6.0+cu124' and report['numpy_version']=='1.26.4'
    assert report['CPU_threads']==report['CPU_interop_threads']==1
    assert report['CUDA_visible_devices']=='' and report['CUDA_initialized'] is False
    assert report['cryptography_version']=='46.0.3'
    assert report['offline_runtime_versions']=={k:v for k,v in plan['offline_runtime_versions'].items() if k!='pip'}
    assert report['exact_consumer_source_members_verified']==plan['consumer_member_count']==17
    assert report['full_source_member_count_verified']==plan['full_source_member_count']==135
    for key in ('training_only_source_roles_verified','all_registered_runtime_wheels_and_members_verified',
        'isolated_offline_runtime_install_succeeded','original_Torch_and_fit_model_bytes_unchanged',
        'capacity_core_exactly_replaced','capacity_target_gate_actual_imported','exact_original_core_overlay_actually_verified',
        'original_init_and_source_namespace_used_without_model_mock','all_registered_original_and_new_consumer_source_bytes_verified',
        'actual_new_exporter_full_dependency_import_succeeded','original_event_envelope_CLI_present','mutually_exclusive_arrival_modes_rejected',
        'absent_local_teacher_inputs_rejected_before_export_output','absent_training_inputs_and_invalid_seed_rejected_before_fit_output',
        'portable_bridge_actual_Linux_import_and_help_passed'):assert report[key] is True
    for key in ('actual_Linux_full_export_or_fit_executed','dataset_read','checkpoint_or_weights_read','optimizer_created',
        'neural_forward_executed','teacher_replay_executed','full_teacher_target_admission','priority_training_executed',
        'full_Stage2_complete','paper_performance_complete'):assert report[key] is False


def readback():
    from clearml import Task
    from clearml.backend_api.session import Session
    from urllib.parse import urlparse,urlunparse
    import requests
    source_gate();row=json.loads(J.read_bytes());plan=row['plan']
    assert row['configuration_verified'] is True and row['source_freeze_sha256']==sha(D/'source-freeze.json')
    assert row['dispatcher_sha256']==sha(__file__)
    task=Task.get_task(task_id=row['task_id']);same_task(task,plan,row['recipe_sha256'])
    status=str(task.status)
    if status not in ('completed','failed','stopped','closed'):
        print(json.dumps(dict(task_id=task.id,status=status,ETA='unknown',no_acceptance_written=True)));return
    success=status=='completed';key='runtime-contract' if success else 'probe-failure'
    if key not in task.artifacts:
        print(json.dumps(dict(task_id=task.id,status=status,missing_terminal_artifact=key,accepted=False,automatic_retry=False)));return
    assert set(task.artifacts)=={key}
    OUT.mkdir(exist_ok=True);asset=task.artifacts[key];path=OUT/(key+'.bytes')
    if not path.exists():
        u=urlparse(asset.url)
        if u.netloc=='10.100.35.118:8081':u=u._replace(netloc='10.100.34.118:8081')
        assert u.scheme=='http' and u.netloc=='10.100.34.118:8081'
        partial=path.with_suffix('.partial');assert not partial.exists(),'preserve live or partial reader; do not overwrite'
        with requests.get(urlunparse(u),headers={'Authorization':'Bearer '+Session().token},timeout=(10,60),stream=True,allow_redirects=False) as response:
            assert response.status_code==200
            with partial.open('xb') as stream:
                for block in response.iter_content(1024**2):stream.write(block)
        assert partial.stat().st_size==asset.size and sha(partial)==asset.hash;partial.rename(path)
    assert path.stat().st_size==asset.size and sha(path)==asset.hash
    report=json.loads(path.read_bytes());assert report['task_id']==task.id
    worker=task.data.last_worker;assert worker in {w['id'] for w in row['worker_snapshot']} and 'L40S' in worker
    proof=dict(task_id=task.id,terminal_status=status,actual_worker=worker,registered_artifacts={key:dict(sha256=asset.hash,bytes=asset.size)},
        bootstrap_sha256=plan['bootstrap_sha256'],source_preparation_sha256=plan['source_preparation_sha256'],
        bridge_sha256=plan['bridge_sha256'],recipe_sha256=row['recipe_sha256'],dispatch_receipt_sha256=sha(J),
        runtime_source_freeze_sha256=sha(D/'source-freeze.json'),verifier_sha256=sha(__file__),dataset_read=False,priority_fit=False,
        full_Stage2_complete=False,paper_performance_complete=False,actual_runtime_contract=report)
    if success:
        validate_report(report,plan,task.id,worker)
        for role in ('original_source','runtime_wheels','runtime_wheel_manifest'):
            check_asset(lambda i:Task.get_task(task_id=i),plan[role])
        proof.update(kind='rbf_final_refit_priority_Linux_CPU_runtime_independent_byte_admission_v1',
            all_source_bytes_verified=True,entrypoints_imported=True,Linux_CPU_only=True,independent_cloud_bytes_verified=True,
            torch_version=report['Torch_version'],numpy_version=report['numpy_version'],full_export_or_fit_admitted=False)
        destination=OUT/'acceptance.json'
    else:
        proof.update(kind='rbf_final_refit_priority_runtime_probe_failure_v1',automatic_retry=False);destination=OUT/'failure-preservation-receipt.json'
    if destination.exists():
        existing=json.loads(destination.read_bytes())
        assert existing['task_id']==proof['task_id'] and existing['registered_artifacts']==proof['registered_artifacts']
        print(json.dumps(dict(existing_receipt_preserved=str(destination),status=status)));return
    new(destination,proof);register(destination,'rbf-final-priority-linux-runtime-readback')
    print(json.dumps(dict(task_id=task.id,status=status,receipt=str(destination),runtime_only_admitted=success,training_started=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('dispatch','readback'));parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.action=='dispatch':dispatch(args.execute)
    else:readback()
