#!/usr/bin/env python3
"""Read-only by default; --execute submits one short four-5090 file probe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
from submit_forest_identity_ddp import IMAGE,PROJECT,available


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    from clearml import Task
    from clearml.backend_api import Session
    from clearml.backend_api.session.client import APIClient
    if urlparse(Session.get_api_server_host()).hostname!='10.100.35.118':
        raise ValueError('only designated private API allowed')
    client=APIClient()
    workers=[w.to_dict() for w in client.workers.get_all(last_seen=120)]
    queues={q.id:q.to_dict() for q in client.queues.get_all()}
    ready=[r for r in available(workers,queues) if r['family']=='5090']
    print(json.dumps(dict(idle_four_5090_workers=ready,training_submitted=False)),flush=True)
    if not args.execute:return
    source=Path(__file__).with_name('probe_clearml_worker_files.py').read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    name='RBF four-5090 file-readiness 20260913 '+digest[:12]
    existing=Task.get_tasks(project_name=PROJECT,task_name='^'+re.escape(name)+'$')
    if existing:
        print(json.dumps(dict(existing_tasks=[dict(id=t.id,status=str(t.status)) for t in existing],
                              duplicate_not_created=True)),flush=True);return
    if not ready:raise ValueError('idle non-overlapping four-5090 worker required')
    task=Task.create(project_name=PROJECT,task_name=name,task_type=Task.TaskTypes.testing,binary='python3.12')
    print('RBF_FILES_PROBE_CREATED '+json.dumps(dict(task_id=task.id,source_sha256=digest)),flush=True)
    task.set_script(repository='',branch='',commit='',working_dir='.',
                    entry_point='probe_clearml_worker_files.py',diff=source.decode())
    task.set_packages(['clearml==2.1.2','cryptography==46.0.5'])
    task.set_base_docker(IMAGE,docker_arguments='-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 1g '
        '--env NVIDIA_DRIVER_CAPABILITIES=compute --env CLEARML_FILES_HOST=http://10.100.35.118:8081')
    task.set_parameters(dict(probe_source_sha256=digest,required_world_size=4,gpu_family='5090',
        SDK_child_timeout_seconds=45,parameter_training=False,raw_data_or_models_requested=False,
        authorization='user-requested-multi-machine-four-GPU-training-readiness-20260913',V100_excluded=True))
    task.add_tags(['Recover-Before-Fuse','worker-readiness','four-5090','not-training-results','no-dataset-read'])
    Task.enqueue(task,queue_name='GPU4-5090')
    print('RBF_FILES_PROBE_SUBMITTED '+json.dumps(dict(task_id=task.id,queue='GPU4-5090',
        status=str(Task.get_task(task_id=task.id).status),training_submitted=False)),flush=True)


if __name__=='__main__':main()
