import datetime
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def dispatcher(monkeypatch):
    root = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    frozen = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-matching-seen-val-joint-GPU-publish-dispatch-v3-live-reservations-20261004')
    monkeypatch.syspath_prepend(str(root)); monkeypatch.syspath_prepend(str(frozen))
    spec = importlib.util.spec_from_file_location('CUDA_state_single_device_dispatch_test',root/'submit_branch_state_cuda_candidate.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fleet():
    now = datetime.datetime(2026,10,5,tzinfo=datetime.timezone.utc)
    queues = [dict(id='q1',name='GPU-1',entries=[]),dict(id='q4',name='GPU4-V100',entries=[]),dict(id='q8',name='GPU8-V100',entries=[])]
    workers = [dict(id='host:gpu'+cards,ip='10.0.0.1',queues=[dict(id=q)],task={},last_activity_time=now.isoformat())
        for cards,q in [('0','q1'),('0,1,2,3','q4'),('4,5,6,7','q4'),('0,1,2,3,4,5,6,7','q8')]]
    return workers,queues,now


def test_smallest_worker_first_with_unrestricted_GPU_model(dispatcher):
    workers,queues,now = fleet()
    choices = dispatcher.eligible(workers,queues,now)
    assert [q['worker_bound_GPU_count'] for q in choices] == [1,4,8]
    queues[1]['name'] = 'GPU4-5090'; queues[2]['name'] = 'GPU8-A100'
    assert len(dispatcher.eligible(workers,queues,now)) == 3


@pytest.mark.parametrize('cause',['eight_busy','eight_pending','reservation','stale','future','L40','disabled'])
def test_unsafe_single_device_dispatch_rejected(dispatcher,cause):
    workers,queues,now = fleet(); reserved = []
    if cause == 'eight_busy': workers[-1]['task'] = dict(id='running')
    elif cause == 'eight_pending': queues[-1]['entries'] = [dict(task='queued')]
    elif cause == 'reservation': reserved = [('10.0.0.1',set(range(8)))]
    elif cause in ('stale','future'):
        for worker in workers: worker['last_activity_time'] = (now+datetime.timedelta(seconds=-91 if cause=='stale' else 1)).isoformat()
    elif cause == 'L40':
        for worker in workers: worker['id'] = worker['id'].replace('host','L40S')
    else:
        for queue in queues: queue['tags'] = ['force_workers:off']
    assert dispatcher.eligible(workers,queues,now,reserved) == []


def test_single_device_work_does_not_allow_overlap_with_other_four_card_worker(dispatcher):
    workers,queues,now = fleet()
    workers[1]['task'] = dict(id='core-job')
    choices = dispatcher.eligible(workers,queues,now)
    assert len(choices) == 1 and choices[0]['bindings'] == [['10.0.0.1',[4,5,6,7]]]
    assert choices[0]['worker_bound_GPU_count'] == 4


def test_missing_core_seed_has_priority(dispatcher,monkeypatch,tmp_path):
    import json
    paths=[]
    for index in range(3):
        p=tmp_path/f'{index}.json'
        p.write_text(json.dumps(dict(jobs=[dict(seed=seed,task_id=f'{index}-{seed}') for seed in (1337,2027,3407) if not(index==2 and seed==3407)])))
        paths.append(p)
    monkeypatch.setattr(dispatcher,'CORE_JOURNALS',paths)
    jobs,missing = dispatcher.core_jobs()
    assert len(jobs) == 8 and missing == [dict(journal=str(paths[2]),seed=3407)]
