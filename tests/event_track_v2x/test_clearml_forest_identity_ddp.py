import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.event_track_v2x.run_clearml_forest_identity_ddp import download_artifact, extract, task_parameters
from tools.event_track_v2x.submit_forest_identity_ddp import MATRIX, available, reject_duplicate_seed, require_capacity, selected_jobs
from tools.event_track_v2x import submit_forest_identity_ddp as submitter


@pytest.mark.parametrize('prefix', ['', 'General/'])
def test_clearml_general_namespace_and_native_values(prefix):
    data = dict(package_task_id='package', package_sha256='0'*64, seed='1337', gpu_family='A100',
                required_world_size='4', global_batch_size='64', epochs='10', class_scope='car')
    assert task_parameters({prefix+k:v for k,v in data.items()}) == data
    with pytest.raises(ValueError, match='duplicate'):
        task_parameters(dict(data, **{'General/seed': '2027'}))
    with pytest.raises(ValueError, match='fixed training contract'):
        task_parameters(dict(data, required_world_size='1'))


def test_artifact_download_never_falls_back_to_another_host_or_credential(tmp_path):
    a = SimpleNamespace(url='http://10.100.35.118:8081/pinned.json', get_local_copy=lambda: str(tmp_path/'pinned.json'))
    assert download_artifact(a) == tmp_path/'pinned.json'
    a.get_local_copy = lambda: None
    with pytest.raises(RuntimeError, match='authenticated ClearML'):
        download_artifact(a)
    a.url='http://10.100.34.118:8081/pinned.json'
    with pytest.raises(ValueError, match='designated private'):
        download_artifact(a)


def test_retry_only_failed_5090_seeds_does_not_require_or_submit_busy_a100():
    jobs = selected_jobs((2027,3407))
    assert [s for s,_,_ in jobs] == [2027,3407]
    require_capacity([{'family':'5090'}, {'family':'5090'}], jobs)
    with pytest.raises(ValueError,match='insufficient'):
        require_capacity([{'family':'5090'}], jobs)
    for bad in ((1337,1337), (), (1,), (True,)):
        with pytest.raises(ValueError, match='distinct'):
            selected_jobs(bad)


def test_a100_fallback_queues_only_remaining_seeds_without_claiming_parallel_capacity():
    jobs=selected_jobs((2027,3407), a100_fallback=True)
    assert jobs==((2027,'A100','GPU4-A100'),(3407,'A100','GPU4-A100'))
    require_capacity([{'family':'A100'}], jobs, a100_fallback=True)
    with pytest.raises(ValueError,match='insufficient'):
        require_capacity([{'family':'5090'}], jobs, a100_fallback=True)
    with pytest.raises(ValueError,match='remaining'):
        selected_jobs((1337,), a100_fallback=True)
    with pytest.raises(ValueError,match='only four-A100'):
        require_capacity([{'family':'5090'}], MATRIX, a100_fallback=True)


@pytest.mark.parametrize('status', ['queued','in_progress','completed'])
def test_same_seed_cannot_restart_on_other_hardware_after_running_or_completion(status):
    record=dict(id='existing-seed-task',status=status)
    with pytest.raises(ValueError,match='active or completed'):
        reject_duplicate_seed([record])
    reject_duplicate_seed([record],own_task_id=record['id'])
    reject_duplicate_seed([dict(id='failed-attempt',status='failed')])


def worker(host, cards, *, busy=False, queue='four'):
    return dict(id=host+':gpu'+','.join(map(str,cards)), task={'id':'other'} if busy else None,
                queues=[dict(id=queue, name='GPU4-A100' if 'A100' in host else 'GPU4-5090')])


def test_excludes_v100_and_idle_worker_overlapping_busy_gpu_group():
    host = '10.100.34.18-A100'
    workers = [worker(host,range(4),busy=True), worker(host,range(4,8)),
               worker(host,range(8)), worker('10.100.34.26-V100',range(4))]
    queues = {'four':dict(entries=[], tags=[])}
    assert [r['cards'] for r in available(workers,queues)] == [[4,5,6,7]]
    workers.append(worker(host,range(8),busy=True))
    assert available(workers,queues) == []


def test_queued_eight_gpu_job_blocks_overlapping_four_gpu_launch():
    host = '10.100.34.130-5090'
    workers = [worker(host,range(4)),worker(host,range(4,8)),worker(host,range(8),queue='eight')]
    queues = {'four':dict(entries=[],tags=[]),'eight':dict(entries=[{'task':'other'}],tags=[])}
    assert not available(workers,queues)
    queues['eight']['entries']=[]
    assert len(available(workers,queues)) == 2
    queues['four']['tags']=['force_workers:off']
    assert not available(workers,queues)
    assert [s for s,_,_ in MATRIX] == [1337,2027,3407]
    assert all(family != 'V100' for _,family,_ in MATRIX)


@pytest.mark.parametrize('name,kind', [('../escape','file'),('/absolute','file'),('link','link'),('bad\\path','file')])
def test_runtime_archive_rejects_path_escape_and_links(tmp_path,name,kind):
    archive = tmp_path/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info = tarfile.TarInfo(name)
        if kind == 'link':
            info.type=tarfile.SYMTYPE; info.linkname='/tmp'
            tar.addfile(info)
        else:
            info.size=1; tar.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe'):
        extract(archive,tmp_path/'out',max_bytes=10)


def test_runtime_archive_extracts_only_regular_files_with_size_cap(tmp_path):
    archive=tmp_path/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('nested/source.py'); info.size=3
        tar.addfile(info,io.BytesIO(b'abc'))
    extract(archive,tmp_path/'ok',max_bytes=3)
    assert (tmp_path/'ok/nested/source.py').read_bytes()==b'abc'
    with pytest.raises(ValueError,match='size cap'):
        extract(archive,tmp_path/'too-small',max_bytes=2)


@pytest.mark.parametrize('execute', [False, True])
@pytest.mark.parametrize('remaining_only', [False, True])
def test_current_identity_cli_is_a100_only_before_any_upload(monkeypatch, tmp_path, capsys,
                                                           execute, remaining_only):
    client = SimpleNamespace(workers=SimpleNamespace(get_all=lambda **kw: []),
                             queues=SimpleNamespace(get_all=lambda: []))
    monkeypatch.setitem(sys.modules, 'clearml', SimpleNamespace(Task=object()))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api', SimpleNamespace(
        Session=SimpleNamespace(get_api_server_host=lambda: 'http://10.100.35.118:8008')))
    monkeypatch.setitem(sys.modules, 'clearml.backend_api.session.client',
                        SimpleNamespace(APIClient=lambda: client))
    ready = [dict(family=f, worker=f, queue='GPU4-'+f, cards=list(range(4)))
             for f in ('A100', '5090')]
    monkeypatch.setattr(submitter, 'available', lambda *args: ready)
    seeds = (2027, 3407) if remaining_only else (1337, 2027, 3407)
    args = ['--a100-fallback', '--seeds', '2027', '3407'] if remaining_only else []
    calls = []

    class StopBeforeUpload(Exception):
        pass

    def check_capacity(actual, selected, *, a100_fallback):
        assert actual == ready[:1]
        assert selected == tuple((seed, 'A100', 'GPU4-A100') for seed in seeds)
        assert a100_fallback is True  # One idle group permits serial execution.
        calls.append(selected)
        raise StopBeforeUpload

    monkeypatch.setattr(submitter, 'require_capacity', check_capacity)
    if execute:
        (tmp_path/'package.json').write_text(json.dumps(dict(kind='forest_identity_ddp_package_v1',
            split='train', class_scope=['car'], raw_GT_included=False, full_official_train=True,
            artifacts=[])))
        with pytest.raises(StopBeforeUpload):
            submitter.main(args + ['--execute', '--package', str(tmp_path),
                                  '--controller', str(tmp_path/'unused.py')])
    else:
        submitter.main(args)
    report = json.loads(capsys.readouterr().out)
    assert report['active_hardware_policy'] == 'A100-only'
    assert report['excluded_families'] == ['V100', '5090']
    assert report['available_four_gpu_workers'] == ready[:1]
    assert report['selected_identity_seeds'] == [[s, 'A100', 'GPU4-A100'] for s in seeds]
    assert report['snapshot_is_reservation'] is False
    assert len(calls) == int(execute)
