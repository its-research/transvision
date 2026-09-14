import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import probe_clearml_worker_files as probe


def test_only_small_pinned_manifest_is_read(tmp_path,monkeypatch):
    path=tmp_path/'package.json'
    path.write_text(json.dumps(dict(kind='forest_identity_ddp_package_v1',split='train',
                                   class_scope=['car'],raw_GT_included=False)))
    monkeypatch.setattr(probe,'PACKAGE_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
    artifact=SimpleNamespace(url='http://10.100.35.118:8081/package.json',get_local_copy=lambda:str(path))
    assert probe.checked_manifest(artifact)['download_verified']
    artifact.url='http://10.100.34.118:8081/package.json'
    artifact.get_local_copy=lambda:pytest.fail('must reject alternate server before SDK download')
    with pytest.raises(ValueError,match='designated'):probe.checked_manifest(artifact)


@pytest.mark.parametrize('mode',['timeout','nonzero','missing','wrong_sha','success'])
def test_child_download_is_bounded_and_cannot_promote_failure(monkeypatch,mode):
    def run(command,**kwargs):
        assert command[-1]=='--package-check' and kwargs['timeout']==45
        assert kwargs['capture_output'] and kwargs['env']['CLEARML_AGENT_FORCE_TASK_INIT']=='0'
        assert kwargs['env']['CLEARML_API_HOST']=='http://10.100.35.118:8008'
        if mode=='timeout':raise subprocess.TimeoutExpired(command,45)
        row=dict(download_verified=True,bytes=120,sha256='0'*64 if mode=='wrong_sha' else probe.PACKAGE_SHA)
        return SimpleNamespace(returncode=2 if mode=='nonzero' else 0,
            stdout='' if mode=='missing' else probe.MARKER+json.dumps(row)+'\n')
    monkeypatch.setattr(probe.subprocess,'run',run)
    assert probe.bounded_download()['download_verified'] is (mode=='success')


def test_sdk_child_does_not_print_exception_payload(monkeypatch,capsys):
    import sys
    class Task:
        @staticmethod
        def get_task(**kwargs):raise RuntimeError('secret-header-must-not-escape')
    monkeypatch.setitem(sys.modules,'clearml',SimpleNamespace(Task=Task))
    assert probe.sdk_child()==2
    output=capsys.readouterr().out
    assert 'secret-header' not in output and 'RuntimeError' in output


def test_network_checks_only_authorized_host_with_five_second_timeouts(monkeypatch):
    observed=[]
    def connect(address,timeout):
        observed.append((address,timeout))
        raise OSError(101,'network unreachable')
    monkeypatch.setattr(probe.socket,'create_connection',connect)
    result=probe.endpoint_reachability()
    assert observed==[(('10.100.35.118',8008),5),(('10.100.35.118',8081),5)]
    assert all(r['reachable'] is False and r['errno']==101 for r in result.values())
