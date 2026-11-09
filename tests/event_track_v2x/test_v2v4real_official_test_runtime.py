import json
import types
import pytest
from tools.event_track_v2x import v2v4real_official_test_runtime as runtime


def valid():
    return dict(interpreter=runtime.sys.executable,dependencies=[dict(distribution=n,expected=v,
        actual=v,module=m,imported=True) for n,v,m in runtime.DEPENDENCIES])


@pytest.mark.parametrize('mutation',['missing','wrong-version','import-failed','wrong-identity'])
def test_runtime_rejects_unusable_native_dependencies(mutation):
    value=valid()
    if mutation=='missing':value['dependencies'].pop()
    elif mutation=='wrong-version':value['dependencies'][7]['actual']='0'
    elif mutation=='import-failed':value['dependencies'][7]['imported']=False
    else:value['dependencies'][7]['module']='substitute'
    with pytest.raises(ValueError):runtime.require_runtime(value)


def test_install_uses_rank_interpreter_and_fresh_probe(tmp_path,monkeypatch):
    bad=valid();bad['dependencies'][7]['imported']=False
    results=iter([bad,valid()]);monkeypatch.setattr(runtime,'inspect_runtime',lambda:next(results))
    commands=[]
    def run(command,**kwargs):
        commands.append(command);assert kwargs['capture_output'] is True
        return types.SimpleNamespace(returncode=0,stdout=b'private-index-secret',stderr=b'private-index-secret')
    monkeypatch.setattr(runtime.subprocess,'run',run)
    uploads=[];task=types.SimpleNamespace(upload_artifact=lambda *a,**k:uploads.append((a,k)))
    receipt=runtime.prepare_runtime(task,tmp_path)
    assert commands==[[runtime.sys.executable,'-m','pip','install','--no-input','--disable-pip-version-check',*runtime.PACKAGES]]
    assert receipt['explicit_install_executed'] and len(uploads)==1
    assert 'private-index-secret' not in (tmp_path/'runtime-dependencies.json').read_text()


def test_failed_install_cannot_attest_runtime(tmp_path,monkeypatch):
    bad=valid();bad['dependencies'][7]['imported']=False
    monkeypatch.setattr(runtime,'inspect_runtime',lambda:bad)
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=1,stderr=b'secret'))
    with pytest.raises(RuntimeError,match='exit 1'):runtime.prepare_runtime(None,tmp_path)
    assert not (tmp_path/'runtime-dependencies.json').exists()


def test_failed_fresh_import_preserves_runtime_evidence(tmp_path,monkeypatch):
    before=valid();before['dependencies'][7]['imported']=False
    after=valid();after['dependencies'][6].update(imported=False,exception_type='OSError',missing_shared_libraries=['libGL.so.1'])
    monkeypatch.setattr(runtime,'repair_system_dependencies',lambda v,root:(v,[]))
    values=iter([before,after]);monkeypatch.setattr(runtime,'inspect_runtime',lambda:next(values))
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0))
    uploads=[];task=types.SimpleNamespace(upload_artifact=lambda *a,**k:uploads.append((a,k)))
    with pytest.raises(ValueError,match='open3d'):runtime.prepare_runtime(task,tmp_path)
    assert len(uploads)==1
    assert json.loads((tmp_path/'runtime-dependencies.json').read_text())['after']==after


def test_system_repair_reuses_only_pinned_bundle_and_reprobes(tmp_path,monkeypatch):
    broken=valid();broken['dependencies'][6].update(imported=False,missing_shared_libraries=['libX11.so.6'])
    monkeypatch.setattr(runtime.platform,'system',lambda:'Linux')
    monkeypatch.setattr(runtime.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(runtime,'inspect_runtime',valid)
    evidence={'archive_sha256':runtime.OFFLINE_ARCHIVE_SHA};calls=[]
    def materialize(root):calls.append(root);return evidence
    monkeypatch.setattr(runtime,'materialize_offline_libraries',materialize)
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:pytest.fail('must not use apt'))
    after,records=runtime.repair_system_dependencies(broken,tmp_path)
    runtime.require_runtime(after)
    assert calls==[tmp_path] and records==[evidence]


def test_unknown_missing_library_never_guesses_OS_package(monkeypatch):
    broken=valid();broken['dependencies'][6].update(imported=False,missing_shared_libraries=['libUnknown.so.1'])
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:pytest.fail('must not install guessed dependency'))
    assert runtime.repair_system_dependencies(broken,None)==(broken,[])


def test_real_existing_offline_bundle_full_member_validation(tmp_path):
    from pathlib import Path
    root=Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-offline-libgl-runtime-20261001')
    proof=runtime.unpack_offline_libraries(root/'libgl.tar.gz',root/'manifest.json',tmp_path/'libs')
    assert len(proof['inventory'])==9 and proof['prior_SPD_runtime_acceptance_inherited'] is False
    bad=tmp_path/'bad.tar.gz';bad.write_bytes((root/'libgl.tar.gz').read_bytes()+b'changed')
    with pytest.raises(ValueError,match='archive differs'):
        runtime.unpack_offline_libraries(bad,root/'manifest.json',tmp_path/'bad-libs')
    assert not (tmp_path/'bad-libs').exists()
