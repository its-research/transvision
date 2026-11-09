"""No-data wiring gates; fake subprocesses do not admit a GPU runtime."""
import ast
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.event_track_v2x import run_v2v4real_official_test_raw_features as export

ARCHIVE=Path('/Volumes/Data/test/recover-before-fuse/historical/live-audit-20260926/fulltrain-native-export-source.tar.gz')


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source'
    files=export.extract_source(ARCHIVE,root)
    return root,files,tmp_path/'extension-build-receipt.json'


def install_fake_build(monkeypatch,root,*,failure=None,mutate_source=False,binary=True):
    calls=[]
    extension=root/'official/opencood/utils'/('box_overlaps'+importlib.machinery.EXTENSION_SUFFIXES[0])
    def fake(command,**kwargs):
        calls.append((command,kwargs))
        rank=len(calls)
        if rank==1:
            assert command==[sys.executable,str(root/'official/opencood/utils/setup.py'),'build_ext','--inplace']
            if binary:extension.write_bytes(b'synthetic extension bytes, never loaded')
            if mutate_source:(root/'official/opencood/utils/box_overlaps.pyx').write_text('changed')
        else:
            assert rank==2
            assert command==[sys.executable,'-c','from opencood.utils.box_overlaps import bbox_overlaps; assert callable(bbox_overlaps)']
            assert kwargs['env']['PYTHONPATH']==str(root/'official')
        assert kwargs['cwd']==root/'official' and kwargs['check'] is True
        assert kwargs['timeout']==(900 if rank==1 else 180)
        if failure==rank:raise subprocess.CalledProcessError(42,command)
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(export.subprocess,'run',fake)
    return calls,extension


def test_original_pinned_builder_commands_and_new_binary_identity(source,monkeypatch):
    root,files,path=source
    monkeypatch.setenv('CLEARML_API_SECRET_KEY','do-not-record-test-sentinel')
    calls,binary=install_fake_build(monkeypatch,root)
    receipt=export.build_frozen_extensions(root,files,path)
    assert len(calls)==2 and receipt['status']=='completed'
    assert receipt['builder_sha256']==export.EXTENSION_BUILDER_SHA
    assert receipt['original_source_count']==403 and receipt['original_sources_unchanged'] is True
    assert receipt['binaries']==[dict(path=str(binary.relative_to(root)),sha256=export.sha(binary),bytes=binary.stat().st_size)]
    assert receipt['subprocess_import_verified'] is True
    assert receipt['independent_runtime_acceptance'] is False
    assert json.loads(path.read_bytes())==receipt
    assert 'do-not-record-test-sentinel' not in path.read_text()
    export.verify_runtime_sources(root,files)
    export.verify_built_extensions(root,receipt)


@pytest.mark.parametrize('failure',[1,2])
def test_compile_or_import_failure_stops_before_admission(source,monkeypatch,failure):
    root,files,path=source
    calls,_=install_fake_build(monkeypatch,root,failure=failure)
    with pytest.raises(subprocess.CalledProcessError):export.build_frozen_extensions(root,files,path)
    receipt=json.loads(path.read_bytes())
    assert len(calls)==failure and receipt['status']=='failed'
    assert receipt['commands'][-1]['returncode']==42
    assert receipt['subprocess_import_verified'] is False and receipt['binaries']==[]


def test_original_source_changes_during_build_are_rejected(source,monkeypatch):
    root,files,path=source
    install_fake_build(monkeypatch,root,mutate_source=True)
    with pytest.raises(ValueError,match='asset bytes differ'):export.build_frozen_extensions(root,files,path)
    assert json.loads(path.read_bytes())['status']=='failed'


def test_missing_extension_despite_successful_exit_is_rejected(source,monkeypatch):
    root,files,path=source
    install_fake_build(monkeypatch,root,binary=False)
    with pytest.raises(ValueError,match='newly built'):export.build_frozen_extensions(root,files,path)
    assert json.loads(path.read_bytes())['subprocess_import_verified'] is False


def test_preexisting_binary_cannot_be_promoted_to_fresh_build(source,monkeypatch):
    root,files,path=source
    calls,binary=install_fake_build(monkeypatch,root)
    binary.write_bytes(b'stale')
    with pytest.raises(ValueError,match='prebuilt'):export.build_frozen_extensions(root,files,path)
    assert calls==[] and not path.exists()


def test_binary_change_after_build_fails_final_identity_gate(source,monkeypatch):
    root,files,path=source
    _,binary=install_fake_build(monkeypatch,root)
    receipt=export.build_frozen_extensions(root,files,path)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset bytes differ'):export.verify_built_extensions(root,receipt)


def test_modified_builder_cannot_execute_even_with_modified_inventory(source,monkeypatch):
    root,files,path=source
    calls,_=install_fake_build(monkeypatch,root)
    builder=root/export.EXTENSION_BUILDER
    builder.write_text(builder.read_text()+'\n# unauthorized change\n')
    files[export.EXTENSION_BUILDER]=export.sha(builder)
    with pytest.raises(ValueError,match='asset bytes differ'):export.build_frozen_extensions(root,files,path)
    assert calls==[]


def test_run_builds_before_opencood_import_and_binds_binary_after_pack():
    tree=ast.parse(Path(export.__file__).read_bytes())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    build=next(n.lineno for n in calls if n.func.id=='build_frozen_extensions')
    imports=[n.lineno for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module.startswith('opencood')]
    assert build<min(imports)
    pack=next(n.lineno for n in calls if n.func.id=='pack')
    gates=[n.lineno for n in calls if n.func.id=='verify_built_extensions']
    assert len(gates)==2 and min(gates)<pack<max(gates)
    source=ast.get_source_segment(Path(export.__file__).read_text(),run)
    assert "receipt['extension_build']" in source and 'extension_receipt_sha' in source
