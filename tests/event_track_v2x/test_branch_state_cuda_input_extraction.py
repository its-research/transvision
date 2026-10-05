import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def consumer():
    p=Path(__file__).resolve().parents[2]/'tools/event_track_v2x/run_branch_state_cuda_candidate.py'
    spec=importlib.util.spec_from_file_location('CUDA_state_input_consumer',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def package(tmp_path, *, mutation=None):
    data=b'known frozen code'
    manifest=dict(kind='rbf_branch_state_CUDA_exact_source_and_admitted_reference_bundle_v1',
        GPU_devices_required=1,actual_GPU_execution=False,
        files={'execution/code.py':dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())})
    if mutation=='manifest_traversal':manifest['files']['../escape']=manifest['files'].pop('execution/code.py')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    ap=tmp_path/'inputs.tar.gz'
    with tarfile.open(ap,'w:gz') as t:
        for name,value in [('manifest.json',mp.read_bytes()),('execution/code.py',data)]:
            if mutation=='missing' and name=='execution/code.py':continue
            info=tarfile.TarInfo(name);info.size=len(value)
            if name=='execution/code.py':
                if mutation=='bytes':value=b'known frozen codf'
                if mutation=='traversal':info.name='../escape'
                if mutation=='symlink':info.type=tarfile.SYMTYPE;info.linkname='/tmp/escape';info.size=0
                if mutation=='bytes':info.size=len(value)
            t.addfile(info,io.BytesIO(value))
            if mutation=='duplicate' and name=='execution/code.py':t.addfile(info,io.BytesIO(value))
    return ap,mp


def test_extracts_only_exact_hash_bound_files(consumer,tmp_path):
    ap,mp=package(tmp_path)
    out=tmp_path/'bundle';m=consumer.unpack_verified(ap,mp,out)
    assert (out/'execution/code.py').read_bytes()==b'known frozen code'
    assert m['GPU_devices_required']==1
    with pytest.raises(AssertionError,match='preserve'):consumer.unpack_verified(ap,mp,out)


@pytest.mark.parametrize('mutation',['manifest_traversal','traversal','symlink','duplicate','missing','bytes'])
def test_rejects_unsafe_or_changed_input_before_GPU_execution(consumer,tmp_path,mutation):
    ap,mp=package(tmp_path,mutation=mutation)
    with pytest.raises(AssertionError):consumer.unpack_verified(ap,mp,tmp_path/'bundle')
    assert not (tmp_path/'escape').exists()


def test_external_manifest_cannot_change_command_or_request_CPU_fallback(consumer):
    with pytest.raises(AssertionError,match='undeclared'):
        consumer.command(dict(command_relative_to_bundle_root=['python','-c','bad']))
