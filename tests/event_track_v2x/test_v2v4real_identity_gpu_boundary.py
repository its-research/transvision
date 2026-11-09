import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import pytest

from tools.event_track_v2x import run_v2v4real_identity_gpu as consumer
from tools.event_track_v2x.prepare_v2v4real_identity_gpu_inputs import verify_archive


def mock_cuda(monkeypatch,*,names=None,uuids=None,architectures=None,processes=''):
    names=names or ['GPU-model']*4;uuids=uuids or ['GPU-'+str(i) for i in range(4)]
    cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:4,
        get_device_properties=lambda i:SimpleNamespace(name=names[i],uuid=uuids[i]),
        get_device_capability=lambda i:(8,6),get_arch_list=lambda:architectures if architectures is not None else ['sm_86'])
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(consumer.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=processes))


def test_physical_boundary_retains_other_GPU_work_and_admits_only_current_UUIDs(monkeypatch):
    mock_cuda(monkeypatch,processes='GPU-other, 123\nGPU-0, '+str(os.getpid())+'\n')
    result=consumer.physical_runtime(4)
    assert result['no_preexisting_compute_process_on_allocated_UUIDs'] is True
    assert result['formal_cost_isolation_claimed'] is False


def test_existing_process_on_allocated_GPU_is_rejected_before_training(monkeypatch):
    mock_cuda(monkeypatch,processes='GPU-2, '+str(os.getpid()+1)+'\n')
    with pytest.raises(ValueError,match='intersects'):consumer.physical_runtime(4)


@pytest.mark.parametrize('kwargs',[dict(names=['L40S']*4),dict(uuids=['GPU-same']*4),dict(architectures=['sm_80'])])
def test_ineligible_or_duplicate_physical_cards_cannot_enter_training(monkeypatch,kwargs):
    mock_cuda(monkeypatch,**kwargs)
    with pytest.raises(ValueError):consumer.physical_runtime(4)


def archive(path,*,name='rows/a',payload=b'original',link=False,duplicate=False):
    with tarfile.open(path,'w:gz') as a:
        m=tarfile.TarInfo(name)
        if link:m.type=tarfile.SYMTYPE;m.linkname='/outside'
        else:m.size=len(payload)
        a.addfile(m,None if link else io.BytesIO(payload))
        if duplicate:a.addfile(m,io.BytesIO(payload))


def test_bundle_readback_rejects_changed_bytes_and_duplicate_inventory(tmp_path):
    p=tmp_path/'good.tar.gz';archive(p)
    inventory={'rows/a':dict(bytes=8,sha256=hashlib.sha256(b'original').hexdigest())}
    verify_archive(p,inventory)
    inventory['rows/a']['sha256']='0'*64
    with pytest.raises(ValueError,match='bytes differ'):verify_archive(p,inventory)
    p=tmp_path/'duplicate.tar.gz';archive(p,duplicate=True)
    with pytest.raises(ValueError,match='inventory'):verify_archive(p,inventory)


@pytest.mark.parametrize('kwargs',[dict(name='../outside'),dict(link=True),dict(duplicate=True)])
def test_gpu_unpack_rejects_unsafe_or_duplicate_payloads(tmp_path,kwargs):
    p=tmp_path/'bad.tar.gz';archive(p,**kwargs)
    with pytest.raises(ValueError):consumer.unpack(p,tmp_path/'output')
    assert not (tmp_path/'outside').exists()
