import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def reader(monkeypatch):
    root = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('CUDA_state_output_reader_test',root/'read_branch_state_cuda_outputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_archive(root, mutation):
    value = b'database bytes'
    device = dict(uuid='GPU-actual-device',used_GPU_count=1)
    receipt = dict(output_files={'candidate.sqlite':dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())},device=device)
    receipt_path = root/'receipt.json'; receipt_path.write_text(json.dumps(receipt))
    command = dict(argv=['python','frozen.py'],shell=False,TF32_enabled=False,working_directory='bundle')
    if mutation == 'metadata': command['TF32_enabled'] = True
    files = [('receipt.json',receipt_path.read_bytes()),('CUDA-candidate/candidate.sqlite',value),
        ('device-before-execution.json',json.dumps(device).encode()),('command.json',json.dumps(command).encode())]
    archive_path = root/'archive.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for name,data in files:
            if name.startswith('CUDA-candidate/') and mutation == 'missing': continue
            info = tarfile.TarInfo(name); info.size = len(data)
            if name.startswith('CUDA-candidate/'):
                if mutation == 'traversal': info.name = '../escaped'
                elif mutation == 'symlink': info.type = tarfile.SYMTYPE; info.linkname = '/tmp/escaped'; info.size = 0
                elif mutation == 'bytes': data = b'database bytez'
            archive.addfile(info,io.BytesIO(data))
            if name.startswith('CUDA-candidate/') and mutation == 'duplicate': archive.addfile(info,io.BytesIO(data))
    return archive_path,receipt_path


def test_complete_byte_bound_output_archive(reader,tmp_path):
    archive,receipt = make_archive(tmp_path,None)
    output = tmp_path/'unpack'
    result = reader.extract(archive,receipt,output)
    assert len(result) == 4 and (output/'CUDA-candidate/candidate.sqlite').read_bytes() == b'database bytes'
    with pytest.raises(AssertionError,match='preserve'): reader.extract(archive,receipt,output)


@pytest.mark.parametrize('mutation',['duplicate','traversal','symlink','bytes','missing','metadata'])
def test_rejects_partial_unsafe_or_changed_output(reader,tmp_path,mutation):
    archive,receipt = make_archive(tmp_path,mutation)
    with pytest.raises(AssertionError): reader.extract(archive,receipt,tmp_path/'unpack')
    assert not (tmp_path/'escaped').exists()
