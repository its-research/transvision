import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

TOOLS = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('seen_val_transport', TOOLS/'package_rbf_seen_val_forest_cache.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def inputs(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    records = []
    for name, data in [('manifest.json', b'{"split":"val"}'), ('frames/a.npz', b'opaque-compressed-query-bytes')]:
        p = root/name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return root, records


def test_whole_archive_independent_readback(tmp_path):
    root, records = inputs(tmp_path)
    archive = tmp_path/'cache.tar'
    transport.create_archive(root, records, archive)
    proof = transport.read_archive(archive, records)
    assert proof == dict(members=2, payload_bytes=sum(r['bytes'] for r in records),
                         all_member_bytes_independently_read=True)
    with pytest.raises(FileExistsError):
        transport.create_archive(root, records, archive)


@pytest.mark.parametrize('change', ['grow', 'shrink', 'same_size', 'duplicate', 'escape', 'symlink'])
def test_source_mutation_and_unsafe_inventory_rejected(tmp_path, change):
    root, records = inputs(tmp_path)
    p = root/records[1]['path']
    if change == 'grow': p.write_bytes(p.read_bytes()+b'x')
    if change == 'shrink': p.write_bytes(p.read_bytes()[:-1])
    if change == 'same_size': p.write_bytes(b'x'*p.stat().st_size)
    if change == 'duplicate': records.append(records[0])
    if change == 'escape': records[1]['path'] = '../outside'
    if change == 'symlink':
        target = tmp_path/'target'
        p.rename(target)
        p.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        transport.create_archive(root, records, tmp_path/'cache.tar')


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'tamper', 'symlink'])
def test_archive_member_tampering_rejected(tmp_path, change):
    root, records = inputs(tmp_path)
    archive = tmp_path/'bad.tar'
    with tarfile.open(archive, 'w') as output:
        for index, record in enumerate(records):
            if change == 'missing' and index == 1: continue
            data = (root/record['path']).read_bytes()
            if change == 'tamper' and index == 1: data = b'x'*len(data)
            item = tarfile.TarInfo('cache/'+record['path'])
            item.size = len(data)
            output.addfile(item, io.BytesIO(data))
        if change in ('extra', 'duplicate', 'symlink'):
            item = tarfile.TarInfo('cache/'+records[0]['path'] if change == 'duplicate' else 'cache/extra')
            if change == 'symlink':
                item.type, item.linkname = tarfile.SYMTYPE, '/etc/hosts'
            output.addfile(item)
    with pytest.raises(ValueError):
        transport.read_archive(archive, records)
