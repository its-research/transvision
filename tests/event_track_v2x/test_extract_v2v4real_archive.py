"""Tiny ZIP fixtures test extraction, not official source authenticity."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from tools.event_track_v2x import extract_v2v4real_archive as tool


def fixture(root, extra=None, *, omit=False, split='train'):
    archive = root / (split + '_01.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for cav in ('0', '1'):
            z.writestr(f'seq/{cav}/000000.yaml', 'lidar_pose: [0,0,0,0,0,0]\n')
            if not (omit and cav == '1'):
                z.writestr(f'seq/{cav}/000000.pcd', 'PCD fixture')
        if extra:
            z.writestr(*extra)
    meta = dict(kind='v2v4real_official_box_metadata_snapshot_v1', dataset='V2V4Real', files=[
        dict(name=archive.name, split=split, reported_sha1=tool.digest_file(archive)[0],
             size_bytes=archive.stat().st_size)])
    release = root / 'release.json'
    release.write_text(json.dumps(meta))
    return archive, release, tool.digest_file(release)[1], root / 'extracted'


def test_extract_full_volume_without_claiming_full_split_or_gt_free(tmp_path):
    args = fixture(tmp_path)
    archive_hash = tool.digest_file(args[0])
    result = tool.extract_volume(*args)
    assert result['payload_verified'] is True
    assert result['full_official_split_verified'] is False
    assert result['raw_yaml_may_contain_GT'] is True
    assert result['inference_ready'] is result['paper_eligible'] is False
    assert result['native_inventory']['paired_frame_count'] == 1
    assert result['native_inventory']['source_frame_count'] == 2
    assert tool.digest_file(args[0]) == archive_hash
    for row in result['files']:
        assert tool.digest_file(args[3] / 'payload' / row['path'])[1] == row['sha256']
    assert json.loads((args[3] / 'receipt.json').read_text()) == result
    with pytest.raises(ValueError, match='fresh'):
        tool.extract_volume(*args)


@pytest.mark.parametrize('name', ['../escape', '/escape', 'seq/0/../bad.yaml', 'seq\\0\\000001.yaml',
                                'seq/0/000000.yaml', 'seq/0/run.py', 'seq/0/000000.yaml/'])
def test_reject_bad_paths_duplicate_and_non_data_members(tmp_path, name):
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', UserWarning)
        args = fixture(tmp_path, (name, 'bad'))
    with pytest.raises((ValueError, FileExistsError)):
        tool.extract_volume(*args)
    assert not args[3].exists()
    assert not (tmp_path / 'escape').exists()


def test_reject_symlink_archive_entry(tmp_path):
    info = zipfile.ZipInfo('seq/0/000001.yaml')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    args = fixture(tmp_path, (info, '/etc/passwd'))
    with pytest.raises(ValueError, match='ordinary ZIP'):
        tool.extract_volume(*args)
    assert not args[3].exists()


@pytest.mark.parametrize('mutation', ['sha', 'length', 'snapshot', 'val', 'source-link', 'missing-pcd', 'size-cap'])
def test_fail_closed_before_publishing(tmp_path, monkeypatch, mutation):
    args = list(fixture(tmp_path, split='val' if mutation == 'val' else 'train', omit=mutation == 'missing-pcd'))
    if mutation in ('sha', 'length'):
        meta = json.loads(args[1].read_text())
        meta['files'][0]['reported_sha1' if mutation == 'sha' else 'size_bytes'] = '0' * 40 if mutation == 'sha' else 0
        args[1].write_text(json.dumps(meta)); args[2] = tool.digest_file(args[1])[1]
    elif mutation == 'snapshot':
        args[2] = '0' * 64
    elif mutation == 'source-link':
        linked = tmp_path / 'link.zip'; linked.symlink_to(args[0]); args[0] = linked
    elif mutation == 'size-cap':
        monkeypatch.setattr(tool, 'MAX_TOTAL_BYTES', 2)
    with pytest.raises(ValueError):
        tool.extract_volume(*args)
    assert not args[3].exists()
    assert not list(tmp_path.glob('.extracted-*'))


def test_source_mutation_during_extraction_is_not_published(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = tool.inventory_native_root
    def mutate(payload):
        result = original(payload)
        with args[0].open('ab') as stream:
            stream.write(b'changed')
        return result
    monkeypatch.setattr(tool, 'inventory_native_root', mutate)
    with pytest.raises(ValueError, match='source changed'):
        tool.extract_volume(*args)
    assert not args[3].exists()
