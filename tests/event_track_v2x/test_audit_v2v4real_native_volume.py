import json
import zipfile

import pytest

from tests.event_track_v2x.test_v2v4real_inputs import _fixture
from tools.event_track_v2x.extract_v2v4real_archive import digest_file, extract_volume
from tools.event_track_v2x.audit_v2v4real_native_volume import audit_volume, main


def volume(tmp_path):
    source, _, _ = _fixture(tmp_path)
    archive = tmp_path / 'train_01.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in source.glob('*/*/*'):
            z.write(path, path.relative_to(source))
    meta = dict(kind='v2v4real_official_box_metadata_snapshot_v1', dataset='V2V4Real', files=[
        dict(name=archive.name, split='train', reported_sha1=digest_file(archive)[0], size_bytes=archive.stat().st_size)])
    release = tmp_path / 'release.json'; release.write_text(json.dumps(meta))
    output = tmp_path / 'volume'
    extract_volume(archive, release, digest_file(release)[1], output)
    return output, digest_file(output / 'receipt.json')[1]


def test_native_audit_uses_all_source_frames_without_identity_or_class_remap(tmp_path):
    root, sha = volume(tmp_path)
    report = audit_volume(root, sha)
    assert report['parsed_source_frames'] == 4 and report['paired_frames'] == 2
    assert report['raw_annotation_counts'] == {'Car': 4}
    assert report['all_volume_file_hashes_verified']
    assert not any(report[k] for k in ('class_mapping_applied', 'identity_mapping_verified',
        'point_cloud_format_validated', 'full_official_split_verified', 'tracking_evaluation_performed', 'paper_eligible'))
    for path, digest in report['source_sha256'].items():
        assert digest_file(path)[1] == digest


@pytest.mark.parametrize('mutation', ['receipt', 'pcd', 'yaml', 'missing', 'duplicate', 'escape', 'output'])
def test_volume_audit_fails_on_inconsistent_inputs(tmp_path, mutation):
    root, sha = volume(tmp_path)
    if mutation == 'receipt': sha = '0' * 64
    elif mutation in ('pcd', 'yaml'):
        path = next((root / 'payload').glob('*/*/*.' + mutation)); path.write_bytes(path.read_bytes() + b'changed')
    elif mutation == 'missing': next((root / 'payload').glob('*/*/*.pcd')).unlink()
    elif mutation in ('duplicate', 'escape'):
        receipt = json.loads((root / 'receipt.json').read_text())
        if mutation == 'duplicate': receipt['files'].append(receipt['files'][0])
        else: receipt['files'][0]['path'] = '../outside.pcd'
        (root / 'receipt.json').write_text(json.dumps(receipt)); sha = digest_file(root / 'receipt.json')[1]
    if mutation == 'output':
        with pytest.raises(ValueError, match='outside immutable'):
            main(['--volume', str(root), '--receipt-sha256', sha, '--output', str(root / 'bad.json')])
    else:
        with pytest.raises(ValueError): audit_volume(root, sha)
