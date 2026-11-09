import hashlib
import json

import pytest

from tools.event_track_v2x.admit_v2v4real_motion_train_labels import admitted_labels


def volume(tmp_path, split='train'):
    rows = []
    for cav in ('0', '1'):
        name = f'scene/{cav}/000000.yaml'
        p = tmp_path / 'payload' / name
        p.parent.mkdir(parents=True)
        p.write_bytes(b'native source label')
        rows.append(dict(path=name, bytes=p.stat().st_size,
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    p = tmp_path / 'receipt.json'
    p.write_text(json.dumps(dict(split=split, files=rows)))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_reject_test_before_payload(tmp_path):
    pin = volume(tmp_path, 'official_test')
    (tmp_path / 'payload/scene/0/000000.yaml').unlink()
    with pytest.raises(ValueError, match='before reading labels'):
        list(admitted_labels(tmp_path, pin, [('scene', '000000')]))


def test_two_sources_and_tampered_bytes(tmp_path):
    pin = volume(tmp_path)
    rows = list(admitted_labels(tmp_path, pin, [('scene', '000000')]))
    assert [r['cav_id'] for r in rows] == ['0', '1']
    (tmp_path / 'payload/scene/1/000000.yaml').write_bytes(b'changed')
    with pytest.raises(ValueError, match='size/SHA'):
        list(admitted_labels(tmp_path, pin, [('scene', '000000')]))


def test_missing_selection_and_symlink(tmp_path):
    pin = volume(tmp_path)
    with pytest.raises(ValueError, match='missing'):
        list(admitted_labels(tmp_path, pin, [('scene', '000001')]))
    p = tmp_path / 'payload/scene/0/000000.yaml'
    p.unlink()
    p.symlink_to(tmp_path / 'receipt.json')
    with pytest.raises(ValueError, match='symlink'):
        list(admitted_labels(tmp_path, pin, [('scene', '000000')]))
