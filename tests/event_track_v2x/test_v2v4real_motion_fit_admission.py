import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest

from tools.event_track_v2x.v2v4real_motion_fit_inputs import read, load
from tools.event_track_v2x.verify_v2v4real_motion_fit import verify


def save(path,value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_and_symlink_inputs_rejected(tmp_path):
    p=tmp_path/'bytes';p.write_bytes(b'admitted');expected=hashlib.sha256(p.read_bytes()).hexdigest()
    assert read(p,expected,{})==b'admitted'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):read(p,expected,{})
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError,match='ordinary'):read(link,hashlib.sha256(p.read_bytes()).hexdigest(),{})


@pytest.mark.parametrize('bad',['detector','acceptance','partition'])
def test_official_test_input_rejected_before_arrays(tmp_path,monkeypatch,bad):
    d=dict(kind='v2v4real_motion_detector_train_input_admission_v1',official_test_used=False,clock='nominal_10hz')
    a=dict(official_test_used=False,clock='nominal_10hz',all_source_geometry_verified=True,
           all_velocity_masks_and_predecessors_verified=True,all_past_identity_velocity_targets_verified=True)
    p=dict(official_split='train',official_test_used_for_partition=False)
    {'detector':d,'acceptance':a,'partition':p}[bad]['official_test_used' if bad!='partition' else 'official_test_used_for_partition']=True
    files=[tmp_path/name for name in ['detector','acceptance','partition']]
    digests=[save(f,v) for f,v in zip(files,[d,a,p])]
    def forbidden(*args,**kwargs):raise AssertionError('arrays opened before train admission')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='train-only'):
        load(files[0],digests[0],files[1],digests[1],files[2],digests[2])


def test_test_motion_candidate_rejected_before_payloads(tmp_path,monkeypatch):
    folder=tmp_path/'candidate';folder.mkdir()
    digest=save(folder/'candidate-receipt.json',dict(kind='v2v4real_motion_covariance_train_fit_candidate_v1',
        fit_split='official_test',official_test_used=True))
    def forbidden(*args,**kwargs):raise AssertionError('online/test arrays opened')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='train-only'):
        verify(SimpleNamespace(candidate=folder,candidate_sha256=digest,output=tmp_path/'independent'))
