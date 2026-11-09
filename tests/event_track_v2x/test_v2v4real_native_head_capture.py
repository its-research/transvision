"""Hook boundary tests with a stub network, not a real GPU experiment."""
import tarfile
import types
import numpy as np
import pytest
import torch
from tools.event_track_v2x.capture_v2v4real_native_heads import capture_predict


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.cls_head=torch.nn.Conv2d(4,2,1);self.reg_head=torch.nn.Conv2d(4,14,1)
    def forward(self,value):
        return dict(psm=self.cls_head(value),rm=self.reg_head(value))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    with tarfile.open('/Volumes/Data/test/recover-before-fuse/historical/live-audit-20260926/fulltrain-native-export-source.tar.gz') as t:
        raw=t.extractfile('project/transvision/models/event_track_v2x/paper_pointpillar.py').read()
    path=tmp_path/'original.py';path.write_bytes(raw)
    core=types.ModuleType('unchanged_core');core.__file__=str(path)
    exec(compile(raw,str(path),'exec'),core.__dict__)
    model=Model().eval();anchors=np.zeros((3,5,2,7),dtype='float32');anchors[...,3:6]=[1,2,3]
    output={'preserved_object':True}
    class Detector:
        def predict(self,points,**kwargs):
            self.model(points)
            return output
    detector=Detector();detector.model=model;detector.anchors=anchors
    return detector,core,torch.ones((1,4,3,5)),output


def assert_clean(model):
    assert not model._forward_hooks and not model.cls_head._forward_pre_hooks


def test_original_result_hooks_are_read_only_and_removed(setup):
    detector,core,points,output=setup
    result,arrays=capture_predict(detector,core,points)
    assert result is output
    assert arrays['psm'].shape==(1,2,3,5) and arrays['regression'].shape==(1,14,3,5)
    assert arrays['observed_anchor_scores'].shape==(30,) and arrays['observed_anchor_scores'].dtype==np.float32
    assert all(not x.flags.writeable for x in arrays.values())
    points.add_(8);np.testing.assert_array_equal(arrays['bev'],1)
    assert_clean(detector.model)


@pytest.mark.parametrize('case',['predict-failure','twice','nonfinite','tampered-core'])
def test_failed_capture_never_leaks_hooks(setup,case):
    detector,core,points,_=setup;old=detector.predict
    def predict(value,**kwargs):
        if case=='predict-failure':raise RuntimeError('synthetic upstream failure')
        result=old(value,**kwargs)
        if case=='twice':old(value,**kwargs)
        if case=='tampered-core':
            with open(core.__file__,'ab') as f:f.write(b'\n# changed\n')
        return result
    detector.predict=predict
    if case=='nonfinite':points[0,0,0,0]=float('nan')
    with pytest.raises((RuntimeError,ValueError)):capture_predict(detector,core,points)
    assert_clean(detector.model)


def test_tf32_and_wrong_core_rejected_before_forward(setup,monkeypatch):
    detector,core,points,_=setup
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',True)
    with pytest.raises(ValueError,match='TF32'):capture_predict(detector,core,points)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    with open(core.__file__,'wb') as f:f.write(b'invalid')
    with pytest.raises(ValueError,match='core'):capture_predict(detector,core,points)
    assert_clean(detector.model)
