from types import SimpleNamespace
import pytest
from tools.event_track_v2x.readback_v2v4real_numeric_gpu import qualify,digest_text,verify_rank

SOURCE=dict(bootstrap_sha256=digest_text('candidate'),execution_bundle_sha256='a'*64)
def task(status='completed',code='candidate'):
    return SimpleNamespace(status=status,data=SimpleNamespace(script=SimpleNamespace(diff=code)),
        get_parameters=lambda:{'General/execution_bundle_sha256':'a'*64,'General/seed':'1337','General/world_size':'4'})

def test_completed_exact_execution_admission():
    assert qualify(task(),SOURCE)==(1337,4)

@pytest.mark.parametrize('status,code',[('in_progress','candidate'),('queued','candidate'),('completed','changed')])
def test_live_or_changed_execution_rejected(status,code):
    with pytest.raises(ValueError):qualify(task(status,code),SOURCE)

def test_missing_rank_runtime_and_bindings_rejected():
    with pytest.raises(ValueError):verify_rank({},seed=1337,rank=0,world=4,source=SOURCE,raw_sha='a'*64,interpreter='/python')
