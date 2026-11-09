from types import SimpleNamespace
import pytest
from tools.event_track_v2x.submit_v2v4real_numeric_gpu import qualify_resume
from tools.event_track_v2x.readback_v2v4real_numeric_gpu import digest_text
M=dict(bootstrap_sha256=digest_text('original'),execution_bundle_sha256='a'*64)
def task(status='created',code='original',artifact=False):
    return SimpleNamespace(status=status,artifacts={'runtime':1} if artifact else {},
        data=SimpleNamespace(script=SimpleNamespace(diff=code)),get_parameters=lambda:{'General/seed':'1337','General/execution_bundle_sha256':'a'*64,'General/accepted_raw_task_id':'c0f709ae40f4409ca71fa01cda5f7eeb'})

def test_same_created_candidate_can_resume_without_new_source():
    assert qualify_resume(task(),seed=1337,manifest=M)

@pytest.mark.parametrize('status',['queued','in_progress','completed','failed'])
def test_live_or_terminal_candidate_never_restarted(status):
    assert not qualify_resume(task(status),seed=1337,manifest=M)

@pytest.mark.parametrize('code,artifact',[('changed',False),('original',True)])
def test_changed_source_or_runtime_outputs_prevent_resume(code,artifact):
    with pytest.raises(ValueError):qualify_resume(task(code=code,artifact=artifact),seed=1337,manifest=M)
