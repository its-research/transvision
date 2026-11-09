from types import SimpleNamespace
import pytest
from tools.event_track_v2x.submit_v2v4real_numeric_gpu import existing_candidate,dispatch

MANIFEST={'execution_bundle_sha256':'a'*64}
def task(**changes):
    p={'General/execution_bundle_sha256':'a'*64,'General/seed':1337,'General/accepted_raw_task_id':'c0f709ae40f4409ca71fa01cda5f7eeb'};p.update(changes)
    return SimpleNamespace(get_parameters=lambda:p)

def test_empty_and_matching_candidate():
    assert existing_candidate([],seed=1337,manifest=MANIFEST) is None
    t=task();assert existing_candidate([t],seed=1337,manifest=MANIFEST) is t

@pytest.mark.parametrize('changes',[{'General/seed':2027},{'General/execution_bundle_sha256':'b'*64},{'General/accepted_raw_task_id':'other'}])
def test_existing_source_or_input_substitution(changes):
    with pytest.raises(ValueError):existing_candidate([task(**changes)],seed=1337,manifest=MANIFEST)

def test_ambiguous_candidate_preserved():
    with pytest.raises(ValueError):existing_candidate([task(),task()],seed=1337,manifest=MANIFEST)

@pytest.mark.parametrize('seed,budget',[(1,1),(1337,0),(1337,-1),(1337,True)])
def test_invalid_seed_budget_before_auth(seed,budget):
    with pytest.raises(ValueError):dispatch('/absent',seed,execute=True,per_rank_bytes=budget,archive_bytes=1,raw_bytes=1)
