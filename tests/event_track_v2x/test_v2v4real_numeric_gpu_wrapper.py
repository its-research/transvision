from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tools.event_track_v2x.run_v2v4real_numeric_capture_gpu_shards import launch,run

@pytest.mark.parametrize('world,budget',[(1,10),(4,0),(8,-1)])
def test_unqualified_cards_and_budget_fail_before_cuda(tmp_path,world,budget):
    with pytest.raises(ValueError):launch(tmp_path,source=None,prepared=None,admission=None,checkpoint=None,
        seed=1337,world=world,raw_index=None,raw_sha=None,raw_root=None,per_rank_bytes=budget)
    assert not list(tmp_path.iterdir())

def test_cuda_count_mismatch_never_launches(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:8)))
    with pytest.raises(ValueError):launch(tmp_path,source=None,prepared=None,admission=None,checkpoint=None,
        seed=1337,world=4,raw_index=None,raw_sha=None,raw_root=None,per_rank_bytes=10)
    assert not list(tmp_path.iterdir())

def test_wrong_registered_seed_fails_before_input_lookup(tmp_path):
    task=SimpleNamespace(get_parameters=lambda:{'General/seed':'99','General/world_size':'4',
        'General/max_trace_bytes_per_rank':'1','General/max_archive_bytes':'1','General/max_raw_unpacked_bytes':'1'})
    with pytest.raises(ValueError):run(task,accepted_proof=tmp_path/'missing',proof_sha='0'*64)
