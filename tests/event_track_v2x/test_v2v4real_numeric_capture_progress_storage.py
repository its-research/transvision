import json,time
from types import SimpleNamespace
import pytest
from tools.event_track_v2x import run_v2v4real_numeric_capture_gpu_shards as wrapper

def test_measured_insufficient_disk_fails_before_download(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    with pytest.raises(ValueError):wrapper.storage_admission(tmp_path,world=4,per_rank_bytes=10,archive_budget=10,raw_budget=10)
    assert not list(tmp_path.iterdir())

def test_progress_waits_all_ranks_and_ignores_nonprogress_logs(tmp_path):
    paths=[tmp_path/f'rank{r}.log' for r in range(4)];offsets={};states={};start=time.monotonic()-10
    paths[0].write_text('private unrelated text\n'+json.dumps(dict(stage='native_numeric_trace_capture',completed_rows=1,total_rows=998))+'\n')
    partial=wrapper.rank_progress(paths,offsets,states,start)
    assert partial['completed_rows']==1 and partial['ETA_seconds'] is None
    for rank,total in [(1,998),(2,996),(3,994)]:
        paths[rank].write_text(json.dumps(dict(stage='native_numeric_trace_capture',completed_rows=1,total_rows=total))+'\n')
    full=wrapper.rank_progress(paths,offsets,states,start)
    assert full['completed_rows']==4 and full['ETA_seconds']>0
    assert wrapper.rank_progress(paths,offsets,states,start)['completed_rows']==4

def test_incomplete_line_and_progress_regression(tmp_path):
    path=tmp_path/'rank';offsets={};states={}
    path.write_text(json.dumps(dict(stage='native_numeric_trace_capture',completed_rows=2,total_rows=3986)))
    assert wrapper.rank_progress([path],offsets,states,time.monotonic())['observed_ranks']==0
    with path.open('a') as f:f.write('\n')
    wrapper.rank_progress([path],offsets,states,time.monotonic())
    with path.open('a') as f:f.write(json.dumps(dict(stage='native_numeric_trace_capture',completed_rows=1,total_rows=3986))+'\n')
    with pytest.raises(ValueError):wrapper.rank_progress([path],offsets,states,time.monotonic())
