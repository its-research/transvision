import datetime
from copy import deepcopy
import pytest
from tools.event_track_v2x.readback_v2v4real_numeric_gpu import verify_rank
from tools.event_track_v2x.v2v4real_official_test_runtime import DEPENDENCIES
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as o
from tools.event_track_v2x.submit_rbf_final_identity import available

SOURCE={'sources':{'tools/event_track_v2x/run_v2v4real_native_numeric_capture.py':{'sha256':'a'*64}}}
def rank(r,w):
    return dict(kind='v2v4real_native_numeric_trace_producer_v1',seed=1337,projection_frames_sha256=o.FRAME_SHA,gt_read=False,TF32_enabled=False,
      trace_layout='compact-cells-v1',producer_sha256='a'*64,accepted_raw_index_sha256='b'*64,
      shard=dict(rank=r,world_size=w,full_source_rows=3986),actual_device=dict(device=f'cuda:{r}',device_type='cuda',logical_index=r,name='fixture-device'),
      rank_runtime=dict(interpreter='/python',dependencies=[dict(distribution=n,expected=v,module=m,actual=v,imported=True) for n,v,m in DEPENDENCIES]),
      rows=2*(1993//w+(r<1993%w)),independent_numeric_acceptance=False,network_forward_verified=False,nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)

@pytest.mark.parametrize('world',[4,8])
def test_complete_original_pair_rank_runtime_cohort(world):
    assert sum(verify_rank(rank(r,world),seed=1337,rank=r,world=world,source=SOURCE,raw_sha='b'*64,interpreter='/python') for r in range(world))==3986

@pytest.mark.parametrize('fault',['row_count','device','package','TF32'])
def test_rank_runtime_or_partition_substitution(fault):
    v=rank(0,4)
    if fault=='row_count':v['rows']-=2
    elif fault=='device':v['actual_device']['logical_index']=1
    elif fault=='package':v['rank_runtime']['dependencies'][0]['imported']=False
    else:v['TF32_enabled']=True
    with pytest.raises(ValueError):verify_rank(v,seed=1337,rank=0,world=4,source=SOURCE,raw_sha='b'*64,interpreter='/python')


def test_busy_overlapping_worker_prevents_dispatch():
    now=datetime.datetime.now(datetime.timezone.utc)
    idle=dict(id='host:gpu0,1,2,3,4,5,6,7',ip='host',last_activity_time=now.isoformat(),queues=[dict(id='q8')],task={})
    busy=dict(id='host:gpu4,5,6,7',ip='host',last_activity_time=now.isoformat(),queues=[dict(id='q4')],task=dict(id='live-original-task'))
    queues=[dict(id='q8',name='GPU8-3090',entries=[]),dict(id='q4',name='GPU4-3090',entries=[])]
    assert available([idle,busy],queues,now)==[]


def test_reserved_cards_prevent_dispatch_and_L40_is_CPU_only():
    now=datetime.datetime.now(datetime.timezone.utc)
    w=dict(id='host:gpu0,1,2,3',ip='host',last_activity_time=now.isoformat(),queues=[dict(id='q')],task={})
    q=dict(id='q',name='GPU4-3090',entries=[])
    assert available([w],[q],now,[('host',{2})])==[]
    q['name']='GPU4-L40S';assert available([w],[q],now)==[]
