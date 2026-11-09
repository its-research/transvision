import pytest
from tools.event_track_v2x.readback_v2v4real_numeric_gpu import verify_execution

def case():
    scope=dict(independent_numeric_acceptance=False,network_forward_verified=False,nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)
    envelope=dict(scope,rows=3986,raw_index_sha256='a'*64,archive_bytes=100)
    params={'General/max_trace_bytes_per_rank':100,'General/max_archive_bytes':200,'General/max_raw_unpacked_bytes':300}
    required=20*1024**3+300+2*4*100+200+3986*1024**2
    e=dict(scope,kind='v2v4real_original_numeric_capture_GPU_candidate_v1',seed=1337,world_size=4,rows=3986,accepted_proof_sha256='b'*64,gt_read=False,trace_transport=envelope,worker_storage_admission=dict(required_free_bytes=required,actual_free_bytes=required,within_budget=True))
    kwargs=dict(params=params,seed=1337,world=4,proof_sha='b'*64,raw_sha='a'*64)
    return e,envelope,kwargs

def test_exact_execution_storage_scope():
    e,m,k=case();verify_execution(e,m,**k)

@pytest.mark.parametrize('fault',['rows','scope','free','archive','budget','proof','gt'])
def test_execution_rejects_partial_or_changed_evidence(fault):
    e,m,k=case()
    if fault=='rows':e['rows']=3985
    elif fault=='scope':e['network_forward_verified']=True
    elif fault=='free':e['worker_storage_admission']['actual_free_bytes']-=1
    elif fault=='archive':m['archive_bytes']=201
    elif fault=='budget':k['params']['General/max_trace_bytes_per_rank']=0
    elif fault=='proof':e['accepted_proof_sha256']='c'*64
    elif fault=='gt':e['gt_read']=True
    with pytest.raises(ValueError):verify_execution(e,m,**k)
