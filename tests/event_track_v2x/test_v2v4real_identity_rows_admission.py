import pytest
from tools.event_track_v2x.prepare_v2v4real_identity_rows import cache_admission


def receipt():
    return dict(kind='v2v4real_native_9d_cache_independent_byte_numeric_readback_v1',split='train',seed=1337,
        native_nine_state_cache_independently_read_back=True,complete_source_frame_coverage_verified=True,frames=14210,online_GT_read=False)


@pytest.mark.parametrize('key,value',[('split','official_test'),('seed',2027),('frames',2034),('online_GT_read',True),
    ('native_nine_state_cache_independently_read_back',False)])
def test_partial_wrong_seed_test_or_unaccepted_cache_rejected(key,value):
    v=receipt();v[key]=value
    with pytest.raises(ValueError,match='before labels'):cache_admission(v,1337)
