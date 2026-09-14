import copy
import hashlib

import pytest

from tools.event_track_v2x.audit_probabilistic_identity_invariance import fingerprint
from transvision.models.event_track_v2x.detection_cache_v2 import canonical


@pytest.fixture
def data():
    anchors=[(0,0),(1,0),(2,2)]
    audits=[];predictions=[];digest=hashlib.sha256()
    for n,(i,root) in enumerate(anchors,1):
        digest.update(canonical((i,root))+b'\n')
        audits.append(dict(sequence_id='s',event_id=str(n),observation_count=n,
            identity_anchor_sha256=digest.hexdigest(),conditional_scans=[dict(index=i,root=root)]))
        predictions.append(dict(sequence_id='s',frame_id=str(n),predictions=[dict(track_id='s:0',
            mean=[float(n)],covariance=[[1.]],score=.9,observation_ids=list(range(n)))]))
    return anchors,audits,predictions


def test_geometric_change_does_not_change_hard_identity_or_output_membership(data):
    before=fingerprint(*data)
    anchors,audits,predictions=copy.deepcopy(data)
    predictions[1]['predictions'][0]['mean']=[100.]
    after=fingerprint(anchors,audits,predictions)
    assert before['events'][1]!=after['events'][1]
    for key in ('conditional_scan_stream_sha256','identity_anchor_stream_sha256','non_state_prediction_stream_sha256'):
        assert before[key]==after[key]
    assert after['every_historical_anchor_hash_reconstructed']


@pytest.mark.parametrize('error',['future_root','non_root','gap','false_hash','decreasing','short_stream','wrong_event','duplicate_event','uncovered'])
def test_anchor_prefix_or_coverage_forgery_rejected(data,error):
    anchors,audits,predictions=data
    if error=='future_root':anchors[0]=(0,1)
    elif error=='non_root':anchors[2]=(2,1)
    elif error=='gap':anchors[1]=(5,0)
    elif error=='false_hash':audits[1]['identity_anchor_sha256']='changed'
    elif error=='decreasing':audits[1]['observation_count']=0
    elif error=='short_stream':predictions.pop()
    elif error=='wrong_event':predictions[0]['frame_id']='different'
    elif error=='duplicate_event':audits[1]['event_id']=predictions[1]['frame_id']='1'
    else:anchors.append((3,3))
    with pytest.raises(ValueError):fingerprint(anchors,audits,predictions)


@pytest.mark.parametrize('field,value',[('track_id','s:other'),('score',.1),('observation_ids',[])])
def test_non_state_fingerprint_detects_hidden_identity_membership_or_score_change(data,field,value):
    before=fingerprint(*data)
    data[2][1]['predictions'][0][field]=value
    after=fingerprint(*data)
    assert before['non_state_prediction_stream_sha256']!=after['non_state_prediction_stream_sha256']
