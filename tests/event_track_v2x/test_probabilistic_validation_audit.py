"""Real tiny replay streams exercise the independent val stream auditor."""
import copy
from dataclasses import asdict
import json

import pytest

from tools.event_track_v2x.audit_probabilistic_validation import audit_streams, digest
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


@pytest.fixture(params=['jpda-ci','jpda-kalman','pkf'])
def replay(replay_inputs,tmp_path,request):
    cache,rows=replay_inputs
    config=PersistentProbabilisticConfig(update_rule=request.param)
    output=tmp_path/'replay'
    receipt=replay_rows(cache,rows,output,config)
    def read(name):
        return [json.loads(line) for line in (output/name).read_bytes().splitlines()]
    return rows,read('predictions.jsonl'),read('tracking.jsonl'),read('frame-timings.jsonl'),asdict(config),receipt


def test_real_replay_stream_audit_agrees_with_final_receipt(replay):
    result=audit_streams(*replay[:5])
    assert result['frames']==2
    assert result['scans']>0
    for k in ('latency_seconds_p50_p95_p99_max','frame_latency_seconds_p50_p95_p99_max'):
        assert result[k]==replay[5][k]
    for sid,head in replay[5]['sequence_heads'].items():
        assert result['sequence_heads'][sid]['prediction_sha256']==head['prediction_sha256']


@pytest.mark.parametrize('stream',[0,1,2,3])
def test_omission_in_any_stream_fails(replay,stream):
    values=copy.deepcopy(replay[:5]);values[stream].pop()
    with pytest.raises(ValueError,match='lengths differ|event identity'):
        audit_streams(*values)


@pytest.mark.parametrize('failure', ['clock','chain','audit_chain','duplicate_id','algorithm','bound','latency','work'])
def test_resigned_but_semantically_invalid_stream_fails(replay,failure):
    rows,predictions,tracking,timings,config=copy.deepcopy(replay[:5])
    p=predictions[0];a=tracking[0]['tracking']
    if failure=='clock': p['decision_timestamp_us']+=1
    elif failure=='chain': p['previous_commit_sha256']='f'*64
    elif failure=='audit_chain': a['previous_audit_sha256']='f'*64
    elif failure=='duplicate_id':
        assert p['predictions']
        p['predictions'].append(dict(p['predictions'][0]))
    elif failure=='algorithm': a['anchor_decoder']='marginal-bayes'
    elif failure=='bound': a['full_history_posterior_bound']=0.
    elif failure=='latency': timings[0]['frame_seconds']=-1.
    elif failure=='work': a['state_updates']+=1
    p['commit_sha256']=digest({k:v for k,v in p.items() if k!='commit_sha256'})
    a['prediction_sha256']=p['commit_sha256']
    with pytest.raises(ValueError):
        audit_streams(rows,predictions,tracking,timings,config)
