"""Independent MHT ledger verification and corrupted-evidence rejection."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from tools.event_track_v2x import audit_mht_validation as audit
from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig, PersistentMHTTracker
from tools.event_track_v2x.run_mht_tracking_v2 import replay_mht_rows,mht_sources
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from test_forest_tracking import observation
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


def synthetic(tmp_path,seed=0,width=4):
    t = PersistentMHTTracker(tmp_path/'mht.sqlite',sequence_id='0003',
        config=PersistentMHTConfig(state=ForestTrackingConfig(active_limit=width)))
    config = asdict(t.config); records = []; rng = np.random.default_rng(seed); n = 0
    for scan in range(4):
        reference = 1000000+scan*200000
        obs = [observation(f'{scan}-{j}',scan*.2+j*3,index=j,source=scan%2,state_us=reference) for j in range(2)]
        rows = [tuple((p,float(rng.normal())) for p in range(-1,n+j)) for j in range(2)]
        result = t.step(obs,rows,reference_us=reference,decision_us=reference+100000,frame_id=str(scan),event_id=str(scan))
        n += 2
        schedule = dict(sequence_id='0003',vehicle_frame=str(scan),infrastructure_frame=str(scan),
                        box_reference_timestamp_us=reference)
        timing = {k:result.prediction[k] for k in ('sequence_id','frame_id','box_reference_timestamp_us','decision_timestamp_us')}
        timing.update(step_seconds=.1,frame_seconds=.2)
        records.append((schedule,result.prediction,dict(tracking=result.audit),timing))
    head = dict(frames=4,prediction_sha256=result.prediction['commit_sha256'])
    path = t.path; t.close()
    return path,head,records,config


@pytest.mark.parametrize('width',[1,2,4])
@pytest.mark.parametrize('seed',range(3))
def test_independent_four_scan_raw_factor_and_identity_ledger(tmp_path,seed,width):
    path,head,records,config = synthetic(tmp_path,seed,width)
    before = sha_file(path)
    result = audit.inspect_sequence(path,'0003',head,records,config)
    assert result['frames'] == 4 and result['observations'] == 8
    assert sha_file(path) == before


def test_v2_replay_audit_is_not_full_validation(replay_inputs,tmp_path):
    cache,rows = replay_inputs; output = tmp_path/'run'
    replay_mht_rows(cache,rows,output,PersistentMHTConfig())
    result = audit.audit_replay(output,sha_file(output/'receipt.json'),rows)
    assert result['frames'] == 2 and result['retained_alias_weights_reconstructed']
    assert not result['full_validation_coverage_verified']
    assert not result['full_top_k_optimality_recomputed'] and not result['gaussian_state_numerics_recomputed']
    assert audit.inference_sources() == mht_sources()


@pytest.mark.parametrize('mutation,error',[
    ('raw_factor','raw factor ledger'),('prefix_root','identity root'),('weight','class weight'),
    ('prefix_sha','prefix digest'),('clock','causal clock'),('predecessor','predecessor beam'),
    ('output_ID','output IDs'),('state_digest','conditional state digest'),
    ('budget','capacity exceeds'),('risk','risk declaration'),('coverage','coverage'),
])
def test_corrupted_ledger_or_published_contract_is_rejected(tmp_path,mutation,error):
    path,head,records,config = synthetic(tmp_path)
    with sqlite3.connect(path) as db:
        if mutation == 'raw_factor': db.execute('UPDATE potentials SET w=w+1 WHERE i=0')
        elif mutation == 'prefix_root': db.execute('UPDATE prefixes SET root=999 WHERE depth=1')
        elif mutation == 'weight': db.execute('UPDATE weights SET value=value+1 WHERE h=(SELECT MIN(h) FROM weights)')
        elif mutation == 'prefix_sha': db.execute("UPDATE prefixes SET sha=? WHERE depth=1",('f'*64,))
        elif mutation == 'coverage': records.pop()
        else:
            row,p,wrapper,timing = records[0]; a = wrapper['tracking']
            if mutation == 'clock': timing['decision_timestamp_us'] += 1
            elif mutation == 'predecessor': a['scans'][0]['ranked_parent_scans'][0]['parent_handle'] = 123456
            elif mutation == 'output_ID':
                p['predictions'][0]['track_id'] = 'invented'
                p['commit_sha256'] = audit.digest({k:v for k,v in p.items() if k != 'commit_sha256'})
                a['prediction_sha256'] = p['commit_sha256']
                for b in a['branches']:
                    if b['handle'] == a['output_handle']: b['state_sha256'] = audit.digest(p['predictions'])
            elif mutation == 'state_digest': a['branches'][0]['state_sha256'] = 'f'*64
            elif mutation == 'budget': a['scans'][0]['ranked_parent_scans'][0]['assignment_solves'] = config['max_assignment_solves']+1
            elif mutation == 'risk': a['decision']['risk_bound'] = .001
            db.execute('UPDATE events SET prediction=?,audit=? WHERE ordinal=0',(canonical(p),canonical(a)))
    with pytest.raises(ValueError,match=error): audit.inspect_sequence(path,'0003',head,records,config)


def test_no_torch_or_tracker_import_for_native_environment():
    result = subprocess.run([sys.executable,'-c',
        'import sys; from tools.event_track_v2x import audit_mht_validation; '
        'assert "torch" not in sys.modules; '
        'assert "tools.event_track_v2x.persistent_mht_tracking" not in sys.modules'],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_empty_schedule_is_never_a_complete_replay(tmp_path):
    with pytest.raises(ValueError,match='nonempty'):
        audit.audit_replay(tmp_path,'a'*64,[])
