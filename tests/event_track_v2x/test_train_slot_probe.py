import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from tools.event_track_v2x.train_forest_identity import FitConfig,fit_dataset
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV
from test_forest_training_data import prepared_rows
from test_train_inference_diagnostic import pair_file


@pytest.mark.parametrize('backend',['slot_bound_joint_beam','sparse_slot_bound_joint_beam',
    'reachable_slot_bound_joint_beam'])
def test_fresh_prefix_probe_cannot_masquerade_as_complete_sequence(prepared_rows,tmp_path,backend):
    data,_,cache,rows=prepared_rows
    metadata,_=pair_file(tmp_path,rows)
    fit=fit_dataset(data,sha_file(data/'manifest.json'),tmp_path/'fit',
        config=FitConfig(epochs=1,batch_size=4,hidden=8,heads=2,dropout=0.),require_full_train=False)
    cp=fit['seeds'][0];checkpoint=(tmp_path/'fit'/cp['checkpoint_manifest']).parent
    output=tmp_path/'probe'
    code='''import sys
from tools.event_track_v2x.probe_train_slot_bound import run_probe
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
c,ch,m,mh,p,ph,o,s,b=sys.argv[1:]
run_probe(VerifiedForestCache(c,ch),m,mh,p,ph,o,sequence=s,events=1,allow_fixture=True,backend=b)
'''
    process=subprocess.run([sys.executable,'-c',code,str(cache.root),cache.manifest_sha256,str(metadata),
        sha_file(metadata),str(checkpoint),cp['checkpoint_sha256'],str(output),rows[0]['sequence_id'],backend],
        cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,**THREAD_ENV),
        capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
    result=json.loads((output/'probe-receipt.json').read_bytes())
    plan=json.loads((output/'plan.json').read_bytes())
    assert result['cohort_mode']=='fixture-only' and not result['complete_sequence_run']
    assert not result['paper_eligible'] and not result['complete_selected_sequence_verified']
    assert not (output/'development-inference-receipt.json').exists()
    assert plan['kind']=='train_prefix_ranking_probe_plan_v1'
    assert plan['backend']==backend
    assert 'tools/event_track_v2x/probe_train_slot_bound.py' in plan['source_sha256']
