import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tools.event_track_v2x.audit_recovery_task_scope import EventAudit, audit_run
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV
from test_forest_training_data import prepared_rows
from test_recovery_task_scope import make_poses,new_stream
from test_persistent_cache_stream import delivery,step

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def completed_scope(prepared_rows,tmp_path):
    data,_,cache,rows=prepared_rows
    pairs=tmp_path/'pairs.json'
    pairs.write_bytes(canonical([dict(vehicle_sequence=r['sequence_id'],infrastructure_sequence=r['sequence_id'],
        vehicle_frame=r['vehicle_frame'],infrastructure_frame=r['infrastructure_frame']) for r in rows]))
    fitted=fit_dataset(data,sha_file(data/'manifest.json'),tmp_path/'fit',
        config=FitConfig(epochs=1,batch_size=4,hidden=8,heads=2,dropout=0.),require_full_train=False)
    cp=fitted['seeds'][0];checkpoint=(tmp_path/'fit'/cp['checkpoint_manifest']).parent
    poses,table=make_poses(cache,tmp_path/'poses')
    common=[str(cache.root),cache.manifest_sha256,str(pairs),sha_file(pairs),str(checkpoint),cp['checkpoint_sha256'],
        str(poses),table.manifest_sha256,rows[0]['sequence_id']]
    code='''import sys
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from tools.event_track_v2x.run_train_inference_diagnostic import run
c,ch,p,ph,m,mh,e,eh,s,b,o=sys.argv[1:]
run(VerifiedForestCache(c,ch),p,ph,m,mh,o,sequence=s,backend=b,allow_fixture=True,
    ego_pose_table=e,ego_pose_table_sha256=eh)
'''
    runs={}
    for mode in ('beam_recovery','beam_recovery_disabled'):
        output=tmp_path/mode
        proc=subprocess.run([sys.executable,'-c',code,*common,mode,str(output)],cwd=ROOT,
            env=dict(os.environ,**THREAD_ENV),capture_output=True,text=True,timeout=60)
        assert proc.returncode==0,proc.stdout+proc.stderr
        runs[mode]=output
    return cache,poses,table,runs


def test_completed_actual_replays_are_independently_audited_and_cannot_be_relabelled(completed_scope,tmp_path):
    cache,poses,table,runs=completed_scope
    for mode,root in runs.items():
        checksum=sha_file(root/'development-inference-receipt.json')
        result=audit_run(root,checksum,cache,poses,table.manifest_sha256,tmp_path/(mode+'-audit'),allow_fixture=True)
        assert result['frames']==1 and result['pose_status_counts']=={'fresh_arrived_pose':1}
        assert result['global_decoder_scope_and_weights_unchanged'] and result['no_future_pose_or_observation_used']
        assert result['all_historical_raw_nodes_in_component_maps'] and result['independent_raw_scope_reconstruction']
        assert result['events'][0]['recent_raw_nodes']==4 and result['events'][0]['task_raw_nodes']==4
        assert not any(result[k] for k in ('paper_eligible','validation','parameter_training','inference_rerun',
            'tracking_metrics_computed','greedy_order_optimality_verified','report_contains_GT'))
        if mode.endswith('disabled'):assert result['extra_search_steps']==0
        with pytest.raises(ValueError,match='provenance'):
            audit_run(root,checksum,cache,poses,table.manifest_sha256,tmp_path/(mode+'-fake-real'))
    root=runs['beam_recovery']
    with (root/'tracking.jsonl').open('ab') as stream:stream.write(b'{}\n')
    with pytest.raises(ValueError,match='artifact identity'):
        audit_run(root,sha_file(root/'development-inference-receipt.json'),cache,poses,table.manifest_sha256,
                  tmp_path/'changed',allow_fixture=True)
    assert not (tmp_path/'changed').exists()


def test_independent_event_reference_detects_hidden_scope_and_work_changes(completed_scope):
    cache,poses,table,runs=completed_scope;root=runs['beam_recovery']
    plan=json.loads((root/'plan.json').read_bytes())
    pose_rows={(r['sequence_id'],r['frame_id']):r for r in json.loads(poses.read_bytes())['poses']}
    original=json.loads((root/'tracking.jsonl').read_bytes())['tracking']
    prediction=json.loads((root/'predictions.jsonl').read_bytes())
    db=sqlite3.connect((root/'sequence-0000.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        for bad in ('GT','missing_pose','pose_position','pose_clock','table_hash','decision_scope','local_scope',
                    'loss_weight','priority','count','search_total','trace_priority','missing_support','depth','future_receipt'):
            a=copy.deepcopy(original);p=copy.deepcopy(prediction)
            if bad=='GT':a['cache_ingestion']['gt_model_inputs']=True
            elif bad=='missing_pose':a['recovery_task_scope']['pose']=None
            elif bad=='pose_position':a['recovery_task_scope']['pose']['ego_translation_world'][0]+=50.
            elif bad=='pose_clock':a['recovery_task_scope']['pose']['arrival_us']+=1
            elif bad=='table_hash':a['recovery_task_scope']['pose_table_sha256']='f'*64
            elif bad=='decision_scope':a['decision_indices'].pop()
            elif bad=='local_scope':a['components'][0]['decision_indices'].pop()
            elif bad=='loss_weight':a['components'][0]['weight']=.123
            elif bad=='priority':a['components'][0]['recovery_priority_weight']=.123
            elif bad=='count':a['components'][0]['recovery_scope_raw_nodes']+=1
            elif bad=='search_total':a['recovery_search_steps']+=1
            elif bad=='trace_priority':
                a['recovery_allocation_trace'].append(dict(component=a['components'][0]['component'],
                    charged_search_steps=0,priority_weight=-1.))
            elif bad=='missing_support':a['components'][0]['complete_raw_support_retained']=False
            elif bad=='depth':a['components'][0]['nodes']+=100
            else:a['cache_ingestion']['new_deliveries'][0]['arrival_us']=p['decision_timestamp_us']+1
            with pytest.raises(ValueError):EventAudit(db,cache,pose_rows,plan).step(a,p)
        # Duplicate events cannot be treated as another valid decision.
        validator=EventAudit(db,cache,pose_rows,plan);validator.step(original,prediction)
        with pytest.raises(ValueError,match='clock'):validator.step(original,prediction)
        future=sqlite3.connect(':memory:')
        try:
            db.backup(future)
            future.execute('UPDATE component_catalog SET created_us=?', (prediction['decision_timestamp_us']+1,))
            with pytest.raises(ValueError,match='future component'):
                EventAudit(future,cache,pose_rows,plan).step(original,prediction)
        finally:future.close()
    finally:db.close()


def test_auditor_refuses_missing_complete_receipt_before_touching_cache(tmp_path):
    with pytest.raises(ValueError):
        audit_run(tmp_path,'a'*64,None,tmp_path/'missing-poses','b'*64,tmp_path/'no-result')
    assert not (tmp_path/'no-result').exists()


def test_historical_component_prefixes_and_receipts_never_borrow_future_data(prepared_rows,tmp_path):
    _,_,cache,_=prepared_rows
    poses,table=make_poses(cache,tmp_path/'history-poses')
    s=new_stream(cache,table,tmp_path/'history.db')
    first=step(s,[delivery(s,'infrastructure-side')])
    second=step(s,[delivery(s,arrival=1_200_000)],reference=1_200_000,event='vehicle')
    third=step(s,[delivery(s,arrival=1_300_000)],reference=1_300_000,event='duplicate')
    fourth=step(s,reference=3_100_000,event='expired')
    plan=dict(configuration=asdict(s.tracker.config),selected_sequence='0003',ego_pose_table_sha256=table.manifest_sha256)
    s.tracker.close()
    pose_rows={(r['sequence_id'],r['frame_id']):r for r in json.loads(poses.read_bytes())['poses']}
    db=sqlite3.connect((tmp_path/'history.db').as_uri()+'?mode=ro',uri=True)
    try:
        verifier=EventAudit(db,cache,pose_rows,plan)
        records=[verifier.step(c.tracking_audit,c.prediction) for c in (first,second,third,fourth)]
        assert records[0]['fallback']=='missing_arrived_pose' and records[0]['recent_raw_nodes']==2
        assert records[1]['fallback'] is None and records[1]['recent_raw_nodes']==4
        assert records[2]['pose_age_us']==300_000 and records[2]['pose_frame']==records[1]['pose_frame']
        assert records[3]['fallback']=='stale_arrived_pose' and records[3]['recent_raw_nodes']==0
        assert verifier.n==4
        # All future data is present in the closed database, but the independent
        # first-event calculation must still reject borrowing the later ego pose.
        bad=copy.deepcopy(first.tracking_audit)
        bad['recovery_task_scope']=copy.deepcopy(second.tracking_audit['recovery_task_scope'])
        bad['cache_ingestion']['recovery_task_scope']=bad['recovery_task_scope']
        with pytest.raises(ValueError,match='latest causally available'):
            EventAudit(db,cache,pose_rows,plan).step(bad,first.prediction)
    finally:db.close()
