#!/usr/bin/env python3
"""Frozen-input finite JPDA/PKF and local scan-MHT; no formal resource claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_bytes());assert sha(__file__)==plan['producer_sha256']
    backend=Path(plan['frozen_backend_root']);toolroot=Path(plan['frozen_mht_tool_root'])
    for name,h in plan['backend_source_sha256'].items():assert sha(backend/name)==h
    for name,h in plan['mht_tool_source_sha256'].items():assert sha(toolroot/name)==h
    for name,root in [('transvision',backend),('transvision.models',backend),('transvision.models.event_track_v2x',backend),('tools',toolroot),('tools.event_track_v2x',toolroot)]:
        module=types.ModuleType(name);module.__path__=[str(root.joinpath(*name.split('.')))];sys.modules[name]=module
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection,PaperForestTrackingConfig
    from transvision.models.event_track_v2x.identity_forest import IdentityNode
    from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig,PersistentProbabilisticTracker
    from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig,PersistentMHTTracker
    args.output.mkdir(exist_ok=False);reports=[]
    for incoming in plan['inputs']:
        path=Path(incoming['path']);assert sha(path)==incoming['sha256'];v=json.loads(path.read_bytes())
        assert set(v)=={'sequence_id','raw','rows','schedule'}
        raw=[RawIdentityDetection(**dict(r,node=IdentityNode(**r['node']))) for r in v['raw']]
        rows=[tuple(tuple(q) for q in row) for row in v['rows']]
        cases=[('jpda',None),('pkf',None)]+[('scan_mht',k) for k in plan['widths']]
        for method,width in cases:
            name=incoming['id']+'-'+method+('' if width is None else '-K'+str(width));out=args.output/name;out.mkdir()
            if method=='scan_mht':
                config=PersistentMHTConfig(state=PaperForestTrackingConfig(active_limit=width));Tracker=PersistentMHTTracker
            else:
                config=PersistentProbabilisticConfig(state=PaperForestTrackingConfig(),association_algorithm='exact',update_rule='pkf' if method=='pkf' else 'jpda-ci');Tracker=PersistentProbabilisticTracker
            assert dataclasses.asdict(config)==plan['configurations'][method]['none' if width is None else str(width)]
            tracker=Tracker(out/'tracker.sqlite',sequence_id=v['sequence_id'],config=config);events=[];ingested=0
            for event in v['schedule']:
                n=event['available_observations'];commit=tracker.step(tuple(raw[ingested:n]),tuple(rows[ingested:n]),**{k:event[k] for k in ['event_id','frame_id','reference_us','decision_us']})
                diagnostic=[]
                if method=='scan_mht':
                    # Deliberately materialize every retained conditional branch
                    # for science readback. These extra calls are NOT a latency
                    # or normal-runtime cost measurement; outputs stay immutable.
                    before=tracker.state_updates
                    for handle in sorted(tracker.active):
                        prediction=tracker._predict(handle,event['reference_us'])
                        diagnostic.append(dict(handle=handle,parents=list(tracker.parents(handle)),predictions=prediction))
                    work=tracker.state_updates-before
                    scan_readbacks=[]
                    for scan in tracker.scan_audits:
                        depth=max(scan['observation_indices'])+1
                        selected=list(tracker.db.execute('SELECT p.h,p.sha,w.value FROM prefixes p JOIN weights w ON w.h=p.h WHERE p.depth=? ORDER BY p.h',(depth,)))
                        assert len(selected)==scan['kept']
                        scan_readbacks.append([dict(handle=h,sha256=s,log_weight=w,parents=list(tracker.parents(h))) for h,s,w in selected])
                    collapsed=[];anchors=[]
                else:
                    work=0;scan_readbacks=[]
                    collapsed=[dict(root=r,reference_us=t,payload=json.loads(b)) for r,t,b in tracker.db.execute('SELECT root,reference_us,payload FROM gaussian_tracks ORDER BY root')]
                    anchors=list(tracker.db.execute('SELECT i,root FROM identity_anchors ORDER BY i'))
                events.append(dict(event_id=event['event_id'],available_observations=n,prediction=json.loads(commit.prediction_json),audit=json.loads(commit.audit_json),
                    prediction_sha256=sha_bytes(commit.prediction_json),audit_sha256=sha_bytes(commit.audit_json),all_retained_branch_state_readbacks=diagnostic,extra_state_readback_work_units=work,
                    selected_histories_per_scan=scan_readbacks,collapsed_gaussian_state_readbacks=collapsed,hard_identity_anchor_readbacks=anchors))
                ingested=n
            first=v['schedule'][0];n=first['available_observations'];duplicate=tracker.step(tuple(raw[:n]),tuple(rows[:n]),**{k:first[k] for k in ['event_id','frame_id','reference_us','decision_us']})
            assert sha_bytes(duplicate.prediction_json)==events[0]['prediction_sha256'] and sha_bytes(duplicate.audit_json)==events[0]['audit_sha256']
            database_sha=tracker.close();report=dict(method=method,width=width,input_id=incoming['id'],input_path=str(path),input_sha256=incoming['sha256'],configuration=dataclasses.asdict(config),events=events,
                database_path=str(out/'tracker.sqlite'),database_sha256=database_sha,original_prediction_and_audit_byte_immutable=True,
                diagnostic_state_cache_materialization=(method=='scan_mht'),normal_runtime_resource_measurement=False,
                inference_read_evaluator_truth_or_RBF_output=False,reproduced_published_MHT_or_PKF=False,formal_Linux_MHT_contract_used=False,full_Stage2_complete=False)
            p=out/'candidate.json';p.open('xb').write(canonical(report)+b'\n');reports.append(dict(run=name,path=str(p),sha256=sha(p)))
            print(json.dumps(dict(new_closed_runs=len(reports),total_runs=len(plan['inputs'])*len(cases))),flush=True)
    for name,h in plan['backend_source_sha256'].items():assert sha(backend/name)==h
    for name,h in plan['mht_tool_source_sha256'].items():assert sha(toolroot/name)==h
    assert not torch.cuda.is_initialized()
    result=dict(kind='rbf_stage2_causal_same_input_exact_scan_JPDA_PKF_local_MHT_candidates',runs=reports,source_plan_sha256=sha(args.plan),producer_sha256=sha(__file__),CUDA_initialized=False,independent_admission=False,same_resource_comparison=False,full_Stage2_complete=False)
    (args.output/'campaign-receipt.json').open('xb').write(canonical(result)+b'\n')


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


if __name__=='__main__':main()
