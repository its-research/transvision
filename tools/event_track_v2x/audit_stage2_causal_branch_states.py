#!/usr/bin/env python3
"""Read every stored root-chain state and every active/output branch digest."""
import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
from pathlib import Path
import numpy as np


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.plan.read_text());assert sha(Path(__file__))==plan['source_sha256']
    helper=Path(plan['helper']);assert sha(helper)==plan['helper_sha256'];spec=importlib.util.spec_from_file_location('independent_9D_state_helper',helper);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    results=[]
    for item in plan['candidates']:
        p=Path(item['path']);assert sha(p)==item['sha256'];data=json.loads(p.read_text())
        if item['method']=='exclusive_RBF':raw=data['raw'];sequence=data['sequence_id']
        else:
            incoming=Path(data['input_path']);assert sha(incoming)==data['input_sha256'];payload=json.loads(incoming.read_text());raw=payload['raw'];sequence=payload['sequence_id']
        state=data['configuration']['state'];names=[r['node']['node_id'] for r in raw];dbpath=Path(data['database_path']);assert sha(dbpath)==data['database_sha256'];cells=branches=branch_tracks=0;maximum=0.
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
            catalog=[cid for cid, in db.execute('SELECT component FROM component_catalog')];assert len(catalog)==1;cid=catalog[0]
            members=[i for i, in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i',(cid,))];assert members==list(range(len(raw)))
            for h,b in db.execute(f'SELECT h,payload FROM pc{int(cid)}_states ORDER BY h'):
                target=json.loads(b);roots=pure.roots(pure.prefix(db,cid,h),names);group=[r for r,identity in zip(raw[:len(roots)],roots) if identity==roots[-1]];clock=max(r['state_us'] for r in group);mean,cov,_=pure.state_for_group(group,clock,state)
                assert np.allclose(mean,target['mean'],atol=1e-8,rtol=1e-8) and np.allclose(cov,target['covariance'],atol=1e-8,rtol=1e-8)
                assert target['first_us']==min(r['state_us'] for r in group) and target['last_us']==clock
                assert target['last_order']==list(max((r['state_us'],r['node']['source_id'],r['node']['node_id']) for r in group)) and target['max_score']==max(r['score'] for r in group)
                maximum=max(maximum,float(np.max(np.abs(mean-target['mean']))),float(np.max(np.abs(cov-target['covariance']))));cells+=1
            for event in data['events']:
                for comp in event['audit']['components']:
                    assert not comp['expired_output_only'] and comp['component']==cid
                    handles={a['handle'] for a in comp['active']}|{comp['output_handle']}
                    assert {b['handle'] for b in comp['branches']}==handles
                    for branch in comp['branches']:
                        h=branch['handle'];roots=pure.roots(pure.prefix(db,cid,h),names);assert len(roots)==comp['nodes'];chain=[];cursor=h
                        while cursor:
                            parent,depth,root=db.execute(f'SELECT parent,depth,root FROM pc{int(cid)}_prefixes WHERE h=?',(cursor,)).fetchone();chain.append((depth-1,root,cursor));cursor=parent
                        latest={}
                        for index,root,cursor in chain:latest.setdefault(root,(index,cursor))
                        output=[];reference=event['prediction']['box_reference_timestamp_us']
                        for root,(index,cursor) in sorted(latest.items()):
                            cache=json.loads(db.execute(f'SELECT payload FROM pc{int(cid)}_states WHERE h=?',(cursor,)).fetchone()[0]);group=[r for r,identity in zip(raw[:len(roots)],roots) if identity==names[root]]
                            score=max(r['score']*state['survival_per_second']**(max(0,reference-r['state_us'])/1e6) for r in group)
                            if cache['max_score']<state['birth_score'] or score<state['prune_score'] or max(0,reference-cache['last_us'])>state['max_age_us']:continue
                            # Projection from the independently verified cache
                            # also reconstructs the original sealed byte digest.
                            dt=(reference-cache['last_us'])/1e6;F=np.eye(9);F[0,7]=F[1,8]=dt
                            mean=F@np.asarray(cache['mean']);cov=F@np.asarray(cache['covariance'])@F.T+np.eye(9)*state['process_noise']*abs(dt)
                            output.append(dict(track_id=sequence+':'+hashlib.sha256(canonical(['forest-birth',names[root]])).hexdigest()[:24],class_label='car',mean=mean.tolist(),covariance=cov.tolist(),score=score))
                        assert hashlib.sha256(canonical(output)).hexdigest()==branch['state_sha256']
                        if h==comp['output_handle']:assert sorted(output,key=lambda t:t['track_id'])==event['prediction']['predictions']
                        branches+=1;branch_tracks+=len(output)
        assert sha(dbpath)==data['database_sha256']
        results.append(dict(run=item['run'],method=item['method'],candidate_sha256=item['sha256'],database_sha256=data['database_sha256'],all_cached_root_chain_states_checked=cells,
            active_and_output_branch_event_digests_checked=branches,complete_branch9D_predictions_checked=branch_tracks,maximum_independent_cached_state_difference=maximum))
        print(json.dumps({'full_branch_state_runs_verified':len(results),'total':len(plan['candidates'])}),flush=True)
    result=dict(kind='rbf_stage2_causal_all_stored_and_active_branch_states_readback_v1',runs=results,
        all_root_chain_cache9D_values_independently_verified=True,all_active_output_branch_prediction_byte_digests_reconstructed=True,
        source_sha256=sha(Path(__file__)),plan_sha256=sha(args.plan),production_modules_imported=False,
        original_runs_not_reexecuted=True,formal_interval_certificate=False,full_Stage2_or_paper_complete=False,goal_status='active')
    with args.output.open('xb') as f:f.write(canonical(result)+b'\n')


if __name__=='__main__':main()
