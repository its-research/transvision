#!/usr/bin/env python3
"""Independent native row features, all causal parents and real label targets.

No producer cache, world transform, identity index, row builder or trainer is
imported. Native labels are offline targets; model features use cache bytes only.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_identity_target_oracle import identities,match_native,supervision

FIELDS={'features','mean','covariance','score','source','node_id','frame_id','cache_sha256','information_us','arrival_us',
    'state_us','detection_index','contexts','positive','known','lengths','decision_us','reason'}
ATOL=RTOL=1e-4


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def observation(state,cov,metadata,appearance,rawscore,score,valid):
    r=np.asarray(metadata['lidar_to_world_row_rotation'],dtype=np.float64);t=np.asarray(metadata['lidar_to_world_translation'],dtype=np.float64)
    s=np.asarray(state,dtype=np.float64).copy();s[[3,4]]=s[[4,3]]
    s[6]=(-s[6]-math.pi/2+math.pi)%(2*math.pi)-math.pi
    j=np.eye(9);j[[3,4]]=j[[4,3]];j[6,6]=-1
    h=np.asarray([math.cos(s[6]),math.sin(s[6]),0.])@r
    d=np.asarray([-math.sin(s[6]),math.cos(s[6]),0.])@r
    jac=np.eye(9);jac[:3,:3]=r.T;jac[7:9,7:9]=r[:2,:2].T
    jac[6,6]=(h[0]*d[1]-h[1]*d[0])/(h[0]*h[0]+h[1]*h[1])
    s[:3]=s[:3]@r+t;s[6]=math.atan2(h[1],h[0]);s[7:9]=s[7:9]@r[:2,:2]
    # Separate Jacobians reproduce the declared legacy-axis and world changes.
    c=jac@(j@np.asarray(cov,dtype=np.float64)@j.T)@jac.T
    pose=np.r_[t/[100,100,20],Rotation.from_matrix(r.T).as_euler('xyz')/math.pi]
    feature=np.r_[s[:6]/[100,100,20,20,20,10],math.sin(s[6]),math.cos(s[6]),s[7:9]/30,
        appearance,[1,0,0],[metadata['box_reference_timestamp_us']/1e8,0,.1,int(metadata['side']=='infrastructure-side')],
        c[np.triu_indices(9)]/100,pose,[1,math.log(2)/8,0,0],[rawscore,score,valid]]
    return s,c,feature


def parents(a,i):
    time=int(a['state_us'][i]);lo=int(np.searchsorted(a['state_us'][:i],max(0,time-2000000),side='left'))
    candidates=np.arange(lo,i,dtype=np.int64)
    if not len(candidates):return []
    mask=(np.abs(a['state_us'][candidates]-time)<=2000000)&(a['arrival_us'][candidates]<=a['decision_us'][i])
    mask &= ~((a['source'][candidates]==a['source'][i])&(a['frame_id'][candidates]==a['frame_id'][i]))
    candidates=candidates[mask]
    dt=(time-a['state_us'][candidates])/1e6
    predicted=a['mean'][candidates,:2]+dt[:,None]*a['mean'][candidates,7:9]
    distances=np.linalg.norm(a['mean'][i,:2]-predicted,axis=1)
    eligible=[(float(d),int(k)) for d,k in zip(distances,candidates) if d<=8.]
    return sorted(k for _,k in sorted(eligible)[:8])


def verify(args):
    pins={};out=Path(args.output).absolute();root=Path(args.rows).absolute();cache=Path(args.cache).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh independent row admission required')
    c=json.loads(read(root/'candidate-receipt.json',args.candidate_sha256,pins))
    if (c['kind']!='v2v4real_full_train_identity_rows_candidate_v1' or c['official_test_labels_read'] is not False
            or c['native_class_scope']!=['vehicle']):raise ValueError('real native train rows required before offline labels')
    m=json.loads(read(root/'manifest.json',c['manifest_sha256'],pins));prov=m['provenance'];protocol=m['row_protocol']
    if (prov['dataset']!='v2v4real' or prov['official_test_labels_read'] is not False or protocol['parent_limit']!=8
            or protocol['max_parent_gap_us']!=2000000 or protocol['gate_distance_m']!=8.
            or protocol['candidate_protocol']!='rbf-all-class-top64-v1' or protocol['minimum_raw_score']!=.05
            or protocol['maximum_detections']!=64 or protocol['old_row_rescoring'] is not False):
        raise ValueError('original native prediction-only row protocol required')
    label=Path(args.labels).absolute();label_acceptance=json.loads(read(args.label_admission,prov['full_train_labels_independent_admission_sha256'],pins))
    if label_acceptance['all_source_geometry_and_native_ids_verified'] is not True:raise ValueError('full independent source geometry required')
    lc=json.loads(read(label/'candidate-receipt.json',label_acceptance['candidate_receipt_sha256'],pins))
    source_rows=[json.loads(x) for x in read(label/'source-labels.jsonl',lc['source_labels_sha256'],pins).splitlines()]
    gt={(r['sequence_id'],r['side'],r['frame_key']):r for r in source_rows};labels=identities(source_rows)
    cm=json.loads(read(cache/'manifest.json',prov['train_cache_sha256'],pins));entries={}
    if cm['dataset']!='v2v4real' or cm['split']!='train' or cm['gt_in_cache'] is not False:raise ValueError('GT-free native train cache required')
    for entry in cm['frames']:
        meta=json.loads(read(cache/entry['metadata']['path'],entry['metadata']['sha256'],pins))
        key=meta['sequence_id'],meta['side'],meta['frame_id']
        if key in entries or meta['arrays_sha256']!=entry['arrays']['sha256']:raise ValueError('unique bound native cache row required')
        entries[key]=entry,meta
    if set(entries)!=set(gt) or m['sequences']!=cm['sequences'] or len(entries)!=14210:raise ValueError('complete native label/cache cohort required')
    counts=Counter();total=0;maximum=0.;results=[]
    for record in m['shards']:
        scene=record['sequence_id'];b=read(root/record['path'],record['sha256'],pins)
        with np.load(io.BytesIO(b),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        n=record['nodes']
        if set(a)!=FIELDS or any(len(x)!=n for x in a.values()) or a['features'].shape!=(n,203):raise ValueError('exact native node and target inventory required')
        node_labels=[];row_number=0
        sequence=sorted([r for r in source_rows if r['sequence_id']==scene],key=lambda r:(r['frame_ordinal'],r['side'],r['frame_key']))
        for target in sequence:
            entry,meta=entries[scene,target['side'],target['frame_key']]
            b=read(cache/entry['arrays']['path'],entry['arrays']['sha256'],pins)
            with np.load(io.BytesIO(b),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
            if np.any(raw['class_indices']!=0):raise ValueError('native vehicle class slot differs')
            matched=match_native(raw['states'],raw['raw_scores'],target['states_source_lwh'])
            selected=sorted([i for i,v in enumerate(raw['raw_scores']) if v>=.05],key=lambda i:(-float(raw['raw_scores'][i]),i))[:64]
            source=int(target['cav_id']);timestamp=target['frame_ordinal']*100000;decision=timestamp+100000
            if meta['box_reference_timestamp_us']!=timestamp:raise ValueError('native nominal time binding differs')
            for index in selected:
                if row_number>=n:raise ValueError('native training shard omitted selected detection')
                for key,value in dict(source=source,frame_id=target['frame_key'],detection_index=index,cache_sha256=entry['frame_sha256'],
                    node_id=digest([scene,source,target['frame_key'],index,entry['frame_sha256']]),information_us=timestamp,
                    state_us=timestamp,arrival_us=decision,decision_us=decision,score=float(raw['scores'][index])).items():
                    if a[key][row_number]!=value:raise ValueError('raw node provenance or time differs: '+key)
                s,cov,f=observation(raw['states'][index],raw['covariances'][index],meta,raw['appearance'][index],
                    float(raw['raw_scores'][index]),float(raw['scores'][index]),bool(raw['appearance_valid'][index]))
                for key,value in [('mean',s),('covariance',cov),('features',f)]:
                    np.testing.assert_allclose(a[key][row_number],value,atol=ATOL,rtol=RTOL)
                    maximum=max(maximum,float(np.max(np.abs(a[key][row_number]-value))))
                j=matched[index]
                node_labels.append((source,None,False,'unmatched_prediction') if j<0 else labels[scene,source,target['object_ids'][j]])
                row_number+=1
        if row_number!=n or np.any(np.diff(a['state_us'])<0):raise ValueError('complete causal selected node order required')
        prior=set();reasons=Counter();cross=temporal=0
        for i,q in enumerate(node_labels):
            parent=parents(a,i);expected=parent+[i];length=int(a['lengths'][i])
            if (a['contexts'].shape!=(n,9) or length!=len(expected) or list(a['contexts'][i,:length])!=expected
                    or np.any(a['contexts'][i,length:]!=-1)):
                raise ValueError('independent complete causal parent context differs at '+scene+':'+str(i))
            positive,known,reason=supervision(q,[node_labels[j] for j in parent],prior)
            if (a['positive'].dtype!=bool or a['known'].dtype!=bool or list(a['positive'][i,:length])!=positive
                    or list(a['known'][i,:length])!=known or a['positive'][i,length:].any() or a['known'][i,length:].any()
                    or str(a['reason'][i])!=reason):raise ValueError('independent real identity target differs at '+scene+':'+str(i))
            for j,p in zip(parent,positive[1:]):
                if p:
                    if node_labels[j][0]==q[0]:temporal+=1
                    else:cross+=1
            prior.add(q);reasons[reason]+=1
        if dict(reasons)!=record['reason_counts'] or reasons['birth']+reasons['parent']!=record['supervised_rows']:
            raise ValueError('all supervised and excluded row counts differ')
        counts.update(reasons);total+=n;results.append(dict(sequence_id=scene,nodes=n,reason_counts=dict(reasons),
            cross_source_positive_parent_edges=cross,temporal_positive_parent_edges=temporal))
        print(json.dumps(dict(stage='independent_native_identity_rows',sequence_id=scene,nodes=n,completed_sequences=len(results),total_sequences=32)),flush=True)
    if total!=c['nodes'] or len(results)!=32:raise ValueError('full native row scope differs')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('row inputs changed during independent audit')
    out.mkdir(parents=True);proof=dict(kind='v2v4real_identity_rows_independent_feature_support_target_admission_v1',seed=c['seed'],
        nodes=total,sequences=32,reason_counts=dict(counts),results=results,maximum_numeric_error=maximum,atol=ATOL,rtol=RTOL,
        all_original_cache_nodes_and_prediction_only_features_verified=True,all_causal_parent_contexts_independently_verified=True,
        all_native_temporal_cross_source_partial_targets_independently_verified=True,official_test_labels_read=False,
        input_sha256=pins,candidate_receipt_sha256=args.candidate_sha256,checkpoint_qualified=False,paper_metrics_complete=False,
        source_sha256={str(Path(__file__)):sha(__file__),str(ROOT/'tools/event_track_v2x/v2v4real_identity_target_oracle.py'):sha(ROOT/'tools/event_track_v2x/v2v4real_identity_target_oracle.py')})
    p=out/'acceptance-receipt.json';p.open('x').write(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),nodes=total)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','candidate-sha256','cache','labels','label-admission','output'):p.add_argument('--'+name,required=True)
    verify(p.parse_args())
