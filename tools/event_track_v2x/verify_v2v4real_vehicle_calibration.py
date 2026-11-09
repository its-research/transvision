#!/usr/bin/env python3
"""Independent train-only vehicle labels and constrained logistic verification.

No producer matcher, GT geometry generator, calibration fitter or probability
application is imported. Existing raw-input geometry admission is reused only
after exact input hashes and the original CAV-0 reference are bound again.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import time

import numpy as np
from scipy.optimize import minimize

RTOL, ATOL = 1e-10, 1e-12  # Existing independent calibration parameter checker limits.
PROJECTED_GRADIENT_TOL = 1e-5  # Original SciPy L-BFGS-B default gtol, not a relaxed limit.
PROJECTION_SHA = 'a536855c2dc267f24d51ef2ddc8a9a181fb087ff5aedbc0db03dd9bd72775ba4'
FRAMES_SHA = 'db03624f5c67ac4a747d0af9866842fa01fd2634f8bd004ef81643a329a3a1e8'
PARTITION_SHA = '02373d0f59ca4e88757b6b3c3c22d9afda0ff94fe47f0f7d10f80211876bbcb2'
GEOMETRY_SHA = '46d7bcbf47e6dafe227d31f8b3246153ec00e7a2941a6ff0bb07fd8358fd30e7'
GEOMETRY_CHECKER_SHA = '92eac33cf266f78264312431d1e95c57dcd6c46a1ac3ab962195c5398aff9a5a'
SOURCE_ARCHIVE_SHA = 'f47f391db33e02813b05e10f598ecf9240877463a8e32b61bc5841a8769d38ea'
EXPORT_SHA = '9b2305d4aef05808b4994f8ae4ecfae61718d678b0a736f9443b1c5673ebfd46'
SEMANTICS = {'materializer':'aae761c404873ad400c05d511aa760aa33bf9b3b8ce19cec3206bd7cd4c4d7f9',
    'box_utils':'961914a1291f57e86d9a7416e7318a3e911f0de8fb98a0c4492e2a9d5785de86',
    'base_postprocessor':'3ad1d15e410965558661a52eea229b148e9c8a175e3d829154c8e7de02b61089'}
VEHICLE_SOURCE = 'params.vehicles_except_exact_obj_type_Pedestrian'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def close(actual,expected,name):
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a,b,rtol=RTOL,atol=ATOL),'independent numeric mismatch: '+name)


def match_labels(states,scores,classes,centers,gt_classes):
    """Python stable score order and scalar XY distances; strict 2 m boundary."""
    need(len(states)==len(scores)==len(classes) and len(centers)==len(gt_classes),'matching shapes differ')
    labels=np.zeros(len(scores),dtype=np.uint8);used=set()
    for i in sorted(range(len(scores)),key=lambda index:(-float(scores[index]),index)):
        choices=[(math.hypot(float(centers[j][0])-float(states[i][0]),
                             float(centers[j][1])-float(states[i][1])),j)
                 for j in range(len(centers)) if j not in used and classes[i]==gt_classes[j]]
        if choices:
            distance,index=min(choices)
            if distance<2.:labels[i]=1;used.add(index)
    return labels


def sigmoid(logits):
    z=np.asarray(logits,dtype=np.float64);out=np.empty_like(z);positive=z>=0
    out[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);out[~positive]=e/(1+e)
    return out


def logit(scores):
    s=np.maximum(1e-7,np.minimum(1-1e-7,np.asarray(scores,dtype=np.float64)))
    return np.log(s)-np.log1p(-s)


def objective(theta,x,y):
    z=theta[0]*x+theta[1]
    loss=float(np.sum(np.maximum(z,0)+np.log1p(np.exp(-np.abs(z)))-y*z)/len(y))
    residual=sigmoid(z)-y
    return loss,np.asarray([np.dot(residual,x)/len(y),np.sum(residual)/len(y)])


def projected_gradient(theta,gradient):
    return np.asarray(theta)-np.clip(np.asarray(theta)-gradient,[0.,-100.],[100.,100.])


def verify_parameters(scores,labels,model):
    theta=np.asarray([model['slope'],model['intercept']],dtype=np.float64)
    need(np.isfinite(theta).all() and 0<=theta[0]<=100 and -100<=theta[1]<=100,'constrained parameter domain differs')
    x,y=logit(scores),np.asarray(labels,dtype=np.float64)
    need(set(y)=={0.,1.},'both binary labels required')
    optimum=minimize(lambda t:objective(t,x,y),[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0.,100.),(-100.,100.)])
    need(optimum.success,'independent constrained optimizer did not converge')
    close(theta,optimum.x,'slope/intercept')
    loss,gradient=objective(theta,x,y);residual=projected_gradient(theta,gradient)
    need(np.max(np.abs(residual))<=PROJECTED_GRADIENT_TOL,'candidate violates original projected-gradient tolerance')
    p=sigmoid(theta[0]*x+theta[1]);reference=sigmoid(optimum.x[0]*x+optimum.x[1])
    close(p,reference,'all candidate calibrated probabilities')
    # A second scalar probability expression guards the vector sigmoid path.
    scalar=np.asarray([1/(1+math.exp(-float(z))) if z>=0 else math.exp(float(z))/(1+math.exp(float(z))) for z in theta[0]*x+theta[1]])
    close(p,scalar,'scalar/vector probabilities')
    weight=p*(1-p);design=np.column_stack((x,np.ones(len(x))))
    hessian=(design.T*weight)@design/len(y)
    need(float(np.linalg.eigvalsh(hessian).min())>=-ATOL,'convex Hessian check failed')
    return p,dict(independent_parameters=optimum.x.tolist(),objective=loss,gradient=gradient.tolist(),
        projected_gradient_inf=float(np.max(np.abs(residual))),projected_gradient_tolerance=PROJECTED_GRADIENT_TOL,
        constrained_convex_stationarity_verified=True,hessian_min_eigenvalue=float(np.linalg.eigvalsh(hessian).min()),
        maximum_probability_difference=float(np.max(np.abs(p-reference))),
        original_optimizer_bounds=[[0,100],[-100,100]],logit_clip=1e-7,L2=0.)


def diagnostics(probabilities,labels):
    p,y=[np.asarray(x,dtype=np.float64) for x in (probabilities,labels)];rows=[];ece=0.
    for bucket in range(10):
        indices=[i for i,value in enumerate(p) if min(math.floor(float(value)*10),9)==bucket]
        n=len(indices);confidence=math.fsum(float(p[i]) for i in indices)/n if n else None
        frequency=math.fsum(float(y[i]) for i in indices)/n if n else None
        if n:ece+=n/len(p)*abs(confidence-frequency)
        rows.append(dict(bin=bucket,count=n,confidence=confidence,frequency=frequency))
    clipped=np.maximum(1e-15,np.minimum(1-1e-15,p))
    return dict(nll=-math.fsum(float(v)*math.log(float(q))+(1-float(v))*math.log1p(-float(q)) for q,v in zip(clipped,y))/len(p),
        brier=math.fsum((float(q)-float(v))**2 for q,v in zip(p,y))/len(p),ece=ece,count=len(p),bins=rows,
        scope='binary_events_not_joint_posterior_TV',joint_probability_certificate=False)


def compare_diagnostics(actual,expected):
    need(set(actual)==set(expected),'diagnostic schema differs')
    for key in ('count','scope','joint_probability_certificate'):need(actual[key]==expected[key],'diagnostic scope/count differs')
    for key in ('nll','brier','ece'):close(actual[key],expected[key],key)
    need(len(actual['bins'])==10,'ten calibration bins required')
    for a,b in zip(actual['bins'],expected['bins']):
        need(set(a)==set(b) and a['bin']==b['bin'] and a['count']==b['count'],'diagnostic bin identity/count differs')
        for key in ('confidence','frequency'):
            if b[key] is None:need(a[key] is None,'empty diagnostic bin differs')
            else:close(a[key],b[key],'bin '+key)


def run(history,candidate_root,execution_sha256,gt_root,gt_sha256,output):
    checker_sha=sha(__file__)
    history,candidate_root,gt_root,output=[Path(x).absolute() for x in (history,candidate_root,gt_root,output)]
    need(not output.exists() and not any(p.is_symlink() for p in (output,*output.parents)),'fresh ordinary output required')
    pins={}
    def pin(path,digest):
        path=Path(path).absolute()
        need(re.fullmatch('[0-9a-f]{64}',digest or '') and path.is_file()
            and not any(p.is_symlink() for p in (path,*path.parents)) and sha(path)==digest,'input bytes differ: '+str(path))
        pins[str(path)]=digest;return path
    def read(path,digest):return json.loads(pin(path,digest).read_bytes())
    execution=read(candidate_root/'candidate-execution-receipt.json',execution_sha256)
    need(execution.get('kind')=='v2v4real_native_vehicle_calibration_three_seed_candidate_execution_v1'
         and execution.get('official_test_used') is False and execution.get('independent_acceptance_verified') is False,'new candidate execution required')
    projection=read(history/'72599160425d47ca9b7e8916bbb77b42/projection-manifest',PROJECTION_SHA)
    need(projection['dataset_split']=='train' and projection['gt_in_projection'] is False
         and set(projection['ego_agents'].values())=={'0'},'frozen train CAV-0 reference required')
    pf=pin(history/'72599160425d47ca9b7e8916bbb77b42/frames.jsonl',FRAMES_SHA)
    projected={}
    for line in pf.read_bytes().splitlines():
        row=json.loads(line);key=(row['sequence_id'],row['frame_key'],row['cav_id'])
        need(key not in projected and row['is_ego'] is (row['cav_id']=='0'),'projection CAV identity differs');projected[key]=row
    partition=read(history/'310283cf29374652a521dbbddf440475/nested-partition',PARTITION_SHA)
    need(partition['official_split']=='train' and partition['official_test_used_for_partition'] is False and partition['gt_used_for_partition'] is False,'train partition required')
    groups=set(partition['calibration_fit']);selected={s for s,g in partition['sequence_to_name_group'].items() if g in groups}
    need(not groups&set(partition['detector_fit']) and not groups&set(partition['identity_selection']),'calibration roles overlap')
    geometry=read(history/'native-bound-calibration-input-independent-admission.json',GEOMETRY_SHA)
    pin(history/'verify_native_bound_calibration_inputs.py',GEOMETRY_CHECKER_SHA)
    need(geometry['verifier_sha256']==GEOMETRY_CHECKER_SHA and geometry['gt_read'] is False and geometry['official_test_read'] is False,'GT-free geometry proof required')
    archive=pin(history/'fulltrain-native-export-source.tar.gz',SOURCE_ARCHIVE_SHA)
    with tarfile.open(archive) as t:
        original=t.extractfile('project/tools/event_track_v2x/run_v2v4real_train_raw_features.py').read()
    need(hashlib.sha256(original).hexdigest()==EXPORT_SHA and b'side="vehicle-side" if row["is_ego"] else "infrastructure-side"' in original,'original ego-to-side exporter differs')
    gt=read(gt_root/'manifest.json',gt_sha256)
    need(gt['kind']=='v2v4real_native_vehicle_gt_projection_v1' and gt['evaluation_protocol']=='v2v4real-official-benchmark-vehicle-v1'
         and gt['split']=='train' and gt['class_scope']==['vehicle'] and gt['native_label_source']==VEHICLE_SOURCE
         and gt['test_payloads_read'] is False and gt['GT_read'] is True and gt['inference_input'] is False
         and gt['partition_sha256']==PARTITION_SHA and set(gt['ego_agents'])==selected and set(gt['ego_agents'].values())=={'0'},'vehicle GT role/ego scope differs')
    pin(gt_root/'audit.jsonl',gt['audit_sha256'])
    frames=[json.loads(x) for x in pin(gt_root/'frames.jsonl',gt['frames_sha256']).read_bytes().splitlines()]
    gt_by_key={};ordinals=Counter()
    for f in frames:
        key=f['sequence_id'],f['frame_key'];need(key not in gt_by_key and f['sequence_id'] in selected and f['ego_cav']=='0'
            and f['frame_ordinal']==ordinals[f['sequence_id']],'GT frame identity/order differs');ordinals[f['sequence_id']]+=1
        need(projected[(*key,'0')]['frame_ordinal']==f['frame_ordinal'],'GT/projection ordinal differs')
        centers=[];ids=[]
        for obj in f['objects']:
            corners=np.asarray(obj['corners_ego'],dtype=float)
            need(corners.shape==(8,3) and np.isfinite(corners).all() and obj['evaluation_class']=='vehicle'
                and isinstance(obj['raw_class'],str) and obj['raw_class'] and obj['raw_class']!='Pedestrian','native vehicle GT object differs')
            centers.append([math.fsum(float(v) for v in corners[:,axis])/8 for axis in range(3)]);ids.append(obj['track_id'])
        need(ids==sorted(set(ids)),'GT identity ordering differs');gt_by_key[key]=(f,np.asarray(centers).reshape(-1,3))
    need(len(frames)==gt['paired_frames']==1017 and len(selected)==6,'full calibration GT cohort required')
    output.mkdir(parents=True);results=[]
    for spec in execution['results']:
        seed=spec['seed'];started=time.monotonic();root=candidate_root/f'seed{seed}'
        model=read(root/'calibration.json',spec['calibration_sha256']);receipt=read(root/'receipt.json',spec['receipt_sha256'])
        need(receipt['kind']=='v2v4real_native_vehicle_existence_calibration_v1' and receipt['calibration_sha256']==spec['calibration_sha256']
            and receipt['gt_manifest_sha256']==gt_sha256 and receipt['partition_sha256']==PARTITION_SHA
            and receipt['independent_acceptance_verified'] is False and receipt['official_test_used'] is False
            and receipt['paper_metric'] is False and receipt['fit_split']=='train' and model['fit_split']=='train'
            and set(model['fit_groups'])==groups and set(receipt['fit_sequences'])==selected,'candidate fit contract differs')
        semantics=read(candidate_root/f'detector-semantics-seed{seed}.json',receipt['detector_semantics_sha256'])
        need(semantics['kind']=='v2v4real_native_vehicle_detector_semantics_v1' and semantics['native_label_source']==VEHICLE_SOURCE
            and semantics['numeric_class_index']==0 and semantics['source_sha256']==SEMANTICS,'detector single-class vehicle meaning differs')
        for key,digest in SEMANTICS.items():pin(semantics['source_paths'][key],digest)
        predroot=history/f'native-bound-calibration-input-seed{seed}'
        pred=read(predroot/'manifest.json',receipt['prediction_sha256'])
        proof=next(v for v in geometry['results'] if v['seed']==seed)
        need(pred['kind']=='v2v4real_calibration_predictions_v1' and pred['split']=='train' and pred['gt_read'] is False and pred['official_test_read'] is False
            and pred['prepared_projection_sha256']==PROJECTION_SHA and pred['partition_sha256']==PARTITION_SHA
            and pred['checkpoint_sha256']==receipt['checkpoint_sha256']==semantics['checkpoint_sha256']
            and receipt['prediction_sha256']==proof['manifest_sha256'] and pred['rows_sha256']==proof['rows_sha256']
            and proof['rows_checked']==2034 and proof['scores_classes_dimensions_exactly_equal_to_parent'] is True,'old geometry proof does not bind exact new input')
        rows=[json.loads(x) for x in pin(predroot/'predictions.jsonl',pred['rows_sha256']).read_bytes().splitlines()]
        rawroot=history/pred['parent_native_export_task_id']
        index=[json.loads(x) for x in pin(rawroot/'native-feature-index',pred['parent_native_feature_index_sha256']).read_bytes().splitlines()]
        parent_proof=read(rawroot/'independent-raw-feature-admission.json',pred['parent_independent_admission_sha256'])
        need(parent_proof['archive_sha256']==pred['parent_native_export_archive_sha256'],'parent raw archive admission differs')
        raw_index={}
        for row in index:
            if row['sequence_id'] not in selected:continue
            key=(row['sequence_id'],row['frame_id'],row['side']);need(key not in raw_index,'duplicate raw-index row');raw_index[key]=row
        by_key={}
        for row in rows:
            key=row['sequence_id'],row['frame_id'],row['side'];need(key not in by_key,'duplicate prediction source frame');by_key[key]=row
        expected={(*key,side) for key in gt_by_key for side in ('vehicle-side','infrastructure-side')}
        need(set(by_key)==set(raw_index)==expected and len(rows)==2034,'full two-side row coverage differs')
        scores=[];targets=[];offsets=[0];side_count=Counter();side_positive=Counter();frame_records=[]
        for number,key in enumerate(sorted(expected,key=lambda k:(k[0],k[1],0 if k[2]=='vehicle-side' else 1)),1):
            row,raw=by_key[key],raw_index[key];f,centers=gt_by_key[key[:2]]
            original_frame=projected[(key[0],key[1],'0' if key[2]=='vehicle-side' else '1')]
            need(raw['pcd_sha256']==original_frame['pcd_sha256'] and raw['source_to_world']==original_frame['source_to_world']
                and row['frame_ordinal']==raw['frame_ordinal']==f['frame_ordinal'],'raw/projected pose/PCD/frame identity differs')
            s=np.asarray(row['raw_scores'],dtype=float);states=np.asarray(row['states_ego'],dtype=float).reshape(-1,9);classes=np.asarray(row['class_indices'])
            need(len(s)<=64 and states.shape==(len(s),9) and classes.shape==s.shape and np.isfinite(states).all() and np.isfinite(s).all()
                and np.all((s>=.05)&(s<=1)) and np.all(states[:,3:6]>0) and np.all(classes==0)
                and (not len(classes) or classes.dtype.kind in 'iu'),'raw candidate state/class/score domain differs')
            labels=match_labels(states,s,classes,centers,np.zeros(len(centers),dtype=int))
            scores.extend(s.tolist());targets.extend(labels.tolist());offsets.append(len(scores));side_count[key[2]]+=len(s);side_positive[key[2]]+=int(labels.sum())
            frame_records.append(dict(sequence_id=key[0],frame_id=key[1],side=key[2],offset=offsets[-2],candidates=len(s),positives=int(labels.sum())))
            if number%250==0 or number==len(expected):
                elapsed=time.monotonic()-started
                print(json.dumps(dict(kind='independent_vehicle_calibration_progress',seed=seed,completed_rows=number,total_rows=len(expected),
                    ETA_seconds=elapsed/number*(len(expected)-number),eta_scope='matching rows only; numerical verification and remaining seeds excluded')),flush=True)
        scores=np.asarray(scores);targets=np.asarray(targets,dtype=np.uint8)
        need(len(scores)==receipt['examples']==spec['examples'] and int(targets.sum())==receipt['positives']==spec['positives']
            and dict(side_count)==receipt['examples_by_side'] and dict(side_positive)==receipt['positives_by_side'],'independent event/positive counts differ')
        probabilities,numeric=verify_parameters(scores,targets,model)
        compare_diagnostics(receipt['raw_diagnostics'],diagnostics(scores,targets))
        compare_diagnostics(receipt['calibrated_diagnostics'],diagnostics(probabilities,targets))
        np.savez_compressed(output/f'seed{seed}-independent-offline-events.npz',raw_scores=scores,labels=targets,calibrated_probabilities=probabilities,row_offsets=np.asarray(offsets,dtype=np.int64))
        (output/f'seed{seed}-independent-frame-index.jsonl').write_bytes(b''.join(canonical(v)+b'\n' for v in frame_records))
        results.append(dict(seed=seed,source_rows=len(rows),events=len(scores),positives=int(targets.sum()),ego_cav='0',
            geometry_receipt_reused_for_exact_bound_bytes=True,all_raw_index_poses_PCDs_and_ego_reference_verified=True,
            candidate_calibration_sha256=spec['calibration_sha256'],candidate_receipt_sha256=spec['receipt_sha256'],
            all_labels_reconstructed_independently=True,all_calibrated_probabilities_verified=True,
            raw_and_calibrated_diagnostics_independently_verified=True,numeric=numeric))
    need({r['seed'] for r in results}=={1337,2027,3407} and len(results)==3,'exactly three frozen seeds required')
    need(all(sha(p)==h for p,h in pins.items()) and sha(__file__)==checker_sha,'input/source changed during independent verification')
    evidence={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()}
    result=dict(kind='v2v4real_vehicle_calibration_three_seed_independent_numeric_acceptance_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=sys.argv,
        checker_sha256=sha(__file__),numeric_tolerance=dict(relative=RTOL,absolute=ATOL,projected_gradient=PROJECTED_GRADIENT_TOL),
        input_sha256=pins,results=results,offline_evidence=evidence,independent_existence_calibration_accepted=True,
        official_test_used=False,GT_only_in_offline_artifacts=True,old_strict_Car_acceptance_inherited=False,
        candidate_receipts_unchanged=True,native_nine_state_cache_admitted=False,velocity_covariance_fitted=False,paper_performance_complete=False)
    (output/'acceptance-receipt.json').write_bytes(canonical(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('history','candidate-root','execution-sha256','gt-root','gt-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=run(a.history,a.candidate_root,a.execution_sha256,a.gt_root,a.gt_sha256,a.output)
    print(json.dumps(dict(accepted_seeds=[r['seed'] for r in result['results']],receipt=str(Path(a.output)/'acceptance-receipt.json'))))


if __name__=='__main__':main()
