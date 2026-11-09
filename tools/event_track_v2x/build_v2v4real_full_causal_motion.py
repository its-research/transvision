#!/usr/bin/env python3
"""Apply an independently checked train model to full GT-free train/test raw rows.

Outputs remain candidates until separately verified. This executes neither a
detector nor tracking/evaluation and never opens test labels or selects on test.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_raw_motion_inputs import load_raw
from tools.event_track_v2x.v2v4real_causal_detector_motion import estimate


def apply(frame,pred,model):
    side=model['sides'][frame['side']];valid=pred['velocity_valid']
    velocity=pred['velocity_source'].copy()
    velocity[~valid]=side['initial_prior']['velocity_source']
    covariance=np.empty((len(valid),9,9),dtype=np.float64)
    for status,mask in (('matched',valid),('unavailable',~valid)):
        cov=np.asarray(side['statuses'][status]['covariance'])
        if cov.shape!=(9,9) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-12,rtol=0):
            raise ValueError('finite symmetric full9 covariance required')
        np.linalg.cholesky(cov);covariance[mask]=cov
    if not np.isfinite(velocity).all():raise ValueError('finite fitted velocity required')
    return velocity.astype(frame['states_source_legacy7'].dtype),covariance


def execute(args):
    pins={};out=Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh separate full motion output required')
    accepted=json.loads(read(args.fit_acceptance,args.fit_acceptance_sha256,pins))
    if (accepted.get('kind')!='v2v4real_motion_train_candidate_independent_numeric_acceptance_v1'
            or accepted.get('official_test_used') is not False or not all(accepted.get(k) is True for k in (
                'all_train_gate_scores_independently_verified','all_train_dependencies_and_velocities_verified',
                'all_initial_population_priors_verified','all_full9_residual_moments_and_payload_covariances_verified'))):
        raise ValueError('complete independent train fitting acceptance required before raw input')
    model_path=Path(args.model).absolute()
    if str(model_path) not in accepted['input_sha256'] or accepted['input_sha256'][str(model_path)]!=args.model_sha256:
        raise ValueError('exact model from independent train receipt required')
    model=json.loads(read(model_path,args.model_sha256,pins))
    detector_receipts=[p for p in accepted['input_sha256'] if p.endswith('/input-admission-receipt.json')]
    if len(detector_receipts)!=1:raise ValueError('exact fitted detector input receipt required')
    detector=json.loads(read(detector_receipts[0],accepted['input_sha256'][detector_receipts[0]],pins))
    fitted_checkpoint=next(x['checkpoint_sha256'] for x in detector['seeds'] if x['seed']==model['seed'])
    raw_proof=json.loads(read(args.raw_acceptance,args.raw_acceptance_sha256,pins))
    if raw_proof.get('kind')=='v2v4real_full_train_raw_feature_independent_byte_and_semantic_admission_v1':
        if raw_proof.get('official_test_read') is not False or not raw_proof.get('complete_projection_coverage_verified'):
            raise ValueError('independent train raw coverage required')
        archive_sha=raw_proof['archive_sha256'];expected_split='train';seed=raw_proof['seed']
    elif raw_proof.get('kind')=='v2v4real_official_test_raw_independent_cloud_byte_coverage_v1':
        if raw_proof.get('independent_byte_schema_coverage_accepted') is not True:
            raise ValueError('independent official-test raw coverage required')
        records=[x for x in raw_proof['cloud_artifacts'] if x['artifact']=='native-features']
        if len(records)!=1:raise ValueError('exact original raw archive registration required')
        archive_sha=records[0]['sha256'];expected_split='official_test';seed=raw_proof['checkpoint_seed']
    else:raise ValueError('explicit original raw byte acceptance required')
    sources={str(ROOT/'tools/event_track_v2x'/n):sha(ROOT/'tools/event_track_v2x'/n) for n in (
        'build_v2v4real_full_causal_motion.py','v2v4real_raw_motion_inputs.py','v2v4real_causal_detector_motion.py',
        'v2v4real_motion_fit_inputs.py')}
    m,index,frames=load_raw(args.raw_root,args.raw_manifest_sha256,archive_sha,pins,
        progress=lambda n:print(json.dumps(dict(stage='full_motion_raw_input',rows=n)),flush=True))
    if (m['checkpoint_seed']!=seed or model['seed']!=seed or m['split']!=expected_split
            or m['checkpoint_sha256']!=fitted_checkpoint):
        raise ValueError('seed/split/model identity differs')
    out.mkdir(parents=True);(out/'payloads').mkdir()
    (out/'model.json').write_bytes(model_path.read_bytes())
    motion_rows=[];measured=0;history={}
    gates=model['gates']
    for i,(row,frame) in enumerate(zip(index,frames)):
        key=frame['sequence_id'],frame['side']
        pred=estimate(frame,history.get(key),max_speed_mps=gates['max_speed_mps'],max_gap_ordinals=gates['max_gap_ordinals'])
        history[key]=frame
        velocity,covariance=apply(frame,pred,model)
        path=out/'payloads'/f'{i:05d}.npz'
        np.savez_compressed(path,velocity_source=velocity,covariances=covariance)
        measured+=int(pred['velocity_valid'].sum())
        motion_rows.append(dict(raw_row_sha256=frame['row_sha256'],arrays=dict(path=str(path.relative_to(out)),sha256=sha(path)),
            dependencies=pred['dependencies'],velocity_measured_valid=pred['velocity_valid'].tolist(),
            initial_prior_applied=(~pred['velocity_valid']).tolist(),predecessor_candidates=pred['predecessor_candidates'].tolist(),
            reasons=pred['reasons'],sequence_id=frame['sequence_id'],side=frame['side'],frame_ordinal=frame['frame_ordinal']))
        if (i+1)%1000==0:
            print(json.dumps(dict(stage='full_causal_motion_candidate',split=m['split'],seed=seed,
                                 completed_rows=i+1,total_rows=len(frames))),flush=True)
    p=out/'motion-frame-evidence.jsonl';p.open('x').write(''.join(json.dumps(x)+'\n' for x in motion_rows))
    # Recheck all inputs after construction; the archive is streamed again to hash.
    for path,digest in {**sources,**pins}.items():
        h=__import__('hashlib').sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        if h.hexdigest()!=digest:raise ValueError('motion source/input changed during application')
    proof=dict(kind='v2v4real_full_split_causal_motion_candidate_v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        split=m['split'],seed=seed,checkpoint_sha256=m['checkpoint_sha256'],source_rows=len(frames),
        candidates=m['total_candidates'],measured_valid=measured,initial_prior_applied=m['total_candidates']-measured,
        raw_root=str(Path(args.raw_root).absolute()),raw_manifest_sha256=args.raw_manifest_sha256,
        raw_archive_sha256=archive_sha,raw_index_sha256=m['rows_sha256'],model_sha256=args.model_sha256,
        input_sha256=pins,source_sha256=sources,evidence_sha256=sha(p),fit_acceptance_sha256=args.fit_acceptance_sha256,
        online_GT_read=False,test_used_for_fitting_or_selection=False,clock='nominal_10hz',
        independent_motion_acceptance=False,nine_state_cache_admitted=False,tracking_accepted=False,paper_metrics_complete=False)
    p=out/'candidate-receipt.json';p.open('x').write(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),source_rows=len(frames),candidates=m['total_candidates'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit-acceptance','fit-acceptance-sha256','model','model-sha256','raw-root','raw-manifest-sha256',
                 'raw-acceptance','raw-acceptance-sha256','output'):
        parser.add_argument('--'+name,required=True)
    execute(parser.parse_args())
