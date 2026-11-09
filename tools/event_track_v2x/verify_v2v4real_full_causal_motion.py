#!/usr/bin/env python3
"""Independent full-split causal dependencies, velocity priors and covariance IO."""
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
from tools.event_track_v2x.v2v4real_causal_detector_motion_oracle import scalar_estimate


def verify(args):
    root=Path(args.candidate).absolute();out=Path(args.output).absolute();pins={}
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh independent full-motion output required')
    candidate=json.loads(read(root/'candidate-receipt.json',args.candidate_sha256,pins))
    if (candidate.get('kind')!='v2v4real_full_split_causal_motion_candidate_v1'
            or candidate.get('split') not in ('train','official_test')
            or candidate.get('online_GT_read') is not False or candidate.get('test_used_for_fitting_or_selection') is not False):
        raise ValueError('GT-free causal candidate required before arrays')
    for path,digest in candidate['source_sha256'].items():read(path,digest,pins)
    accept_paths=[p for p in candidate['input_sha256'] if 'motion-train-fit-independent-' in p and p.endswith('/acceptance-receipt.json')]
    if len(accept_paths)!=1:raise ValueError('exact independent training fit receipt required')
    fit=json.loads(read(accept_paths[0],candidate['fit_acceptance_sha256'],pins))
    if (fit.get('kind')!='v2v4real_motion_train_candidate_independent_numeric_acceptance_v1'
            or fit['official_test_used'] is not False or not fit['all_full9_residual_moments_and_payload_covariances_verified']):
        raise ValueError('independent train-only initial and full covariance fit required')
    model=json.loads(read(root/'model.json',candidate['model_sha256'],pins))
    model_paths=[p for p,h in fit['input_sha256'].items() if h==candidate['model_sha256'] and p.endswith('/model.json')]
    if len(model_paths)!=1:raise ValueError('exact seed model from independent train fit required')
    if read(model_paths[0],candidate['model_sha256'],pins)!=(root/'model.json').read_bytes():
        raise ValueError('applied model differs from independently fitted model')
    if model['seed']!=candidate['seed'] or model['gates']!={k:fit['gate_selection'][k] for k in ('max_speed_mps','max_gap_ordinals')}:
        raise ValueError('selected gate or seed changed before prediction')
    m,index,frames=load_raw(candidate['raw_root'],candidate['raw_manifest_sha256'],candidate['raw_archive_sha256'],pins,
        progress=lambda n:print(json.dumps(dict(stage='independent_full_motion_raw_input',seed=candidate['seed'],rows=n)),flush=True))
    if (m['split']!=candidate['split'] or m['checkpoint_seed']!=model['seed']
            or m['checkpoint_sha256']!=candidate['checkpoint_sha256'] or m['rows_sha256']!=candidate['raw_index_sha256']):
        raise ValueError('original raw model/seed/split binding differs')
    evidence=[json.loads(x) for x in read(root/'motion-frame-evidence.jsonl',candidate['evidence_sha256'],pins).splitlines()]
    if len(evidence)!=len(frames):raise ValueError('complete source motion evidence required')
    if {p.name for p in root.iterdir()}!={'payloads','model.json','motion-frame-evidence.jsonl','candidate-receipt.json'}:
        raise ValueError('exact full-motion candidate inventory required')
    if {p.name for p in (root/'payloads').iterdir()}!={f'{i:05d}.npz' for i in range(len(frames))}:
        raise ValueError('complete one-file-per-source motion inventory required')
    sources={str(ROOT/'tools/event_track_v2x'/n):sha(ROOT/'tools/event_track_v2x'/n) for n in (
        'verify_v2v4real_full_causal_motion.py','v2v4real_causal_detector_motion_oracle.py',
        'v2v4real_raw_motion_inputs.py','v2v4real_motion_fit_inputs.py')}
    history={};valid_count=0;max_velocity=max_covariance=0.;minimum_eigenvalue=float('inf')
    for side in model['sides'].values():
        for value in side['statuses'].values():
            cov=np.asarray(value['covariance']);np.linalg.cholesky(cov)
            minimum_eigenvalue=min(minimum_eigenvalue,float(np.linalg.eigvalsh(cov)[0]))
    for i,(frame,item) in enumerate(zip(frames,evidence)):
        key=frame['sequence_id'],frame['side'];pred=scalar_estimate(frame,history.get(key),**model['gates'])
        history[key]=frame
        if (item['raw_row_sha256']!=frame['row_sha256'] or item['dependencies']!=pred['dependencies']
                or item['reasons']!=pred['reasons'] or item['predecessor_candidates']!=pred['predecessor_candidates']
                or item['velocity_measured_valid']!=pred['velocity_valid']
                or item['initial_prior_applied']!=[not v for v in pred['velocity_valid']]
                or item['arrays']['path']!=f'payloads/{i:05d}.npz'):
            raise ValueError('causal support, assignment or unavailable/measurement semantics differ')
        path=root/item['arrays']['path'];read(path,item['arrays']['sha256'],pins)
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'velocity_source','covariances'}:raise ValueError('exact GT-free motion payload required')
            velocity,cov=z['velocity_source'].copy(),z['covariances'].copy()
        n=len(frame['states_source_legacy7']);valid=np.asarray(pred['velocity_valid'],dtype=bool)
        if velocity.shape!=(n,2) or velocity.dtype!=frame['states_source_legacy7'].dtype or cov.shape!=(n,9,9):
            raise ValueError('original dtype/order and full covariance shape required')
        if not np.isfinite(velocity).all() or not np.isfinite(cov).all():raise ValueError('finite motion payload required')
        side=model['sides'][frame['side']]
        ref=np.asarray(pred['velocity_source'],dtype=float).reshape(-1,2)
        ref[~valid]=side['initial_prior']['velocity_source'];ref=ref.astype(velocity.dtype)
        np.testing.assert_allclose(velocity,ref,atol=1e-4,rtol=1e-4)
        if n:
            max_velocity=max(max_velocity,float(np.max(np.abs(velocity-ref))))
            np.testing.assert_allclose(cov,cov.transpose(0,2,1),atol=1e-12,rtol=0)
            np.linalg.cholesky(cov)
        for status,mask in (('matched',valid),('unavailable',~valid)):
            expected=np.broadcast_to(np.asarray(side['statuses'][status]['covariance']),(int(mask.sum()),9,9))
            np.testing.assert_allclose(cov[mask],expected,atol=1e-4,rtol=1e-4)
            if mask.any():max_covariance=max(max_covariance,float(np.max(np.abs(cov[mask]-expected))))
        valid_count+=int(valid.sum())
        if (i+1)%1000==0:
            print(json.dumps(dict(stage='independent_full_causal_motion',split=m['split'],seed=model['seed'],
                completed_rows=i+1,total_rows=len(frames))),flush=True)
    for path,digest in {**pins,**sources,**candidate['input_sha256']}.items():
        h=__import__('hashlib').sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        if h.hexdigest()!=digest:raise ValueError('source/input/payload changed during independent verification')
    if valid_count!=candidate['measured_valid'] or m['total_candidates']!=candidate['candidates']:
        raise ValueError('full measured/unavailable candidate population differs')
    proof=dict(kind='v2v4real_full_split_causal_motion_independent_numeric_acceptance_v1',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),split=m['split'],seed=model['seed'],
        candidate_receipt_sha256=args.candidate_sha256,fit_acceptance_sha256=candidate['fit_acceptance_sha256'],
        model_sha256=candidate['model_sha256'],source_rows=len(frames),sequences=len(m['sequence_ids']),
        candidates=m['total_candidates'],measured_valid=valid_count,initial_prior_applied=m['total_candidates']-valid_count,
        max_abs_velocity_error=max_velocity,max_abs_covariance_error=max_covariance,minimum_model_eigenvalue=minimum_eigenvalue,
        causal_source_only_verified=True,initial_population_prior_application_verified=True,
        all_full9_payload_covariances_verified=True,independent_acceptance_verified=True,
        online_GT_read=False,test_used_for_fitting_or_selection=False,clock='nominal_10hz',
        input_sha256=pins,source_sha256=sources,first_observation_velocity_physically_measured=False,
        nine_state_cache_admitted=False,tracking_accepted=False,paper_metrics_complete=False)
    out.mkdir(parents=True);p=out/'acceptance-receipt.json';p.open('x').write(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),source_rows=len(frames),candidates=m['total_candidates'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','candidate-sha256','output'):parser.add_argument('--'+name,required=True)
    verify(parser.parse_args())
