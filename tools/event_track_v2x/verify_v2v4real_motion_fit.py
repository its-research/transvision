#!/usr/bin/env python3
"""Independently recompute the complete train motion candidate and selection."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import load, read, sha
from tools.event_track_v2x.v2v4real_causal_detector_motion_oracle import scalar_estimate
from tools.event_track_v2x.v2v4real_motion_fit_oracle import match, initial_prior, errors, moments, SIDES

ATOL = RTOL = 1e-4


def verify(args):
    output = Path(args.output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh separate independent receipt required')
    root = Path(args.candidate).absolute()
    pins = {}
    receipt = json.loads(read(root/'candidate-receipt.json', args.candidate_sha256, pins))
    if (receipt.get('kind') != 'v2v4real_motion_covariance_train_fit_candidate_v1'
            or receipt.get('fit_split') != 'train' or receipt.get('official_test_used') is not False):
        raise ValueError('train-only motion candidate required before any arrays')
    for p, expected in receipt['source_sha256'].items():
        read(p, expected, pins)
    recipe = json.loads(read(root/'recipe-before-fitting.json', receipt['recipe_sha256'], pins))['recipe']
    # These constants belong to this oracle, rather than being imported from the producer.
    if (recipe['speed_candidates_mps'] != [5.,10.,20.,40.] or recipe['gap_candidates_ordinals'] != [1,3,10]
            or recipe['shrinkage'] != .05 or recipe['variance_floor'] != 1e-6
            or recipe['role'] != 'calibration_fit' or recipe['official_test_used'] is not False
            or recipe['atol'] != ATOL or recipe['rtol'] != RTOL
            or recipe['minimum_prior_samples'] != 10 or recipe['minimum_covariance_samples_per_side_status'] != 10):
        raise ValueError('independent frozen recipe constants differ')
    selection = json.loads(read(root/'selection.json', receipt['selection_sha256'], pins))
    inputpins = receipt['input_sha256']
    detector_paths = [p for p in inputpins if p.endswith('/input-admission-receipt.json')]
    target_paths = [p for p in inputpins if 'motion-train-target-independent-' in p and p.endswith('/acceptance-receipt.json')]
    partition_paths = [p for p in inputpins if p.endswith('/nested-partition')]
    if tuple(map(len,(detector_paths,target_paths,partition_paths))) != (1,1,1):
        raise ValueError('exact original motion input declarations required')
    seeds, targets, groups, admitted, _ = load(detector_paths[0],inputpins[detector_paths[0]],
        target_paths[0],inputpins[target_paths[0]],partition_paths[0],inputpins[partition_paths[0]])
    if admitted != inputpins or groups != receipt['fit_groups'] or groups != selection['groups']:
        raise ValueError('complete fit/selection input coverage differs')
    pins.update(admitted)
    initial = initial_prior(targets,set(groups))
    folds = {g: initial_prior(targets,set(groups)-{g}) for g in groups}
    for group in groups:
        for side in SIDES:
            if folds[group][side]['samples'] != selection['fold_priors'][group][side]['samples']:
                raise ValueError('held-out prior sample count differs')
            np.testing.assert_allclose(folds[group][side]['velocity_source'],
                selection['fold_priors'][group][side]['velocity_source'],atol=ATOL,rtol=RTOL)
    oracle_sources = {str(ROOT/'tools/event_track_v2x'/n):sha(ROOT/'tools/event_track_v2x'/n) for n in (
        'verify_v2v4real_motion_fit.py','v2v4real_motion_fit_oracle.py',
        'v2v4real_causal_detector_motion_oracle.py','v2v4real_motion_fit_inputs.py')}
    output.mkdir(parents=True)
    scores = {(s,g):[] for s in (5.,10.,20.,40.) for g in (1,3,10)}
    results = []
    max_velocity_error = max_covariance_error = 0.
    for seed, rows in sorted(seeds.items()):
        info = next(x for x in receipt['seeds'] if x['seed'] == seed)
        folder = root/f'seed{seed}'
        artifacts = info['artifacts']
        model = json.loads(read(folder/'model.json',artifacts['model.json']['sha256'],pins))
        with np.load(folder/'motion-arrays.npz',allow_pickle=False) as z:
            read(folder/'motion-arrays.npz',artifacts['motion-arrays.npz']['sha256'],pins)
            a = {k:z[k].copy() for k in z.files}
        expected_fields={'velocity_source','velocity_measured_valid','covariances','initial_prior_applied',
                         'offsets','predecessor_candidates','raw_velocity_source'}
        if set(a) != expected_fields or len(a['offsets']) != len(rows)+1:
            raise ValueError('exact full train motion output required')
        index = [json.loads(x) for x in read(folder/'motion-frame-index.jsonl',
            artifacts['motion-frame-index.jsonl']['sha256'],pins).splitlines()]
        if len(index) != len(rows): raise ValueError('full train source coverage differs')
        residual_samples = {(side,status):[] for side in SIDES for status in ('matched','unavailable')}
        accum = {(s,g):{group:dict(losses=[],samples=0) for group in groups}
                 for s in (5.,10.,20.,40.) for g in (1,3,10)}
        history = {}
        matched_count = 0
        for n, (row, actual_index) in enumerate(zip(rows,index)):
            frame = {k:row[k] for k in ('sequence_id','side','frame_ordinal','source_to_world',
                                      'states_source_legacy7','class_indices','row_sha256')}
            key = row['sequence_id'],row['side']; previous=history.get(key)
            gt_matches=match(row['states_source_legacy7'],row['raw_scores'],row['target_states_source'])
            gt_ids=[(i,j) for i,j in enumerate(gt_matches) if j>=0 and row['target_velocity_valid'][j]]
            selected_pred = None
            for speed in (5.,10.,20.,40.):
                base=scalar_estimate(frame,previous,max_speed_mps=speed,max_gap_ordinals=10)
                for gap in (1,3,10):
                    pred=base
                    if previous is not None and row['frame_ordinal']-previous['frame_ordinal']>gap:
                        pred=deepcopy(base); count=len(row['raw_scores'])
                        pred.update(velocity_source=[[math.nan,math.nan] for _ in range(count)],
                            velocity_valid=[False]*count,predecessor_candidates=[-1]*count,
                            reasons=['gap_exceeded']*count)
                    errors2=[]
                    for i,j in gt_ids:
                        v=(pred['velocity_source'][i] if pred['velocity_valid'][i]
                           else folds[row['group']][row['side']]['velocity_source'])
                        y=row['target_velocity_source'][j]
                        errors2.append(math.fsum((float(v[k])-float(y[k]))**2 for k in range(2)))
                    record=accum[speed,gap][row['group']]
                    record['losses'].append(math.fsum(errors2));record['samples']+=len(gt_ids)
                    if dict(max_speed_mps=speed,max_gap_ordinals=gap)==selection['selected']:
                        selected_pred=pred
            pred=selected_pred
            if pred is None: raise ValueError('selected gate outside original candidates')
            history[key]=frame
            start,end=a['offsets'][n:n+2]
            if (actual_index['raw_row_sha256']!=row['row_sha256'] or actual_index['offset']!=start
                    or end-start!=len(row['raw_scores']) or actual_index['dependencies']!=pred['dependencies']
                    or actual_index['reasons']!=pred['reasons']):
                raise ValueError('row dependencies, reasons or candidate coverage differ')
            valid=np.asarray(pred['velocity_valid'],dtype=bool)
            np.testing.assert_array_equal(a['velocity_measured_valid'][start:end],valid)
            np.testing.assert_array_equal(a['initial_prior_applied'][start:end],~valid)
            np.testing.assert_array_equal(a['predecessor_candidates'][start:end],pred['predecessor_candidates'])
            raw=np.asarray(pred['velocity_source'],dtype=float).reshape(-1,2)
            np.testing.assert_allclose(a['raw_velocity_source'][start:end],raw,atol=ATOL,rtol=RTOL,equal_nan=True)
            fitted=raw.copy();fitted[~valid]=initial[row['side']]['velocity_source']
            np.testing.assert_allclose(a['velocity_source'][start:end],fitted,atol=ATOL,rtol=RTOL)
            if len(fitted): max_velocity_error=max(max_velocity_error,float(np.max(np.abs(a['velocity_source'][start:end]-fitted))))
            matched_count+=int(valid.sum())
            for status, samples in errors(row,pred,gt_matches,initial).items():
                residual_samples[row['side'],status].extend(samples)
            if (n+1)%250==0:
                print(json.dumps(dict(stage='independent_train_motion_all_gate_reconstruction',
                                     seed=seed,completed_rows=n+1,total_rows=len(rows))),flush=True)
        for gate,group_records in accum.items():
            for group,record in group_records.items():
                loss=math.fsum(record['losses']);count=record['samples']
                scores[gate].append(dict(seed=seed,group=group,samples=count,
                    sum_squared_vector_error=loss,vector_MSE=loss/count))
        for side in SIDES:
            actual=model['sides'][side]
            if actual['initial_prior']['samples']!=initial[side]['samples']:
                raise ValueError('full train initial prior sample count differs')
            np.testing.assert_allclose(actual['initial_prior']['velocity_source'],initial[side]['velocity_source'],atol=ATOL,rtol=RTOL)
            for status in ('matched','unavailable'):
                ref=moments(residual_samples[side,status]); produced=actual['statuses'][status]
                if ref['samples']!=produced['samples']:raise ValueError('full9 residual sample count differs')
                for name in ('residual_mean','residual_second_moment','covariance'):
                    np.testing.assert_allclose(produced[name],ref[name],atol=ATOL,rtol=RTOL)
                covariance=np.asarray(ref['covariance'])
                max_covariance_error=max(max_covariance_error,float(np.max(np.abs(covariance-np.asarray(produced['covariance'])))))
                # Verify each of the persisted covariance matrices, rather than only model metadata.
                for n,row in enumerate(rows):
                    if row['side']!=side:continue
                    start,end=a['offsets'][n:n+2];valid=a['velocity_measured_valid'][start:end]
                    mask=valid if status=='matched' else ~valid
                    expected=np.broadcast_to(covariance,(int(mask.sum()),9,9))
                    np.testing.assert_allclose(a['covariances'][start:end][mask],expected,atol=ATOL,rtol=RTOL)
        if matched_count!=info['measured_valid'] or len(a['velocity_source'])!=info['candidates']:
            raise ValueError('candidate/validity summary differs')
        result=dict(seed=seed,source_rows=len(rows),candidates=len(a['velocity_source']),measured_valid=matched_count,
            full9_residual_samples={s:{k:len(residual_samples[s,k]) for k in ('matched','unavailable')} for s in SIDES})
        results.append(result);print(json.dumps(dict(stage='independent_motion_seed_complete',**result)),flush=True)
        del a
    reference_scores=[]
    for gate,fold_scores in scores.items():
        actual=next(x for x in selection['scores'] if (x['max_speed_mps'],x['max_gap_ordinals'])==gate)
        for ref in fold_scores:
            item=next(x for x in actual['folds'] if (x['seed'],x['group'])==(ref['seed'],ref['group']))
            if ref['samples']!=item['samples']:raise ValueError('gate scoring population changed')
            for key in ('sum_squared_vector_error','vector_MSE'):
                np.testing.assert_allclose(item[key],ref[key],atol=ATOL,rtol=RTOL)
        score=math.fsum(x['vector_MSE'] for x in fold_scores)/len(fold_scores)
        np.testing.assert_allclose(actual['mean_seed_group_vector_MSE'],score,atol=ATOL,rtol=RTOL)
        reference_scores.append(dict(max_speed_mps=gate[0],max_gap_ordinals=gate[1],mean_seed_group_vector_MSE=score))
    chosen=min(reference_scores,key=lambda x:(x['mean_seed_group_vector_MSE'],x['max_speed_mps'],x['max_gap_ordinals']))
    if selection['selected']!={k:chosen[k] for k in ('max_speed_mps','max_gap_ordinals')}:
        raise ValueError('independent gate selection differs')
    for path,digest in {**pins,**oracle_sources}.items():
        if sha(path)!=digest:raise ValueError('input/source changed during independent reconstruction')
    proof=dict(kind='v2v4real_motion_train_candidate_independent_numeric_acceptance_v1',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),candidate_receipt_sha256=args.candidate_sha256,
        input_sha256=pins,oracle_source_sha256=oracle_sources,seeds=results,gate_selection=chosen,
        max_abs_velocity_error=max_velocity_error,max_abs_covariance_error=max_covariance_error,atol=ATOL,rtol=RTOL,
        all_train_gate_scores_independently_verified=True,all_train_dependencies_and_velocities_verified=True,
        all_initial_population_priors_verified=True,all_full9_residual_moments_and_payload_covariances_verified=True,
        first_frame_velocity_physically_measured=False,false_positive_uncertainty_calibrated=False,
        official_test_used=False,official_test_motion_verified=False,complete_train_motion_contract_verified=False,
        nine_state_cache_admitted=False,tracking_accepted=False,paper_metrics_complete=False)
    p=output/'acceptance-receipt.json';p.open('x').write(json.dumps(proof,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),results=results)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','candidate-sha256','output'):parser.add_argument('--'+name,required=True)
    verify(parser.parse_args())
