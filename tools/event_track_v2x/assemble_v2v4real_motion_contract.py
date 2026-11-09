#!/usr/bin/env python3
"""Package independently checked full motion into the existing bridge contract.

This copies accepted arrays unchanged and derives metadata from independent
receipts. It does not estimate velocity, refit noise, or accept a tracking cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha

CONFIG_SHA='138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1'
SOURCE_SHA='f47f391db33e02813b05e10f598ecf9240877463a8e32b61bc5841a8769d38ea'


def write(p,v):
    p.open('x').write(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    return dict(path=p.name,sha256=sha(p))


def assemble(args):
    candidate=Path(args.candidate).absolute();out=Path(args.output).absolute();pins={}
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh separate motion contract required')
    admission_path=Path(args.independent_acceptance).absolute()
    admitted=json.loads(read(admission_path,args.independent_sha256,pins))
    if (admitted.get('kind')!='v2v4real_full_split_causal_motion_independent_numeric_acceptance_v1'
            or not all(admitted.get(k) is True for k in ('causal_source_only_verified',
                'initial_population_prior_application_verified','all_full9_payload_covariances_verified','independent_acceptance_verified'))
            or admitted.get('online_GT_read') is not False or admitted.get('test_used_for_fitting_or_selection') is not False):
        raise ValueError('full independent source motion required before packaging')
    receipt=json.loads(read(candidate/'candidate-receipt.json',admitted['candidate_receipt_sha256'],pins))
    if (admitted['split'],admitted['seed'],admitted['source_rows'],admitted['candidates'])!=(
            receipt['split'],receipt['seed'],receipt['source_rows'],receipt['candidates']):
        raise ValueError('full source admission scope differs')
    model_bytes=read(candidate/'model.json',admitted['model_sha256'],pins)
    model=json.loads(model_bytes)
    fit_paths=[p for p in receipt['input_sha256'] if 'motion-train-fit-independent-' in p and p.endswith('/acceptance-receipt.json')]
    if len(fit_paths)!=1:raise ValueError('exact independent training receipt required')
    fit=json.loads(read(fit_paths[0],admitted['fit_acceptance_sha256'],pins))
    if fit['official_test_used'] is not False or not fit['all_full9_residual_moments_and_payload_covariances_verified']:
        raise ValueError('independent full9 training fit required')
    candidate_fit_paths=[p for p in fit['input_sha256'] if 'motion-train-fit-candidate-' in p and p.endswith('/candidate-receipt.json')]
    if len(candidate_fit_paths)!=1:raise ValueError('exact original fitted model receipt required')
    training=json.loads(read(candidate_fit_paths[0],fit['candidate_receipt_sha256'],pins))
    raw=Path(receipt['raw_root']);manifest=json.loads(read(raw/'native-feature-manifest',receipt['raw_manifest_sha256'],pins))
    read(raw/'native-feature-index',receipt['raw_index_sha256'],pins)
    if manifest['checkpoint_seed']!=admitted['seed']:raise ValueError('checkpoint seed differs')
    projection_sha=args.projection_frames_sha256
    read(args.projection_frames,projection_sha,pins)
    source=read(args.original_source,SOURCE_SHA,pins)
    # Only the original SHA-pinned config is copied, never a guessed or rewritten YAML.
    import io
    with tarfile.open(fileobj=io.BytesIO(source),mode='r:gz') as t:
        member=t.getmember('official/opencood/hypes_yaml/point_pillar_late_fusion.yaml')
        if not member.isfile():raise ValueError('ordinary original config required')
        config=t.extractfile(member).read()
    if hashlib.sha256(config).hexdigest()!=CONFIG_SHA:raise ValueError('original detector config SHA differs')
    evidence=[json.loads(x) for x in read(candidate/'motion-frame-evidence.jsonl',receipt['evidence_sha256'],pins).splitlines()]
    if len(evidence)!=admitted['source_rows']:raise ValueError('full motion evidence required')
    out.mkdir(parents=True);(out/'payloads').mkdir();(out/'implementation').mkdir()
    (out/'model.json').write_bytes(model_bytes);(out/'detector-config.yaml').write_bytes(config)
    for p,digest in receipt['source_sha256'].items():
        b=read(p,digest,pins);(out/'implementation'/Path(p).name).write_bytes(b)
    implementation=out/'implementation/build_v2v4real_full_causal_motion.py'
    method=write(out/'method.json',dict(kind='v2v4real_causal_motion_covariance_method_v1',
        velocity_method='past_only_adjacent_source_unique_mutual_nearest_3D_world_centre_nominal_10hz_train_selected_gates',
        initial_observation_policy=model['recipe']['initial_policy']+'; '+model['recipe']['first_observation_transfer'],
        full_covariance_method=model['recipe']['covariance_policy'],
        implementation=dict(path='implementation/'+implementation.name,sha256=sha(implementation))))
    model_record=dict(path='model.json',sha256=sha(out/'model.json'))
    fit_record=write(out/'fit-receipt.json',dict(kind='v2v4real_motion_covariance_train_fit_v1',fit_split='train',
        official_test_used=False,fit_groups=training['fit_groups'],seed=admitted['seed'],model_sha256=model_record['sha256'],
        method_sha256=method['sha256'],original_candidate_receipt_sha256=fit['candidate_receipt_sha256'],
        independent_train_numeric_admission_sha256=admitted['fit_acceptance_sha256'],
        fit_selection_role='calibration_fit_leave_one_frozen_name_group_out',
        first_observation_population_prior=True,first_observation_velocity_physically_measured=False,
        false_positive_uncertainty_calibrated=False))
    (out/'original-fit-independent-admission.json').write_bytes(Path(fit_paths[0]).read_bytes())
    (out/'original-full-motion-independent-admission.json').write_bytes(admission_path.read_bytes())
    independent=write(out/'independent-admission.json',dict(kind='v2v4real_motion_covariance_independent_admission_v1',
        causal_source_only_verified=True,initial_observation_policy_verified=True,
        full_nine_state_covariance_verified=True,independent_acceptance_verified=True,
        method_sha256=method['sha256'],model_sha256=model_record['sha256'],fit_receipt_sha256=fit_record['sha256'],
        original_fit_numeric_admission_sha256=sha(out/'original-fit-independent-admission.json'),
        original_full_motion_numeric_admission_sha256=sha(out/'original-full-motion-independent-admission.json'),
        split=admitted['split'],seed=admitted['seed'],source_rows=admitted['source_rows'],
        numerical_policy_application_only=True,first_observation_velocity_physically_measured=False,
        nine_state_cache_admitted=False,tracking_accepted=False,paper_metrics_complete=False))
    clock=write(out/'clock.json',dict(protocol_id='v2v4real-nominal-10hz-formal-v1',dataset='v2v4real',
        nominal_rate_hz=10,period_us=100000,measured=False,origin='zero_relative_nominal_time_per_published_sequence'))
    frames=[]
    for item in evidence:
        p=candidate/item['arrays']['path'];digest=item['arrays']['sha256']
        if admitted['input_sha256'].get(str(p))!=digest:
            raise ValueError('payload not covered by full independent admission')
        read(p,digest,pins);target=out/item['arrays']['path'];shutil.copyfile(p,target)
        if sha(target)!=digest:raise ValueError('motion copy bytes differ')
        frames.append({k:item[k] for k in ('raw_row_sha256','arrays','dependencies')})
    contract=write(out/'motion-contract.json',dict(kind='v2v4real_source_local_causal_motion_covariance_v1',
        dataset='v2v4real',split=admitted['split'],raw_manifest_sha256=receipt['raw_manifest_sha256'],
        raw_index_sha256=receipt['raw_index_sha256'],raw_archive_sha256=receipt['raw_archive_sha256'],
        projection_frames_sha256=projection_sha,clock_sha256=clock['sha256'],
        state_layout='mmdet3d-legacy-gravity-dimxyz-yaw-vxy',coordinate_system='source_lidar',
        velocity_units='metres_per_second',reference_time='source_capture',gt_read_online=False,cross_source_inputs=False,
        detector_config=dict(path='detector-config.yaml',sha256=CONFIG_SHA),method=method,model=model_record,
        fit_receipt=fit_record,independent_admission=independent,
        sequence_origin_us={s:0 for s in manifest['sequence_ids']},frames=frames))
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('packaging inputs changed')
    write(out/'assembly-receipt.json',dict(kind='v2v4real_motion_contract_independent_evidence_packaging_v1',
        contract_sha256=contract['sha256'],input_sha256=pins,source_sha256=sha(__file__),
        source_rows=len(frames),split=admitted['split'],seed=admitted['seed'],
        motion_bytes_unchanged=True,motion_math_rerun=False,nine_state_cache_admitted=False,paper_metrics_complete=False))
    print(json.dumps(dict(contract=str(out/'motion-contract.json'),sha256=contract['sha256'],frames=len(frames))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','independent-acceptance','independent-sha256','projection-frames',
                 'projection-frames-sha256','original-source','output'):
        parser.add_argument('--'+name,required=True)
    assemble(parser.parse_args())
