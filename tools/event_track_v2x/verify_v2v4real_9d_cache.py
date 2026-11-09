#!/usr/bin/env python3
"""Independent cache byte mapping and scalar probability transformation.

No bridge, cache-loader or calibration application functions are imported.
Motion mathematics is reused only for the exact NPZ bytes already accepted by
the separate full scalar motion oracle. This is not a tracking metric result.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha

ARRAYS={'states','raw_scores','scores','class_indices','covariances','appearance','appearance_valid'}
META_FIELDS={'kind','schema_version','sequence_id','frame_id','side','box_reference_timestamp_us',
    'source_image_timestamp_us','coordinate_system','lidar_to_world_row_rotation','lidar_to_world_translation',
    'image_sha256','box_layout','agent_mask','dataset_split','dataset_sha256','detector_config_sha256',
    'detector_checkpoint_sha256','feature_checkpoint_sha256','feature_method','calibration_sha256',
    'calibration_fit_split','raw_manifest_sha256','raw_arrays_sha256','raw_metadata_sha256','arrays_sha256'}


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def probability(values,slope,intercept):
    result=[]
    for x in values:
        p=min(1-1e-7,max(1e-7,float(x)));z=slope*(math.log(p)-math.log1p(-p))+intercept
        if z>=0:result.append(1/(1+math.exp(-z)))
        else:
            e=math.exp(z);result.append(e/(1+e))
    return np.asarray(result,dtype=np.float64)


def verify(args):
    scope=Path(args.scope).absolute();out=Path(args.output).absolute();pins={}
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh separate independent cache receipt required')
    bridge_path=scope/'cache-raw-bridge-receipt.json'
    bridge=json.loads(read(bridge_path,args.bridge_sha256,pins))
    if (bridge.get('kind')!='v2v4real_raw_top64_explicit_motion_contract_bridge_v1' or bridge.get('fixture') is not False
            or bridge.get('gt_payload_opened') is not False or not bridge.get('motion_independent_admission_imported')):
        raise ValueError('actual GT-free bridge with independent motion required')
    contract_path=scope/'contract/motion-contract.json'
    contract=json.loads(read(contract_path,bridge['motion_contract_sha256'],pins))
    def asset(name):
        spec=contract[name];return json.loads(read(contract_path.parent/spec['path'],spec['sha256'],pins))
    declaration=asset('independent_admission');model=asset('model');fit=asset('fit_receipt')
    method=asset('method')
    full_path=contract_path.parent/'original-full-motion-independent-admission.json'
    full=json.loads(read(full_path,declaration['original_full_motion_numeric_admission_sha256'],pins))
    training_path=contract_path.parent/'original-fit-independent-admission.json'
    training=json.loads(read(training_path,declaration['original_fit_numeric_admission_sha256'],pins))
    if (full['independent_acceptance_verified'] is not True or full['model_sha256']!=contract['model']['sha256']
            or full['fit_acceptance_sha256']!=declaration['original_fit_numeric_admission_sha256']
            or training['official_test_used'] is not False or fit['official_test_used'] is not False
            or model['seed']!=full['seed'] or method['kind']!='v2v4real_causal_motion_covariance_method_v1'):
        raise ValueError('original full motion/train numeric admission differs')
    original_candidates=[p for p in full['input_sha256'] if p.endswith('/candidate-receipt.json')]
    if len(original_candidates)!=1:raise ValueError('exact original full-motion candidate required')
    original_candidate_path=Path(original_candidates[0])
    original_candidate=json.loads(read(original_candidate_path,full['candidate_receipt_sha256'],pins))
    original_evidence=[json.loads(x) for x in read(original_candidate_path.parent/'motion-frame-evidence.jsonl',
        original_candidate['evidence_sha256'],pins).splitlines()]
    original_motion={x['raw_row_sha256']:x for x in original_evidence}
    raw=Path(args.raw_root).absolute();m=json.loads(read(raw/'native-feature-manifest',bridge['raw_manifest_sha256'],pins))
    rows=[json.loads(x) for x in read(raw/'native-feature-index',m['rows_sha256'],pins).splitlines()]
    bysha={hashlib.sha256(canonical(r)).hexdigest():r for r in rows}
    motion={x['raw_row_sha256']:x for x in contract['frames']}
    if len(rows)!=len(bysha) or set(motion)!=set(bysha):raise ValueError('complete unique raw/motion support required')
    cache=scope/'cache';cm=json.loads(read(cache/'manifest.json',bridge['cache_sha256'],pins))
    if (cm['split']!=m['split'] or cm['gt_in_cache'] is not False or cm['fixture'] is not False
            or cm['frame_count']!=len(rows) or cm['detection_count']!=m['total_candidates']):
        raise ValueError('full native cache scope differs')
    clock=json.loads(read(scope/'contract/clock.json',contract['clock_sha256'],pins))
    if (clock.get('period_us')!=100000 or clock.get('nominal_rate_hz')!=10 or clock.get('measured') is not False):
        raise ValueError('original nominal 10 Hz clock required')
    entries={};allfiles={'manifest.json'}
    for entry in cm['frames']:
        spec=entry['metadata'];mb=read(cache/spec['path'],spec['sha256'],pins);meta=json.loads(mb)
        if (set(meta)!=META_FIELDS or mb!=canonical(meta) or meta['raw_metadata_sha256'] not in bysha
                or meta['raw_metadata_sha256'] in entries):
            raise ValueError('unique canonical raw-bound cache metadata required')
        r=bysha[meta['raw_metadata_sha256']];pose=np.asarray(r['source_to_world'])
        stamp=contract['sequence_origin_us'][r['sequence_id']]+r['frame_ordinal']*clock['period_us']
        for k,v in dict(kind='detection_cache_v2',schema_version=2,sequence_id=r['sequence_id'],frame_id=r['frame_id'],side=r['side'],dataset_split=m['split'],
            raw_manifest_sha256=bridge['raw_manifest_sha256'],raw_arrays_sha256=r['payload_sha256'],
            dataset_sha256=m['prepared_projection_sha256'],detector_config_sha256=contract['detector_config']['sha256'],
            agent_mask=1 if r['side']=='vehicle-side' else 2,
            image_sha256=hashlib.sha256(b'v2v4real-lidar-only-no-image-v1').hexdigest(),
            detector_checkpoint_sha256=m['checkpoint_sha256'],feature_checkpoint_sha256=m['checkpoint_sha256'],
            box_reference_timestamp_us=stamp,source_image_timestamp_us=stamp,coordinate_system='source_lidar',
            box_layout='mmdet3d-legacy-gravity-dimxyz-yaw-vxy',feature_method='rbf-pointpillar-bev128-bilinear-l2-v1',
            calibration_fit_split='train').items():
            if meta[k]!=v:raise ValueError('cache provenance/time/feature differs: '+k)
        if not np.array_equal(meta['lidar_to_world_row_rotation'],pose[:3,:3].T) or not np.array_equal(meta['lidar_to_world_translation'],pose[:3,3]):
            raise ValueError('original pose changed in cache')
        entries[r['payload']]=(entry,meta,mb)
        allfiles.update([entry[k]['path'] for k in ('metadata','arrays')])
    if len(entries)!=len(rows) or allfiles!={p.name for p in cache.iterdir()}:
        raise ValueError('complete cache inventory including empty source frames required')
    paths=[p for p in bridge['input_sha256'] if p.endswith('/calibration.json')]
    if len(paths)!=1:raise ValueError('exact fitted existence calibration required')
    cal=json.loads(read(paths[0],bridge['input_sha256'][paths[0]],pins))
    if cal['fit_split']!='train':raise ValueError('train-only calibration required')
    original=raw/'native-features';h=hashlib.sha256()
    with original.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    if h.hexdigest()!=bridge['raw_archive_sha256']:raise ValueError('original raw archive changed')
    completed=count=0;maximum_score_error=0.;seen=set()
    with tarfile.open(original,'r:*') as tar:
        for member in tar:
            if member.name not in entries:continue
            if not member.isfile() or member.name in seen:raise ValueError('ordinary unique original raw member required')
            seen.add(member.name);entry,meta,mb=entries[member.name];row=bysha[meta['raw_metadata_sha256']]
            if meta['calibration_sha256']!=bridge['input_sha256'][paths[0]]:
                raise ValueError('calibration model byte provenance differs')
            b=tar.extractfile(member).read()
            if hashlib.sha256(b).hexdigest()!=row['payload_sha256']:raise ValueError('original raw payload changed')
            with np.load(io.BytesIO(b),allow_pickle=False) as z:raw_arrays={k:z[k].copy() for k in z.files}
            spec=entry['arrays'];read(cache/spec['path'],spec['sha256'],pins)
            with np.load(cache/spec['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            if set(a)!=ARRAYS:raise ValueError('GT-free native array inventory differs')
            item=motion[meta['raw_metadata_sha256']];p=contract_path.parent/item['arrays']['path']
            original_item=original_motion[meta['raw_metadata_sha256']]
            original_payload=original_candidate_path.parent/original_item['arrays']['path']
            if (item['arrays']['sha256']!=original_item['arrays']['sha256']
                    or item['dependencies']!=original_item['dependencies']
                    or full['input_sha256'].get(str(original_payload))!=item['arrays']['sha256']):
                raise ValueError('copied motion payload not covered by full independent acceptance')
            read(p,item['arrays']['sha256'],pins)
            with np.load(p,allow_pickle=False) as z:v,c=z['velocity_source'].copy(),z['covariances'].copy()
            for name in ('raw_scores','class_indices','appearance','appearance_valid'):
                if (a[name].dtype!=raw_arrays[name].dtype or a[name].shape!=raw_arrays[name].shape
                        or a[name].tobytes()!=raw_arrays[name].tobytes()):
                    raise ValueError('original raw array bytes changed: '+name)
            s=a['states'];r=raw_arrays['states_source_legacy7']
            if (s.shape!=(len(r),9) or s.dtype!=r.dtype or s[:,:7].tobytes()!=r.tobytes()
                    or s[:,7:].tobytes()!=v.tobytes() or a['covariances'].tobytes()!=c.tobytes()
                    or a['covariances'].dtype!=c.dtype or a['covariances'].shape!=c.shape):
                raise ValueError('state or admitted full covariance bytes changed')
            expected=probability(raw_arrays['raw_scores'],cal['slope'],cal['intercept'])
            if a['scores'].shape!=expected.shape:raise ValueError('calibrated probability shape differs')
            np.testing.assert_allclose(a['scores'],expected,atol=1e-12,rtol=1e-10)
            if len(expected):maximum_score_error=max(maximum_score_error,float(np.max(np.abs(a['scores']-expected))))
            h=hashlib.sha256(mb)
            for k in sorted(a):
                x=a[k];h.update(k.encode());h.update(x.dtype.str.encode());h.update(canonical(list(x.shape)));h.update(x.tobytes())
            if h.hexdigest()!=entry['frame_sha256']:raise ValueError('decoded native frame digest differs')
            if not np.isfinite(s).all() or not np.isfinite(a['covariances']).all():raise ValueError('finite full9 cache required')
            np.linalg.cholesky(a['covariances']);completed+=1;count+=len(r)
            if completed%1000==0:print(json.dumps(dict(stage='independent_9d_cache_readback',completed_rows=completed,total_rows=len(rows))),flush=True)
    if completed!=len(rows) or count!=m['total_candidates']:raise ValueError('complete original candidate coverage differs')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('cache/contract input changed during independent readback')
    proof=dict(kind='v2v4real_native_9d_cache_independent_byte_numeric_readback_v1',split=m['split'],seed=full['seed'],
        frames=completed,candidates=count,cache_manifest_sha256=bridge['cache_sha256'],bridge_receipt_sha256=args.bridge_sha256,
        original_raw_arrays_and_seven_state_columns_byte_identical=True,admitted_velocities_and_full_covariances_byte_identical=True,
        all_calibrated_probabilities_independently_recomputed=True,maximum_score_error=maximum_score_error,
        complete_source_frame_coverage_verified=True,native_nine_state_cache_independently_read_back=True,
        online_GT_read=False,tracking_accepted=False,paper_metrics_complete=False,input_sha256=pins,source_sha256=sha(__file__))
    out.mkdir(parents=True);p=out/'acceptance-receipt.json';p.open('x').write(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),frames=completed,candidates=count)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scope','bridge-sha256','raw-root','output'):parser.add_argument('--'+name,required=True)
    verify(parser.parse_args())
