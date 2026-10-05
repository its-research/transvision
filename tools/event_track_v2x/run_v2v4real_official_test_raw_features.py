#!/usr/bin/env python3
"""GT-free official-test raw export using unchanged frozen native inference.

Local admitted assets are required. This program never downloads, trains,
calibrates, reads labels or dispatches ClearML tasks. It exports legacy 7D
candidates/BEV features, not fabricated velocity/covariance or paper metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import types
import numpy as np

SOURCE_ARCHIVE_SHA='f47f391db33e02813b05e10f598ecf9240877463a8e32b61bc5841a8769d38ea'
CORE_SHA='5138ba83aa381b722828265b4dfb4934bd53cc533685fd004db862b3158cd48b'
PROJECTION_TASK='4e93e12cd9cd4ea3864c9b9af57182f9'
PROJECTION_SHA='d0e13e20f5ff2cafeb713fb66cd642ef861962fbf8ce99bef8e2ee9bd50a5601'
FRAME_SHA='5662c02a999eac83e8523f4cd0f483721fd49b2f5cfd19ea8bdf7471af42d61a'
ADMISSION_SHA='c540c3625e67fbbbc44b20253bf67a45302cfab2162ebc0de67866be55e0781c'
PARTITION_SHA='02373d0f59ca4e88757b6b3c3c22d9afda0ff94fe47f0f7d10f80211876bbcb2'
CHECKPOINTS={1337:'c6982b189ff2852f62d583ecbc271b688aeb4d4ada3822ecf02ed6bd01488907',
             2027:'2b00c50b9759ccefd72dc60fb6b81c0115b49fb5ade9c48678280373eb0d9ed6',
             3407:'0882c4d76db9cca3fd17f833c62b97cba6560b2c7bf75ac01e093ab8404c7e4e'}
PROTOCOL_ID='v2v4real-nominal-10hz-formal-v1'
KIND='v2v4real_official_test_raw_native_features_v1'
PACKAGE='_rbf_frozen_official_test_native_core'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def asset(path,expected):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file() or sha(path)!=expected:
        raise ValueError('asset bytes differ: '+str(path))
    return path


def extract_source(archive,directory):
    """Verify the original 4 MB source archive, then copy only regular files."""
    archive=asset(archive,SOURCE_ARCHIVE_SHA); directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False);seen=set();copied={}
    with tarfile.open(archive) as stream:
        for member in stream:
            relative=PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or member.name in seen or member.issym() or member.islnk():
                raise ValueError('unsafe frozen source archive')
            seen.add(member.name)
            if member.isdir():continue
            if not member.isfile() or relative.parts[0] not in ('project','official'):
                raise ValueError('unexpected frozen source archive member')
            target=directory/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            raw=stream.extractfile(member).read()
            if len(raw)!=member.size:raise ValueError('truncated source')
            target.write_bytes(raw);copied[member.name]=hashlib.sha256(raw).hexdigest()
    if sha(directory/'project/transvision/models/event_track_v2x/paper_pointpillar.py')!=CORE_SHA:
        raise ValueError('frozen inference core differs')
    if sha(archive)!=SOURCE_ARCHIVE_SHA:raise ValueError('source changed while reading')
    return copied


def frozen_core(directory):
    """Private package prevents ambient NMS/old-checkpoint code substitution."""
    if any(n==PACKAGE or n.startswith(PACKAGE+'.') for n in sys.modules):
        raise ValueError('frozen native namespace already occupied; use a fresh process')
    package=types.ModuleType(PACKAGE)
    package.__path__=[str(Path(directory)/'project/transvision/models/event_track_v2x')]
    sys.modules[PACKAGE]=package
    return importlib.import_module(PACKAGE+'.paper_pointpillar')


def verify_runtime_sources(directory,files):
    for relative,digest in files.items():
        asset(Path(directory)/relative,digest)


def device_identity(torch,device):
    """Record runtime identity without claiming independent physical admission."""
    selected=torch.device(device)
    result=dict(device=device,device_type=selected.type,logical_index=selected.index,
        uuid=None,physical_identity_verified=False,resource_admission_verified=False)
    if selected.type=='cuda':
        index=selected.index if selected.index is not None else torch.cuda.current_device()
        properties=torch.cuda.get_device_properties(index)
        uuid=getattr(properties,'uuid',None)
        if isinstance(uuid,bytes):uuid=uuid.hex()
        result.update(logical_index=index,name=properties.name,uuid=str(uuid) if uuid else None,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            physical_identity_status='uuid_observed_not_independently_verified' if uuid else 'uuid_unavailable')
    else:result['physical_identity_status']='not_a_cuda_device'
    return result


def validate_scope(projection,admission,records):
    if (projection.get('dataset_split')!='test' or projection.get('kind')!='v2v4real_pose_lidar_projection_v2'
            or projection.get('gt_in_projection') is not False or projection.get('frames_sha256')!=FRAME_SHA
            or (projection.get('sequence_count'),projection.get('paired_frame_count'),projection.get('source_frame_count'))!=(9,1993,3986)):
        raise ValueError('original official-test GT-free projection required')
    required={'kind':'v2v4real_official_split_admission_v1','split':'official_test',
        'projection_manifest_sha256':PROJECTION_SHA,'protocol_id':PROTOCOL_ID,
        'official_split_membership_verified':True,'official_original_split_preserved':True,
        'official_test_unchanged':True,'overlap_disclosure_required':True,'statistical_independence_claimed':False}
    if any(admission.get(k)!=v for k,v in required.items()):raise ValueError('official split admission differs')
    if len(records)!=3986 or set(r['sequence_id'] for r in records)!=set(projection['ego_agents']):
        raise ValueError('complete official-test cohort required')


class IdentityCalibration:
    def apply(self,scores):return np.asarray(scores,dtype=np.float64)


def export_rows(records,inputs_root,detector,lidar_range,read_pcd,mask_ego,mask_range,output,*,progress=None):
    """No row is skipped when the detector returns zero candidates."""
    grouped={};out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'payloads').mkdir()
    for row in records:grouped.setdefault((row['sequence_id'],row['frame_ordinal']),[]).append(row)
    rows=[];valid_count=0;started=time.monotonic()
    for (sequence,_),pair in sorted(grouped.items()):
        if len(pair)!=2 or sum(r['is_ego'] is True for r in pair)!=1 or len({r['frame_key'] for r in pair})!=1:
            raise ValueError('one ego and one partner for every original frame required')
        for row in sorted(pair,key=lambda r:not r['is_ego']):
            cloud=read_pcd(Path(inputs_root)/row['pcd_path'],expected_sha256=row['pcd_sha256'])
            points=mask_range(mask_ego(cloud.xyzi),lidar_range)
            arrays=detector.predict(points,lidar_range=lidar_range,covariance_diagonal=np.ones(9),calibration=IdentityCalibration())
            # Original adapter pads velocity and covariance; export neither.
            n=len(arrays['states']);scores=arrays['raw_scores'];features=arrays['appearance'];valid=arrays['appearance_valid']
            if (not 0<=n<=64 or arrays['states'].shape!=(n,9) or features.shape!=(n,128)
                    or scores.shape!=(n,) or arrays['class_indices'].shape!=(n,) or valid.shape!=(n,)
                    or valid.dtype!=np.bool_ or not np.issubdtype(arrays['class_indices'].dtype,np.integer)
                    or not all(np.isfinite(arrays[k]).all() for k in ('states','raw_scores','appearance','class_indices'))
                    or np.any(arrays['class_indices']!=0) or np.any(arrays['states'][:,3:6]<=0)
                    or np.any(scores<.05) or np.any(scores>1) or np.any(np.diff(scores)>0)
                    or not np.allclose(np.linalg.norm(features[valid],axis=1),1,atol=1e-6,rtol=0)
                    or np.any(features[~valid]!=0)):
                raise ValueError('frozen raw top64 output contract differs')
            payload=out/'payloads'/f'frame-{len(rows):06d}.npz'
            np.savez_compressed(payload,states_source_legacy7=arrays['states'][:,:7],raw_scores=scores,
                class_indices=arrays['class_indices'],appearance=features,appearance_valid=valid)
            rows.append(dict(sequence_id=sequence,frame_id=row['frame_key'],frame_ordinal=row['frame_ordinal'],
                side='vehicle-side' if row['is_ego'] else 'infrastructure-side',source_to_world=row['source_to_world'],
                pcd_sha256=row['pcd_sha256'],payload=payload.relative_to(out).as_posix(),payload_sha256=sha(payload),candidates=n))
            valid_count+=int(valid.sum())
            if progress and (len(rows)%25==0 or len(rows)==len(records)):
                elapsed=time.monotonic()-started
                progress(dict(stage='official_test_raw_export',completed_rows=len(rows),total_rows=len(records),
                    ETA_seconds=elapsed/len(rows)*(len(records)-len(rows)),eta_scope='detector export only'))
    if len(rows)!=len(records):raise ValueError('source frame coverage differs')
    return rows,valid_count


def pack(work,output,manifest,rows,projection_path,admission_path):
    """Consumer envelope keeps auxiliary provenance outside the raw archive."""
    work,output=Path(work),Path(output)
    raw=b''.join(canonical(r)+b'\n' for r in rows)
    if hashlib.sha256(raw).hexdigest()!=manifest['rows_sha256']:raise ValueError('raw index differs')
    (work/'manifest.json').write_bytes(canonical(manifest));(work/'predictions.jsonl').write_bytes(raw)
    (output/'native-feature-manifest').write_bytes(canonical(manifest));(output/'native-feature-index').write_bytes(raw)
    shutil.copyfile(projection_path,output/'projection-manifest');shutil.copyfile(admission_path,output/'split-admission')
    names=['anchors.npy','manifest.json','predictions.jsonl',*(r['payload'] for r in rows)]
    with tarfile.open(output/'native-features','w:gz') as t:
        for name in names:t.add(work/name,arcname=name,recursive=False)
    with tarfile.open(output/'native-features') as t:
        members=t.getmembers()
        if [m.name for m in members]!=names or any(not m.isfile() for m in members):raise ValueError('raw archive inventory differs')
        for m in members:
            if hashlib.sha256(t.extractfile(m).read()).hexdigest()!=sha(work/m.name):raise ValueError('archive readback differs')
    return sha(output/'native-features')


def run(source_archive,inputs_root,admission,checkpoint,seed,output,*,device='cuda:0'):
    producer_sha=sha(__file__)
    if type(seed) is not int or seed not in CHECKPOINTS:raise ValueError('frozen seed required')
    source_archive=asset(source_archive,SOURCE_ARCHIVE_SHA);checkpoint=asset(checkpoint,CHECKPOINTS[seed])
    inputs_root=Path(inputs_root).absolute();projection_path=asset(inputs_root/'manifest.json',PROJECTION_SHA)
    admission=asset(admission,ADMISSION_SHA);asset(inputs_root/'frames.jsonl',FRAME_SHA)
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('fresh output directory required')
    import torch
    import yaml
    output.mkdir(parents=True);files=extract_source(source_archive,output/'frozen-runtime');core=frozen_core(output/'frozen-runtime')
    source=output/'frozen-runtime/official'
    loader=importlib.import_module(PACKAGE+'.v2v4real_inputs')
    projection,records=loader.load_prepared_frames(inputs_root,expected_manifest_sha256=PROJECTION_SHA)
    validate_scope(projection,json.loads(admission.read_bytes()),records)
    if any(n=='opencood' or n.startswith('opencood.') for n in sys.modules):raise ValueError('unverified detector namespace already loaded')
    sys.path.insert(0,str(source))
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    from opencood.models.point_pillar import PointPillar
    from opencood.utils.pcd_utils import mask_ego_points,mask_points_by_range
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=load_point_pillar_params(yaml.safe_load((source/'opencood/hypes_yaml/point_pillar_late_fusion.yaml').read_text()))
    anchors=np.asarray(build_postprocessor(config['postprocess'],train=False).generate_anchor_box())
    if anchors.ndim!=4 or anchors.shape[-1]!=7 or not np.isfinite(anchors).all():raise ValueError('invalid official anchors')
    detector=core.FrozenPointPillar(PointPillar(config['model']['args']),SpVoxelPreprocessor(config['preprocess'],train=False),anchors,
        checkpoint=checkpoint,checkpoint_sha256=CHECKPOINTS[seed],device=device,formal_seed=seed,formal_protocol=PROTOCOL_ID,
        official_commit='5a821e13753bafc611f95c47bc1a306acdcb0f7c')
    pcd=importlib.import_module(PACKAGE+'.v2v4real_pcd')
    rows,valid=export_rows(records,inputs_root,detector,config['preprocess']['cav_lidar_range'],pcd.read_native_pcd,
        mask_ego_points,mask_points_by_range,output/'raw',progress=lambda v:print(json.dumps(v),flush=True))
    np.save(output/'raw/anchors.npy',anchors,allow_pickle=False)
    rows.sort(key=lambda r:(r['sequence_id'],r['frame_ordinal'],r['side']))
    index=b''.join(canonical(r)+b'\n' for r in rows)
    sequences=sorted(projection['ego_agents'])
    manifest=dict(kind=KIND,protocol_id=PROTOCOL_ID,split='official_test',cohort='official_test_raw_native_features',
        candidate_protocol='rbf-all-class-top64-v1',matching_candidate_rule='raw_score_ge_0.05_stable_top64_before_nms',
        checkpoint_seed=seed,checkpoint_sha256=CHECKPOINTS[seed],anchor_generation='official_build_postprocessor_generate_anchor_box',
        anchor_array_sha256=hashlib.sha256(canonical(dict(shape=list(anchors.shape),dtype=str(anchors.dtype)))+np.ascontiguousarray(anchors).tobytes()).hexdigest(),
        anchor_file_sha256=sha(output/'raw/anchors.npy'),state_layout='source_lidar_legacy_xyz_width_length_height_neg_yaw_minus_pi_over_2',
        velocity_exported=False,covariance_exported=False,feature_method='rbf-pointpillar-bev128-bilinear-l2-v1',
        downstream_covariance_velocity_and_identity_admission_pending=True,prepared_projection_sha256=PROJECTION_SHA,
        split_admission_sha256=ADMISSION_SHA,partition_sha256=PARTITION_SHA,export_groups=sequences,sequence_ids=sequences,
        rows=len(rows),rows_sha256=hashlib.sha256(index).hexdigest(),total_candidates=sum(r['candidates'] for r in rows),
        appearance_valid_candidates=valid,gt_read=False,official_test_read=True,paper_metric=False,
        projection_manifest=dict(path='projection-manifest',sha256=PROJECTION_SHA),split_admission=dict(path='split-admission',sha256=ADMISSION_SHA))
    verify_runtime_sources(output/'frozen-runtime',files)
    for p,h in ((source_archive,SOURCE_ARCHIVE_SHA),(checkpoint,CHECKPOINTS[seed]),(projection_path,PROJECTION_SHA),(inputs_root/'frames.jsonl',FRAME_SHA),(admission,ADMISSION_SHA)):
        asset(p,h)
    archive_sha=pack(output/'raw',output,manifest,rows,projection_path,admission)
    verify_runtime_sources(output/'frozen-runtime',files)
    for p,h in ((source_archive,SOURCE_ARCHIVE_SHA),(checkpoint,CHECKPOINTS[seed]),(projection_path,PROJECTION_SHA),(inputs_root/'frames.jsonl',FRAME_SHA),(admission,ADMISSION_SHA)):
        asset(p,h)
    if sha(__file__)!=producer_sha:raise ValueError('producer source changed during execution')
    actual_device=device_identity(torch,device)
    receipt=dict(kind='v2v4real_official_test_raw_export_v1',split='official_test',gt_read=False,official_test_read=True,paper_metric=False,
        raw_manifest_sha256=sha(output/'native-feature-manifest'),raw_archive_sha256=archive_sha,rows_sha256=manifest['rows_sha256'],
        checkpoint_sha256=CHECKPOINTS[seed],checkpoint_seed=seed,projection_manifest_sha256=PROJECTION_SHA,
        projection_frames_sha256=FRAME_SHA,split_admission_sha256=ADMISSION_SHA,projection_task_id=PROJECTION_TASK,
        producer_sha256=producer_sha,command=sys.argv,source_archive_sha256=SOURCE_ARCHIVE_SHA,inference_core_sha256=CORE_SHA,actual_device=actual_device,TF32_enabled=False,
        rows=len(rows),source_frames_including_empty_retained=True,raw_archive_local_readback=True,
        original_checkpoint_known_overlap_disclosed=True,statistical_independence_claimed=False,
        original_partition_sha256_is_training_provenance_only=True,independent_raw_acceptance=False,
        covariance_velocity_and_identity_admission=False,formal_metrics_complete=False)
    (output/'raw-export-receipt.json').write_bytes(canonical(receipt));return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive','inputs-root','admission','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,choices=sorted(CHECKPOINTS),required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(run(a.source_archive,a.inputs_root,a.admission,a.checkpoint,a.seed,a.output,device=a.device)))


if __name__=='__main__':main()
