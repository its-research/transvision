"""CPU full-cache content audit against independently accepted raw poses."""
import json,hashlib
from pathlib import Path
import numpy as np
from spd_canonical_oof_fit_feature_cache_primitives import verify_cache
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
POSE_TABLE_SHA={0:'6674234084d40b3beceac4b1d81f9d1eb87211c18dd682f41f8c9bdc546b75e6',1:'33cb20df958246b36ff512e824d5aa11f1a42216d27e5470ea52f38eb9224b61',2:'285e9f63c504a5ec56dbac9c3b3b6cd529fcfb9940aed6d7cb584273fe5ec307',3:'66f2a3aac510bc518fa6353714f15f51bea5ab6709d4a5af782bc48d9158e564',4:'209da7b812ef7b47205a6ae9cd2a97049093bf822a12d1be4c76cf6ad75f9ce9'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def check_raw_pose(actual,expected):
 for key,shape in (('lidar_to_world_row_rotation',(3,3)),('lidar_to_world_translation',(3,))):
  got=np.asarray(actual[key],dtype=np.float64);ref=np.asarray(expected[key],dtype=np.float64)
  need(got.shape==shape and np.isfinite(got).all(),'invalid raw pose shape/value')
  need(np.all(np.abs(got-ref)<=1e-10),'pose differs from unmodified raw calibration composition')

def audit(root,inputs,package_manifest,poses):
 checked=verify_cache(root,inputs,package_manifest);m=json.loads((root/'raw-cache-manifest.json').read_bytes())
 input_manifest=json.loads((inputs/'input-manifest.json').read_bytes())
 sources=input_manifest['source_input_manifests']
 need({r['fold_id'] for r in sources}==set(range(5))-{m['fold_id']} and len(sources)==4,'fit raw pose source complement differs')
 lookup={};pose_shas={}
 for source in sources:
  fold=source['fold_id'];path=poses/('fold-%d-raw-poses.json'%fold)
  need(sha(path)==POSE_TABLE_SHA[fold],'raw pose admission differs');pose_shas[fold]=sha(path)
  table=json.loads(path.read_bytes())
  hp=ROOT/'artifacts/spd-canonical-oof-heldout-inference-inputs-20260930'/('fold-%d/input-manifest.json'%fold)
  need(sha(hp)==source['sha256'],'shared fit source manifest differs')
  hm=json.loads(hp.read_bytes());need(table['input_manifest_sha256']==hm['source_heldout_input_manifest_sha256'],'raw pose original source lineage differs')
  for row in table['frames']:
   key=(row['side'],row['sequence_id'],row['frame_id'])
   need(key not in lookup and row['sequence_id'] in input_manifest['fit_sequence_ids'] and row['sequence_id'] not in input_manifest['excluded_held_out_sequence_ids'],'duplicate or held-out pose in fit export')
   lookup[key]=row
 need(m['kind']=='eventtrack_fit_feature_raw_detector_cache_v1' and m['canonical_oof_fit_feature_export'] is True and m['held_out_selection_scoring_eligible'] is False,'cache is not fit-only feature scope')
 need(m.get('metadata_pose_source')=='raw-calibration-composition-float64-v2','raw pose metadata source not admitted')
 need(m['weights_unchanged'] is True and m['raw_head_frames_verified']==m['frame_count'] and m['legacy_placeholder_count']==0 and m['gt_inputs'] is False and m['optimizer_created'] is False and m['covariance_calibrated'] is False and m['covariance_source'] is None and all(m[k] is False for k in ('preselection_roi','preselection_nms','preselection_topk','test_payloads_read','val_payloads_read')),'raw scope/state differs')
 poses_verified=0
 for item in m['frames']:
  meta=json.loads((root/item['metadata']['path']).read_bytes());key=(meta['side'],meta['sequence_id'],meta['frame_id']);need(key in lookup,'unknown pose frame');check_raw_pose(meta,lookup[key]);poses_verified+=1
  with np.load(root/item['arrays']['path'],allow_pickle=False) as data:
   b=data['boxes_lidar_bottom_xyz_dims_xyz_yaw_vxy'];g=data['gravity_centers_lidar'];expected=b[:,:3].copy();expected[:,2]+=b[:,5]*np.float32(.5)
   tolerance=2*np.maximum(np.abs(np.spacing(expected)),np.finfo(np.float32).eps)
   need(np.all(np.abs(g.astype(np.float64)-expected.astype(np.float64))<=tolerance),'bottom/gravity box geometry differs')
   placeholder=(data['scores']==.5)&np.all(b==np.asarray([0,0,0,1,1,1,0,0,0]),axis=1);need(not placeholder.any(),'legacy placeholder in raw cache')
 need(poses_verified==checked['frames_verified'],'pose coverage incomplete')
 return dict(checked,raw_poses_verified=poses_verified,raw_pose_table_sha256_by_source_fold=pose_shas,held_out_selection_scoring_eligible=False,bottom_gravity_geometry_verified=True,pose_precision_scope='original float64 raw homogeneous composition; no quaternion projection',covariance_calibrated=False,paper_eligible=False)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('root','inputs','package-manifest','poses'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();print(json.dumps(audit(a.root,a.inputs,a.package_manifest,a.poses)))
