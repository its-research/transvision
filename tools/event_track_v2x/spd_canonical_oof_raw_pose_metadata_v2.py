"""Compose original calibration bytes for metadata without quaternion projection."""
import json,hashlib
from pathlib import Path
import numpy as np

def raw_pose_lookup(inputs,side):
 inputs=Path(inputs);m=json.loads((inputs/'input-manifest.json').read_bytes());payloads={r['path']:r for r in m['payload_inventory']}
 def load(relative):
  rel=Path(relative)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe raw calibration path')
  p=inputs/side/rel;record=payloads[side+'/'+rel.as_posix()]
  if p.is_symlink() or p.stat().st_size!=record['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('raw calibration payload differs')
  return json.loads(p.read_bytes())
 def homogeneous(value):
  r=np.asarray(value['rotation'],dtype=np.float64);t=np.asarray(value['translation'],dtype=np.float64).reshape(3)
  if r.shape!=(3,3) or not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('invalid raw calibration')
  matrix=np.eye(4,dtype=np.float64);matrix[:3,:3]=r;matrix[:3,3]=t;return matrix
 poses={}
 for row in json.loads((inputs/side/'data_info.json').read_bytes()):
  if side=='vehicle-side':matrix=homogeneous(load(row['calib_novatel_to_world_path']))@homogeneous(load(row['calib_lidar_to_novatel_path'])['transform'])
  elif side=='infrastructure-side':matrix=homogeneous(load(row['calib_virtuallidar_to_world_path']))
  else:raise ValueError('unknown side')
  if row['frame_id'] in poses:raise ValueError('duplicate pose frame')
  poses[row['frame_id']]={'sequence_id':row['sequence_id'],'lidar_to_world_row_rotation':matrix[:3,:3].T.tolist(),'lidar_to_world_translation':matrix[:3,3].tolist()}
 return poses
