#!/usr/bin/env python3
"""Read back every OOF pose via independent homogeneous-matrix composition."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def homogeneous(value):
    t=np.eye(4);t[:3,:3]=value['rotation']
    t[:3,3]=np.asarray(value['translation']).reshape(3)
    return t


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','poses','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();folds=[];total=0;max_error=0.
    for fid in range(5):
        root=a.inputs / ('fold-'+str(fid));mp=root/'input-manifest.json'
        manifest=json.loads(mp.read_bytes());pp=a.poses / ('fold-'+str(fid)+'-raw-poses.json')
        poses=json.loads(pp.read_bytes())
        if poses['input_manifest_sha256'] != sha(mp) or poses['fold_id'] != fid:
            raise ValueError('input binding differs')
        if any(poses[k] is not False for k in ('gt_inputs','val_or_test_read',
            'independent_pose_arrival_created','poses_renormalized','detector_predictions_available','formal_paper_eligible')):
            raise ValueError('unsupported pose scope')
        expected={};sources={}
        def load(side,relative):
            key=side+'/'+relative;p=root/key;sources[key]=sha(p)
            return json.loads(p.read_bytes())
        for side in ('vehicle-side','infrastructure-side'):
            rows=json.loads((root/side/'data_info.json').read_bytes())
            idx=json.loads((root/side/'frame-index.json').read_bytes())
            lookup={(r['sequence_id'],r['frame_id']):r for r in idx}
            for row in rows:
                if side=='vehicle-side':
                    t=homogeneous(load(side,row['calib_novatel_to_world_path'])) @ homogeneous(
                        load(side,row['calib_lidar_to_novatel_path'])['transform'])
                else:
                    t=homogeneous(load(side,row['calib_virtuallidar_to_world_path']))
                key=(side,row['sequence_id'],row['frame_id'])
                expected[key]=(t,lookup[key[1:]])
        actual={}
        for row in poses['frames']:
            key=(row['side'],row['sequence_id'],row['frame_id'])
            if key in actual or key not in expected:
                raise ValueError('extra or duplicate pose frame')
            actual[key]=row;t,index=expected[key]
            if any(row[k]!=v for k,v in index.items()) or row['coordinate_system']!='source_lidar':
                raise ValueError('frame metadata changed')
            composed=np.eye(4);composed[:3,:3]=np.asarray(row['lidar_to_world_row_rotation']).T
            composed[:3,3]=row['lidar_to_world_translation']
            error=float(np.max(np.abs(t-composed)));max_error=max(max_error,error)
            if not np.isfinite(composed).all() or error>1e-10:
                raise ValueError('raw composition differs')
            if not np.allclose(np.linalg.inv(t)@composed,np.eye(4),rtol=0,atol=1e-9):
                raise ValueError('inverse rigid transform differs')
        if set(actual)!=set(expected) or sources!=poses['source_calibration_sha256']:
            raise ValueError('pose frame or source inventory incomplete')
        total+=len(actual);folds.append({'fold_id':fid,'frames':len(actual),'pose_table_sha256':sha(pp)})
        print('POSE_READBACK fold='+str(fid)+' passed; remaining ETA=unknown',flush=True)
    report={'kind':'spd_oof_raw_pose_independent_homogeneous_readback_v1',
        'status':'all_frames_and_source_compositions_verified','folds':folds,
        'total_frames':total,'maximum_absolute_composition_error':max_error,
        'gt_or_val_test_read':False,'independent_pose_arrival_created':False,
        'formal_paper_eligible':False,'scope':'raw pose table only; no detector predictions or evaluator result',
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    with a.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
