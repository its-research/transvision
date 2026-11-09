#!/usr/bin/env python3
"""Independent original AST source geometry for every full-train label row."""
import argparse
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_gt_oracle import verified_sources,definitions
from transvision.models.event_track_v2x.v2v4real_inputs import load_raw_yaml


def oracle(root):
    trees=verified_sources(root);transform=dict(np=np,torch=torch)
    definitions(trees['transformation_utils.py'],('x_to_world','x1_to_x2'),transform,'original/transformation_utils.py')
    common=dict(np=np,torch=torch)
    definitions(trees['common_utils.py'],('check_numpy_to_torch','rotate_points_along_z'),common,'original/common_utils.py')
    box=dict(np=np,torch=torch,sys=sys,x1_to_x2=transform['x1_to_x2'],common_utils=types.SimpleNamespace(**common))
    definitions(trees['box_utils.py'],('corner_to_center','boxes_to_corners_3d','get_mask_for_boxes_within_range_torch',
        'mask_boxes_outside_range_numpy','create_bbx','project_world_objects'),box,'original/box_utils.py')
    def apply(m,cav):
        selected={k:v for k,v in m['vehicles'].items() if v['obj_type']!='Pedestrian'};result={}
        box['project_world_objects'](selected,result,np.eye(4),[-100,-40,-5,100,40,3],'lwh')
        return dict(states_source_lwh=np.asarray([v['coord'][0] for v in result.values()]).reshape(-1,7),
            object_ids=[str(k) for k in result],association_ids=[str(v['ass_id']) for v in result.values()],
            native_mapped_ids=[int(v['ass_id']) if int(v['ass_id'])!=-1 else int(k)+100*int(cav) for k,v in result.items()],
            raw_classes=[m['vehicles'][k]['obj_type'] for k in result],pose=np.asarray(m['lidar_pose'],dtype=np.float64))
    return apply


def verify(args):
    candidate=Path(args.candidate).absolute();out=Path(args.output).absolute();pins={}
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh independent label proof required')
    c=json.loads(read(candidate/'candidate-receipt.json',args.candidate_sha256,pins))
    if c['kind']!='v2v4real_full_train_identity_label_candidate_v1' or c['split']!='train' or c['official_test_read'] is not False:
        raise ValueError('full official train label candidate required before reading labels')
    data=read(candidate/'source-labels.jsonl',c['source_labels_sha256'],pins)
    original=oracle(args.oracle_root);seen=set();objects=0;maximum=0.;projection_paths=[p for p in c['input_sha256'] if p.endswith('/frames.jsonl')]
    if len(projection_paths)!=1:raise ValueError('exact original GT-free full train projection required')
    projected={}
    for line in read(projection_paths[0],c['input_sha256'][projection_paths[0]],pins).splitlines():
        r=json.loads(line);projected[r['sequence_id'],r['frame_key'],r['cav_id']]=r
    for n,line in enumerate(data.splitlines(),1):
        row=json.loads(line);key=row['sequence_id'],row['frame_key'],row['cav_id']
        if key in seen or key not in projected:raise ValueError('unique native label/projection binding required')
        seen.add(key);binding=projected[key]
        if any(row[k]!=binding[k] for k in ('frame_ordinal','source_to_world','pcd_sha256')):
            raise ValueError('original pose, PCD or ordinal changed')
        if c['input_sha256'].get(row['label_path'])!=row['label_sha256']:raise ValueError('label not in original byte admission')
        m=load_raw_yaml(read(row['label_path'],row['label_sha256'],pins));expected=original(m,row['cav_id'])
        signed=[dict(object_id=str(k),original_signed_extent=v['extent']) for k,v in m['vehicles'].items()
            if v['obj_type']!='Pedestrian' and any(float(x)<0 for x in v['extent'])]
        if signed!=row['signed_extent_objects']:raise ValueError('original signed extent audit differs')
        for k in ('object_ids','association_ids','native_mapped_ids','raw_classes'):
            if row[k]!=expected[k]:raise ValueError('original retained source identity differs: '+k)
        actual=np.asarray(row['states_source_lwh']).reshape(-1,7);reference=expected['states_source_lwh']
        np.testing.assert_allclose(actual,reference,atol=1e-4,rtol=0)
        np.testing.assert_allclose(row['source_to_world'],expected['pose'],atol=1e-4,rtol=0)
        if len(actual):maximum=max(maximum,float(np.max(np.abs(actual-reference))))
        objects+=len(actual)
        if n%500==0:print(json.dumps(dict(stage='independent_full_train_source_label_geometry',source_rows=n,total=14210)),flush=True)
    if seen!=set(projected) or len(seen)!=c['source_rows'] or objects!=c['objects']:raise ValueError('full train label support differs')
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('original inputs changed during independent geometry readback')
    out.mkdir(parents=True);v=dict(kind='v2v4real_full_train_identity_label_independent_geometry_v1',candidate_receipt_sha256=args.candidate_sha256,
        source_rows=len(seen),objects=objects,all_source_geometry_and_native_ids_verified=True,maximum_abs_geometry_error_m=maximum,
        geometry_atol_m=1e-4,official_test_read=False,physical_identity_continuity_verified=False,
        input_sha256=pins,source_sha256=sha(__file__),identity_row_supervision_accepted=False,identity_model_qualified=False)
    p=out/'acceptance-receipt.json';p.open('x').write(json.dumps(v,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),objects=objects)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('candidate','candidate-sha256','oracle-root','output'):p.add_argument('--'+n,required=True)
    verify(p.parse_args())
