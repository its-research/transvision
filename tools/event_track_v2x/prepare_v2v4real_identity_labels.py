#!/usr/bin/env python3
"""Offline full official-train source labels, separate from inference caches.

Native association IDs and source object IDs remain distinct. This candidate
does not qualify a checkpoint or assert physical identity continuity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.admit_v2v4real_motion_train_labels import admitted_labels
from transvision.models.event_track_v2x.v2v4real_inputs import load_raw_yaml
from transvision.models.event_track_v2x.v2v4real_ground_truth import matrix_pose
from tools.event_track_v2x.v2v4real_identity_source_geometry import source_boxes


def inputs(plan):
    pins={};part=json.loads(read(plan['partition']['path'],plan['partition']['sha256'],pins))
    if (part['official_split']!='train' or part['official_test_used_for_partition'] is not False
            or len(part['sequence_to_name_group'])!=32):
        raise ValueError('original full official-train partition required before labels')
    projection=[json.loads(x) for x in read(plan['projection']['path'],plan['projection']['sha256'],pins).splitlines()]
    bykey={};ordinals={}
    for r in projection:
        k=r['sequence_id'],r['frame_key'],r['cav_id']
        if k in bykey or r['sequence_id'] not in part['sequence_to_name_group'] or r['cav_id'] not in ('0','1'):
            raise ValueError('unique official train source projection required')
        bykey[k]=r;ordinals.setdefault((k[0],k[2]),[]).append(r['frame_ordinal'])
    if len(bykey)!=14210 or any(sorted(v)!=list(range(len(v))) for v in ordinals.values()):
        raise ValueError('full paired nominal train schedule required')
    for scene in part['sequence_to_name_group']:
        if sorted(ordinals.get((scene,'0'),[]))!=sorted(ordinals.get((scene,'1'),[])) or not ordinals.get((scene,'0')):
            raise ValueError('both complete native CAV sources required')
    inventories={};members={}
    for spec in plan['volumes']:
        p=Path(spec['path'])/'receipt.json';v=json.loads(read(p,spec['sha256'],pins))
        if v['split']!='train' or v['payload_verified'] is not True:
            raise ValueError('verified official train volume required before label payload')
        inventories[str(p.parent)]=v
        for item in v['files']:
            if not item['path'].endswith('.yaml'):continue
            scene,cav,name=item['path'].split('/');k=scene,Path(name).stem,cav
            if k in members:raise ValueError('duplicate native train label member')
            members[k]=(spec,item)
    if set(members)!=set(bykey):raise ValueError('native labels and full train projection differ')
    return pins,part,bykey,members


def prepare(args):
    out=Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('fresh offline supervision output required')
    plan=json.loads(Path(args.plan).read_bytes());pins,part,projected,members=inputs(plan)
    pins[str(Path(args.plan).absolute())]=sha(args.plan)
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'tools/event_track_v2x/admit_v2v4real_motion_train_labels.py',
        ROOT/'tools/event_track_v2x/v2v4real_motion_fit_inputs.py',ROOT/'transvision/models/event_track_v2x/v2v4real_ground_truth.py',
        ROOT/'transvision/models/event_track_v2x/v2v4real_inputs.py',ROOT/'transvision/models/event_track_v2x/v2v4real_numpy_yaml.py',
        ROOT/'transvision/models/event_track_v2x/paper_evaluation_policy.py',
        ROOT/'tools/event_track_v2x/v2v4real_identity_source_geometry.py']}
    out.mkdir(parents=True);rows=objects=0;classes=Counter();duplicates=0
    with (out/'source-labels.jsonl').open('x') as stream:
        for spec in plan['volumes']:
            selected=sorted({(s,k) for s,k,c in members if members[s,k,c][0]==spec})
            for label in admitted_labels(spec['path'],spec['sha256'],selected):
                key=label['sequence_id'],label['frame_key'],label['cav_id'];r=projected[key]
                p=Path(spec['path'])/'payload'/label['label_path'];pins[str(p)]=label['label_sha256']
                metadata=load_raw_yaml(label['label_bytes']);pose=matrix_pose(metadata['lidar_pose'])
                if not np.array_equal(pose,r['source_to_world']):raise ValueError('original label/raw projection pose differs')
                boxes,counts,signed=source_boxes(metadata,label['cav_id'])
                value=dict(sequence_id=key[0],frame_key=key[1],cav_id=key[2],side='vehicle-side' if key[2]=='0' else 'infrastructure-side',
                    frame_ordinal=r['frame_ordinal'],source_to_world=r['source_to_world'],pcd_sha256=r['pcd_sha256'],
                    label_path=str(p),label_sha256=label['label_sha256'],name_group=part['sequence_to_name_group'][key[0]],
                    states_source_lwh=[b['box_lwh'].tolist() for b in boxes],
                    object_ids=[b['object_id'] for b in boxes],association_ids=[b['associated_id'] for b in boxes],
                    native_mapped_ids=[b['track_id'] for b in boxes],raw_classes=[b['raw_class'] for b in boxes],
                    native_annotation_counts=counts,signed_extent_objects=signed)
                stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
                classes.update(value['raw_classes']);objects+=len(boxes);rows+=1
                duplicates+=sum(n>1 for n in Counter(value['native_mapped_ids']).values())
                if rows%500==0:print(json.dumps(dict(stage='full_train_native_identity_label_geometry',source_rows=rows,total=14210)),flush=True)
    if rows!=14210:raise ValueError('complete native train labels required')
    for p,h in {**pins,**sources}.items():
        if sha(p)!=h:raise ValueError('source or train labels changed during preparation')
    v=dict(kind='v2v4real_full_train_identity_label_candidate_v1',split='train',source_rows=rows,paired_frames=rows//2,sequences=32,
        objects=objects,raw_classes=dict(classes),duplicate_native_ids_within_source_frames=duplicates,
        source_labels_sha256=sha(out/'source-labels.jsonl'),input_sha256=pins,source_sha256=sources,
        official_test_read=False,nominal_rate_hz=10,labels_in_model_inputs=False,
        native_identity_semantics='association_id_unless_minus_one_else_source_object_id_plus_100_cav',
        temporal_supervision_basis='native_source_object_id',cross_source_supervision_basis='explicit_native_association_id',
        signed_extent_policy='original_signed_values_into_ordered_corners_then_native_edge_lengths_no_abs_repair',
        physical_identity_continuity_verified=False,independent_geometry_accepted=False,identity_model_trained=False,paper_metrics_complete=False)
    p=out/'candidate-receipt.json';p.open('x').write(json.dumps(v,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),source_rows=rows,objects=objects)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','output'):parser.add_argument('--'+n,required=True)
    prepare(parser.parse_args())
