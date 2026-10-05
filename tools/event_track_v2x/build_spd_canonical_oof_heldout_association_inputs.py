#!/usr/bin/env python3
"""Full vehicle-reference car-only GT-free input inventory, retaining source events.

Public cooperative frame links are metadata, never labels. Unpaired infrastructure
frames remain explicit pending source events; this inventory is not tracking or
channel replay and cannot supply final paper metrics.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as stream:json.dump(v,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def build(spec_path,output):
    S=Path(__file__).parent
    for row in json.loads((S/'source-inventory.json').read_text()):assert sha(S/row['path'])==row['sha256']
    sys.path.insert(0,str(S/'runtime'))
    from transvision.models.event_track_v2x.canonical_oof_predicted_features import predicted_features,load_fold_calibration
    from transvision.models.event_track_v2x.canonical_oof_prediction_gate import world_predictions,geometry_gate
    spec=json.loads(spec_path.read_text());ap=Path(spec['coverage_audit']['path']);assert sha(ap)==spec['coverage_audit']['sha256'];audit=json.loads(ap.read_text());fold=spec['fold_id'];binding=next(r for r in audit['folds'] if r['fold_id']==fold);assert not binding['missing_pair_raw_identity']
    cal=load_fold_calibration(Path(binding['calibration']['path']),binding['calibration']['sha256'],fold_id=fold);root=Path(binding['raw_root']);frames={};sources=[];origins={};started=time.monotonic()
    for row in binding['raw_manifests']:
        mp=Path(row['path']);assert sha(mp)==row['sha256'];m=json.loads(mp.read_text());assert m['gt_inputs'] is False and m['fold_id']==fold
        for r in m['frames']:
            p=mp.parent/r['metadata']['path'];assert sha(p)==r['metadata']['sha256'];meta=json.loads(p.read_text());key=(meta['side'],meta['sequence_id'],meta['frame_id']);assert key not in frames and meta['sequence_id'] in cal['canonical_oof_binding']['held_out_sequence_ids'];arrays_path=mp.parent/r['arrays']['path'];assert sha(arrays_path)==r['arrays']['sha256'];frames[key]=(arrays_path,meta,r)
            sources.append(dict(side=key[0],sequence_id=key[1],frame_id=key[2],arrays_path=str(arrays_path),metadata_path=str(p),arrays_sha256=r['arrays']['sha256'],metadata_sha256=r['metadata']['sha256'],source_image_timestamp_us=meta['source_image_timestamp_us'],box_reference_timestamp_us=meta['box_reference_timestamp_us']))
            if key[0]=='vehicle-side':origins[key[1]]=min(origins.get(key[1],meta['box_reference_timestamp_us']),meta['box_reference_timestamp_us'])
    pairmap={(p['sequence_id'],p['vehicle_frame_id']):p['infrastructure_frame_id'] for p in binding['pairs']};assert len(pairmap)==binding['admitted_pairs']
    vehicles=sorted([k for k in frames if k[0]=='vehicle-side'],key=lambda k:(k[1],frames[k][1]['box_reference_timestamp_us'],k[2]));assert len(vehicles)==binding['vehicle_frames'];assert not output.exists();output.mkdir();(output/'features').mkdir();rows=[];counts=dict(vehicle_reference_frames=0,paired_frames=0,unpaired_vehicle_frames=0,vehicle_unavailable_frames=0,left_car_queries=0,right_car_queries=0,gate_pairs=0)
    for key in vehicles:
        seq,vehicle_frame=key[1:];frame=pairmap.get((seq,vehicle_frame));left=frames[key];right=frames.get(('infrastructure-side',seq,frame)) if frame is not None else None;decision=left[1]['box_reference_timestamp_us']+100000;available=[all(v[1][k]<=decision for k in ['source_image_timestamp_us','box_reference_timestamp_us']) if v else False for v in [left,right]];selected=[];features=[];world=[]
        for value in [left,right]:
            if value is not None and available[0] and all(value[1][k]<=decision for k in ['source_image_timestamp_us','box_reference_timestamp_us']):
                with np.load(value[0],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                sel,feat=predicted_features(arrays,value[1],left[1],cal,role='held_out',decision_time_us=decision,origin_us=origins[seq]);keep=arrays['class_indices'][sel]==0;sel,feat=sel[keep],feat[keep];physical=world_predictions(arrays,value[1],sel,cal,decision_time_us=decision)
            else:sel=np.empty(0,np.int64);feat=np.empty((0,203),np.float32);physical=(np.empty((0,9)),np.empty((0,9,9)),np.empty(0,np.int64))
            selected.append(sel);features.append(feat);world.append(physical)
        gate,d2=geometry_gate(*world,probability=.99);name=seq+'-'+vehicle_frame+'.npz';p=output/'features'/name;np.savez_compressed(p,left=features[0],right=features[1],left_query_indices=selected[0],right_query_indices=selected[1],left_classes=world[0][2],right_classes=world[1][2],geometry_gate=gate,innovation_distance_squared=d2)
        rows.append(dict(sequence_id=seq,vehicle_frame_id=vehicle_frame,infrastructure_frame_id=frame,decision_time_us=decision,origin_us=origins[seq],available=available,features=dict(path=str(p.relative_to(output)),sha256=sha(p),bytes=p.stat().st_size),raw_inputs=[dict(arrays_sha256=v[2]['arrays']['sha256'],metadata_sha256=v[2]['metadata']['sha256']) if v else None for v in [left,right]]))
        counts['vehicle_reference_frames']+=1;counts['paired_frames']+=int(frame is not None);counts['unpaired_vehicle_frames']+=int(frame is None);counts['vehicle_unavailable_frames']+=int(not available[0]);counts['left_car_queries']+=len(selected[0]);counts['right_car_queries']+=len(selected[1]);counts['gate_pairs']+=int(gate.sum())
        done=len(rows)
        if done%100==0 or done==len(vehicles):print(json.dumps(dict(kind='rbf_experiment_progress_v1',stage='canonical_heldout_car_association_input',fold_id=fold,completed=done,total=len(vehicles),eta_seconds=(time.monotonic()-started)/done*(len(vehicles)-done))),flush=True)
    assert counts['paired_frames']==binding['admitted_pairs'] and counts['unpaired_vehicle_frames']==len(binding['unpaired_vehicle_frame_ids'])
    save(output/'examples.json',rows);save(output/'source-events.json',sorted(sources,key=lambda x:(x['sequence_id'],max(x['source_image_timestamp_us'],x['box_reference_timestamp_us']),x['side'],x['frame_id'])))
    save(output/'manifest.json',dict(kind='canonical_heldout_full_vehicle_reference_car_only_association_input_inventory',fold_id=fold,held_out_sequence_ids=cal['canonical_oof_binding']['held_out_sequence_ids'],excluded_fit_sequence_ids=cal['fit_sequences'],calibration=binding['calibration'],coverage_audit=spec['coverage_audit'],raw_manifests=binding['raw_manifests'],counts=counts,source_frames=len(sources),source_event_index_sha256=sha(output/'source-events.json'),example_index_sha256=sha(output/'examples.json'),source_inventory_sha256=sha(S/'source-inventory.json'),specification_sha256=sha(spec_path),class_scope='car',feature_dimension=203,candidate_policy='class0 view after unchanged raw-score>=0.05/all-class-top64',availability='clean-link/source-complete-at-vehicle-box-time-plus-100ms',unpaired_infrastructure_frame_ids=binding['unpaired_infrastructure_frame_ids'],unpaired_infrastructure_events_preserved_for_full_tracker_consumer=True,held_out_GT_read=False,annotation_payloads_read=False,optimizer_created=False,network_trace_used=False,complete_channel_tracking_replay=False,paper_eligible=False,independent_input_acceptance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--specification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.specification,a.output)
