#!/usr/bin/env python3
"""Independent full held-out feature/source/availability and geometry readback."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import chi2


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(data,source,output):
    assert not output.exists()
    for r in json.loads((source/'source-inventory.json').read_text()):assert sha(source/r['path'])==r['sha256']
    sys.path.insert(0,str(source/'runtime'))
    from transvision.models.event_track_v2x.prediction_features import raw_state,choose_candidates,encode_features,transform_state_covariance,wrap_angle,CLASSES
    m=json.loads((data/'manifest.json').read_text());fold=m['fold_id'];assert sha(source/'source-inventory.json')==m['source_inventory_sha256'] and sha(data/'examples.json')==m['example_index_sha256'] and sha(data/'source-events.json')==m['source_event_index_sha256'];calpath=Path(m['calibration']['path']);assert sha(calpath)==m['calibration']['sha256'];cal=json.loads(calpath.read_text());assert cal['canonical_oof_binding']['held_out_sequence_ids']==m['held_out_sequence_ids'] and set(m['held_out_sequence_ids']).isdisjoint(m['excluded_fit_sequence_ids'])
    ap=Path(m['coverage_audit']['path']);assert sha(ap)==m['coverage_audit']['sha256'];audit=next(x for x in json.loads(ap.read_text())['folds'] if x['fold_id']==fold);mapping={(r['sequence_id'],r['vehicle_frame_id']):r['infrastructure_frame_id'] for r in audit['pairs']};raw={};origins={}
    for bind in m['raw_manifests']:
        mp=Path(bind['path']);assert sha(mp)==bind['sha256']
        for r in json.loads(mp.read_text())['frames']:
            p=mp.parent/r['metadata']['path'];assert sha(p)==r['metadata']['sha256'];meta=json.loads(p.read_text());k=(meta['side'],meta['sequence_id'],meta['frame_id']);assert k not in raw and k[1] in m['held_out_sequence_ids'];npz=mp.parent/r['arrays']['path'];assert sha(npz)==r['arrays']['sha256'];raw[k]=(npz,meta,r)
            if k[0]=='vehicle-side':origins[k[1]]=min(origins.get(k[1],meta['box_reference_timestamp_us']),meta['box_reference_timestamp_us'])
    events=json.loads((data/'source-events.json').read_text());event_keys=[(r['side'],r['sequence_id'],r['frame_id']) for r in events];assert len(event_keys)==len(set(event_keys))==len(raw)==m['source_frames'] and set(event_keys)==set(raw)
    for event in events:
        value=raw[event['side'],event['sequence_id'],event['frame_id']];assert event['arrays_sha256']==value[2]['arrays']['sha256'] and event['metadata_sha256']==value[2]['metadata']['sha256'] and event['source_image_timestamp_us']==value[1]['source_image_timestamp_us'] and event['box_reference_timestamp_us']==value[1]['box_reference_timestamp_us']
    rows=json.loads((data/'examples.json').read_text());seen=set();counts=dict(vehicle_reference_frames=0,paired_frames=0,unpaired_vehicle_frames=0,vehicle_unavailable_frames=0,left_car_queries=0,right_car_queries=0,gate_pairs=0);start=time.monotonic()
    for row in rows:
        key=(row['sequence_id'],row['vehicle_frame_id']);assert key not in seen and row['infrastructure_frame_id']==mapping.get(key);seen.add(key);v=raw['vehicle-side',*key];infra=raw['infrastructure-side',key[0],row['infrastructure_frame_id']] if row['infrastructure_frame_id'] is not None else None;deadline=v[1]['box_reference_timestamp_us']+100000;available=[all(x[1][k]<=deadline for k in ['source_image_timestamp_us','box_reference_timestamp_us']) if x else False for x in [v,infra]];assert deadline==row['decision_time_us'] and available==row['available'] and row['origin_us']==origins[key[0]];p=data/row['features']['path'];assert sha(p)==row['features']['sha256'] and p.stat().st_size==row['features']['bytes']
        with np.load(p,allow_pickle=False) as z:features={k:z[k] for k in z.files}
        assert set(features)=={'left','right','left_classes','right_classes','left_query_indices','right_query_indices','geometry_gate','innovation_distance_squared'};world=[]
        for i,value in enumerate([v,infra]):
            prefix=['left','right'][i]
            if value is None or not available[0] or not available[i]:expected=np.empty((0,203),np.float32);sel=np.empty(0,np.int64);classes=np.empty(0,np.int64);s=np.empty((0,9));c=np.empty((0,9,9))
            else:
                with np.load(value[0],allow_pickle=False) as z:a={k:z[k] for k in z.files}
                meta=value[1];states=raw_state(a);sel=choose_candidates(a['scores'],dict(minimum_raw_score=.05,maximum_per_side=64));sel=sel[a['class_indices'][sel]==0];cov=np.asarray([cal['sides'][meta['side']][CLASSES[int(k)]]['covariance']['matrix'] for k in a['class_indices']],float);expected=encode_features(states,cov,a,sel,meta,v[1],(meta['box_reference_timestamp_us']-v[1]['box_reference_timestamp_us'])/1e6,origins[key[0]],cal['sides'][meta['side']]);s=states[sel].copy();c=cov[sel].copy();s[:,[3,4]]=s[:,[4,3]];s[:,6]=wrap_angle(-s[:,6]-np.pi/2);j=np.eye(9);j[[3,4]]=j[[4,3]];j[6,6]=-1;c=j@c@j.T;s,c,_,_=transform_state_covariance(s,c,meta,dict(lidar_to_world_row_rotation=np.eye(3),lidar_to_world_translation=np.zeros(3)));delta=(deadline-meta['box_reference_timestamp_us'])/1e6;f=np.eye(9);f[0,7]=delta;f[1,8]=delta;s=s@f.T;c=f@c@f.T+np.eye(9)*.1*abs(delta);classes=a['class_indices'][sel]
            assert np.array_equal(expected,features[prefix]) and np.array_equal(sel,features[prefix+'_query_indices']) and np.array_equal(classes,features[prefix+'_classes']);world.append((s,c,classes));assert row['raw_inputs'][i]==(dict(arrays_sha256=value[2]['arrays']['sha256'],metadata_sha256=value[2]['metadata']['sha256']) if value else None)
        l,r=world;innovation=l[0][:,None,:3]-r[0][None,:,:3];bound=2*(l[1][:,None,:3,:3]+r[1][None,:,:3,:3]);d2=np.einsum('ijk,ijk->ij',innovation,np.linalg.solve(bound,innovation[...,None])[...,0]);gate=(l[2][:,None]==r[2][None,:])&(d2<=chi2.ppf(.99,3));assert np.array_equal(features['innovation_distance_squared'],d2) and np.array_equal(features['geometry_gate'],gate)
        counts['vehicle_reference_frames']+=1;counts['paired_frames']+=int(infra is not None);counts['unpaired_vehicle_frames']+=int(infra is None);counts['vehicle_unavailable_frames']+=int(not available[0]);counts['left_car_queries']+=len(l[0]);counts['right_car_queries']+=len(r[0]);counts['gate_pairs']+=int(gate.sum())
        if len(seen)%100==0 or len(seen)==len(rows):print(json.dumps(dict(stage='independent_heldout_input_full_readback',fold_id=fold,completed=len(seen),total=len(rows),eta_seconds=(time.monotonic()-start)/len(seen)*(len(rows)-len(seen)))),flush=True)
    assert seen=={k[1:] for k in raw if k[0]=='vehicle-side'} and counts==m['counts']
    proof=dict(fold_id=fold,manifest_sha256=sha(data/'manifest.json'),example_index_sha256=sha(data/'examples.json'),source_event_index_sha256=sha(data/'source-events.json'),counts=counts,source_frames=len(raw),all_source_frames_and_vehicle_reference_examples_covered=True,all_feature_bytes_encoder_values_gate_distances_and_deadlines_verified=True,held_out_GT_read=False,producer_feature_gate_helpers_called=False,independent_verifier_sha256=sha(Path(__file__)),full_channel_tracking_replay=False,paper_eligible=False)
    with output.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.data,a.source,a.output)
