#!/usr/bin/env python3
"""Second-process full readback of real predicted fit features/identity labels.

Reconstructs the common-time gate and masked supervision without calling the
producer's gate or target helpers. Reuses only the frozen encoder/matcher.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import chi2


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def verify(data, source, output):
    assert not output.exists()
    for r in json.loads((source / 'source-inventory.json').read_text()):
        assert sha(source / r['path']) == r['sha256']
    sys.path.insert(0, str(source / 'runtime'))
    from transvision.models.event_track_v2x.prediction_features import raw_state, match_predictions, choose_candidates, encode_features, transform_state_covariance, wrap_angle, CLASSES
    manifest = json.loads((data / 'manifest.json').read_text()); fold = manifest['fold_id']
    assert manifest['source_inventory_sha256'] == sha(source / 'source-inventory.json')
    assert manifest['example_index_sha256'] == sha(data / 'examples.json')
    specpath = source / f'fold-{fold}-specification.json'; assert sha(specpath) == manifest['specification_sha256']
    spec = json.loads(specpath.read_text()); root = Path(spec['root'])
    calpath = Path(spec['calibration']['path']); assert sha(calpath) == spec['calibration']['sha256']
    cal = json.loads(calpath.read_text())
    fitter = root / 'source-freezes/spd-canonical-calibration-runtime-v5-config-role-input-gate-closure-20261001'
    sys.path.insert(0, str(fitter / 'tools/event_track_v2x'))
    from spd_canonical_oof_calibration_supervision import load_fit_supervision
    package = root / f'artifacts/spd-official-oof-fivefold-20260930/fold-{fold}-package/package-manifest.json'
    gt, gtproof = load_fit_supervision(root / f'artifacts/spd-official-oof-fivefold-20260930/remote-conversion/fold-{fold}-converted', package,
                                     root / f'artifacts/spd-single-fit-overlay-materializer-readback-20261001/job-fold-{fold}/fold-{fold}')
    assert gtproof == manifest['converted_GT_supervision_readback']
    mappingpath = Path(spec['mapping']); assert sha(mappingpath) == manifest['mapping_sha256']
    mappings = {}
    for line in mappingpath.read_text().splitlines():
        row = json.loads(line); k = (row['sequence_id'], row['vehicle_frame_id'], row['infrastructure_frame_id'])
        assert k not in mappings; mappings[k] = row
    raw = {}; origins = {}
    for binding in manifest['raw_manifests']:
        path = Path(binding['path']); assert sha(path) == binding['sha256']
        for r in json.loads(path.read_text())['frames']:
            mp = path.parent / r['metadata']['path']; assert sha(mp) == r['metadata']['sha256']
            meta = json.loads(mp.read_text()); key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            assert key not in raw; raw[key] = (path.parent, r, meta)
            if meta['side'] == 'vehicle-side':
                origins[meta['sequence_id']] = min(origins.get(meta['sequence_id'], meta['box_reference_timestamp_us']), meta['box_reference_timestamp_us'])
    assert set(raw) == set(gt)
    rows = json.loads((data / 'examples.json').read_text()); seen = set(); start = time.monotonic()
    totals = dict(pairs=0, available_bilateral_pairs=0, vehicle_unavailable_pairs=0, positive_gate_pairs=0,
                  negative_gate_pairs=0, unknown_gate_pairs=0, left_assignment_rows=0, right_assignment_rows=0)
    for row in rows:
        key = (row['sequence_id'], row['vehicle_frame_id'], row['infrastructure_frame_id'])
        assert key in mappings and key not in seen; seen.add(key); mapping = mappings[key]
        paths = []
        for name in ('features', 'offline_targets'):
            p = data / row[name]['path']; assert p.is_file() and not p.is_symlink() and sha(p) == row[name]['sha256'] and p.stat().st_size == row[name]['bytes']; paths.append(p)
        with np.load(paths[0], allow_pickle=False) as z: features = {k: z[k] for k in z.files}
        with np.load(paths[1], allow_pickle=False) as z: targets = {k: z[k] for k in z.files}
        assert set(features) == {'left','right','left_query_indices','right_query_indices','geometry_gate','innovation_distance_squared','left_classes','right_classes'}
        values = []
        for side, frame in zip(('vehicle-side','infrastructure-side'), key[1:]):
            folder, r, meta = raw[side,key[0],frame]; p = folder / r['arrays']['path']; assert sha(p) == r['arrays']['sha256']
            with np.load(p, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
            values.append((arrays,meta,gt[side,key[0],frame],r))
        deadline = values[0][1]['box_reference_timestamp_us'] + 100000
        available = [all(v[1][k] <= deadline for k in ('source_image_timestamp_us','box_reference_timestamp_us')) for v in values]
        assert row['decision_time_us'] == deadline and row['available'] == available and row['origin_us'] == origins[key[0]]
        worlds = []; identities = []; known = []
        for side_index, (arrays, meta, truth, rawrow) in enumerate(values):
            prefix = ('left','right')[side_index]
            assert row['raw_inputs'][side_index] == dict(arrays_sha256=rawrow['arrays']['sha256'],metadata_sha256=rawrow['metadata']['sha256'])
            states = raw_state(arrays)
            sel = choose_candidates(arrays['scores'],dict(minimum_raw_score=.05,maximum_per_side=64)) if available[0] and available[side_index] else np.empty(0,np.int64)
            covariance = np.asarray([cal['sides'][meta['side']][CLASSES[int(c)]]['covariance']['matrix'] for c in arrays['class_indices']],float)
            expected = encode_features(states,covariance,arrays,sel,meta,values[0][1],(meta['box_reference_timestamp_us']-values[0][1]['box_reference_timestamp_us'])/1e6,origins[key[0]],cal['sides'][meta['side']]) if len(sel) else np.empty((0,203),np.float32)
            assert np.array_equal(features[prefix],expected) and np.array_equal(features[prefix+'_query_indices'],sel)
            matches = match_predictions(states[sel],arrays['scores'][sel],arrays['class_indices'][sel],truth['state'],truth['classes'])
            assert np.array_equal(targets[prefix+'_matches'],matches)
            by_token = {v['annotation_token']:int(v['cooperative_identity_id']) for v in mapping['source_identity_bindings'] if v['side']==meta['side']}
            ids = [None if j==-1 else by_token.get(truth['tokens'][int(j)]) for j in matches]
            identities.append(ids); known.append(np.array([j==-1 or uid is not None for j,uid in zip(matches,ids)],bool))
            s = states[sel].copy(); c = covariance[sel].copy(); s[:,[3,4]]=s[:,[4,3]];s[:,6]=wrap_angle(-s[:,6]-np.pi/2)
            jac=np.eye(9);jac[[3,4]]=jac[[4,3]];jac[6,6]=-1;c=jac@c@jac.T
            s,c,_,_=transform_state_covariance(s,c,meta,dict(lidar_to_world_row_rotation=np.eye(3),lidar_to_world_translation=np.zeros(3)))
            delta=(deadline-meta['box_reference_timestamp_us'])/1e6;transition=np.eye(9);transition[0,7]=delta;transition[1,8]=delta
            s=s@transition.T;c=transition@c@transition.T+np.eye(9)*.1*abs(delta)
            classes=arrays['class_indices'][sel];assert np.array_equal(features[prefix+'_classes'],classes);worlds.append((s,c,classes))
        left,right=worlds;n,m=len(left[0]),len(right[0]);innovation=left[0][:,None,:3]-right[0][None,:,:3]
        bound=2*(left[1][:,None,:3,:3]+right[1][None,:,:3,:3]);distance=np.einsum('ijk,ijk->ij',innovation,np.linalg.solve(bound,innovation[...,None])[...,0])
        gate=(left[2][:,None]==right[2][None,:])&(distance<=chi2.ppf(.99,3))
        assert np.array_equal(features['geometry_gate'],gate) and np.array_equal(features['innovation_distance_squared'],distance)
        y=np.array([[int(a is not None and a==b) for b in identities[1]] for a in identities[0]],np.uint8).reshape(n,m)
        mask=gate&known[0][:,None]&known[1][None,:]
        lm=known[0]&np.all(~gate|known[1][None,:],axis=1);rm=known[1]&np.all(~gate|known[0][:,None],axis=0)
        la=np.full(n,m,np.int64);ra=np.full(m,n,np.int64);i,j=np.nonzero(y.astype(bool)&gate);la[i]=j;ra[j]=i
        expected_targets=dict(targets=y,supervised_pair_mask=mask,left_assignment=la,right_assignment=ra,left_assignment_mask=lm,right_assignment_mask=rm,left_identity_known=known[0],right_identity_known=known[1])
        assert set(targets)==set(expected_targets)|{'left_matches','right_matches'}
        assert all(np.array_equal(targets[k],v) for k,v in expected_targets.items())
        positive=int((y.astype(bool)&mask).sum());totals['pairs']+=1;totals['available_bilateral_pairs']+=int(all(available));totals['vehicle_unavailable_pairs']+=int(not available[0]);totals['positive_gate_pairs']+=positive;totals['negative_gate_pairs']+=int(mask.sum())-positive;totals['unknown_gate_pairs']+=int((gate&~mask).sum());totals['left_assignment_rows']+=int(lm.sum());totals['right_assignment_rows']+=int(rm.sum())
        if len(seen)%100==0 or len(seen)==len(rows):
            elapsed=time.monotonic()-start;print(json.dumps(dict(stage='independent_association_examples_readback',fold_id=fold,completed=len(seen),total=len(rows),eta_seconds=elapsed/len(seen)*(len(rows)-len(seen)))),flush=True)
    assert seen==set(mappings) and totals==manifest['counts']
    receipt=dict(kind='canonical_fit_prediction_features_and_offline_identity_targets_independent_full_readback',fold_id=fold,manifest_sha256=sha(data/'manifest.json'),example_index_sha256=sha(data/'examples.json'),counts=totals,source_sha256=sha(Path(__file__)),all_example_bytes_features_gates_targets_and_deadline_availability_verified=True,GT_fields_in_feature_artifacts=False,producer_gate_and_target_helpers_called=False,hard_negative_scenario_coverage_certified=False,GPU_training_completed=False,paper_eligible=False,checked_at_utc=datetime.now(timezone.utc).isoformat())
    with output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(dict(accepted=True,fold_id=fold,receipt_sha256=sha(output),counts=totals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.data,a.source,a.output)
