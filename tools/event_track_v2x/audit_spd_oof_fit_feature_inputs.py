"""Read every fit feature byte and info value against independent held-out sources."""
import argparse,io,json,time
from pathlib import Path
from audit_spd_oof_inference_infos import Restricted,equal
from spd_canonical_oof_input_gate import digest,validate_inputs
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
ADMISSION='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args();need(not a.receipt.exists(),'receipt is create-once')
 ap=ROOT/'receipts/spd-canonical-oof-heldout-inference-inputs-independent-readback-20260930.json';need(digest(ap)==ADMISSION,'original input admission differs');ad=json.loads(ap.read_bytes());sources={}
 for row in ad['folds']:
  fold=row['fold_id'];r=ROOT/('artifacts/spd-canonical-oof-heldout-inference-inputs-20260930/fold-%d'%fold);mp=ROOT/('artifacts/spd-official-oof-fivefold-20260930/fold-%d-package/package-manifest.json'%fold);original=validate_inputs(r,json.loads(mp.read_bytes()),row['input_manifest_sha256']);sources[fold]=(r,original)
 records=[];total=0;start=time.monotonic()
 for fold in range(5):
  r=a.inputs/('fold-%d'%fold);mp=ROOT/('artifacts/spd-official-oof-fivefold-20260930/fold-%d-package/package-manifest.json'%fold);package=json.loads(mp.read_bytes());m=json.loads((r/'input-manifest.json').read_bytes());fit=package['fit_sequence_ids'];held=package['held_out_sequence_ids'];selected=[sources[x] for x in range(5) if x!=fold]
  need(m['kind']=='spd_canonical_oof_fit_feature_inputs_v1' and m['cohort']=='canonical-oof-fit-feature-inputs' and m['fit_sequence_ids']==fit and m['excluded_held_out_sequence_ids']==held and m['fold_id']==fold and m['training_package_manifest_sha256']==digest(mp) and m['canonical_fivefold_manifest_sha256']==package['canonical_fivefold_manifest_sha256'] and m['official_split_sha256']==package['official_split_sha256'],'wrong training fold role or package')
  need(all(m[k] is False for k in ['gt_payloads_in_package','gt_payloads_read','val_payloads_read','test_payloads_read','held_out_selection_scoring_eligible','predictions_generated','paper_eligible']),'fit view crossed role/label boundary')
  need(m['source_all_train_input_admission_sha256']==ADMISSION and m['source_input_manifests']==[{'fold_id':sm['fold_id'],'sha256':digest(sr/'input-manifest.json')} for sr,sm in selected],'source admission differs')
  expected={}
  for sr,sm in selected:
   for item in sm['payload_inventory']:
    rel=Path(item['path'])
    if len(rel.parts)>=3 and rel.parts[1] in ('image','calib'):
     need(item['path'] not in expected,'ambiguous source payload');expected[item['path']]=item
  inventory={x['path']:x for x in m['payload_inventory']};need(len(inventory)==len(m['payload_inventory']),'duplicate fit payload')
  for path,item in expected.items():need(inventory.get(path)==item,'original image/calibration bytes relabeled')
  for item in inventory.values():
   rel=Path(item['path']);q=r/rel;need(not rel.is_absolute() and '..' not in rel.parts and not q.is_symlink() and q.is_file() and q.stat().st_size==item['bytes'] and digest(q)==item['sha256'],'fit payload byte identity changed')
  allowed=set(expected)|{'train-split.json'}
  for side in ('vehicle-side','infrastructure-side'):
   orig_index=[];orig_meta=[];orig_infos=[]
   for sr,sm in selected:
    orig_index+=json.loads((sr/side/'frame-index.json').read_bytes());orig_meta+=json.loads((sr/side/'data_info.json').read_bytes());orig_infos+=Restricted(io.BytesIO((sr/side/'image-pose-infos.pkl').read_bytes())).load()['infos']
   orig_index.sort(key=lambda x:(x['sequence_id'],x['frame_id']));orig_meta.sort(key=lambda x:(x['sequence_id'],x['frame_id']))
   observed_index=json.loads((r/side/'frame-index.json').read_bytes());observed_meta=json.loads((r/side/'data_info.json').read_bytes())
   need(observed_index==orig_index and observed_meta==orig_meta and len(observed_index)==m['frames'][side] and {x['sequence_id'] for x in observed_index}==set(fit) and not set(fit)&set(held),'fit frame/time/raw metadata coverage changed')
   expected_infos={x['token']:x for x in orig_infos};actual=Restricted(io.BytesIO((r/side/'image-pose-infos.pkl').read_bytes())).load();need(actual['metadata']=={'version':'v1.0-trainval'} and len(actual['infos'])==len(orig_index) and len(expected_infos)==len(orig_index),'fit infos missing or duplicate')
   for value,index in zip(actual['infos'],orig_index):
    need(value['token']==index['frame_id'] and value['scene_token']==index['sequence_id'] and equal(value,expected_infos[index['frame_id']]),'portable info changed value/type/shape/dtype')
   for name,key in [('data_info.json','metadata_sha256'),('frame-index.json','frame_index_sha256'),('image-pose-infos.pkl','infos_sha256')]:
    path=side+'/'+name;allowed.add(path);need(digest(r/path)==m[side][key] and inventory[path]['sha256']==m[side][key],'metadata binding differs')
  need(set(inventory)==allowed,'unexpected fit feature payload')
  observed={q.relative_to(r).as_posix() for q in r.rglob('*') if q.is_file() or q.is_symlink()};need(observed==allowed|{'input-manifest.json'} and not any(q.is_symlink() for q in r.rglob('*')),'fit view has extra/missing/symlink files')
  need(json.loads((r/'train-split.json').read_bytes())=={'batch_split':{'train':fit,'val':[],'test':[],'test_A':[]}},'wrong fit split')
  # Existing OOF scoring input gate must reject this distinct training role.
  try:validate_inputs(r,package,digest(r/'input-manifest.json'))
  except ValueError:pass
  else:raise ValueError('fit view was incorrectly admitted as held-out inference')
  records.append({'fold_id':fold,'input_manifest_sha256':digest(r/'input-manifest.json'),'frames':m['frames'],'files_verified':len(allowed),'fit_sequence_count':len(fit),'held_out_scoring_gate_rejected':True});total+=sum(m['frames'].values());print('FIT_FEATURE_INPUTS_INDEPENDENTLY_READ',fold,'eta_seconds',round((time.monotonic()-start)/(fold+1)*(4-fold),1),flush=True)
 report={'kind':'spd_canonical_oof_fit_feature_inputs_independent_readback_v1','folds':records,'total_subject_frames_verified':total,'all_payload_bytes_and_all_info_values_verified':True,'source_all_train_input_admission_sha256':ADMISSION,'held_out_selection_scoring_eligible':False,'predictions_generated':False,'paper_eligible':False}
 with a.receipt.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print('FIT_FEATURE_INPUTS_INDEPENDENTLY_ACCEPTED',digest(a.receipt),flush=True)
if __name__=='__main__':main()
