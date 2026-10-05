"""Create GT-free fit feature views, never inputs eligible for held-out scoring."""
import argparse,copy,io,json,os,time
from pathlib import Path
from prepare_spd_oof_inference_infos import Portable
from audit_spd_oof_inference_infos import Restricted
from spd_canonical_oof_input_gate import digest,validate_inputs
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
ADMISSION='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
def write(p,value):
 with p.open('x') as f:json.dump(value,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
def load_sources():
 p=ROOT/'receipts/spd-canonical-oof-heldout-inference-inputs-independent-readback-20260930.json'
 if digest(p)!=ADMISSION:raise ValueError('original all-train GT-free admission changed')
 sources=[]
 for rec in json.loads(p.read_bytes())['folds']:
  fold=rec['fold_id'];root=ROOT/('artifacts/spd-canonical-oof-heldout-inference-inputs-20260930/fold-%d'%fold)
  package=json.loads((ROOT/('artifacts/spd-official-oof-fivefold-20260930/fold-%d-package/package-manifest.json'%fold)).read_bytes())
  manifest=validate_inputs(root,package,rec['input_manifest_sha256']);sources.append((root,manifest))
 return sources
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();sources=load_sources()
 if a.output.exists():raise FileExistsError('fit feature inputs are create-once')
 a.output.mkdir(parents=True);records=[];start=time.monotonic()
 for fold in range(5):
  package_path=ROOT/('artifacts/spd-official-oof-fivefold-20260930/fold-%d-package/package-manifest.json'%fold)
  package=json.loads(package_path.read_bytes());fit=package['fit_sequence_ids'];held=package['held_out_sequence_ids'];selected=[(r,m) for r,m in sources if m['fold_id']!=fold]
  if set(fit)!={seq for r,m in selected for seq in m['train_sequences']} or set(fit)&set(held):raise ValueError('frozen fit/held membership differs')
  target=a.output/('fold-%d'%fold);target.mkdir();inventory={};side_meta={};frames={}
  for root,manifest in selected:
   for row in manifest['payload_inventory']:
    rel=Path(row['path'])
    # Reuse only original image/calibration payloads, not per-fold infos/split.
    if len(rel.parts)<3 or rel.parts[1] not in ('image','calib'):continue
    if row['path'] in inventory:raise ValueError('duplicate fit payload identity')
    src=root/rel;dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);inventory[row['path']]=copy.deepcopy(row)
  for side in ('vehicle-side','infrastructure-side'):
   metadata=[];index=[];infos=[]
   for root,manifest in selected:
    metadata.extend(json.loads((root/side/'data_info.json').read_bytes()))
    index.extend(json.loads((root/side/'frame-index.json').read_bytes()))
    values=Restricted(io.BytesIO((root/side/'image-pose-infos.pkl').read_bytes())).load()
    infos.extend(values['infos'])
   metadata.sort(key=lambda r:(r['sequence_id'],r['frame_id']));index.sort(key=lambda r:(r['sequence_id'],r['frame_id']));by_token={r['token']:r for r in infos}
   if len(by_token)!=len(infos) or len({r['frame_id'] for r in index})!=len(index) or len(metadata)!=len(index):raise ValueError('duplicate/missing fit frame')
   infos=[by_token[r['frame_id']] for r in index]
   if {r['scene_token'] for r in infos}!=set(fit) or any(r['scene_token'] in held for r in infos):raise ValueError('held-out frame entered fit inputs')
   write(target/side/'data_info.json',metadata);write(target/side/'frame-index.json',index)
   q=target/side/'image-pose-infos.pkl'
   with q.open('xb') as f:Portable(f,protocol=2).dump({'infos':infos,'metadata':{'version':'v1.0-trainval'}})
   side_meta[side]={'metadata_sha256':digest(target/side/'data_info.json'),'frame_index_sha256':digest(target/side/'frame-index.json'),'infos_sha256':digest(q)};frames[side]=len(index)
   for name in ('data_info.json','frame-index.json','image-pose-infos.pkl'):
    q=target/side/name;rel=side+'/'+name;inventory[rel]={'path':rel,'bytes':q.stat().st_size,'sha256':digest(q)}
  q=target/'train-split.json';write(q,{'batch_split':{'train':fit,'val':[],'test':[],'test_A':[]}});inventory[q.name]={'path':q.name,'bytes':q.stat().st_size,'sha256':digest(q)}
  manifest={'kind':'spd_canonical_oof_fit_feature_inputs_v1','cohort':'canonical-oof-fit-feature-inputs','fold_id':fold,'fit_sequence_ids':fit,'excluded_held_out_sequence_ids':held,'training_package_manifest_sha256':digest(package_path),'canonical_fivefold_manifest_sha256':package['canonical_fivefold_manifest_sha256'],'official_split_sha256':package['official_split_sha256'],'source_all_train_input_admission_sha256':ADMISSION,'source_input_manifests':[{'fold_id':m['fold_id'],'sha256':digest(r/'input-manifest.json')} for r,m in selected],'frames':frames,**side_meta,'payload_inventory':sorted(inventory.values(),key=lambda r:r['path']),'infos_encoding':'pickle-protocol2-numpy-public-array-v1','gt_payloads_in_package':False,'gt_payloads_read':False,'val_payloads_read':False,'test_payloads_read':False,'held_out_selection_scoring_eligible':False,'predictions_generated':False,'paper_eligible':False,'purpose':'fit-only score/covariance/association/VoI feature inference; supervision remains separately isolated'}
  write(target/'input-manifest.json',manifest);records.append({'fold_id':fold,'manifest_sha256':digest(target/'input-manifest.json'),'frames':frames});print('FIT_FEATURE_INPUTS_PREPARED',fold,frames,'eta_seconds',round((time.monotonic()-start)/(fold+1)*(4-fold),1),flush=True)
 write(a.output/'preparation-receipt.json',{'kind':'spd_canonical_oof_fit_feature_preparation_v1','folds':records,'independent_readback_passed':False,'held_out_selection_scoring_eligible':False,'predictions_generated':False})
if __name__=='__main__':main()
