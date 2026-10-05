"""Reconstruct admitted fit inputs from metadata overlay and GT-free shared assets."""
import argparse,hashlib,json,os,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
from spd_canonical_oof_fit_feature_input_gate import validate_inputs,digest

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('package','shared-inputs','training-packages','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();proof=a.package/'independent-overlay-readback.json'
 if digest(proof)!='a5707455ccd7310e0315378b388e5d9d3172a5829715d1be0005a882f228817b':raise ValueError('overlay admission changed')
 admitted=json.loads(proof.read_bytes());mp=a.package/'overlay-manifest.json';m=json.loads(mp.read_bytes());ar=a.package/m['archive']['path']
 if admitted['overlay_manifest_sha256']!=digest(mp) or admitted['archive']!=m['archive'] or not admitted['all_members_read']:raise ValueError('overlay proof differs')
 if m['kind']!='spd_canonical_oof_fit_feature_metadata_overlay_v1' or m['held_out_selection_scoring_eligible'] is not False or ar.stat().st_size!=m['archive']['bytes'] or digest(ar)!=m['archive']['sha256']:raise ValueError('wrong overlay bytes/role')
 allowed={'input-manifest.json','train-split.json'}|{s+'/'+n for s in ('vehicle-side','infrastructure-side') for n in ('data_info.json','frame-index.json','image-pose-infos.pkl')}
 expected={r['path']:r for r in m['inventory']}
 if len(expected)!=40 or set(expected)!={'fold-%d/'%f+n for f in range(5) for n in allowed}:raise ValueError('wrong overlay inventory')
 payloads={}
 with tarfile.open(ar,'r:gz') as t:
  for member in t:
   if not member.isfile() or member.name not in expected or member.name in payloads:raise ValueError('unsafe archive member')
   raw=t.extractfile(member).read();r=expected[member.name]
   if len(raw)!=r['bytes'] or hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('overlay member differs')
   payloads[member.name]=raw
 if set(payloads)!=set(expected):raise ValueError('missing metadata')
 if a.output.exists():raise FileExistsError('reconstructed inputs are create-once')
 a.output.mkdir(parents=True);results=[];started=time.monotonic()
 for rec in sorted(m['folds'],key=lambda r:r['fold_id']):
  fold=rec['fold_id'];prefix='fold-%d/'%fold;root=a.output/('fold-%d'%fold);root.mkdir()
  fit=json.loads(payloads[prefix+'input-manifest.json'])
  shared={}
  for src in fit['source_input_manifests']:
   sr=a.shared_inputs/('fold-%d'%src['fold_id']);sp=sr/'input-manifest.json'
   if digest(sp)!=src['sha256']:raise ValueError('shared source manifest changed')
   hm=json.loads(sp.read_bytes())
   if hm['kind']!='eventtrack_train_image_pose_inputs_v1' or hm['cohort']!='canonical-oof-held-out' or hm['fold_id']!=src['fold_id']:raise ValueError('shared source role differs')
   for row in hm['payload_inventory']:
    rel=Path(row['path'])
    if len(rel.parts)<3 or rel.parts[1] not in ('image','calib'):continue
    if rel.is_absolute() or '..' in rel.parts or row['path'] in shared:raise ValueError('unsafe/duplicate shared asset')
    shared[row['path']]=(sr/rel,row)
  required={r['path']:r for r in fit['payload_inventory'] if len(Path(r['path']).parts)>=3 and Path(r['path']).parts[1] in ('image','calib')}
  if {name:row for name,(_,row) in shared.items()}!=required:raise ValueError('shared payloads differ from exact fit complement')
  for name,(source,row) in shared.items():
   if source.is_symlink() or not source.is_file() or source.stat().st_size!=row['bytes']:raise ValueError('unsafe shared source')
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
  for name in allowed:
   q=root/name;q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('xb') as f:f.write(payloads[prefix+name])
  package_path=a.training_packages/('fold-%d-package/package-manifest.json'%fold)
  parsed=validate_inputs(root,json.loads(package_path.read_bytes()),rec['input_manifest_sha256'],package_manifest_sha256=digest(package_path))
  results.append({'fold_id':fold,'input_manifest_sha256':digest(root/'input-manifest.json'),'files_verified':len(parsed['payload_inventory'])+1,'frames':parsed['frames'],'held_out_selection_scoring_eligible':False})
  print('FIT_OVERLAY_RECONSTRUCTED_AND_VERIFIED',fold,'eta_seconds',round((time.monotonic()-started)/(fold+1)*(4-fold),1),flush=True)
 receipt={'kind':'spd_canonical_oof_fit_feature_overlay_materialization_actual_readback_v1','status':'passed','folds':results,'overlay_admission_sha256':digest(proof),'all_reconstructed_payload_bytes_verified':True,'predictions_generated':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 with (a.output/'materialization-acceptance.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('FIT_OVERLAY_MATERIALIZATION_ACCEPTED',digest(a.output/'materialization-acceptance.json'),flush=True)

if __name__=='__main__':main()
