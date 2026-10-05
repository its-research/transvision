"""Package exact fit metadata, referencing shared admitted GT-free image assets."""
import argparse,gzip,json,tarfile
from pathlib import Path
from spd_canonical_oof_input_gate import digest

ROOT=Path('/Volumes/Data/test/recover-before-fuse')
PINS={
 'spd-canonical-oof-fit-feature-inputs-independent-readback-20261001.json':'3ff6673dc459939051b5d16ac59f305594f0570e195940dca4d547cd95fc7a76',
 'spd-canonical-oof-fit-feature-infos-legacy-runtime-readback-20261001.json':'1e02c8869eef4ff26a7a45434bcd791a683d6326ac870af470759c4c17d8c44d',
 'spd-canonical-oof-fivefold-heldout-inference-package-admission-20261001.json':'a38ed021adb55158c9546d33f94280efe83cbd44e88dd68cefad79f7c374b98c'}
FILES=['input-manifest.json','train-split.json']+[side+'/'+name for side in ('vehicle-side','infrastructure-side') for name in ('data_info.json','frame-index.json','image-pose-infos.pkl')]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 receipts={}
 for name,h in PINS.items():
  q=ROOT/'receipts'/name
  if digest(q)!=h:raise ValueError('source admission changed')
  receipts[name]=json.loads(q.read_bytes())
 fit=receipts[next(iter(PINS))]['folds'];held=receipts['spd-canonical-oof-fivefold-heldout-inference-package-admission-20261001.json']['folds']
 held_inventory={}
 for row in held:
  q=ROOT/('artifacts/spd-canonical-oof-heldout-inference-packages-20261001/fold-%d/package-manifest.json'%row['fold_id'])
  if digest(q)!=row['package_manifest_sha256']:raise ValueError('shared source package changed')
  held_inventory[row['fold_id']]={r['path']:r for r in json.loads(q.read_bytes())['input_inventory']}
 if a.output.exists():raise FileExistsError('overlay output is create-once')
 a.output.mkdir(parents=True);inventory=[];folds=[]
 archive=a.output/'fit-feature-overlay.tar.gz'
 with archive.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tar:
  for row in fit:
   fold=row['fold_id'];root=ROOT/('artifacts/spd-canonical-oof-fit-feature-inputs-20261001/fold-%d'%fold)
   if digest(root/'input-manifest.json')!=row['input_manifest_sha256']:raise ValueError('fit metadata differs from admission')
   m=json.loads((root/'input-manifest.json').read_bytes())
   if m['held_out_selection_scoring_eligible'] is not False or m['kind']!='spd_canonical_oof_fit_feature_inputs_v1':raise ValueError('wrong fit role')
   shared={path:r for f,items in held_inventory.items() if f!=fold for path,r in items.items() if len(Path(path).parts)>=3 and Path(path).parts[1] in ('image','calib')}
   expected={r['path']:r for r in m['payload_inventory'] if len(Path(r['path']).parts)>=3 and Path(r['path']).parts[1] in ('image','calib')}
   if shared!=expected:raise ValueError('shared source payload coverage differs from admitted fit')
   for name in FILES:
    q=root/name
    if q.is_symlink() or not q.is_file():raise ValueError('unsafe metadata')
    arc='fold-%d/'%fold+name;info=tar.gettarinfo(str(q),arcname=arc);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
    with q.open('rb') as f:tar.addfile(info,f)
    inventory.append({'path':arc,'bytes':q.stat().st_size,'sha256':digest(q)})
   folds.append({'fold_id':fold,'input_manifest_sha256':row['input_manifest_sha256'],'shared_source_folds':[f for f in range(5) if f!=fold],'shared_payload_count':len(shared),'frames':row['frames']})
 manifest={'kind':'spd_canonical_oof_fit_feature_metadata_overlay_v1','archive':{'path':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},'inventory':inventory,'folds':folds,'source_admission_sha256':PINS,'shared_heldout_packages':held,'images_or_calibrations_in_overlay':False,'materialized_fit_inputs_verified':False,'source_uploaded':False,'predictions_generated':False,'held_out_selection_scoring_eligible':False,'paper_eligible':False}
 with (a.output/'overlay-manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
 print('FIT_FEATURE_OVERLAY_PACKAGED',archive.stat().st_size,digest(archive),flush=True)

if __name__=='__main__':main()
