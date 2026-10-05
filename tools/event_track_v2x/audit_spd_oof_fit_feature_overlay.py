"""Read every overlay member; this does not accept reconstructed GPU inputs."""
import argparse,hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path
from spd_canonical_oof_input_gate import digest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);a=p.parse_args()
 mp=a.package/'overlay-manifest.json';m=json.loads(mp.read_bytes());ar=a.package/m['archive']['path']
 if m['kind']!='spd_canonical_oof_fit_feature_metadata_overlay_v1' or m['images_or_calibrations_in_overlay'] is not False or m['held_out_selection_scoring_eligible'] is not False:raise ValueError('wrong overlay contract')
 if ar.stat().st_size!=m['archive']['bytes'] or digest(ar)!=m['archive']['sha256']:raise ValueError('archive differs')
 root=Path('/Volumes/Data/test/recover-before-fuse');fit_receipt=root/'receipts/spd-canonical-oof-fit-feature-inputs-independent-readback-20261001.json'
 if digest(fit_receipt)!='3ff6673dc459939051b5d16ac59f305594f0570e195940dca4d547cd95fc7a76':raise ValueError('fit admission changed')
 source={r['fold_id']:r for r in json.loads(fit_receipt.read_bytes())['folds']}
 allowed={'input-manifest.json','train-split.json'}|{s+'/'+n for s in ('vehicle-side','infrastructure-side') for n in ('data_info.json','frame-index.json','image-pose-infos.pkl')}
 expected={r['path']:r for r in m['inventory']}
 if len(expected)!=40 or set(expected)!={'fold-%d/'%f+n for f in range(5) for n in allowed}:raise ValueError('unexpected overlay payloads')
 seen=set();total=0
 with tarfile.open(ar,'r:gz') as t:
  for member in t:
   if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('unsafe or extra archive member')
   b=t.extractfile(member).read();r=expected[member.name]
   if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('member bytes differ')
   fold=int(member.name.split('/')[0][5:]);rel=member.name.split('/',1)[1]
   original=root/('artifacts/spd-canonical-oof-fit-feature-inputs-20261001/fold-%d'%fold)/rel
   if b!=original.read_bytes():raise ValueError('packaged metadata differs from admitted source')
   if rel=='input-manifest.json' and hashlib.sha256(b).hexdigest()!=source[fold]['input_manifest_sha256']:raise ValueError('wrong fold input identity')
   seen.add(member.name);total+=len(b)
 if seen!=set(expected):raise ValueError('incomplete archive')
 receipt={'kind':'spd_canonical_oof_fit_feature_metadata_overlay_independent_readback_v1','status':'passed','overlay_manifest_sha256':digest(mp),'archive':m['archive'],'members_verified':len(seen),'uncompressed_bytes_verified':total,'all_members_read':True,'materialized_fit_inputs_verified':False,'actual_inference_complete':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 with (a.package/'independent-overlay-readback.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('FIT_OVERLAY_INDEPENDENTLY_READ_BACK',digest(a.package/'independent-overlay-readback.json'))

if __name__=='__main__':main()
