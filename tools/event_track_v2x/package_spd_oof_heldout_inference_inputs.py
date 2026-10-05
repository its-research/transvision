"""Package admitted label-free held-out payloads for later GPU inference."""
import argparse,gzip,hashlib,json,tarfile,time
from pathlib import Path
from spd_canonical_oof_input_gate import validate_inputs,digest
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 admission=ROOT/'receipts/spd-canonical-oof-heldout-inference-inputs-independent-readback-20260930.json'
 if digest(admission)!='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df':raise ValueError('held-out admission changed')
 legacy=ROOT/'receipts/spd-canonical-oof-infos-legacy-runtime-readback-20261001.json'
 if digest(legacy)!='da71c87d0a25f47007dd4a19bec0daa03359c5dc8780fe371ed58ee752c74ae7':raise ValueError('legacy infos admission changed')
 accepted=next(r for r in json.loads(admission.read_bytes())['folds'] if r['fold_id']==a.fold)
 inputs=ROOT/('artifacts/spd-canonical-oof-heldout-inference-inputs-20260930/fold-%d'%a.fold)
 package=ROOT/('artifacts/spd-official-oof-fivefold-20260930/fold-%d-package/package-manifest.json'%a.fold)
 m=validate_inputs(inputs,json.loads(package.read_bytes()),accepted['input_manifest_sha256'])
 if a.output.exists():raise FileExistsError('inference package is create-once')
 a.output.mkdir(parents=True)
 inventory=[{'path':p.relative_to(inputs).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(inputs.rglob('*')) if p.is_file()]
 archive=a.output/'heldout-inputs.tar.gz';started=time.monotonic()
 with archive.open('xb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as compressed:
   with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT) as t:
    for i,row in enumerate(inventory):
     path=inputs/row['path'];info=t.gettarinfo(str(path),arcname='inputs/'+row['path']);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
     with path.open('rb') as f:t.addfile(info,f)
     if (i+1)%5000==0:print('INFERENCE_PACKAGE_PROGRESS '+json.dumps({'fold':a.fold,'files':i+1,'total':len(inventory),'eta_seconds':(time.monotonic()-started)*(len(inventory)-i-1)/(i+1)}),flush=True)
 manifest={'kind':'spd_canonical_oof_heldout_inference_package_v1','fold_id':a.fold,'held_out_sequence_ids':m['train_sequences'],'excluded_fit_sequence_ids':m['excluded_fit_sequence_ids'],'canonical_fivefold_manifest_sha256':m['canonical_fivefold_manifest_sha256'],'official_split_sha256':m['official_split_sha256'],'input_manifest_sha256':accepted['input_manifest_sha256'],'input_admission_sha256':digest(admission),'legacy_infos_admission_sha256':digest(legacy),'archive':{'path':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)},'input_inventory':inventory,'frames':m['frames'],'gt_or_val_test_included':False,'predictions_generated':False,'paper_eligible':False}
 with (a.output/'package-manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,separators=(',',':'));f.write('\n')
 print('INFERENCE_INPUT_PACKAGE_CREATED '+json.dumps({'fold':a.fold,'files':len(inventory),'archive':manifest['archive'],'independent_archive_readback_passed':False}),flush=True)
if __name__=='__main__':main()
