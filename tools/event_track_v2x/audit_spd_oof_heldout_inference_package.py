"""Independently read every archive member against accepted held-out input identity."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from spd_canonical_oof_input_gate import digest
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True,type=Path);a=p.parse_args();out=a.package/'independent-archive-readback.json'
 if out.exists():raise FileExistsError('archive receipt is create-once')
 m=json.loads((a.package/'package-manifest.json').read_bytes());fold=m['fold_id'];admission=ROOT/'receipts/spd-canonical-oof-heldout-inference-inputs-independent-readback-20260930.json'
 if digest(admission)!='29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df':raise ValueError('source admission changed')
 accepted=next(r for r in json.loads(admission.read_bytes())['folds'] if r['fold_id']==fold)
 if m['input_manifest_sha256']!=accepted['input_manifest_sha256'] or m['frames']!=accepted['frames'] or m['gt_or_val_test_included'] is not False:raise ValueError('archive source identity differs')
 ar=a.package/m['archive']['path']
 if ar.stat().st_size!=m['archive']['bytes'] or digest(ar)!=m['archive']['sha256']:raise ValueError('archive bytes changed')
 expected={r['path']:r for r in m['input_inventory']};assert len(expected)==len(m['input_inventory'])
 seen=set();records={};embedded=None;split=None
 with tarfile.open(ar,'r|gz') as t:
  for member in t:
   if not member.isfile() or not member.name.startswith('inputs/'):raise ValueError('non-file or out-of-scope archive member')
   rel=member.name[len('inputs/'):]
   if Path(rel).is_absolute() or '..' in Path(rel).parts or rel not in expected or rel in seen:raise ValueError('unsafe/duplicate/unexpected archive member')
   if member.size!=expected[rel]['bytes']:raise ValueError('archive member length differs')
   h=hashlib.sha256();chunks=[];stream=t.extractfile(member)
   for b in iter(lambda:stream.read(1024*1024),b''):
    h.update(b)
    if rel in ('input-manifest.json','train-split.json'):chunks.append(b)
   if h.hexdigest()!=expected[rel]['sha256']:raise ValueError('archive member hash differs')
   if rel=='input-manifest.json':embedded=json.loads(b''.join(chunks))
   if rel=='train-split.json':split=json.loads(b''.join(chunks))
   records[rel]=h.hexdigest();seen.add(rel)
 if seen!=set(expected) or records['input-manifest.json']!=accepted['input_manifest_sha256']:raise ValueError('incomplete archive or altered embedded input manifest')
 allowed={r['path'] for r in embedded['payload_inventory']}|{'input-manifest.json'}
 for side in ('vehicle-side','infrastructure-side'):
  for name,key in [('data_info.json','metadata_sha256'),('frame-index.json','frame_index_sha256')]:
   rel=side+'/'+name;allowed.add(rel)
   if records[rel]!=embedded[side][key]:raise ValueError('metadata identity differs')
 for r in embedded['payload_inventory']:
  if records[r['path']]!=r['sha256'] or expected[r['path']]['bytes']!=r['bytes']:raise ValueError('embedded payload identity differs')
 if allowed!=seen or embedded['cohort']!='canonical-oof-held-out' or embedded['fold_id']!=fold or embedded['train_sequences']!=m['held_out_sequence_ids'] or embedded['excluded_fit_sequence_ids']!=m['excluded_fit_sequence_ids']:raise ValueError('held-out archive boundary changed')
 if split!={'batch_split':{'train':m['held_out_sequence_ids'],'val':[],'test':[],'test_A':[]}}:raise ValueError('split contains held-out test/val')
 receipt={'kind':'spd_canonical_oof_heldout_inference_package_independent_readback_v1','fold_id':fold,'package_manifest_sha256':digest(a.package/'package-manifest.json'),'archive':m['archive'],'files_verified':len(seen),'frames':m['frames'],'all_archive_payloads_verified':True,'source_input_admission_sha256':digest(admission),'predictions_generated':False,'paper_eligible':False}
 with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('INFERENCE_PACKAGE_INDEPENDENTLY_ACCEPTED '+digest(out),flush=True)
if __name__=='__main__':main()
