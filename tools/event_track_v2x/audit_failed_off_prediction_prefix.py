"""Compare sealed failed-run prediction prefixes; not a full ablation acceptance."""
import argparse,json,hashlib,tarfile,sqlite3
from pathlib import Path
DOC=Path(__file__).resolve().parents[2]/'docs/recover-before-fuse'
BASE=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed1337')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(p):return json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seed',type=int,choices=(1337,3407),default=1337);a=p.parse_args();read=a.root/'registered-byte-readback.json';d=json.loads(read.read_text());assert d['status']=='failed'
 if a.seed==1337:
  index=json.loads((DOC/'SPD-seed1337-real-recovery-candidate-index-current.json').read_text());reports={s['sequence_id']:s for s in index['sequence_reports']};base=BASE
 else:
  accepted=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-forest-seed3407-after-prefetch-continuation-v1-20261005/independent-acceptance')
  assert sha(accepted/'acceptance.json')=='028611d23af082106dfe722d37f802af72ddf12229adcc7b498da0b8c03d4e20'
  receipt=json.loads((accepted/'acceptance.json').read_text());assert receipt['seed']==3407 and receipt['completed_sequences']==46 and receipt['completed_events']==7445
  reports={}
  for q in sorted(accepted.glob('sequence-*.json')):
   v=json.loads(q.read_text());assert v['fresh_state']['database_sha256']==v['database_sha256'];reports[v['sequence_id']]=v
  assert len(reports)==46
  base=BASE.parent/'seed3407'
 results=[]
 for item in d['artifacts']:
  archive=Path(item['path']);assert sha(archive)==item['sha256'];assert archive.stat().st_size==item['bytes']
  with tarfile.open(archive,'r:gz') as t:
   for member in t:
    if not member.isfile() or not member.name.endswith('/predictions.jsonl'):continue
    raw=t.extractfile(member).read();predictions=[json.loads(v) for v in raw.splitlines()];assert predictions
    sid=predictions[0]['sequence_id'];paths=list(base.glob(f'rank*-unpack/rank-*/{sid}/sequence-*.sqlite'));assert len(paths)==1 and sha(paths[0])==reports[sid]['database_sha256']
    c=sqlite3.connect(paths[0].as_uri()+'?mode=ro&immutable=1',uri=True);previous=None;different=[]
    for ordinal,off in enumerate(predictions):
     assert off['sequence_id']==sid
     assert hashlib.sha256(canonical({k:v for k,v in off.items() if k!='commit_sha256'})).hexdigest()==off['commit_sha256']
     if previous is not None:assert off['previous_commit_sha256']==previous
     previous=off['commit_sha256'];row=c.execute('select prediction from events where ordinal=?',(ordinal,)).fetchone();assert row
     on=json.loads(row[0]);assert all(on[k]==off[k] for k in ('sequence_id','frame_id','box_reference_timestamp_us'))
     if canonical(on['predictions'])!=canonical(off['predictions']):
      main_by_id={r['track_id']:r for r in on['predictions']};off_by_id={r['track_id']:r for r in off['predictions']}
      assert len(main_by_id)==len(on['predictions']) and len(off_by_id)==len(off['predictions'])
      common=set(main_by_id)&set(off_by_id);max_mean=max((abs(x-y) for tid in common for x,y in zip(main_by_id[tid]['mean'],off_by_id[tid]['mean'])),default=0.)
      different.append({'event':ordinal,'frame':off['frame_id'],'main_prediction_count':len(on['predictions']),'off_prediction_count':len(off['predictions']),
        'main_only_track_IDs':sorted(set(main_by_id)-set(off_by_id)),'off_only_track_IDs':sorted(set(off_by_id)-set(main_by_id)),
        'same_track_ID_set':set(main_by_id)==set(off_by_id),'same_sorted_prediction_bytes':canonical(sorted(on['predictions'],key=lambda r:r['track_id']))==canonical(sorted(off['predictions'],key=lambda r:r['track_id'])),
        'common_track_class_changes':[tid for tid in sorted(common) if main_by_id[tid]['class_label']!=off_by_id[tid]['class_label']],
        'common_track_max_mean_abs_difference':max_mean})
    c.close();results.append({'sequence':sid,'member_sha256':hashlib.sha256(raw).hexdigest(),'sealed_records_aligned':len(predictions),'different_prediction_events':different,'SQLite_off_transactions_verified':False})
 out={'failed_task_id':d['task_id'],'registered_readback_sha256':sha(read),'source_sha256':sha(Path(__file__)),'results':results,'same_resource_comparison_accepted':False,'formal_metric_comparison':False,'full_recovery_off_accepted':False,'recovery_benefit_proven':False}
 (a.root/'sealed-prediction-common-prefix-difference-classification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'sequence':r['sequence'],'aligned':r['sealed_records_aligned'],'different':len(r['different_prediction_events'])} for r in results]))
if __name__=='__main__':main()
