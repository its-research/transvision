"""Failure-selected paired geometric diagnostic, never formal tracking metrics."""
import json,hashlib,tarfile,sqlite3,zipfile,math
from pathlib import Path
from audit_rbf_positive_recovery_gt import DOC,ARCHIVE,sha,transform
from audit_rbf_recovery_xy_flow import solve
from check_rbf_matching_certificate import certify

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 root=DOC/'recovery-off-3407-failed-output-readback-20261011';read=root/'registered-byte-readback.json';registered=json.loads(read.read_text());assert registered['status']=='failed'
 accepted=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-forest-seed3407-after-prefetch-continuation-v1-20261005/independent-acceptance')
 assert sha(accepted/'acceptance.json')=='028611d23af082106dfe722d37f802af72ddf12229adcc7b498da0b8c03d4e20'
 reports={v['sequence_id']:v for q in accepted.glob('sequence-*.json') for v in [json.loads(q.read_text())]};assert len(reports)==46
 base=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed3407')
 assert sha(ARCHIVE)=='10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1'
 results=[]
 with zipfile.ZipFile(ARCHIVE) as z:
  for item in registered['artifacts']:
   archive=Path(item['path']);assert sha(archive)==item['sha256'] and archive.stat().st_size==item['bytes']
   with tarfile.open(archive,'r:gz') as t:
    members=[m for m in t.getmembers() if m.isfile() and m.name.endswith('/predictions.jsonl')];assert len(members)==1
    raw=t.extractfile(members[0]).read();offrows=[json.loads(line) for line in raw.splitlines()];sid=offrows[0]['sequence_id'];paths=list(base.glob(f'rank*-unpack/rank-*/{sid}/sequence-*.sqlite'));assert len(paths)==1 and sha(paths[0])==reports[sid]['database_sha256']
    c=sqlite3.connect(paths[0].as_uri()+'?mode=ro&immutable=1',uri=True);events=[];previous={'main':None,'off':None}
    for ordinal,off in enumerate(offrows):
     on=json.loads(c.execute('select prediction from events where ordinal=?',(ordinal,)).fetchone()[0]);assert all(on[k]==off[k] for k in ('sequence_id','frame_id','box_reference_timestamp_us','decision_timestamp_us'))
     for label,p in [('main',on),('off',off)]:
      assert hashlib.sha256(canonical({k:v for k,v in p.items() if k!='commit_sha256'})).hexdigest()==p['commit_sha256']
      if previous[label] is not None:assert p['previous_commit_sha256']==previous[label]
      previous[label]=p['commit_sha256']
     fid=off['frame_id'];source={};hashes={}
     for key,path in [('GT',f'cooperative/label/{fid}.json'),('lidar',f'vehicle-side/calib/lidar_to_novatel/{fid}.json'),('world',f'vehicle-side/calib/novatel_to_world/{fid}.json')]:
      b=z.read('V2X-Seq-SPD/'+path);hashes[key]=hashlib.sha256(b).hexdigest();source[key]=json.loads(b)
     w=source['world'];l=source['lidar'].get('transform',source['lidar']);ego=[float(v[0]) for v in w['translation']];gt=[]
     for row in source['GT']:
      assert int(row['veh_pointcloud_timestamp'])==int(off['box_reference_timestamp_us'])
      if row['type'] not in ('Car','Truck','Van','Bus'):continue
      xyz=transform(w,transform(l,[float(row['3d_location'][k]) for k in ('x','y','z')]))
      if math.hypot(xyz[0]-ego[0],xyz[1]-ego[1])<50:gt.append((row['track_id'],xyz))
     result={'event':ordinal,'frame':fid,'members_sha256':hashes}
     for label,p in [('main',on),('off',off)]:
      pred=[v for v in p['predictions'] if v['class_label']=='car' and math.hypot(v['mean'][0]-ego[0],v['mean'][1]-ego[1])<50]
      matrix=[[math.hypot(v['mean'][0]-xyz[0],v['mean'][1]-xyz[1]) for tid,xyz in gt] for v in pred];cost,pairs=solve(matrix);certify(matrix,pairs,len(gt))
      result[label]={'matched':len(pairs),'unmatched_predictions':len(pred)-len(pairs),'unmatched_GT':len(gt)-len(pairs),'matches':[{'prediction_id':pred[i]['track_id'],'GT_id':gt[j][0],'distance_m':d} for i,j,d in pairs]}
     events.append(result)
    c.close();results.append({'sequence':sid,'database_sha256':reports[sid]['database_sha256'],'events':events});print('completed',sid,len(events),flush=True)
 out={'registered_readback_sha256':sha(read),'reader_sha256':sha(Path(__file__)),'results':results,'events':sum(len(v['events']) for v in results),'scope':'failure-selected sealed common prefix; car strict50m ROI XY<2m geometric diagnostic, not formal FP/FN or tracking metrics','recovery_benefit_proven':False,'full_recovery_off_accepted':False}
 (root/'sealed-prefix-GT-geometric-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
