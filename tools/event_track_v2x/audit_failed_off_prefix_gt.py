"""Same-frame GT diagnostic on sealed failed-run prefixes, never paper metrics."""
import json,hashlib,tarfile,zipfile,math
from pathlib import Path
from audit_rbf_positive_recovery_gt import DOC,ARCHIVE,sha,transform
from audit_rbf_recovery_xy_flow import solve
from check_rbf_matching_certificate import certify

def main():
 root=DOC/'recovery-off-1337-failed-output-readback-20261011';readback=root/'registered-byte-readback.json';registered=json.loads(readback.read_text())
 original=DOC/'SPD-seed1337-full-train-GT-assignment-diagnostic-current.json';full=json.loads(original.read_text());lookup={(s['sequence'],e['frame']):e for s in full['results'] for e in s['events']};results=[]
 assert sha(ARCHIVE)=='10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1'
 with zipfile.ZipFile(ARCHIVE) as z:
  for item in registered['artifacts']:
   archive=Path(item['path']);assert sha(archive)==item['sha256']
   with tarfile.open(archive,'r:gz') as t:
    for member in t:
     if not member.isfile() or not member.name.endswith('/predictions.jsonl'):continue
     raw=t.extractfile(member).read();events=[];previous=None
     for ordinal,line in enumerate(raw.splitlines()):
      p=json.loads(line);assert hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='commit_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==p['commit_sha256']
      if previous is not None:assert p['previous_commit_sha256']==previous
      previous=p['commit_sha256'];sid=p['sequence_id'];fid=p['frame_id'];on=lookup[(sid,fid)];assert on['event']==ordinal
      members={}
      for key,path in [('GT',f'cooperative/label/{fid}.json'),('lidar',f'vehicle-side/calib/lidar_to_novatel/{fid}.json'),('world',f'vehicle-side/calib/novatel_to_world/{fid}.json')]:
       b=z.read('V2X-Seq-SPD/'+path);assert hashlib.sha256(b).hexdigest()==on['members_sha256'][key];members[key]=json.loads(b)
      w=members['world'];l=members['lidar'].get('transform',members['lidar']);ego=[float(v[0]) for v in w['translation']];gt=[]
      for row in members['GT']:
       assert int(row['veh_pointcloud_timestamp'])==int(p['box_reference_timestamp_us'])
       if row['type'] not in ('Car','Truck','Van','Bus'):continue
       xyz=transform(w,transform(l,[float(row['3d_location'][k]) for k in ('x','y','z')]))
       if math.hypot(xyz[0]-ego[0],xyz[1]-ego[1])<50:gt.append((row['track_id'],xyz))
      pred=[r for r in p['predictions'] if r['class_label']=='car' and math.hypot(r['mean'][0]-ego[0],r['mean'][1]-ego[1])<50]
      matrix=[[math.hypot(r['mean'][0]-xyz[0],r['mean'][1]-xyz[1]) for tid,xyz in gt] for r in pred];cost,pairs=solve(matrix);certify(matrix,pairs,len(gt))
      events.append({'event':ordinal,'frame':fid,'main_matched':len(on['matches']),'off_matched':len(pairs),'main_unmatched_predictions':len(on['unmatched_predictions']),'off_unmatched_predictions':len(pred)-len(pairs),'main_unmatched_GT':len(on['unmatched_GT']),'off_unmatched_GT':len(gt)-len(pairs),'off_matches':[{'prediction_id':pred[i]['track_id'],'GT_id':gt[j][0],'distance_m':d} for i,j,d in pairs]})
     results.append({'sequence':sid,'events':events,'sealed_prediction_member_sha256':hashlib.sha256(raw).hexdigest()});print('completed',sid,len(events),flush=True)
 out={'registered_readback_sha256':sha(readback),'main_diagnostic_sha256':sha(original),'reader_sha256':sha(Path(__file__)),'results':results,'events':sum(len(r['events']) for r in results),'scope':'failure-selected common prefix, car strict50m ROI, XY<2m geometric diagnostic; not official FP/FN or metrics','SQLite_off_transaction_status_verified':False,'recovery_benefit_proven':False,'full_recovery_off_accepted':False}
 (root/'sealed-prefix-GT-geometric-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
