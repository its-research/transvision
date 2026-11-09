import json,sqlite3,hashlib,zipfile,math,datetime,sys
from pathlib import Path
root=Path('/Users/lbin/transvision');index=json.loads((root/'docs/recover-before-fuse/SPD-seed1337-real-recovery-candidate-index-current.json').read_text());sid=sys.argv[1];record=next(r for r in index['sequence_reports'] if r['sequence_id']==sid);assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256'];base=Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed1337'); matches=list(base.glob('rank*-unpack/rank-*/'+sid+'/sequence-*.sqlite')); assert len(matches)==1; db=matches[0];h=hashlib.sha256()
with db.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==record['database_sha256'];conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
def component(n,k):return next(v for v in json.loads(conn.execute('select audit from events where ordinal=?',(n,)).fetchone()[0])['components'] if v['component']==k)
def chain(k,h):
 rows={}
 while h is not None:
  parent,depth,r=conn.execute(f'select parent,depth,root from pc{k}_prefixes where h=?',(h,)).fetchone()
  if depth:rows[depth-1]=r
  h=parent
 return rows
results=[]
with zipfile.ZipFile('/Volumes/Data/test/recover-before-fuse/artifacts/spd-mht-k4-gt-source-recovery-20260929/V2X-Seq-SPD.zip') as archive:
 for case in sorted([r for r in index['candidate_events'] if r['sequence_id']==sid],key=lambda r:r['event_ordinal']):
  n=case['event_ordinal'];k=case['current_component'];now=component(n,k);prev=component(n-1,k);assert now['output_handle']==case['output_handle'];old=chain(k,prev['output_handle']);new=chain(k,now['output_handle']);changed=[i for i in old if old[i]!=new[i]];ids=[]
  for r in sorted({v for i in changed for v in [old[i],new[i]]}):
   g=conn.execute('select global_i from component_members where component=? and local_i=?',(k,r)).fetchone()[0];node=conn.execute('select node_id from observations where i=?',(g,)).fetchone()[0];ids.append(sid+':'+hashlib.sha256(json.dumps(['forest-birth',node],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:24])
  prediction=json.loads(conn.execute('select prediction from events where ordinal=?',(n,)).fetchone()[0]);assert hashlib.sha256(json.dumps({k:v for k,v in prediction.items() if k!='commit_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==prediction['commit_sha256'];frame=prediction['frame_id'];raw=archive.read('V2X-Seq-SPD/vehicle-side/calib/novatel_to_world/'+frame+'.json');ego=[float(v[0]) for v in json.loads(raw)['translation']];visible=[]
  for p in prediction['predictions']:
   if p['track_id'] in ids:visible.append({'track_id':p['track_id'],'class':p['class_label'],'XY_from_ego_m':math.hypot(p['mean'][0]-ego[0],p['mean'][1]-ego[1]),'car_inside_ROI':p['class_label']=='car' and math.hypot(p['mean'][0]-ego[0],p['mean'][1]-ego[1])<50})
  results.append({'event_ordinal':n,'component':k,'frame':frame,'changed_old_rows':len(changed),'visible_outputs':visible,'calibration_member_sha256':hashlib.sha256(raw).hexdigest(),'screen_positive':any(v['car_inside_ROI'] for v in visible)})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sequence':sid,'database_SHA_verified':True,'report_SHA_verified':True,'reader_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates_checked':len(results),'positive_events':sum(r['screen_positive'] for r in results),'all_results':results,'GT_correctness_or_recovery_advantage_proven':False};p=root/('docs/recover-before-fuse/SPD-seed1337-sequence'+sid+'-recovery-ROI-screen-current.json');p.write_text(json.dumps(out,indent=2)+'\n');print('candidates',len(results),'positive',out['positive_events'])
