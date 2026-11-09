"""Inventory preserved failed archives without extracting or recovering SQLite."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath

def main():
 p=argparse.ArgumentParser();p.add_argument('readback',type=Path);p.add_argument('output',type=Path);a=p.parse_args();raw=a.readback.read_bytes();d=json.loads(raw);out=[]
 for item in d['artifacts']:
  path=Path(item['path']);h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
  assert h.hexdigest()==item['sha256'] and path.stat().st_size==item['bytes']
  files=[]
  with tarfile.open(path,'r:gz') as archive:
   for member in archive:
    name=PurePosixPath(member.name)
    assert not name.is_absolute() and '..' not in name.parts
    assert member.isdir() or member.isfile(),'unexpected link/device member'
    if member.isfile():files.append({'name':member.name,'bytes':member.size})
  out.append({'artifact':item['key'],'sha256':item['sha256'],'files':files,'SQLite_members':[v for v in files if '.sqlite' in v['name']],'journal_members':[v for v in files if v['name'].endswith(('-journal','-wal'))]})
 a.output.write_text(json.dumps({'task_id':d['task_id'],'readback_sha256':hashlib.sha256(raw).hexdigest(),'archives':out,'SQLite_extracted':False,'transactions_recovered':False,'committed_event_count_verified':False,'full_recovery_off_accepted':False},indent=2)+'\n')
 print(json.dumps([{'artifact':v['artifact'],'SQLite_members':len(v['SQLite_members']),'journal_members':len(v['journal_members'])} for v in out]))
if __name__=='__main__':main()
