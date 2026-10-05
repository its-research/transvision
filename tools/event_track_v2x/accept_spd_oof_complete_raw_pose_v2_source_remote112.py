"""Independently rehash all published forward source artifacts on trusted Linux112."""
import argparse,json,shlex,subprocess
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime,timezone
from spd_canonical_oof_input_gate import digest

ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);a=p.parse_args();out=a.package/'clearml-independent-readback.json'
 if out.exists():raise FileExistsError('acceptance is create-once')
 from clearml import Task
 from clearml.storage.helper import StorageHelper
 upload=json.loads((a.package/'clearml-upload-acceptance.json').read_bytes());mp=a.package/'source-manifest.json';m=json.loads(mp.read_bytes());proof=a.package/'independent-source-archive-readback.json'
 if digest(mp)!='04681ea607045761eea078de04a62e764e351a0d64aa2d0b07760d0107fc0dd7' or upload['source_manifest_sha256']!=digest(mp) or upload['status']!='completed_registered_identity_verified':raise ValueError('wrong source admission')
 if json.loads(proof.read_bytes())['source_manifest_sha256']!=digest(mp) or m['actual_gpu_execution_verified'] is not False:raise ValueError('wrong source role')
 task=Task.get_task(task_id=upload['task_id']);params=task.get_parameters()
 if task.status!='completed' or params.get('General/source_manifest_sha256')!=digest(mp) or params.get('General/source_admission_sha256')!=digest(proof):raise ValueError('task binding differs')
 expected={r['name']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in upload['artifacts']}
 if set(expected)!={'source','source-manifest','independent-source-archive-readback'} or expected['source']!={k:m['archive'][k] for k in ('bytes','sha256')}:raise ValueError('artifact contract differs')
 artifacts=[]
 for name,row in expected.items():
  item=task.artifacts[name];parsed=urlparse(item.url)
  if item.hash!=row['sha256'] or item.size!=row['bytes'] or parsed.scheme!='http' or parsed.netloc not in {'10.100.35.118:8081','10.100.34.118:8081'}:raise ValueError('registered identity or file host differs')
  helper=StorageHelper.get(item.url);container=helper._driver._containers['http://'+parsed.netloc]
  artifacts.append(dict(row,name=name,url=item.url,headers=dict(container.get_headers(item.url))))
 source=Path(__file__).with_name('readback_spd_oof_package_remote_bytes.py');frozen=ROOT/'source-freezes/spd-canonical-oof-complete-raw-pose-v2-source-remote-readback-20261001'
 for file in (source,Path(__file__)):
  if file.read_bytes()!=(frozen/file.name).read_bytes():raise ValueError('reader source changed')
 proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','10.100.35.112','python3 -u -c '+shlex.quote(source.read_text())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
 proc.stdin.write(json.dumps({'artifacts':artifacts}));proc.stdin.close();result=None
 for line in proc.stdout:
  if line.startswith('REMOTE_READBACK_RESULT '):result=json.loads(line.split(' ',1)[1])
  elif line.startswith('REMOTE_READBACK_PROGRESS '):print(line,end='',flush=True)
 if proc.wait()!=0 or result is None or result['artifacts']!=expected:raise RuntimeError('independent full byte readback failed')
 current=Task.get_task(task_id=task.id)
 if current.status!='completed' or any(current.artifacts[n].hash!=r['sha256'] or current.artifacts[n].size!=r['bytes'] for n,r in expected.items()):raise ValueError('task changed during readback')
 receipt={'kind':'spd_canonical_oof_complete_raw_pose_v2_source_clearml_independent_readback','status':'independent_bytes_verified','task_id':task.id,'source_manifest_sha256':digest(mp),'artifacts':expected,'execution_host':result['execution_host'],'execution_platform':result['execution_platform'],'acceptor_sha256':digest(Path(__file__)),'remote_reader_sha256':digest(source),'credentials_persisted':False,'actual_gpu_execution_verified':False,'paper_eligible':False,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
 with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('COMPLETE_RAW_POSE_V2_SOURCE_REMOTE_BYTES_ACCEPTED',digest(out),flush=True)

if __name__=='__main__':main()
