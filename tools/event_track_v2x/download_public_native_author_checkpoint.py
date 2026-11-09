#!/usr/bin/env python3
"""Read public author-pinned checkpoint bytes; preserve failures, no inference."""
import argparse,datetime,hashlib,json,os,time
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();plan=json.loads(a.plan.read_bytes());row=next(r for r in plan['files'] if r['name']==a.name);root=Path(plan['output_root']);output=root/a.name;part=root/(a.name+'.partial');receipt=root/(a.name+'.byte-receipt.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['producer_source_sha256']
    assert not output.exists() and not part.exists() and not receipt.exists();h=hashlib.sha256();md5=hashlib.md5();count=0;start=last=time.monotonic()
    with requests.get(row['url'],stream=True,timeout=(15,120)) as r:
        r.raise_for_status();content_type=r.headers.get('Content-Type','');assert 'text/html' not in content_type and 'text/plain' not in content_type, 'author download returned a page, not checkpoint bytes'
        declared=int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
        with part.open('xb') as f:
            for block in r.iter_content(8*1024**2):
                if not block:continue
                f.write(block);h.update(block);md5.update(block);count+=len(block);now=time.monotonic()
                if now-last>=30:
                    print(json.dumps(dict(name=a.name,actual_bytes=count,total_bytes=row['expected_bytes'] or declared,elapsed_seconds=now-start)),flush=True);last=now
            f.flush();os.fsync(f.fileno())
    if declared is not None:assert count==declared
    if row['expected_bytes'] is not None:assert count==row['expected_bytes']
    if row['expected_sha256'] is not None:assert h.hexdigest()==row['expected_sha256']
    if row['expected_author_md5'] is not None:assert md5.hexdigest()==row['expected_author_md5']
    assert count>1024;os.rename(part,output)
    # Re-read the saved file independently from the network streaming hashes.
    again=hashlib.sha256();second_count=0
    with output.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):again.update(block);second_count+=len(block)
    assert second_count==count and again.hexdigest()==h.hexdigest()
    v=dict(kind='rbf_public_native_author_checkpoint_complete_byte_acquisition',name=a.name,method=row['method'],path=str(output),bytes=count,sha256=h.hexdigest(),md5=md5.hexdigest(),source=row,download_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),saved_file_independently_reread=True,checkpoint_tensors_qualified=False,native_predictions_produced=False,native_method_accepted=False,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt.open('x').write(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps({'name':a.name,'bytes':count,'sha256':h.hexdigest(),'author_byte_acquisition_complete':True}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__,'partial_files_preserved':True,'no_automatic_retry':True}),flush=True);raise
