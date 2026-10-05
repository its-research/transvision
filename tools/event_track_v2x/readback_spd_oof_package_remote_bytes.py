#!/usr/bin/env python3
"""Hash exact ClearML artifact bytes; request credentials exist only in stdin."""
import hashlib
import json
import platform
import socket
import sys
import time
import urllib.parse
import urllib.request


def main():
    payload=json.load(sys.stdin);results={}
    for item in payload['artifacts']:
        url=item['url'];parsed=urllib.parse.urlparse(url)
        if parsed.scheme!='http' or parsed.netloc!='10.100.35.118:8081':
            raise ValueError('registered file-service identity differs')
        size=item['bytes'];offset=0;digest=hashlib.sha256();started=time.monotonic()
        while offset<size:
            end=min(offset+64*1024*1024,size)-1
            headers=dict(item['headers']);headers['Accept-Encoding']='identity'
            headers['Range']=f'bytes={offset}-{end}'
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                        if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{size}'
                                or int(response.headers.get('Content-Length','-1'))!=end-offset+1
                                or response.headers.get('Content-Encoding','identity') not in ('identity','')):
                            raise ValueError('strict identity Range mismatch')
                        candidate=digest.copy();received=0
                        while True:
                            block=response.read(1024*1024)
                            if not block:break
                            received+=len(block);candidate.update(block)
                        if received!=end-offset+1:
                            raise ValueError('short artifact Range')
                    digest=candidate;offset+=received;break
                except Exception:
                    if attempt==2:raise
            elapsed=max(time.monotonic()-started,.001)
            if offset%(256*1024*1024)<64*1024*1024 or offset==size:
                print('REMOTE_READBACK_PROGRESS '+json.dumps({'artifact':item['name'],'bytes_verified':offset,
                    'expected_bytes':size,'eta_seconds':(size-offset)*elapsed/offset}),flush=True)
        if digest.hexdigest()!=item['sha256']:
            raise ValueError('complete independent artifact SHA differs')
        results[item['name']]={'bytes':offset,'sha256':digest.hexdigest()}
    print('REMOTE_READBACK_RESULT '+json.dumps({'artifacts':results,'execution_host':socket.gethostname(),
        'execution_platform':platform.platform(),'python_version':platform.python_version()}),flush=True)


if __name__=='__main__':
    main()
