"""Read selected checkpoint-save sources while hashing the whole registered archive."""
import sys,json,hashlib,time,tarfile,requests
spec=json.load(sys.stdin)
response=requests.get(spec['url'],headers=spec['headers'],stream=True,timeout=(15,60))
if response.status_code!=200:raise RuntimeError('runtime archive HTTP access failed')
class Reader:
 def __init__(self):self.digest=hashlib.sha256();self.count=0;self.start=time.monotonic();self.marker=0
 def read(self,n=1024*1024):
  b=response.raw.read(n if n>0 else 1024*1024);self.digest.update(b);self.count+=len(b)
  if self.count>spec['bytes']:raise ValueError('runtime archive exceeds registration')
  if self.count//(256*1024*1024)>self.marker:
   self.marker=self.count//(256*1024*1024);elapsed=time.monotonic()-self.start
   print('REMOTE_RUNTIME_ARCHIVE_PROGRESS '+json.dumps({'bytes':self.count,'expected':spec['bytes'],'eta_seconds':(spec['bytes']-self.count)*elapsed/self.count}),flush=True)
  return b
reader=Reader();selected={}
with tarfile.open(fileobj=reader,mode='r|gz') as archive:
 for member in archive:
  if member.name in spec['members']:
   if not member.isfile() or member.name in selected:raise ValueError('invalid selected runtime member')
   b=archive.extractfile(member).read();selected[member.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source':b.decode('utf-8')}
while reader.read():pass
response.close()
if reader.count!=spec['bytes'] or reader.digest.hexdigest()!=spec['sha256'] or set(selected)!=set(spec['members']):raise ValueError('runtime archive bytes or selected inventory differs')
print('REMOTE_RUNTIME_SOURCE_RESULT '+json.dumps({'archive_bytes':reader.count,'archive_sha256':reader.digest.hexdigest(),'members':selected}),flush=True)
