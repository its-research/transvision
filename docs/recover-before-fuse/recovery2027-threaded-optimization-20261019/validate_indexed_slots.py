import sqlite3,json,time,hashlib
from pathlib import Path
from indexed_observation_slots_candidate import INDEXED_SQL
c=sqlite3.connect('file:/baseline/baseline.sqlite?mode=ro',uri=True)
n=c.execute('SELECT count(*) FROM pc86_observations').fetchone()[0]
cases=[]
for d in sorted(set([0,1,n-1,*range(0,n,max(1,n//128))])):
 s,f=c.execute('SELECT source,frame FROM pc86_observations WHERE i=?',(d,)).fetchone()
 for depth in (0,d,d+1):cases.append((s,f,depth))
cases.extend([(999,'missing',n),(0,'missing',n)])
results=[];outputs=[]
for optimized in (False,True):
 start=time.perf_counter();out=[]
 for s,f,d in cases:
  out.append(c.execute(INDEXED_SQL,(s,f,86,d)).fetchall() if optimized else c.execute('SELECT i FROM pc86_observations WHERE source=? AND frame=? AND i<?',(s,f,d)).fetchall())
 results.append({'indexed':optimized,'seconds':time.perf_counter()-start});outputs.append(out)
assert outputs[0]==outputs[1]
r={'passed':True,'cases':len(cases),'results':results,'output_sha256':hashlib.sha256(json.dumps(outputs[0]).encode()).hexdigest(),'original_plan':c.execute('EXPLAIN QUERY PLAN SELECT i FROM pc86_observations WHERE source=? AND frame=? AND i<?',(0,'000730',15000)).fetchall(),'indexed_plan':c.execute('EXPLAIN QUERY PLAN '+INDEXED_SQL,(0,'000730',86,15000)).fetchall()}
Path('/output/indexed-query-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));c.close()
