import ast,json,math,random
from pathlib import Path
from dataclasses import dataclass
from prepared_threaded_factors_v2_candidate import PreparedThreadedFactors, METRICS
ns=dict(dataclass=dataclass,math=math);R=Path('/runtime/transvision/models/event_track_v2x')
for fn,names in [('identity_forest.py',{'IdentityNode','ForestFactors'}),('recovery_off_tracking.py',{'HistoryRestriction','RestrictedFactors'})]:
 nodes=[n for n in ast.parse((R/fn).read_text()).body if isinstance(n,ast.ClassDef) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),fn,'exec'),ns)
rng=random.Random(2027);checks=0;rejections=0
for trial in range(8):
 n=16;ids=tuple(range(n));nodes=tuple(ns['IdentityNode'](str(i),i%2,i,i,str(i//4)) for i in range(n))
 factors=ns['ForestFactors'](nodes,tuple(tuple((p,0.) for p in range(-1,i)) for i in range(n)))
 restriction=ns['HistoryRestriction']([] if trial%2 else [(ids,(ids,(0,)*n))]);original=ns['RestrictedFactors'](factors,restriction)
 for threads in (1,4,8):
  candidate=PreparedThreadedFactors(original,'/native/native.so',threads);prefix=()
  for d in range(n):
   expected=original.children(prefix);assert candidate.children(prefix)==expected
   assert candidate.roots(prefix)==original.roots(prefix)
   for child in expected:assert candidate.roots(child)==original.roots(child)
   checks+=1
   if not expected:break
   prefix=rng.choice(expected)
  for bad in ((False,),(-1.,),(99,),[-1],(-1,0,0,0)):
   try:original.roots(bad)
   except ValueError:
    try:candidate.roots(bad)
    except ValueError:rejections+=1
    else:raise AssertionError(('invalid prefix admitted',bad))
assert METRICS['max_native_workers']==8
print(json.dumps({'native_metrics':METRICS,'passed':True,'children_checks':checks,'invalid_prefix_rejections':rejections,'threads':[1,4,8]}))
