import ast,gc,hashlib,pathlib,random,types,unittest,weakref,json,sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from incremental_projection_candidate import projected_kernel_choice
P=pathlib.Path(__file__).resolve().parent
raw=(P.parent/'teacher-compact-storage-20261014/recovery2027-frozen-source-inspection-20261015/transvision/models/event_track_v2x/recovery_off_tracking.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4525e65a2a85a37c19aa5618a8cd5993d271fa4650266fbe3dcc61c1b3d465b2'
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='HistoryRestriction')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_HistoryRestriction','exec'),namespace)
HistoryRestriction=namespace['HistoryRestriction']
class Kernel:
    def __init__(self,roots):self.roots=roots;self.ancestor_calls=0
    def _prefix(self,h):return types.SimpleNamespace(depth=h,parent_handle=h-1,root=self.roots[h-1] if h else -1)
    def ancestor(self,h,d):self.ancestor_calls+=1;assert 0<=d<=h;return d

class ProjectionTests(unittest.TestCase):
    def test_frozen_sqlite_ancestor_methods_with_small_cache(self):
        path=P/'runtime/transvision/models/event_track_v2x/persistent_forest.py'
        manifest=json.loads((P/'runtime-package-manifest.json').read_text())['files']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest['transvision/models/event_track_v2x/persistent_forest.py'])
        tree=ast.parse(path.read_bytes());prefix=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_Prefix')
        tracker=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PersistentForestTracker')
        methods=[x for x in tracker.body if isinstance(x,ast.FunctionDef) and x.name in ('_prefix','ancestor')]
        ns={'dataclass':dataclass,'json':json}
        exec(compile(ast.Module(body=[prefix,*methods],type_ignores=[]),'frozen_SQL_methods','exec'),ns)
        SQLKernel=type('SQLKernel',(),{n:ns[n] for n in ('_prefix','ancestor')})
        kernels=[]
        for _ in range(2):
            k=SQLKernel();k.cache=OrderedDict();k.config=types.SimpleNamespace(prefix_cache_entries=8);k.db=sqlite3.connect(':memory:');kernels.append(k)
            k.db.execute('CREATE TABLE prefixes(h INTEGER PRIMARY KEY,parent,depth,choice,root,previous_root,jumps,sha)')
            for depth in range(41):
                jumps=[depth-(1<<bit) for bit in range(depth.bit_length())]
                k.db.execute('INSERT INTO prefixes VALUES(?,?,?,?,?,?,?,?)',(depth,depth-1 if depth else None,depth,-1,depth-1,None,json.dumps(jumps),'fixture'))
            k.db.commit()
        try:
            counts=[0,0]
            for index,k in enumerate(kernels):
                def trace(sql,index=index):
                    if sql.startswith('SELECT'):counts[index]+=1
                k.db.set_trace_callback(trace)
            before=[k.db.serialize() for k in kernels]
            ids=tuple(range(40));a=HistoryRestriction([(ids,(ids,(0,)*40))]);b=HistoryRestriction([(ids,(ids,(0,)*40))])
            for root in range(40):self.assertEqual(a.kernel_choice(kernels[0],39,root),projected_kernel_choice(b,kernels[1],39,root))
            self.assertLess(counts[1],counts[0])
            self.assertEqual(before,[k.db.serialize() for k in kernels])
            self.assertTrue(all(len(k.cache)<=8 for k in kernels))
        finally:
            for k in kernels:k.db.close()
    def test_random_correlated_disjoint_histories(self):
        rng=random.Random(3407)
        for trial in range(24):
            n=24;clauses=[];roots=list(range(n))
            for side in (0,1):
                ids=tuple(range(side,n,2));variants=set()
                for _ in range(8):
                    v=[]
                    for j,i in enumerate(ids):v.append(i if j==0 or rng.randrange(3)==0 else v[rng.randrange(j)])
                    variants.add(tuple(v))
                variants=tuple(sorted(variants));clauses.append((ids,variants))
                selected=rng.choice(variants)
                for i,value in zip(ids,selected):roots[i]=value
            r=HistoryRestriction(clauses);k=Kernel(roots)
            for depth in range(n+1):
                for root in range(depth+1):
                    self.assertEqual(r.kernel_choice(k,depth,root),projected_kernel_choice(r,k,depth,root))
    def test_does_not_retain_kernel_or_reuse_another_owners_projection(self):
        ids=(0,1,2);r=HistoryRestriction([(ids,(ids,(0,0,0)))])
        k=Kernel([0,1,2]);reference=weakref.ref(k)
        self.assertTrue(projected_kernel_choice(r,k,2,2))
        del k;gc.collect();self.assertIsNone(reference())
        other=Kernel([0,0,0])
        self.assertFalse(projected_kernel_choice(r,other,2,2))
        self.assertTrue(projected_kernel_choice(r,other,2,0))
    def test_equivalence_and_handle_owner_invalidation(self):
        rng=random.Random(2027)
        for n in (1,4,16,64):
            ids=tuple(range(n));variants=(ids,) if n==1 else (ids,(0,)*n)
            r=HistoryRestriction([(ids,variants)])
            for trial in range(10):
                roots=[]
                for i in range(n):roots.append(i if rng.randrange(2) else (roots[rng.randrange(i)] if i else 0))
                k=Kernel(roots)
                for depth in range(n+1):
                    for root in range(depth+1):
                        self.assertEqual(r.kernel_choice(k,depth,root),projected_kernel_choice(r,k,depth,root))
    def test_reuses_one_projection_and_preserves_rejection(self):
        n=100;ids=tuple(range(n));r=HistoryRestriction([(ids,(ids,))]);k=Kernel(list(ids))
        for root in range(n):self.assertEqual(projected_kernel_choice(r,k,99,root),root==99)
        self.assertEqual(k.ancestor_calls,99)
        r.clauses=(((0,),((0,),)),)
        self.assertTrue(projected_kernel_choice(r,k,99,0))
    def test_disjoint_clauses_and_unrestricted_depth(self):
        r=HistoryRestriction([((0,2),((0,2),(0,0))),((1,3),((1,3),(1,1)))])
        k=Kernel([0,1,0,1,4])
        for depth in range(6):
            for root in range(depth+1):self.assertEqual(r.kernel_choice(k,depth,root),projected_kernel_choice(r,k,depth,root))
class IncrementalTests(unittest.TestCase):
    def test_chain_then_backward_branch_and_owner(self):
        ids=tuple(range(200));r=HistoryRestriction([(ids,(ids,(0,)*200))]);k=Kernel(list(ids))
        for depth in range(201):
            for root in (0,depth):
                self.assertEqual(r.kernel_choice(k,depth,root),projected_kernel_choice(r,k,depth,root))
        k.ancestor_calls=0
        r=HistoryRestriction([(ids,(ids,(0,)*200))])
        for depth in range(201):projected_kernel_choice(r,k,depth,depth)
        self.assertEqual(k.ancestor_calls,0)
        for depth in (80,20,21,22,190,0,1):
            for root in (0,depth):
                self.assertEqual(r.kernel_choice(k,depth,root),projected_kernel_choice(r,k,depth,root))
    def test_sibling_handles_same_depth(self):
        class Tree:
            def __init__(self):
                self.nodes={0:(None,0,-1),1:(0,1,0),2:(1,2,1),3:(1,2,0),4:(2,3,2),5:(3,3,0)}
            def _prefix(self,h):
                p,d,r=self.nodes[h];return types.SimpleNamespace(parent_handle=p,depth=d,root=r)
            def ancestor(self,h,d):
                while self.nodes[h][1]>d:h=self.nodes[h][0]
                return h
        k=Tree();r=HistoryRestriction([((0,1,2,3),((0,1,2,3),(0,0,0,0)))])
        for h in (0,1,2,4,3,5,2,3,4,5):
            for root in range(4):self.assertEqual(r.kernel_choice(k,h,root),projected_kernel_choice(r,k,h,root))
if __name__=='__main__':unittest.main()
