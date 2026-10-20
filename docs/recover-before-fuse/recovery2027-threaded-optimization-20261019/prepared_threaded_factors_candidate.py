"""Isolated 8-thread integration candidate; original factor semantics retained."""
import ctypes as C
from array import array
from bisect import bisect_left
import time
from threaded_factors import ThreadedRestrictedFactors

METRICS = {'wrappers':0,'calls':0,'max_native_workers':0}
def timing(name, wall, cpu):
    METRICS[name+'_wall_seconds']=METRICS.get(name+'_wall_seconds',0.)+time.perf_counter()-wall
    METRICS[name+'_CPU_seconds']=METRICS.get(name+'_CPU_seconds',0.)+time.process_time()-cpu

def ints(values):
    owner=array('q',values)
    if owner.itemsize != C.sizeof(C.c_int64):
        raise RuntimeError('64-bit integer buffer required')
    return (C.c_int64*len(owner)).from_buffer(owner)

class PreparedThreadedFactors(ThreadedRestrictedFactors):
    def __init__(self, original, library, threads=8):
        w,c=time.perf_counter(),time.process_time()
        super().__init__(original,library,threads)
        self.slots={}
        for i,node in enumerate(self.nodes):
            if node.source_id!=-1:self.slots.setdefault((node.source_id,node.frame_id),[]).append(i)
        self.last_prefix=None;self.last_roots=None
        METRICS['wrappers']+=1;timing('setup',w,c)

    def roots(self,prefix):
        if (isinstance(prefix,tuple) and prefix==self.last_prefix
                and all(type(p) is int for p in prefix)):return self.last_roots
        result=super().roots(prefix)
        self.last_prefix,self.last_roots=prefix,result
        return result

    def children(self,prefix):
        w,c=time.perf_counter(),time.process_time();roots=self.roots(prefix);timing('roots',w,c)
        depth=len(prefix)
        if depth==len(self.nodes):return ()
        w,c=time.perf_counter(),time.process_time()
        parents=tuple(p for p,_ in self.original.factors.rows[depth])
        node=self.nodes[depth];blocked=(C.c_uint8*(depth+1))()
        if node.source_id!=-1:
            positions=self.slots.get((node.source_id,node.frame_id),())
            for j in range(bisect_left(positions,depth)):blocked[roots[positions[j]]]=1
        roots_buffer,parents_buffer=ints(roots),ints(parents)
        output=(C.c_uint8*len(parents))()
        offsets,indices,voffsets,variants,counts=self.arrays
        timing('prepare',w,c)
        w,c=time.perf_counter(),time.process_time()
        workers=self.fn(roots_buffer,depth,parents_buffer,len(parents),blocked,
                        offsets,self.clauses,indices,voffsets,variants,counts,self.threads,output)
        timing('native',w,c)
        if workers<0:raise RuntimeError('native candidate check failed: '+str(workers))
        self.max_native_workers=max(self.max_native_workers,workers)
        METRICS['max_native_workers']=max(METRICS['max_native_workers'],workers);METRICS['calls']+=1
        w,c=time.perf_counter(),time.process_time()
        admitted=tuple(p for p,ok in zip(parents,output) if ok)
        self.cached_prefix,self.cached_roots=prefix,roots;self.cached_parents=frozenset(admitted)
        result=tuple(prefix+(p,) for p in admitted);timing('materialize',w,c)
        return result
