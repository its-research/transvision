"""Candidate v2: packed setup buffers and binary support lookup; integer-only changes."""
import ctypes as C
from array import array
from bisect import bisect_left
import time
from prepared_threaded_factors_candidate import PreparedThreadedFactors as Previous, METRICS, timing

class PreparedThreadedFactors(Previous):
    def __init__(self,original,library,threads=8):
        w,c=time.perf_counter(),time.process_time()
        if type(threads) is not int or not 1<=threads<=64:raise ValueError('threads must be explicitly bounded to 1..64')
        self.original,self.nodes,self.threads=original,original.nodes,threads
        self.lib=C.CDLL(str(library));self.fn=self.lib.restricted_children
        i64=C.POINTER(C.c_int64);u8=C.POINTER(C.c_uint8)
        self.fn.argtypes=[i64,C.c_int64,i64,C.c_int64,u8,i64,C.c_int64,i64,i64,i64,i64,C.c_int,u8];self.fn.restype=C.c_int
        offsets,indices,voffsets,variants,counts=[array('q') for _ in range(5)];offsets.append(0)
        for ids,choices in original.restriction.clauses:
            indices.extend(ids);offsets.append(len(indices));voffsets.append(len(variants));counts.append(len(choices))
            for choice in choices:variants.extend(choice)
        owners=(offsets,indices,voffsets,variants,counts)
        if any(a.itemsize!=8 for a in owners):raise RuntimeError('64-bit integer buffer required')
        self.arrays=tuple((C.c_int64*len(a)).from_buffer(a) for a in owners)
        self.clauses=len(counts);self.cached_prefix=None;self.cached_roots=();self.cached_parents=frozenset();self.max_native_workers=0
        self.last_prefix=None;self.last_roots=None;self.slots={}
        for i,node in enumerate(self.nodes):
            if node.source_id!=-1:self.slots.setdefault((node.source_id,node.frame_id),[]).append(i)
        METRICS['wrappers']+=1;timing('setup',w,c)

    def roots(self,prefix):
        if not isinstance(prefix,tuple) or len(prefix)>len(self.nodes):raise ValueError('invalid prefix length/type')
        if any(type(p) is not int for p in prefix):raise ValueError('parent outside original support')
        if prefix==self.last_prefix:return self.last_roots
        if (self.cached_prefix is not None and len(prefix)==len(self.cached_prefix)+1
                and prefix[:-1]==self.cached_prefix and prefix[-1] in self.cached_parents):
            parent=prefix[-1];result=self.cached_roots+(len(self.cached_roots) if parent==-1 else self.cached_roots[parent],)
        else:
            roots=[];slots={}
            for i,parent in enumerate(prefix):
                row=self.original.factors.rows[i];position=bisect_left(row,(parent,))
                if position==len(row) or row[position][0]!=parent:raise ValueError('parent outside original support')
                root=i if parent==-1 else roots[parent];node=self.nodes[i]
                if node.source_id!=-1:
                    slot=(node.source_id,node.frame_id);seen=slots.setdefault(root,set())
                    if slot in seen:raise ValueError('identity collision: multiple observations from one source/frame')
                    seen.add(slot)
                roots.append(root)
            result=tuple(roots)
            if not self.original.restriction.allows(result):raise ValueError('action would recover an irreversibly pruned history')
        self.last_prefix,self.last_roots=prefix,result
        return result
