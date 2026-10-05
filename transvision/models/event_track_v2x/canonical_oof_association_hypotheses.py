"""Gate-aware retained joint assignments for canonical predicted OOF logits.

Weights normalize the retained energy alternatives, not calibrated full posterior
mass. Keep explicit unmatched and preserve logits for later exact replay.
"""
import heapq
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def k_best_assignments(cost, count):
    cost=np.asarray(cost,float)
    if cost.ndim!=2 or cost.shape[0]>cost.shape[1] or np.isnan(cost).any() or count not in (1,3,5):raise ValueError('invalid assignment cost/count')
    n=cost.shape[0]
    if not n:return [(0.,())]
    serial=itertools.count();heap=[];result=[];seen=set()
    def push(fixed,forbidden):
        c=cost.copy()
        for i,j in forbidden:c[i,j]=np.inf
        for i,j in fixed:
            value=c[i,j];c[i,:]=np.inf;c[:,j]=np.inf;c[i,j]=value
        try:rows,cols=linear_sum_assignment(c)
        except ValueError:return
        if len(rows)==n and np.isfinite(c[rows,cols]).all():heapq.heappush(heap,(float(cost[rows,cols].sum()),tuple(cols.tolist()),next(serial),fixed,forbidden))
    push((),frozenset())
    while heap and len(result)<count:
        energy,columns,_,fixed,forbidden=heapq.heappop(heap)
        if columns not in seen:result.append((energy,columns));seen.add(columns)
        fixed_rows={i for i,_ in fixed};prefix=list(fixed)
        for i in range(n):
            if i not in fixed_rows:push(tuple(prefix),forbidden|{(i,columns[i])});prefix.append((i,columns[i]))
    return result


def retained_hypotheses(pair,left_dustbin,right_dustbin,gate,*,top_h=5):
    pair=np.asarray(pair,float);ld=np.asarray(left_dustbin,float);rd=np.asarray(right_dustbin,float);gate=np.asarray(gate)
    if pair.ndim!=2:raise ValueError('pair matrix required')
    n,m=pair.shape
    if ld.shape!=(n,) or rd.shape!=(m,) or gate.shape!=(n,m) or gate.dtype.kind!='b' or top_h not in (1,3,5) or not all(np.isfinite(x).all() for x in [pair,ld,rd]):raise ValueError('invalid logits, gate or Top-H')
    if not n or not m:return [dict(pairs=[],unmatched_left=list(range(n)),unmatched_right=list(range(m)),energy=0.,weight=1.)]
    pair=np.where(gate,pair,-np.inf);row=np.c_[pair,ld];row-=logsumexp(row,axis=1)[:,None];col=np.c_[pair.T,rd];col-=logsumexp(col,axis=1)[:,None];cost=np.full((n,m+n),np.inf);cost[:,:m]=-row[:,:m]-col[:,:n].T+col[:,-1][None,:];cost[np.arange(n),m+np.arange(n)]=-row[:,-1]
    choices=k_best_assignments(cost,top_h);null=tuple(m+i for i in range(n))
    if null not in {r[1] for r in choices}:choices.append((float(cost[np.arange(n),null].sum()),null))
    energies=np.array([r[0] for r in choices])-col[:,-1].sum();weights=np.exp(-energies-logsumexp(-energies));output=[]
    for (_,columns),energy,weight in zip(choices,energies,weights):
        pairs=[(i,j) for i,j in enumerate(columns) if j<m];assert all(gate[i,j] for i,j in pairs)
        output.append(dict(pairs=pairs,unmatched_left=[i for i,j in enumerate(columns) if j>=m],unmatched_right=sorted(set(range(m))-{j for _,j in pairs}),energy=float(energy),weight=float(weight)))
    return output
