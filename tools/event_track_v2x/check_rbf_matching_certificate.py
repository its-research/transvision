"""Check a matching certificate without running the production flow solver."""
from fractions import Fraction

def certify(matrix, pairs, gt_count):
    n=len(matrix);m=gt_count;s=n+m;t=s+1;edges=[]
    pi={i for i,j,d in pairs};gj={j for i,j,d in pairs}
    assert len(pi)==len(gj)==len(pairs)
    chosen={(i,j) for i,j,d in pairs}
    for i in range(n):edges.append((i,s,0) if i in pi else (s,i,0))
    for j in range(m):edges.append((t,n+j,0) if j in gj else (n+j,t,0))
    for i,row in enumerate(matrix):
        assert len(row)==m
        for j,d in enumerate(row):
            if d<2:
                edges.append((n+j,i,-Fraction(d)) if (i,j) in chosen else (i,n+j,Fraction(d)))
    for i,j,d in pairs:assert matrix[i][j]==d and 0<=d<2
    reachable={s}
    while True:
        new=reachable|{b for a,b,c in edges if a in reachable}
        if new==reachable:break
        reachable=new
    assert t not in reachable,'augmenting path: not maximum cardinality'
    # An artificial zero-cost source reaches every residual vertex. A
    # relaxation on iteration |V| witnesses a negative cycle and a cheaper
    # matching with the same cardinality. Exact rationals avoid roundoff.
    distances=[Fraction(0)]*(t+1)
    for iteration in range(t+1):
        changed=False
        for a,b,c in edges:
            if distances[b]>distances[a]+c:
                distances[b]=distances[a]+c;changed=True
        if not changed:return
    raise AssertionError('negative residual cycle: not minimum distance')
