"""Independent adjacency-component and scalar matching/partial-label oracle.

Does not import producer identity index, matching, row construction or training.
"""
from collections import Counter,defaultdict
import math


def identities(rows):
    graph=defaultdict(set);frames=defaultdict(list);associated=defaultdict(set);bad=set();pairs=defaultdict(dict);linked=set()
    for r in rows:
        src=int(r['cav_id']);scene=r['sequence_id'];frame=r['frame_key'];aids=r['association_ids']
        tracks=[(scene,src,oid) for oid in r['object_ids']]
        pairs[scene,frame][src]=(tracks,aids,r['native_mapped_ids'])
        aliases=Counter(r['native_mapped_ids'])
        if any(int(mid)!=(int(aid) if int(aid)>=0 else int(oid)+100*src) for oid,aid,mid in zip(r['object_ids'],aids,r['native_mapped_ids'])):
            raise ValueError('independent author-native identity key differs')
        for track,aid,mid in zip(tracks,aids,r['native_mapped_ids']):
            graph[track];frames[track].append((src,frame))
            if int(aid)>=0:associated[track].add(aid)
            if aliases[mid]>1:bad.add(track)
    for pair in pairs.values():
        if set(pair)!={0,1}:raise ValueError('paired native source target support required')
        # Independent directed lookup: each explicit association points to
        # the opposite source's author-native key. Graph closure joins tracks.
        for src in (0,1):
            tracks,aids,mids=pair[src];other,oa,om=pair[1-src]
            counts=Counter(mids);oc=Counter(om)
            for t,aid,mid in zip(tracks,aids,mids):
                if int(aid)<0 or counts[mid]!=1 or oc[mid]!=1:continue
                u=other[om.index(mid)]
                graph[t].add(u);graph[u].add(t);linked.update((t,u))
    bad.update(t for t,values in associated.items() if len(values)>1)
    result={};remaining=set(graph)
    while remaining:
        start=min(remaining);stack=[start];component=set()
        while stack:
            t=stack.pop()
            if t in component:continue
            component.add(t);stack.extend(graph[t]-component)
        remaining.difference_update(component)
        slots=Counter(slot for t in component for slot in frames[t])
        ambiguous=bool(component&bad) or any(n>1 for n in slots.values());cross=bool(component&linked)
        for t in component:result[t]=(t[1],None if ambiguous else start,cross,'ambiguous' if ambiguous else 'matched')
    return result


def match_native(boxes,scores,gt):
    taken=set();out=[-1]*len(boxes)
    for i in sorted(range(len(boxes)),key=lambda n:(-float(scores[n]),n)):
        distances=[(math.hypot(float(boxes[i][0])-float(g[0]),float(boxes[i][1])-float(g[1])),j)
            for j,g in enumerate(gt) if j not in taken]
        if not distances:continue
        d,j=min(distances)
        if d<2:out[i]=j;taken.add(j)
    return out


def relation(query,previous):
    if query[3]!='matched':return None
    if previous[3]=='unmatched_prediction':return False
    if previous[3]!='matched':return None
    if query[1]==previous[1]:return True
    if query[0]==previous[0] or query[2] and previous[2]:return False
    return None


def supervision(query,parents,prior):
    n=len(parents)+1;empty=[False]*n
    if query[3]!='matched':return empty,empty,query[3]
    values=[relation(query,p) for p in parents]
    if any(v is True for v in values):return [False]+[v is True for v in values],[True]+[v is not None for v in values],'parent'
    history=[relation(query,p) for p in prior]
    if any(v is True for v in history):return empty,empty,'out_of_support'
    if any(v is None for v in history+values):return empty,empty,'uncertain_birth'
    return [True]+[False]*len(parents),[True]*n,'birth'
