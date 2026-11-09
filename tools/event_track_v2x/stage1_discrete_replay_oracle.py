"""Independent sealed-association state replay and rectangular assignment.

No production tracker, covariance helper, SciPy assignment, GT or previous
recorded output supplies state. Sealed association hypotheses are the only
discrete input. Real-input acceptance is separate from these equations.
"""
import math
import numpy as np
import stage1_fusion_numeric_oracle as numeric

GATE = 11.34486673014436
CERTIFICATE_ATOL = 1e-8


def assignment(cost):
    """Shortest augmenting paths with a checked primal/dual certificate."""
    c=np.asarray(cost,dtype=float)
    if c.ndim!=2 or c.shape[0]>c.shape[1] or np.isnan(c).any() or np.isneginf(c).any():
        raise ValueError('finite-or-positive-infinite rectangular costs required')
    n,m=c.shape;u=np.zeros(n+1);v=np.zeros(m+1);owner=np.zeros(m+1,dtype=int);way=np.zeros(m+1,dtype=int)
    for i in range(1,n+1):
        owner[0]=i;j0=0;minimum=np.full(m+1,np.inf);used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True;i0=owner[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                reduced=c[i0-1,j-1]-u[i0]-v[j]
                if reduced<minimum[j]:minimum[j]=reduced;way[j]=j0
                if minimum[j]<delta:delta=minimum[j];j1=j
            if not math.isfinite(delta):raise ValueError('no finite full row assignment')
            for j in range(m+1):
                if used[j]:u[owner[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if owner[j0]==0:break
        while True:
            j1=way[j0];owner[j0]=owner[j1];j0=j1
            if j0==0:break
    columns=np.full(n,-1,dtype=int)
    for j in range(1,m+1):
        if owner[j]:columns[owner[j]-1]=j-1
    if np.any(columns<0) or len(set(columns.tolist()))!=n:raise ValueError('incomplete one-to-one assignment')
    chosen=c[np.arange(n),columns];primal=math.fsum(float(x) for x in chosen)
    dual=math.fsum(float(x) for x in u[1:])+math.fsum(float(x) for x in v[1:])
    reduced=c-u[1:,None]-v[None,1:];finite=np.isfinite(c)
    slack=float(np.min(reduced[finite])) if finite.any() else 0.
    tight=float(np.max(np.abs(reduced[np.arange(n),columns]))) if n else 0.
    if abs(primal-dual)>CERTIFICATE_ATOL or slack < -CERTIFICATE_ATOL or tight>CERTIFICATE_ATOL or np.any(v[1:]>CERTIFICATE_ATOL):
        raise ValueError('full assignment primal/dual certificate fails')
    return columns,dict(primal=primal,dual=dual,duality_gap=abs(primal-dual),minimum_reduced_cost=slack,maximum_selected_slack=tight)


def hypotheses(values,left,right,mode):
    if not values or abs(math.fsum(x['weight'] for x in values)-1)>1e-12:
        raise ValueError('complete normalized sealed alternatives required')
    for h in values:
        pairs=h['pairs'];ls=[x[0] for x in pairs];rs=[x[1] for x in pairs]
        if any(type(x) is not int or x<0 or x>=left for x in ls) or any(type(x) is not int or x<0 or x>=right for x in rs):raise ValueError('sealed pair outside selected source')
        if len(set(ls))!=len(ls) or len(set(rs))!=len(rs):raise ValueError('sealed alternative not one-to-one')
        if sorted(h['unmatched_left'])!=sorted(set(range(left))-set(ls)) or sorted(h['unmatched_right'])!=sorted(set(range(right))-set(rs)):raise ValueError('sealed unmatched complement differs')
        if not math.isfinite(h['weight']) or h['weight']<0:raise ValueError('bad original posterior mass')
    if mode=='M1' and (len(values)!=1 or values[0]['pairs'] or values[0]['weight']!=1):raise ValueError('M1 all-unmatched control differs')


def nodes(vehicle,road,alternatives,mode):
    """Sources: list of (mean9, covariance9x9, score, class index)."""
    if mode not in ['M0','M1','M2','M3','M4']:raise ValueError('unknown original mechanism')
    hypotheses(alternatives,len(vehicle),len(road),mode);weights=[x['weight'] for x in alternatives];out=[]
    for i,(vm,vp,vs,vc) in enumerate(vehicle):
        states=[];scores=[]
        for h in alternatives:
            partner=dict(h['pairs']).get(i)
            if partner is None:state=(vm,vp);score=vs
            else:
                rm,rp,rs,rc=road[partner]
                if vc!=rc:raise ValueError('sealed cross-class pair')
                state=(rm,rp) if mode=='M2' else numeric.ci(vm,vp,rm,rp);score=max(vs,rs)
            states.append(state);scores.append(score)
        m,p=numeric.mixture([x[0] for x in states],[x[1] for x in states],weights)
        score=math.fsum(float(w)*float(s) for w,s in zip(weights,scores))
        out.append(dict(mean=m,covariance=p,score=score,class_index=vc,birth_score=score))
    for j,(rm,rp,rs,rc) in enumerate(road):
        mass=math.fsum(h['weight'] for h in alternatives if j in h['unmatched_right'])
        if mass>0:
            score=rs if mode=='M3' else mass*rs
            out.append(dict(mean=rm.copy(),covariance=rp.copy(),score=score,class_index=rc,birth_score=rs if mode=='M4' else score))
    return out


def transport(mean,covariance,delta):
    m,p=numeric.gaussian(mean,covariance);f=np.eye(9);f[0,7]=f[1,8]=delta
    m[0]+=m[7]*delta;m[1]+=m[8]*delta
    return m,f@p@f.T+np.eye(9)*.1*abs(delta)


class Replay:
    def __init__(self,sequence):
        self.sequence=sequence;self.tracks={};self.next_id=1;self.last_target=None

    def step(self,target,values):
        if type(target) is not int or (self.last_target is not None and target<=self.last_target):raise ValueError('native increasing reference timestamps required')
        delta=0. if self.last_target is None else (target-self.last_target)/1e6;events=[];eligible=[]
        for tid,t in sorted(self.tracks.items()):
            prior=dict(t);prior['mean'],prior['covariance']=transport(t['mean'],t['covariance'],delta)
            prior['score']=t['score']*.95**delta;age=(target-t['last_update_us'])/1e6
            if age>2 or prior['score']<.05:events.append(dict(event='kill_before_assignment',track_id=tid))
            else:eligible.append(prior)
        n,m=len(eligible),len(values);cost=np.full((n,m+n),np.inf)
        for i,t in enumerate(eligible):
            js=[j for j,x in enumerate(values) if x['class_index']==t['class_index']]
            if js:
                innovation=np.asarray([values[j]['mean'][:3]-t['mean'][:3] for j in js])
                covariance=2*(np.asarray([values[j]['covariance'][:3,:3] for j in js])+t['covariance'][:3,:3])
                solved=np.linalg.solve(covariance,innovation[...,None])[...,0]
                d2=np.sum(innovation*solved,axis=1)
                for j,distance in zip(js,d2):
                    if distance<=GATE:cost[i,j]=.5*float(distance)-math.log(max(values[j]['score'],1e-12))
            cost[i,m+i]=12.
        chosen,certificate=assignment(cost);current={};matched=set();matches=[]
        for i,j in enumerate(chosen):
            t=eligible[i];tid=t['track_id']
            if j<m:
                node=values[j];t['mean'],t['covariance']=numeric.ci(t['mean'],t['covariance'],node['mean'],node['covariance'])
                t['score']=max(t['score'],node['score']);t['last_update_us']=target
                possibilities=sorted([(eligible[k]['track_id'],float(cost[k,j])) for k in range(n) if math.isfinite(cost[k,j])],key=lambda x:(x[1],x[0]))
                masses=[math.exp(-c) for _,c in possibilities]+[math.exp(-12.)];total=math.fsum(masses);masses=[x/total for x in masses]
                keep=min(3,len(possibilities));t['identity_hypotheses']=[dict(track_id=possibilities[k][0],probability=masses[k]) for k in range(keep)];t['other_identity_probability']=math.fsum(masses[keep:])
                matched.add(int(j));matches.append(dict(track_id=tid,node_index=int(j)))
            else:t['score']*=.75;events.append(dict(event='miss',track_id=tid))
            if t['score']>=.05:current[tid]=t
            else:events.append(dict(event='kill_after_miss',track_id=tid))
        for j,node in enumerate(values):
            if j in matched:continue
            if node['birth_score']<.3:events.append(dict(event='birth_rejected_score',node_index=j));continue
            tid=f'{self.sequence}:{self.next_id:06d}';self.next_id+=1
            current[tid]=dict(track_id=tid,class_index=node['class_index'],mean=node['mean'].copy(),covariance=node['covariance'].copy(),score=node['birth_score'],last_update_us=target,identity_hypotheses=[dict(track_id=tid,probability=1.)],other_identity_probability=0.)
            events.append(dict(event='birth',track_id=tid,node_index=j))
        self.tracks=current;self.last_target=target
        return current,matches,events,certificate
