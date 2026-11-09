#!/usr/bin/env python3
"""Independent exhaustive scan, history and full9D readback; no model imports."""
import argparse
import decimal
import hashlib
import itertools
import json
import math
import sqlite3
from pathlib import Path
import numpy as np


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
    import importlib.util
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def close(a,b):
    if not np.asarray(a).size or not np.asarray(b).size:
        assert not np.asarray(a).size and not np.asarray(b).size;return 0.
    assert np.allclose(a,b,atol=1e-8,rtol=1e-8),(a,b)
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.asarray(a).size else 0.
def chain(db,h):
    out=[];seen=set()
    while h:
        assert h not in seen;seen.add(h)
        parent,depth,choice=db.execute('SELECT parent,depth,choice FROM prefixes WHERE h=?',(h,)).fetchone()
        out.append(choice);h=parent
    return tuple(reversed(out))
def root_indices(parents):
    out=[]
    for i,p in enumerate(parents):out.append(i if p<0 else out[p])
    return tuple(out)
def logsum(values):
    largest=max(values);return largest+math.log(math.fsum(math.exp(x-largest) for x in values))
def predict(mean,cov,dt,q):
    F=np.eye(9);F[0,7]=F[1,8]=dt
    return F@mean,F@cov@F.T+np.eye(9)*q*abs(dt)
def scan_reference(raw,rows,indices,anchors):
    slot=tuple(raw[indices[0]]['node'][k] for k in ('source_id','frame_id'))
    occupied={r for i,r in anchors.items() if tuple(raw[i]['node'][k] for k in ('source_id','frame_id'))==slot}
    groups=[];births=[]
    for index in indices:
        g={};birth=[]
        for p,w in rows[index]:
            if p<0:birth.append(w)
            elif tuple(raw[p]['node'][k] for k in ('source_id','frame_id'))!=slot:
                r=anchors[p]
                if r not in occupied:g.setdefault(r,[]).append((p,w))
        assert len(birth)==1;births.append(birth[0]);groups.append(g)
    roots=sorted({r for g in groups for r in g});pair=np.zeros((len(roots),len(indices)));allowed=np.zeros_like(pair,dtype=bool)
    options=[]
    for j,g in enumerate(groups):
        options.append([(None,births[j])]+[(r,logsum([w for p,w in g[r]])) for r in sorted(g)])
        for i,r in enumerate(roots):
            if r in g:pair[i,j]=logsum([w for p,w in g[r]]);allowed[i,j]=True
    actions=[];weights=[]
    for choices in itertools.product(*options):
        a=tuple(x[0] for x in choices);assigned=[x for x in a if x is not None]
        if len(set(assigned))==len(assigned):actions.append(a);weights.append(math.fsum(x[1] for x in choices))
    Z=logsum(weights);probs=[math.exp(w-Z) for w in weights];marginal=np.zeros_like(pair);left=np.zeros(len(roots));right=np.zeros(len(indices))
    for a,p in zip(actions,probs):
        for i,r in enumerate(roots):
            if r not in a:left[i]+=p
        for j,r in enumerate(a):
            if r is None:right[j]+=p
            else:marginal[roots.index(r),j]+=p
    maximum=max(weights);best={a for a,w in zip(actions,weights) if math.isclose(w,maximum,abs_tol=1e-11,rel_tol=0)}
    return roots,groups,births,pair,allowed,marginal,left,right,Z,best
def update(prior,measurements,weights,miss,method,omega):
    m,P=prior
    if method=='pkf':
        # Expanded weighted observation model and Joseph covariance, independent
        # of the producer's information-form precision accumulation.
        H=np.vstack([math.sqrt(w)*np.eye(9) for w in weights]);R=np.zeros((9*len(weights),9*len(weights)))
        residual=np.concatenate([math.sqrt(w)*(z-m) for (z,C),w in zip(measurements,weights)])
        for j,(z,C) in enumerate(measurements):R[9*j:9*(j+1),9*j:9*(j+1)]=C
        S=H@P@H.T+R;K=np.linalg.solve(S,H@P).T;A=np.eye(9)-K@H
        return m+K@residual,(A@P@A.T+K@R@K.T)
    components=[(m,P)]
    for z,C in measurements:
        invP=np.linalg.inv(P);invC=np.linalg.inv(C);cov=np.linalg.inv(omega*invP+(1-omega)*invC)
        components.append((cov@(omega*invP@m+(1-omega)*invC@z),cov))
    w=np.asarray([miss,*weights]);assert abs(math.fsum(w)-1)<1e-8;w=w/math.fsum(w)
    mean=sum((p*z for p,(z,C) in zip(w,components)),np.zeros(9))
    cov=sum((p*(C+np.outer(z-mean,z-mean)) for p,(z,C) in zip(w,components)),np.zeros((9,9)))
    return mean,cov
def compare_tracks(tracks,identities,raw,reference,sequence,state,state_lookup,metadata=True):
    remaining={t['track_id']:t for t in tracks};maximum=0.
    for root in sorted(set(identities)):
        group=[r for r,a in zip(raw,identities) if a==root];last=max(r['state_us'] for r in group);first=min(r['state_us'] for r in group)
        score=max(r['score']*state['survival_per_second']**(max(0,reference-r['state_us'])/1e6) for r in group)
        if max(r['score'] for r in group)<state['birth_score'] or max(0,reference-last)>state['max_age_us'] or score<state['prune_score']:continue
        identity=sequence+':'+digest(['forest-birth',raw[root]['node']['node_id']])[:24];t=remaining.pop(identity)
        assert t['class_label']=='car'
        if metadata:assert t['birth_state_us']==first and t['last_update_us']==last and t['observation_ids']==[r['node']['node_id'] for r in group]
        m,P=state_lookup(root,group);maximum=max(maximum,close(m,t['mean']),close(P,t['covariance']),close(score,t['score']))
        assert np.linalg.eigvalsh(P).min()>0
    assert not remaining;return maximum


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_bytes());assert sha(__file__)==plan['auditor_sha256']
    for k in ('independent_SQL_state_helper','independent_beam_helper','independent_oracle'):assert sha(plan[k])==plan[k+'_sha256']
    pure=load(plan['independent_SQL_state_helper'],'pure_9D');beamhelper=load(plan['independent_beam_helper'],'pure_history_math');oracle=load(plan['independent_oracle'],'pure_full_oracle').oracle
    campaign=json.loads(args.campaign.read_bytes());assert campaign['source_plan_sha256']==sha(args.plan) and len(campaign['runs'])==24
    decimal.getcontext().prec=80;D=decimal.Decimal;checks=[]
    for run in campaign['runs']:
        assert sha(run['path'])==run['sha256'];data=json.loads(Path(run['path']).read_bytes());inp=next(v for v in plan['inputs'] if v['id']==data['input_id']);assert sha(inp['path'])==inp['sha256']==data['input_sha256']
        payload=json.loads(Path(inp['path']).read_bytes());raw=payload['raw'];rows=payload['rows'];names=[r['node']['node_id'] for r in raw];state=data['configuration']['state'];method=data['method'];width=data['width']
        assert data['configuration']==plan['configurations'][method]['none' if width is None else str(width)] and all(r['mean'][6]==0 for r in raw)
        truthref=plan['evaluator_only_original_candidates'][data['input_id']];assert sha(truthref['path'])==truthref['sha256'];external=json.loads(Path(truthref['path']).read_bytes());assert external['raw']==raw and external['rows']==rows;truth=external['evaluator_only_true_roots']
        nodes=[dict(node_id=r['node']['node_id'],source_id=r['node']['source_id'],frame_id=r['node']['frame_id'],information_us=r['node']['information_us'],arrival_us=r['node']['arrival_us'],choices=[dict(parent_id=None if p<0 else names[p],log_potential=w) for p,w in row]) for r,row in zip(raw,rows)]
        dbpath=Path(data['database_path']);assert sha(dbpath)==data['database_sha256'];anchors={};states={};beam={()};old_n=0;previous='0'*64;previous_audit='0'*64;events=[];initial=None;previous_roots=None;expiry_record=None;expiry_count=0
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
            assert list(db.execute('SELECT i,p,w FROM potentials ORDER BY i,p'))==sorted((i,p,w) for i,row in enumerate(rows) for p,w in row)
            observations=list(db.execute('SELECT raw,sha FROM observations ORDER BY i'));assert len(observations)==len(raw)
            for r,(b,h) in zip(raw,observations):assert canonical(r)==b and hashlib.sha256(b).hexdigest()==h
            stored=list(db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal'));assert len(stored)==len(data['events'])==len(payload['schedule'])
            for log,schedule,(event_id,pb,ab) in zip(data['events'],payload['schedule'],stored):
                pred,audit=json.loads(pb),json.loads(ab);assert pred==log['prediction'] and audit==log['audit'] and event_id==log['event_id']==schedule['event_id'];assert hashlib.sha256(pb).hexdigest()==log['prediction_sha256'] and hashlib.sha256(ab).hexdigest()==log['audit_sha256']
                body=dict(pred);commit=body.pop('commit_sha256');assert digest(body)==commit and body['previous_commit_sha256']==previous;previous=commit
                assert audit['previous_audit_sha256']==previous_audit;previous_audit=digest(audit)
                n=schedule['available_observations'];reference=schedule['reference_us'];assert n==audit['observation_count'] and audit['appended_rows']==rows[old_n:n]
                fh=hashlib.sha256()
                for i,r in enumerate(raw[:n]):fh.update(canonical([i,digest(r),sorted(rows[i])])+b'\n')
                assert fh.hexdigest()==audit['factor_rows_sha256'] and digest(data['configuration'])==audit['configuration_sha256'] and audit['prediction_sha256']==commit
                assert pred['sequence_id']==payload['sequence_id'] and pred['frame_id']==schedule['frame_id'] and pred['box_reference_timestamp_us']==reference and pred['decision_timestamp_us']==schedule['decision_us']
                assert all(r['node']['arrival_us']<=schedule['decision_us'] for r in raw[:n]);maximum=0.;scan_checks=0;conditional_ties=0
                if method!='scan_mht':
                    scans=audit['conditional_scans'];assert sum(len(s['indices']) for s in scans)==n-old_n
                    for scan in scans:
                        indices=scan['indices'];roots,groups,births,pair,allowed,beta,miss,right,Z,best=scan_reference(raw,rows,indices,anchors)
                        assert roots==scan['conditioned_track_roots'] and allowed.tolist()==scan['allowed'];close(pair,scan['log_pair']);close(births,scan['log_birth'])
                        marg=scan['marginals'];close(beta,marg['pair']);close(miss,marg['left_unmatched']);close(right,marg['right_unmatched']);close(Z,marg['log_partition']);assert marg['algorithm']=='jpda-exact-frontier-v1'
                        actual=tuple(None if a['root']==a['index'] else a['root'] for a in scan['anchors']);assert actual in best;conditional_ties+=int(len(best)>1)
                        projected=[predict(np.asarray(raw[i]['mean']),np.asarray(raw[i]['covariance']),(reference-raw[i]['state_us'])/1e6,state['process_noise']) for i in indices]
                        for j,r in enumerate(roots):
                            t,m,P=states[r];assert reference>=t;prior=predict(m,P,(reference-t)/1e6,state['process_noise']);columns=np.flatnonzero(allowed[j]);m,P=update(prior,[projected[c] for c in columns],beta[j,columns],miss[j],method,state['ci_weight']);states[r]=(reference,m,(P+P.T)/2)
                        for j,a in enumerate(scan['anchors']):
                            index=indices[j];root=actual[j] if actual[j] is not None else index;assert a['index']==index and a['root']==root
                            assert a['representative_parent']==(-1 if actual[j] is None else min(p for p,w in groups[j][root]));anchors[index]=root
                            if actual[j] is None:states[root]=(reference,*projected[j])
                        scan_checks+=1
                    chosen=tuple(anchors[i] for i in range(n));assert root_indices(chain(db,audit['output_handle']))==chosen
                    snapshot=log['collapsed_gaussian_state_readbacks'];assert len(snapshot)==len(states)
                    gh=hashlib.sha256()
                    for x in snapshot:
                        t,m,P=states[x['root']];assert t==x['reference_us'];maximum=max(maximum,close(m,x['payload']['mean']),close(P,x['payload']['covariance']));gh.update(canonical([x['root'],x['reference_us'],x['payload']])+b'\n')
                    assert gh.hexdigest()==audit['gaussian_state_sha256'];assert log['hard_identity_anchor_readbacks']==[[i,anchors[i]] for i in range(n)]
                    ah=hashlib.sha256()
                    for pair_anchor in log['hard_identity_anchor_readbacks']:ah.update(canonical(pair_anchor)+b'\n')
                    assert ah.hexdigest()==audit['identity_anchor_sha256'] and sum(audit['state_work_breakdown'].values())==audit['state_updates']
                    def lookup(root,group):
                        t,m,P=states[root];return predict(m,P,(reference-t)/1e6,state['process_noise'])
                    maximum=max(maximum,compare_tracks(pred['predictions'],chosen,raw[:n],reference,payload['sequence_id'],state,lookup));retained=None;omitted=None;branch_count=len(pred['predictions'])
                else:
                    if data['history_expiry_enabled']:
                        contract=plan['fixed_history_expiry_contract'];transaction=audit['fixed_history_expiry_transaction']
                        assert audit['fixed_history_expiry_contract']==contract
                        if transaction is not None:
                            assert expiry_record is None and expiry_count==0 and n==old_n==contract['initial_observation_count']
                            assert schedule['event_id']=='history-expiry-tick' and reference==schedule['decision_us']==contract['deadline_us']
                            assert contract['deadline_us']-contract['initial_reference_us']==contract['retention_us']
                            before={tuple(names[r] for r in root_indices(chain(db,x['handle']))) for x in transaction['previous_retained']};assert before==beam and ('a','b','b') in before
                            assert tuple(names[r] for r in root_indices(chain(db,transaction['committed_prefix_handle'])))==previous_roots==('a','b','a')
                            assert transaction['committed_parents']==list(chain(db,transaction['committed_prefix_handle'])) and transaction['committed_prefix_sha256']==beamhelper.class_digest(payload['sequence_id'],previous_roots,names,rows)
                            for x in transaction['previous_retained']:assert x['parents']==list(chain(db,x['handle'])) and x['sha256']==beamhelper.class_digest(payload['sequence_id'],tuple(names[r] for r in root_indices(chain(db,x['handle']))),names,rows)
                            assert transaction['discarded_handles']==sorted(x['handle'] for x in transaction['previous_retained'] if x['handle']!=transaction['committed_prefix_handle'])
                            assert transaction['contract_sha256']==digest(contract) and len(transaction['discarded_handles'])>0
                            beam={previous_roots};expiry_record=transaction;expiry_count+=1
                        assert audit['fixed_history_expiry_record']==expiry_record
                        if reference>=contract['deadline_us']:assert expiry_record is not None
                    assert len(log['selected_histories_per_scan'])==len(audit['scans']) and not audit['restored_ancestors'] and not audit['frontier'] and not audit['recovery_enabled']
                    depth_before=old_n
                    for scan,selected in zip(audit['scans'],log['selected_histories_per_scan']):
                        indices=scan['observation_indices'];assert indices==list(range(depth_before,max(indices)+1));depth=max(indices)+1
                        full_depth=oracle(nodes[:depth]);distribution=beamhelper.classes(full_depth);eligible={a:p for a,p in distribution.items() if a[:depth_before] in beam}
                        assert len(selected)==min(width,len(eligible));threshold=sorted(eligible.values(),reverse=True)[len(selected)-1];next_beam=set()
                        for x in selected:
                            parents=chain(db,x['handle']);assert list(parents)==x['parents'];identities=tuple(names[r] for r in root_indices(parents));assert identities in eligible and eligible[identities]+D('1e-70')>=threshold
                            assert x['sha256']==beamhelper.class_digest(payload['sequence_id'],identities,names,rows)
                            assert abs(float(D(full_depth['log_partition_decimal'])+eligible[identities].ln())-x['log_weight'])<1e-11;next_beam.add(identities)
                        assert len(next_beam)==len(selected) and all(a in next_beam for a,p in eligible.items() if p>threshold+D('1e-70'));beam=next_beam;depth_before=depth;scan_checks+=1
                    active={tuple(names[r] for r in root_indices(chain(db,a['handle']))) for a in audit['active']};assert active==beam
                    ordered=sorted(audit['active'],key=lambda x:(-x['log_weight'],x['sha256']));assert audit['output_handle']==ordered[0]['handle'] and audit['decision']['policy']=='MAP_of_retained_global_histories'
                    chosen=root_indices(chain(db,audit['output_handle']));diagnostics=log['all_retained_branch_state_readbacks'];assert {x['handle'] for x in diagnostics}=={x['handle'] for x in audit['active']};branch_count=0
                    for x in diagnostics:
                        parents=chain(db,x['handle']);assert list(parents)==x['parents'];identities=root_indices(parents)
                        def lookup(root,group):return pure.state_for_group(group,reference,state)[:2]
                        maximum=max(maximum,compare_tracks(x['predictions'],identities,raw[:n],reference,payload['sequence_id'],state,lookup,False));branch_count+=len(x['predictions'])
                    output=next(x['predictions'] for x in diagnostics if x['handle']==audit['output_handle']);assert output==pred['predictions']
                    full=oracle(nodes[:n]);posterior=beamhelper.classes(full);retained=sum((posterior[a] for a in beam),D(0));omitted=1-retained
                    for a in audit['active']:
                        identities=tuple(names[r] for r in root_indices(chain(db,a['handle'])));assert abs(float(D(full['log_partition_decimal'])+posterior[identities].ln())-a['log_weight'])<1e-11
                loss=[i for i in range(n) if raw[i]['state_us']>=max(0,reference-state['window_us'])]
                full=oracle(nodes[:n],[names[i] for i in loss]);posterior=beamhelper.classes(full);named=tuple(names[i] for i in chosen);risk=beamhelper.risk(named,posterior,loss);regret=risk-D(full['bayes_risk_decimal'])
                assert regret>=-D('1e-70');truth_loss=sum(named[i]!=truth[i] for i in loss)/max(1,len(loss))
                if initial is None:initial=sorted(beam) if method=='scan_mht' else [named]
                events.append(dict(event_id=event_id,chosen_roots=named,exact_full_Z_log=full['log_partition_decimal'],exact_full_model_regret_decimal=str(regret),evaluator_only_identity_loss=truth_loss,
                    retained_full_history_mass_decimal=None if retained is None else str(retained),omitted_full_history_mass_decimal=None if omitted is None else str(omitted),independently_verified_scans=scan_checks,conditional_MAP_ties=conditional_ties,
                    full9D_max_abs_difference=maximum,prediction_states=len(pred['predictions']),all_retained_branch_states_checked=branch_count));old_n=n;previous_roots=named
            if method!='scan_mht':
                assert list(db.execute('SELECT i,root FROM identity_anchors ORDER BY i'))==list(anchors.items())
                for r,t,b in db.execute('SELECT root,reference_us,payload FROM gaussian_tracks ORDER BY root'):
                    expected_t,m,P=states[r];assert t==expected_t;v=json.loads(b);close(m,v['mean']);close(P,v['covariance'])
            cache_cells=0
            if method=='scan_mht':
                for h,b in db.execute('SELECT h,payload FROM states ORDER BY h'):
                    parents=chain(db,h);ids=root_indices(parents);root=ids[-1];group=[r for r,a in zip(raw,ids) if a==root];v=json.loads(b)
                    assert v['first_us']==min(r['state_us'] for r in group) and v['last_us']==max(r['state_us'] for r in group)
                    assert v['last_order']==max([r['state_us'],r['node']['source_id'],r['node']['node_id']] for r in group)
                    m,P,s=pure.state_for_group(group,v['last_us'],state);close(m,v['mean']);close(P,v['covariance']);close(max(r['score'] for r in group),v['max_score']);cache_cells+=1
        assert sha(dbpath)==data['database_sha256'];target=('a','b','b');pruned=target not in initial
        if pruned:assert all(e['chosen_roots'][2]!='b' for e in events)
        if data['history_expiry_enabled']:
            assert expiry_count==1 and all(e['chosen_roots'][2]=='a' for e in events) and ('a','b','b') in initial
        else:assert any(e['chosen_roots'][2]=='b' for e in events)
        checks.append(dict(history_expiry_enabled=data['history_expiry_enabled'],exactly_one_expiry_transaction=expiry_count,expired_record=expiry_record,run=run['run'],method=method,width=width,input_id=data['input_id'],events=events,initial_retained_root_classes=initial,evaluator_target_initially_permanently_pruned=pruned,candidate_sha256=run['sha256'],database_sha256=data['database_sha256'],all_SQL_branch_state_cache_cells_checked=cache_cells,no_recovery_claim=True))
        print(json.dumps(dict(independently_accepted_runs=len(checks),total=24)),flush=True)
    result=dict(kind='rbf_stage2_fixed_deadline_identity_history_support_expiry_independent_admission_v1',checks=checks,production_modules_imported=False,all24_full_event_scan_state_history_checks_passed=True,
        PKF_independent_expanded_measurement_Joseph_derivation=True,JPDA_full_mixture_between_mean_covariance_checked=True,only_frozen_zero_yaw_fixtures=True,expiry_is_actual_discard_of_previously_retained_alternative_support=True,not_output_age_or_initial_K1_pruning=True,
        conditional_MAP_ties_accept_any_exact_optimal_legal_action=True,MHT_pruning_boundary_ties_accept_any_exact_top_K_set=True,full_history_regret_is_evaluator_only_not_probabilistic_backend_certificate=True,
        same_resource_comparison_complete=False,published_MHT_PKF_reproduction=False,formal_Linux_MHT_admission=False,full_Stage2_complete=False,source_sha256=sha(__file__),source_plan_sha256=sha(args.plan))
    args.output.open('xb').write(canonical(result)+b'\n')


if __name__=='__main__':main()
