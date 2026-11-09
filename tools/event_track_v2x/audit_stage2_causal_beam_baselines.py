#!/usr/bin/env python3
"""Independent exhaustive beam/pruning, decision and full9D SQL readback."""
import argparse
from collections import defaultdict
import decimal
import hashlib
import importlib.util
import json
import math
import sqlite3
from pathlib import Path
import numpy as np


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def classes(reference):
    result=defaultdict(decimal.Decimal)
    for h in reference['history_probabilities']:
        result[tuple(h['roots'])]+=decimal.Decimal(h['probability_decimal'])
    return dict(result)


def class_digest(sequence,roots,names,rows):
    """Independent canonical minimum-parent root-class SHA recipe."""
    digest=hashlib.sha256(canonical(['persistent-forest-root',sequence])).hexdigest()
    for i,root in enumerate(roots):
        parent=-1 if root==names[i] else min(p for p,w in rows[i] if p>=0 and roots[p]==root)
        digest=hashlib.sha256(canonical([digest,i,parent,names.index(root)])).hexdigest()
    return digest


def risk(action,distribution,loss):
    D=decimal.Decimal
    return sum((p*D(sum(action[i]!=truth[i] for i in loss))/D(max(1,len(loss))) for truth,p in distribution.items()),D(0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text());assert sha(Path(__file__))==plan['auditor_sha256']
    helper=Path(plan['independent_SQL_state_helper']);assert sha(helper)==plan['independent_SQL_state_helper_sha256'];pure=load(helper,'pure_sql_and_9D')
    oraclepath=Path(plan['independent_oracle']);assert sha(oraclepath)=='29c038639fb1db1f5604a4f1ac150827752e22f86ccefc3bfcdd9356bc5821c0';oracle=load(oraclepath,'full_history_reference').oracle
    campaign=json.loads(args.campaign.read_text());assert campaign['source_plan_sha256']==sha(args.plan)
    assert len(campaign['runs'])==len(plan['inputs'])*len(plan['methods'])*len(plan['widths'])
    inputrefs={r['id']:r for r in plan['inputs']};truthrefs=plan['evaluator_only_original_candidates'];D=decimal.Decimal;decimal.getcontext().prec=80;checks=[]
    for row in campaign['runs']:
        path=Path(row['path']);assert sha(path)==row['sha256'];data=json.loads(path.read_text());ref=inputrefs[data['input_id']];incoming=Path(ref['path']);assert sha(incoming)==ref['sha256']==data['input_sha256'];payload=json.loads(incoming.read_text());raw=payload['raw'];factors=payload['rows'];names=[r['node']['node_id'] for r in raw];state=data['configuration']['state'];width=data['width']
        gtref=truthrefs[data['input_id']];gtpath=Path(gtref['path']);assert sha(gtpath)==gtref['sha256'];original=json.loads(gtpath.read_text());truth=original['evaluator_only_true_roots'];assert original['raw']==raw and original['rows']==factors
        assert data['configuration']==plan['configurations'][data['method']][str(width)]
        nodes=[dict(node_id=r['node']['node_id'],source_id=r['node']['source_id'],frame_id=r['node']['frame_id'],information_us=r['node']['information_us'],arrival_us=r['node']['arrival_us'],choices=[dict(parent_id=None if p<0 else names[p],log_potential=w) for p,w in weights]) for r,weights in zip(raw,factors)]
        dbpath=Path(data['database_path']);assert sha(dbpath)==data['database_sha256'];beam={()};old_n=0;previous='0'*64;events=[];initial_beam=None
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
            for item,(b,h) in zip(raw,db.execute('SELECT raw,sha FROM observations ORDER BY i')):
                assert b==canonical(item) and hashlib.sha256(b).hexdigest()==h
            assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==len(raw)
            assert list(db.execute('SELECT i,p,w FROM potentials ORDER BY i,p'))==sorted((i,p,w) for i,weights in enumerate(factors) for p,w in weights)
            stored=list(db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal'))
            assert len(stored)==len(data['events'])==len(payload['schedule'])
            for logged,schedule,(event_id,pb,ab) in zip(data['events'],payload['schedule'],stored):
                pred,audit=json.loads(pb),json.loads(ab);assert pred==logged['prediction'] and audit==logged['audit'] and event_id==schedule['event_id']==logged['event_id']
                assert hashlib.sha256(pb).hexdigest()==logged['prediction_sha256'] and hashlib.sha256(ab).hexdigest()==logged['audit_sha256'];body=dict(pred);commit=body.pop('commit_sha256');assert hashlib.sha256(canonical(body)).hexdigest()==commit and body['previous_commit_sha256']==previous;previous=commit
                n=schedule['available_observations'];assert n==audit['observation_count'];assert pred['decision_timestamp_us']==schedule['decision_us'] and pred['box_reference_timestamp_us']==schedule['reference_us']
                assert audit['appended_rows']==factors[old_n:n] and all(r['node']['arrival_us']<=schedule['decision_us'] for r in raw[:n])
                assert len(audit['components'])==1  # verified single connected component, no merge claim
                comp=audit['components'][0];cid=comp['component'];members=[i for i, in db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i',(cid,))][:comp['nodes']];assert members==list(range(n))
                full=oracle(nodes[:n],[names[i] for i in audit['decision_indices']]);posterior=classes(full)
                if n>old_n:
                    depths=range(old_n+1,n+1) if data['method']=='node_beam' else [n]
                    depth_before=old_n
                    for depth in depths:
                        distribution=classes(oracle(nodes[:depth]))
                        eligible={roots:p for roots,p in distribution.items() if roots[:depth_before] in beam}
                        ordered=sorted(eligible,key=lambda roots:(-eligible[roots],class_digest(payload['sequence_id'],roots,names,factors)))
                        beam=set(ordered[:width]);depth_before=depth
                else:
                    assert audit['beam_expansions']==audit['candidate_evaluations']==0
                active={pure.roots(pure.prefix(db,cid,a['handle']),names[:n]) for a in comp['active']}
                assert active==beam and 0<len(active)<=width
                assert not comp['frontier'] and not comp['recovery_events'] and not audit['recovery_enabled'] and not audit['archived_prefixes_used_for_recovery']
                for a in comp['active']:
                    roots=pure.roots(pure.prefix(db,cid,a['handle']),names[:n]);assert a['sha256']==class_digest(payload['sequence_id'],roots,names,factors)
                    exact_log=D(full['log_partition_decimal'])+posterior[roots].ln();assert math.isclose(float(exact_log),a['log_weight'],abs_tol=1e-11,rel_tol=0)
                retained=sum((posterior[roots] for roots in active),D(0));conditional={roots:posterior[roots]/retained for roots in active};omitted=sum((p for roots,p in posterior.items() if roots not in active),D(0))
                chosen=pure.roots(pure.prefix(db,cid,comp['output_handle']),names[:n]);assert chosen in active;loss=audit['decision_indices'];conditional_risk=risk(chosen,conditional,loss);conditional_best=min(risk(a,conditional,loss) for a in active)
                assert conditional_risk-conditional_best<=D('1e-11') and math.isclose(float(conditional_risk),comp['decision']['conditional_risk'],abs_tol=1e-11,rel_tol=0)
                full_risk=risk(chosen,posterior,loss);regret=full_risk-D(full['bayes_risk_decimal']);assert regret>=-D('1e-70');assert float(omitted)<=audit['product_omitted_mass_upper']+1e-11 and float(regret)<=audit['model_regret_upper']+1e-11
                tracks={t['track_id']:t for t in pred['predictions']};maximum=0.
                for root in set(chosen):
                    group=[r for r,identity in zip(raw[:n],chosen) if identity==root];ident=payload['sequence_id']+':'+hashlib.sha256(canonical(['forest-birth',root])).hexdigest()[:24];target=tracks.pop(ident);assert target['class_label']=='car';mean,cov,score=pure.state_for_group(group,pred['box_reference_timestamp_us'],state)
                    assert np.allclose(mean,target['mean'],atol=1e-8,rtol=1e-8) and np.allclose(cov,target['covariance'],atol=1e-8,rtol=1e-8) and math.isclose(score,target['score'],abs_tol=1e-8,rel_tol=1e-8)
                    maximum=max(maximum,float(np.max(np.abs(mean-target['mean']))),float(np.max(np.abs(cov-target['covariance']))),abs(score-target['score']))
                assert not tracks
                if initial_beam is None:initial_beam=sorted(active)
                events.append(dict(event_id=event_id,available_observations=n,chosen_roots=chosen,
                    exact_full_Z_log=full['log_partition_decimal'],exact_full_model_regret_decimal=str(regret),
                    exact_full_omitted_mass_decimal=str(omitted),conditional_retained_Bayes_risk_decimal=str(conditional_risk),
                    retained_classes=len(active),legal_full_histories=full['legal_histories'],beam_ranking_and_pruning_independently_verified=True,
                    deleted_prior_classes_used_for_inference=False,full9D_max_abs_difference=maximum,
                    evaluator_only_identity_loss=sum(chosen[i]!=truth[i] for i in loss)/max(1,len(loss)),
                    prediction_states=len(pred['predictions'])))
                old_n=n
        assert sha(dbpath)==data['database_sha256'] and sha(path)==row['sha256']
        target_initial=('a','b','b');target_discarded=target_initial not in initial_beam
        if target_discarded:assert all(e['chosen_roots'][2]!='b' for e in events)
        checks.append(dict(run=row['run'],method=data['method'],width=width,input_id=data['input_id'],candidate_sha256=row['sha256'],database_sha256=data['database_sha256'],events=events,
            initial_retained_root_classes=initial_beam,evaluator_target_initially_permanently_pruned=target_discarded,
            recovery_claimed=False,no_new_rows_compute_events_cannot_restore_deleted_class=True,
            reproduced_classical_MHT=False,full_Stage2_complete=False))
        print(json.dumps({'independently_accepted_beam_runs':len(checks),'total':len(campaign['runs'])}),flush=True)
    result=dict(kind='rbf_stage2_causal_irreversible_beam_full_exhaustive_admission_v1',checks=checks,
        all_runs_full_events_ranked_history_and_9D_verified=True,production_modules_imported=False,
        single_connected_component_no_merge_cases_only=True,tie_break_prefix_SHA_independently_reconstructed=True,
        formal_numeric_interval_certificate=False,same_resource_comparison_complete=False,learned_priority_executed=False,
        full_Stage2_complete=False,goal_status='active',source_sha256=sha(Path(__file__)))
    with args.output.open('xb') as f:f.write(canonical(result)+b'\n')


if __name__=='__main__':main()
