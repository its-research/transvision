#!/usr/bin/env python3
"""Source-bound comparison of 102 accepted same-input finite causal runs."""
import argparse,hashlib,json,decimal
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();plan=json.loads(a.plan.read_bytes());assert sha(__file__)==plan['source_sha256'];D=decimal.Decimal;rows=[];traces=[]
    for family,reference in plan['admissions'].items():
        assert sha(reference['path'])==reference['sha256'];admission=json.loads(Path(reference['path']).read_bytes())
        for c in admission['checks']:
            candidate=Path(reference['path']).parent/'candidate'/c['run']/'candidate.json';assert sha(candidate)==c['candidate_sha256'];v=json.loads(candidate.read_bytes())
            input_id=c.get('input_id',c['run'].rsplit('-K',1)[0]);ref=next(x for x in plan['inputs'] if x['id']==input_id);assert sha(ref['path'])==ref['sha256'];incoming=json.loads(Path(ref['path']).read_bytes())
            if 'input_path' in v:assert v['input_sha256']==ref['sha256'] and json.loads(Path(v['input_path']).read_bytes())==incoming
            else:assert v['raw']==incoming['raw'] and v['rows']==incoming['rows']
            assert len(c['events'])==len(incoming['schedule']);method=c.get('method','RBF');width=c.get('width',int(c['run'].rsplit('-K',1)[1]) if family=='RBF' else None)
            paired=[]
            assert len(v['events'])==len(c['events'])
            for e,s,observed in zip(c['events'],incoming['schedule'],v['events']):
                assert e['event_id']==s['event_id'];regret=e.get('exact_full_model_regret_decimal',e.get('exact_same_action_model_regret'));z=e.get('exact_full_Z_log',e.get('exact_Z_log'))
                assert observed['event_id']==s['event_id'] and observed['available_observations']==s['available_observations'] and observed['prediction']['box_reference_timestamp_us']==s['reference_us'] and observed['prediction']['decision_timestamp_us']==s['decision_us']
                paired.append(dict(event_id=e['event_id'],reference_us=s['reference_us'],decision_us=s['decision_us'],available_observations=s['available_observations'],chosen_roots=e['chosen_roots'],full_posterior_log_Z=z,full_posterior_model_regret=str(regret),evaluator_only_identity_loss=e['evaluator_only_identity_loss']))
            corrected=next((x['event_id'] for x in paired if x['chosen_roots'][2]=='b'),None);first_new=next(x for x in paired if x['available_observations']>3)
            rows.append(dict(input_id=input_id,method=method,width=width,run=c['run'],candidate_sha256=c['candidate_sha256'],first_new_data_model_regret=first_new['full_posterior_model_regret'],final_model_regret=paired[-1]['full_posterior_model_regret'],final_evaluator_identity_loss=paired[-1]['evaluator_only_identity_loss'],first_current_decision_c_to_B=corrected,recovery_mechanism_claim=family=='RBF' and c['recovery_claimed']))
            traces.append(dict(input_id=input_id,method=method,width=width,events=paired))
    assert len(rows)==102 and len(traces)==102
    for input_id in [x['id'] for x in plan['inputs']]:
        selected=[t for t in traces if t['input_id']==input_id];assert len(selected)==17
        for i in range(len(selected[0]['events'])):
            anchor=selected[0]['events'][i]
            for t in selected[1:]:
                e=t['events'][i];assert all(e[k]==anchor[k] for k in ['event_id','reference_us','decision_us','available_observations']);assert abs(D(e['full_posterior_log_Z'])-D(anchor['full_posterior_log_Z']))<D('1e-11')
    out=dict(kind='rbf_stage2_102_accepted_finite_same_input_causal_comparison_v1',source_sha256=sha(__file__),source_plan_sha256=sha(a.plan),rows=rows,paired_traces=traces,unique_frozen_inputs=6,accepted_runs=102,new_runs=30,reused_RBF_and_beam_runs=72,
        identity_loss_is_evaluator_only_not_tracking_HOTA=True,seeds_are_factor_jitter_not_trained_model_seeds=True,Gaussian_state_protocol_differs_for_JPDA_PKF=True,width_is_not_same_resource=True,no_uniform_recovery_advantage_claim=True,expired_history_case_included=False,
        formal_Linux_MHT_contract_satisfied=False,published_native_method_reproduction=False,same_resource_comparison_complete=False,full_Stage2_complete=False,goal_status='active')
    a.output.open('x').write(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'qualified_same_input_runs':len(rows),'new_runs':30,'source_bound_rows':len(rows)}))
if __name__=='__main__':main()
