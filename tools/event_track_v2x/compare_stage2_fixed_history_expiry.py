#!/usr/bin/env python3
"""Pair actual fixed-deadline support deletion with matched retained controls."""
import argparse,decimal,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_bytes());assert sha(__file__)==plan['source_sha256'];ref=plan['independent_admission'];assert sha(ref['path'])==ref['sha256'];proof=json.loads(Path(ref['path']).read_bytes());checks=proof['checks'];assert len(checks)==24;pairs=[];D=decimal.Decimal
    for expired in [c for c in checks if c['history_expiry_enabled']]:
        control=next(c for c in checks if not c['history_expiry_enabled'] and c['input_id']==expired['input_id'] and c['width']==expired['width'])
        paths=[Path(ref['path']).parent/'candidate'/c['run']/'candidate.json' for c in [control,expired]]
        for path,c in zip(paths,[control,expired]):assert sha(path)==c['candidate_sha256']
        cv,ev=[json.loads(p.read_bytes()) for p in paths];assert cv['input_sha256']==ev['input_sha256'] and cv['configuration']==ev['configuration'] and cv['events'][0]['prediction_sha256']==ev['events'][0]['prediction_sha256']
        assert len(control['events'])==len(expired['events']);paired=[]
        for c,e in zip(control['events'],expired['events']):
            assert c['event_id']==e['event_id'] and abs(D(c['exact_full_Z_log'])-D(e['exact_full_Z_log']))<D('1e-11')
            paired.append(dict(event_id=c['event_id'],retained_chosen_roots=c['chosen_roots'],expired_chosen_roots=e['chosen_roots'],retained_evaluator_identity_loss=c['evaluator_only_identity_loss'],expired_evaluator_identity_loss=e['evaluator_only_identity_loss'],retained_full_model_regret=c['exact_full_model_regret_decimal'],expired_full_model_regret=e['exact_full_model_regret_decimal']))
        assert control['events'][-1]['evaluator_only_identity_loss']==0 and expired['events'][-1]['evaluator_only_identity_loss']>0
        assert expired['exactly_one_expiry_transaction']==1 and ('a','b','b') in [tuple(r) for r in expired['initial_retained_root_classes']]
        pairs.append(dict(input_id=expired['input_id'],K=expired['width'],control_candidate_sha256=control['candidate_sha256'],expired_candidate_sha256=expired['candidate_sha256'],expiry_record=expired['expired_record'],events=paired))
    assert len(pairs)==12
    result=dict(kind='rbf_stage2_12_paired_actual_fixed_history_expiry_counterexamples',source_sha256=sha(__file__),plan_sha256=sha(a.plan),pairs=pairs,first_predictions_byte_identical=True,actual_transaction_previously_retained_B_history_removed=True,
        all_future_inference_restricted_to_expired_MAP_prefix=True,same_raw_factors_config_and_empty_tick_schedule=True,extra_budget_to_fix_failure=False,full_Stage2_complete=False,published_MHT_reproduction=False,formal_Linux_MHT_admission=False,same_resource_comparison_complete=False,goal_status='active')
    a.output.open('x').write(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'qualified_matched_expiry_pairs':len(pairs)}))
if __name__=='__main__':main()
