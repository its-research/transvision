#!/usr/bin/env python3
"""Summarize qualified same-input finite cases without resource or win claims."""
import argparse
from collections import defaultdict
import hashlib
import json
import statistics
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.plan.read_text());assert sha(Path(__file__))==plan['source_sha256']
    loaded={}
    for key,ref in plan['inputs'].items():
        p=Path(ref['path']);assert sha(p)==ref['sha256'];loaded[key]=json.loads(p.read_text())
    beam=loaded['beam'];rbf=loaded['RBF'];witness=loaded['RBF_SQL_witness'];branches=loaded['all_branch_states']
    assert beam['all_runs_full_events_ranked_history_and_9D_verified'] and rbf['all_new_runs_full_events_independently_verified']
    assert branches['all_active_output_branch_prediction_byte_digests_reconstructed'] and len(branches['runs'])==72
    branch_pins={(r['candidate_sha256'],r['database_sha256']) for r in branches['runs']};rbf_by_run={r['run']:r for r in rbf['checks']};witness_by_run={r['run']:r for r in witness['runs']};groups=defaultdict(list);pairs=[]
    for b in beam['checks']:
        incoming=b['input_id'];seed=int(incoming.rsplit('seed',1)[1]);case=incoming.rsplit('-seed',1)[0];width=b['width'];r=rbf_by_run[incoming+f'-K{width}'];assert (b['candidate_sha256'],b['database_sha256']) in branch_pins and (r['candidate_sha256'],r['database_sha256']) in branch_pins
        assert [e['event_id'] for e in b['events']]==[e['event_id'] for e in r['events']]
        assert all(len(a['chosen_roots'])==len(c['chosen_roots']) for a,c in zip(b['events'],r['events']))
        first_b=next((e['event_id'] for e in b['events'] if e['chosen_roots'][2]=='b'),None);first_r=next((e['event_id'] for e in r['events'] if e['chosen_roots'][2]=='b'),None)
        last=b['events'][-1];groups[(case,b['method'],width)].append(dict(fixture_seed=seed,
            final_exact_full_model_regret=float(last['exact_full_model_regret_decimal']),final_evaluator_identity_loss=last['evaluator_only_identity_loss'],
            current_c_identity_b=last['chosen_roots'][2]=='b',initial_target_permanently_pruned=b['evaluator_target_initially_permanently_pruned'],
            first_current_c_b_event=first_b,previously_frontier_only_current_output_recovery=False))
        pairs.append(dict(case=case,fixture_seed=seed,width=width,beam_method=b['method'],beam_final_regret=groups[(case,b['method'],width)][-1]['final_exact_full_model_regret'],RBF_final_regret=r['events'][-1]['exact_same_action_model_regret'],beam_first_current_c_b_event=first_b,RBF_first_current_c_b_event=first_r,same_resource_comparison=False))
    for r in rbf['checks']:
        input_id,width_s=r['run'].rsplit('-K',1);case=input_id.rsplit('-seed',1)[0];seed=int(input_id.rsplit('seed',1)[1]);width=int(width_s);last=r['events'][-1];w=witness_by_run[r['run']]
        groups[(case,'exclusive_RBF',width)].append(dict(fixture_seed=seed,
            final_exact_full_model_regret=last['exact_same_action_model_regret'],final_evaluator_identity_loss=last['evaluator_only_identity_loss'],
            current_c_identity_b=last['chosen_roots'][2]=='b',first_current_c_b_event=next((e['event_id'] for e in r['events'] if e['chosen_roots'][2]=='b'),None),
            previously_frontier_only_current_output_recovery=w['accepted_recovery']))
    rows=[]
    for (case,method,width),values in sorted(groups.items()):
        assert sorted(v['fixture_seed'] for v in values)==[1337,2027,3407];regrets=[v['final_exact_full_model_regret'] for v in values];loss=[v['final_evaluator_identity_loss'] for v in values]
        rows.append(dict(case=case,method=method,width=width,fixture_runs=values,
            final_model_regret_mean=statistics.mean(regrets),final_model_regret_fixture_sample_SD=statistics.stdev(regrets),
            final_evaluator_identity_loss_mean=statistics.mean(loss),correct_current_c_identity_runs=sum(v['current_c_identity_b'] for v in values),
            previously_frontier_only_current_output_recovery_runs=sum(v['previously_frontier_only_current_output_recovery'] for v in values)))
    output=dict(kind='rbf_stage2_causal_same_input_72_run_pruning_recovery_comparison_v1',rows=rows,paired_traces=pairs,
        input_sha256={k:ref['sha256'] for k,ref in plan['inputs'].items()},source_sha256=sha(Path(__file__)),
        complete_new_beam_runs=54,reused_RBF_runs=18,total_actual_events=1044,
        actual_single_connected_component_only=True,no_component_merge_claim=True,
        rootclass_ranking_not_classical_MHT=True,fixture_variability_not_trained_seed_variability=True,
        confidence_intervals_inferred=False,same_resource_costs_complete=False,learned_priority_executed=False,
        support_recovery_value_uniformly_positive=False,real_data_performance=False,full_Stage2_complete=False,goal_status='active')
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'qualified_comparison_rows':len(rows),'actual_configs':72,'uniform_recovery_advantage':False}))


if __name__=='__main__':main()
