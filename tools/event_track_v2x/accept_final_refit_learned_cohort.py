"""Full learned exclusive cohort: forest and checkpoint-bound allocation trajectory.

Structure/causality/recovery are checked by this new read-only oracle; continuous
states and emitted predictions use the original frozen NumPy oracle unchanged.
Bounded full legal action search, all raw-cache 203 feature values and exact first-arrival parent contexts are independently reconstructed.
Whole original supplied train schedule only. Complete posterior optimality, learned Stage2, physical recovery and metrics remain unproved.
"""
import argparse,datetime,hashlib,importlib.util,json,time,sys
from pathlib import Path
from clearml import Task
from rbf_nested_seen_val_v2_common import register
STRUCTURE_D=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002')
sys.path.insert(0,str(STRUCTURE_D))
from oracle import audit_database,sha,canonical,ATOL,RTOL,ROOT
sys.path.insert(0,str(Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-independent-legal-action-search-v1-20261002')))
from decoder_capacity import verify_database as action_database
from causal import verify_database as causal_database
FRESH_PATH=ROOT/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py'
FRESH_SHA='46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58'
D=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-independent-action-capacity-undecided-v2-20261002')
ACTION_D=ROOT/'source-freezes/rbf-independent-legal-action-search-v1-20261002'
CACHE_D=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-independent-runtime-cache203-context-v1-20261002')
sys.path.insert(0,str(CACHE_D))
from final_cache203 import CacheAdmission,verify_database as feature_database

from rbf_final_refit_learned_forest_binding import (validate, source_gate, arguments, check_sequence, ordering, ordering_summary, KIND)

def main():
 p=argparse.ArgumentParser();arguments(p);args=p.parse_args()
 assert args.output.resolve().is_relative_to(ROOT/'artifacts')
 gate=ROOT/'artifacts/rbf-independent-exclusive-forest-oracle-software-gate-v1-20261002/software-gate-v2.json';g=json.loads(gate.read_text())
 assert g['completed_cases']==297 and g['completed_events']==1548 and len(g['rejection_mutations'])==6 and g['independent_output_class_recoveries']==9
 for name,h in g['sources'].items():assert sha(STRUCTURE_D/name)==h
 action_gate=ROOT/'artifacts/rbf-independent-legal-action-search-software-gate-v1-20261002/software-gate.json';ag=json.loads(action_gate.read_text())
 assert ag['component_decisions']==1908 and len(ag['rejection_mutations'])==8
 for name,h in ag['sources'].items():assert sha(ACTION_D/name)==h
 own_gate=ROOT/'artifacts/rbf-exclusive-nonretained-action-CPU-v1-20261002/acceptance.json';og=json.loads(own_gate.read_text())
 assert og['completed_cases']==1 and og['actions_outside_retained_classes']==2 and og['chosen_action_strictly_better_than_all_retained_actions'] is True
 assert og['all_selected_states_freshly_reconstructed_from_raw_history'] is True and og['decoder_sha256']==sha(ACTION_D/'decoder.py')
 gate_task=Task.get_task(task_id=og['task_id']);assert str(gate_task.status)=='completed'
 for k,a in og['registered_artifacts'].items():assert gate_task.artifacts[k].hash==a['sha256'] and gate_task.artifacts[k].size==a['bytes']
 freeze=json.loads((D/'source-freeze.json').read_text());assert freeze['future_cohort_driver']=='accept_real_cohort.py'
 for name,h in freeze['sources'].items():assert sha(D/name)==h
 feature_gate=ROOT/'artifacts/rbf-independent-runtime-cache203-context-software-v1-20261002/final-gate/software-gate.json';fg=json.loads(feature_gate.read_text())
 assert fg['kind']=='rbf_independent_runtime_cache203_context_verifier_software_gate_v1' and len(fg['rejection_mutations'])==9
 assert fg['fixture']['rows']==2959 and fg['full_schedule_requirement_rejects_historical_prefix'] is True and fg['real_exclusive_cohort_admitted'] is False
 for name,h in fg['sources'].items():assert sha(CACHE_D/name)==h
 capacity_gate=ROOT/'artifacts/rbf-independent-action-capacity-undecided-software-v1-20261002/software-gate.json'
 cg=json.loads(capacity_gate.read_text());assert len(cg['accepted_software_cases'])==3 and len(cg['rejected_semantic_mutations'])==21
 for name,h in cg['sources'].items():assert sha(D/name)==h
 control=json.loads((D/'source-control.json').read_text());assert control['original_decoder_sha256']==sha(ACTION_D/'decoder.py') and control['in_capacity_verifier_AST_identical_except_name'] is True
 final_driver_binding=source_gate();v=json.loads(args.byte_admission.read_text());job=validate(v,args,Task)
 assert sha(args.checkpoint)==job['plan']['checkpoint']['sha256']
 feature_admission=CacheAdmission(v['seed'],args.checkpoint)
 for key,a in v['artifacts'].items():
  file=args.byte_admission.parent/(key+('.json' if key in ('receipt','exclusive-source-manifest','priority-input-binding') else '.tar.gz'));assert sha(file)==a['sha256'] and file.stat().st_size==a['bytes']
 assert sha(FRESH_PATH)==FRESH_SHA
 spec=importlib.util.spec_from_file_location('original_frozen_independent_fresh_state',FRESH_PATH);fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
 assert not args.output.exists() and not any(p.is_symlink() for p in (args.output,*args.output.parents));args.output.mkdir(parents=True)
 sources={p.name:sha(p) for p in D.glob('*.py')};binding=dict(task_id=v['task_id'],seed=v['seed'],recipe_sha256=v['recipe_sha256'],byte_admission_sha256=sha(args.byte_admission),sources=sources,fresh_oracle_sha256=FRESH_SHA,software_gate_sha256=sha(gate),capacity_action_software_gate_sha256=sha(capacity_gate),capacity_source_control_sha256=sha(D/'source-control.json'),action_search_software_gate_sha256=sha(action_gate),runtime_cache203_context_software_gate_sha256=sha(feature_gate),runtime_cache203_source_sha256=sha(CACHE_D/'cache203.py'),nonretained_own_state_acceptance_sha256=sha(own_gate),frozen_structure_sources=g['sources'],atol=ATOL,rtol=RTOL,complete_online_method_accepted=False,learned_Stage2_complete=False,identity_decoder_full_search_and_regret_admitted=False,physical_recovery_correctness_or_metrics_admitted=False,same_resource_performance_accepted=False,paper_performance_complete=False)
 binding.update(final_driver_binding);binding.update(feature_admission.final_model_binding);binding['old_model_full_forest_acceptance_inherited']=False
 (args.output/'binding.json').write_text(json.dumps(binding,indent=2)+'\n');started=time.monotonic();results=[]
 try:
  for index,s in enumerate(v['sequences']):
   files=list(args.byte_admission.parent.glob(f"rank*-unpack/rank-*/{s['sequence_id']}/receipt.json"));assert len(files)==1
   db=check_sequence(files[0].parent,s,job)
   def progress(data):
    payload=dict(data);payload.update(scope_stage='full_independent_exclusive_forest_scope',completed_sequences=index,total_sequences=46);print(json.dumps(payload),flush=True)
   features=feature_database(db,s['database_sha256'],feature_admission,progress=progress);structure=audit_database(db,s['database_sha256'],progress);causal=causal_database(db,s['database_sha256']);state=fresh.verify_database(db,s['database_sha256'],progress);action=action_database(db,s['database_sha256'],progress)
   learned_order=ordering(db,s,args,job,progress)
   assert structure['events']==causal['events']==state['events']==action['events']==features['events']==learned_order['events']==s['events'] and causal['observations']==features['rows']==learned_order['observations']==s['nodes']
   # The fresh oracle independently reconstructs every output state chosen by
   # the same recovery commits. Correct physical identity still requires GT.
   result=dict(sequence_id=s['sequence_id'],database_sha256=s['database_sha256'],structure=structure,causal=causal,fresh_state=state,conditional_action_search=action,runtime_raw_cache203_context=features,learned_search_trajectory=learned_order)
   results.append(result);(args.output/f'sequence-{index:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps(dict(stage='full_independent_exclusive_forest_sequence_complete',completed_sequences=len(results),total_sequences=46,ETA_seconds=None,ETA_reason='heterogeneous component/branch history cost; no whole-cohort timing model')),flush=True)
  assert len(results)==46 and sum(s['structure']['events'] for s in results)==7445
  assert sources=={p.name:sha(p) for p in D.glob('*.py')} and sha(FRESH_PATH)==FRESH_SHA
  assert source_gate()==final_driver_binding
  assert sha(args.byte_admission)==binding['byte_admission_sha256']
  assert validate(v,args,Task)==job
  priority_summary=ordering_summary(results,job)
  final=dict(binding,kind=KIND,completed_sequences=46,completed_events=7445,strict_declared_support_partition_coverage=True,padded_float64_mass_arithmetic_verified=True,output_class_recovery_provenance_verified=True,model_output_class_recoveries=sum(x['structure']['output_class_recoveries_verified'] for x in results),all_raw_factor_and_causal_commits_verified=True,fresh_conditional_states=sum(x['fresh_state']['states'] for x in results),fresh_predictions=sum(x['fresh_state']['predictions'] for x in results),max_fresh_state_abs_error=max(x['fresh_state']['max_abs_error'] for x in results),max_mass_abs_error=max(x['structure']['max_abs_arithmetic_error'] for x in results),all_decisions_search_or_capacity_undecided_semantics_verified=True,capacity_undecided_component_decisions=sum(x['conditional_action_search']['capacity_undecided_component_decisions'] for x in results),conditional_full_legal_action_search_and_declared_gap_verified=not any(x['conditional_action_search']['capacity_undecided_component_decisions'] for x in results),actions_outside_retained_classes=sum(x['conditional_action_search']['actions_outside_retained_classes'] for x in results),incomplete_component_action_searches=sum(x['conditional_action_search']['incomplete_component_searches'] for x in results),max_conditional_action_arithmetic_error=max(x['conditional_action_search']['max_abs_arithmetic_error'] for x in results),full_model_posterior_optimum_regret_independently_computed=False,all_203_feature_recipe_values_independently_verified=True,all_scorer_contexts_bound_to_original_causal_raw_history=True,max_cache203_abs_error=max(max(x['runtime_raw_cache203_context']['max_abs_error'].values()) for x in results),formal_interval_certificate=False,true_posterior_guarantee=False,elapsed_seconds=time.monotonic()-started,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  final.update(priority_summary)
  (args.output/'acceptance.json').write_text(json.dumps(final,indent=2)+'\n');register(args.output/'acceptance.json',final['kind']);print(json.dumps(final),flush=True)
 except BaseException as e:
  (args.output/'failure.json').write_text(json.dumps(dict(binding,type=type(e).__name__,message=str(e),completed_sequences=len(results),experiment_accepted=False),indent=2)+'\n');raise
if __name__=='__main__':main()
