"""Portable receipt gate for final-model teacher export and priority fitting.

This consumes completed numerical evidence. It never computes or fabricates
that evidence. Training-only modules have their own source binding and cannot
be represented as modules executed by the teacher.
"""
from pathlib import Path
import hashlib
import json
import math

BACKEND='exclusive_root_partition_regions_v1'
MAIN_KIND='rbf_final_refit_full_exclusive_cohort_independent_structure_causal_mass_recovery_state_action_scope_acceptance_v1'
BYTE_KIND='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'
TARGET_KIND='rbf_final_refit_full_teacher_independent_model_bound_targets_v1'
MAIN_FREEZE='f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
TARGET_FREEZE='3a1b58959479a205b56235f82fd83af78ba3a0a419e87b0678d66de3e0a6c620'
ORIGINAL_NUMERICAL_FREEZE='026a6693e032906fea9daf3cc6c384297d02ee2c685ad6eff0a7a920e7e540d2'
WITNESS_RECIPE='exclusive_raw_search_before_after_probe_witness_v1'
INITIAL_RECIPE='exclusive_all_live_initial_search_and_catalog_v1'
TRAINING_ONLY={
    'transvision/models/event_track_v2x/exclusive_allocation_training.py',
    'transvision/models/event_track_v2x/exclusive_priority_admission.py'}
EXPORTERS={
    'tools/event_track_v2x/train_exclusive_paper_priority.py',
    'tools/event_track_v2x/spd_teacher_schedule_coverage.py',
    'tools/event_track_v2x/spd_causal_teacher_events.py',
    'tools/event_track_v2x/seal_spd_paper_schedule.py'}
FILES={'main-independent-admission.json','main-byte-admission.json',
       'teacher-target-admission.json','teacher-collection-binding.json'}


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'admission bytes changed')
    value=json.loads(raw);require(isinstance(value,dict),'receipt object required')
    return value,raw


def verify_teacher_sources(plan_sources,consumer_sources):
    require(TRAINING_ONLY <= set(consumer_sources),'explicit training-only source roles required')
    require(not (TRAINING_ONLY & set(plan_sources)),'unexpected training module in original teacher source map')
    runtime={k:v for k,v in consumer_sources.items() if k not in TRAINING_ONLY}
    require(bool(runtime) and all(plan_sources.get(k)==v for k,v in runtime.items()),
        'executed teacher feature or solver sources differ')
    return dict(teacher_runtime_sources=runtime,
        training_only_sources={k:consumer_sources[k] for k in sorted(TRAINING_ONLY)})


def validate(plan,receipt_sha256,m,b,t,c,*,main_sha256,byte_sha256):
    seed=plan['model_binding']['seed'];model=plan['model_binding']['model_sha256']
    require(type(seed) is int and seed in (1337,2027,3407),'supported seed required')
    require(m['kind']==MAIN_KIND and b['kind']==BYTE_KIND and t['kind']==TARGET_KIND,'final-model receipts required')
    require(m['driver_freeze_sha256']==MAIN_FREEZE and t['source_freeze_sha256']==TARGET_FREEZE,
        'qualified final-model numerical driver required')
    require(t['unchanged_target_driver_freeze_sha256']==ORIGINAL_NUMERICAL_FREEZE,'original target numerics required')
    require(m['seed']==b['seed']==t['seed']==c['seed']==seed,'mixed seed')
    require(m['task_id']==b['task_id'] and m['recipe_sha256']==b['recipe_sha256'],'main byte task identity')
    require(m['byte_admission_sha256']==byte_sha256 and b['method']=='rbf','main forest byte proof required')
    require(m['completed_sequences']==46 and m['completed_events']==7445,'full final main required')
    require(m['final_model_sha256']==b['final_model_sha256']==t['final_model_sha256']==c['final_model_sha256']==model,
        'final-model lineage mismatch')
    require(m['original_model_sha256']!=model and m['old_model_full_forest_acceptance_inherited'] is False,
        'legacy model acceptance cannot be inherited')
    require(m['checkpoint_sha256']==b['final_checkpoint_sha256']==t['checkpoint_sha256']==c['checkpoint_sha256']==plan['model_binding']['checkpoint_sha256'],
        'checkpoint identity differs')
    for key in ('strict_declared_support_partition_coverage','padded_float64_mass_arithmetic_verified',
        'output_class_recovery_provenance_verified','all_raw_factor_and_causal_commits_verified',
        'all_decisions_search_or_capacity_undecided_semantics_verified','all_203_feature_recipe_values_independently_verified',
        'all_scorer_contexts_bound_to_original_causal_raw_history'):
        require(m[key] is True,'missing main scope: '+key)
    require(b['all_registered_bytes_verified'] is True and b['all_46_sequences_7445_events_and_final_model_factor_nodes_verified'] is True,
        'full main bytes and factors required')
    seqs=b['sequences'];identities=sorted(s['sequence_id'] for s in seqs)
    require(len(seqs)==len(set(identities))==46 and sum(s['events'] for s in seqs)==7445,'full original sequence coverage')
    require(plan['expected_sequences']==t['completed_sequences']==identities and plan['expected_events']==t['completed_events']==7445,
        'teacher must cover the complete original schedule')
    require(plan['protocol']['dataset']==plan['model_binding']['dataset']=='spd'
        and plan['protocol']['split']==plan['model_binding']['fit_split']=='train','train-only SPD required')
    require(plan['fixture'] is c['fixture'] is False,'real teacher required')
    configuration=plan['configuration'];original=c['original_plan']
    require(configuration==t['configuration']==original['configuration'],'teacher configuration differs')
    require(configuration['method']=='rbf' and configuration['allocation']=='teacher'
        and configuration['backend']==BACKEND and configuration['limits']['residual_partition_version']==1
        and configuration['state']['candidate_protocol']=='rbf-all-class-top64-v1','exclusive all-class teacher required')
    require(t['source_sha256']==plan['source_sha256']==c['expected_runtime_sources'],'teacher runtime source identity')
    require(t['teacher_replay_receipt_sha256']==receipt_sha256,'actual teacher receipt binding')
    require(t['main_replay_admission_sha256']==c['main_replay_admission_sha256']==original['final_main_acceptance_sha256']==main_sha256,
        'teacher main acceptance differs')
    require(t['main_byte_admission_sha256']==c['main_byte_admission_sha256']==original['final_main_byte_admission_sha256']==byte_sha256,
        'teacher main byte admission differs')
    require(t['task_id']==c['task_id'] and t['recipe_sha256']==c['recipe_sha256'],'teacher task identity')
    require(t['recipe_sha256']==hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        'teacher registered plan differs')
    require(t['registered_artifacts']==c['registered_artifacts'],'teacher registered bytes differ')
    require(original['checkpoint']['sha256']==t['checkpoint_sha256'] and original['final_refit_model_sha256']==model,
        'teacher producer model differs')
    require(plan['cache_sha256']==original['cache_manifest']['sha256'],'teacher cache identity')
    require(t['scorer_signature']==plan['scorer_signature'] and t['frozen_cache_identity']==plan['model_binding']['frozen_cache_identity'],
        'scorer or upstream cache identity differs')
    prerequisite=t['main_prerequisite']
    require(prerequisite['kind']=='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1'
        and prerequisite['main_prerequisite_verified'] is True and prerequisite['teacher_runtime_or_targets_admitted'] is False,
        'independently admitted main prerequisite required')
    require(prerequisite['main_task_id']==m['task_id'] and prerequisite['seed']==seed
        and prerequisite['final_model_sha256']==model and prerequisite['main_acceptance_sha256']==main_sha256
        and prerequisite['byte_admission_sha256']==byte_sha256 and len(prerequisite['sequence_proof_sha256'])==46,
        'main prerequisite lineage differs')
    for key in ('full_real_teacher_target_admission','all_causal_features_and_counterfactual_targets_verified',
        'all_18_features_and_signed_targets_independently_recomputed','all_probe_operations_and_charged_steps_reconstructed',
        'all_203_feature_values_and_first_arrival_histories_verified','full_fresh_conditional_states_verified'):
        require(t[key] is True,'missing teacher target scope: '+key)
    require(type(t['labels']) is int and t['labels']==c['labels']>0,'all teacher labels required')
    for item in (t,c):require(item['GT_read'] is item['test_read'] is False,'GT/test input forbidden')
    require(t['target_atol']==t['target_rtol']==t['fresh_atol']==t['fresh_rtol']==m['atol']==m['rtol']==1e-8,
        'unchanged numerical tolerances required')
    for key in ('max_target_error','max_fresh_state_error'):require(math.isfinite(t[key]) and t[key]>=0,'invalid error summary')
    for value in (m['capacity_undecided_component_decisions'],t['capacity_undecided_component_decisions']):
        require(type(value) is int and value>=0,'capacity count required')
    require(m['conditional_full_legal_action_search_and_declared_gap_verified'] is (m['capacity_undecided_component_decisions']==0),
        'capacity undecided is not a searched optimum')
    require(t['capacity_undecided_counted_as_optimal'] is False and t['exact_identity_loss_or_posterior_optimum_target_claimed'] is False
        and m['full_model_posterior_optimum_regret_independently_computed'] is False,'model-bound labels cannot claim exact identity regret')
    require(t['strict_pipeline_isolated_selection'] is False,'this does not establish isolated upstream selection')


def verify(plan,receipt_sha256,*,main,main_sha256,byte,byte_sha256,target,target_sha256,collection_binding,collection_binding_sha256):
    m,mraw=load(main,main_sha256);b,braw=load(byte,byte_sha256);t,traw=load(target,target_sha256)
    c,craw=load(collection_binding,collection_binding_sha256)
    validate(plan,receipt_sha256,m,b,t,c,main_sha256=main_sha256,byte_sha256=byte_sha256)
    return {'main-independent-admission.json':mraw,'main-byte-admission.json':braw,
        'teacher-target-admission.json':traw,'teacher-collection-binding.json':craw}


def verify_exported(root,manifest):
    require(manifest['raw_probe_witness_recipe']==WITNESS_RECIPE and manifest['initial_search_witness_recipe']==INITIAL_RECIPE,
        'raw witness training recipe required')
    exports=manifest['exporter_source_sha256'];require(set(exports)==EXPORTERS,'complete exporter source binding')
    source_root=Path(__file__).resolve().parents[3]
    for name,expected in exports.items():require(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==expected,'export source changed')
    refs=manifest['independent_admission_files'];require(set(refs)==FILES,'all final-model admission files required')
    plan=manifest['independent_admission_plan'];binding=manifest['binding']
    require(dict(state=plan['configuration']['state'],**plan['configuration']['limits'])==binding['configuration']
        and binding['solver_backend']==BACKEND and plan['model_binding']['seed']==manifest['upstream_identity_seed']
        and plan['scorer_signature']==binding['factor_scorer_signature']
        and plan['model_binding']['frozen_cache_identity']==binding['frozen_cache_identity'],'exported training binding differs')
    verify_teacher_sources(plan['source_sha256'],manifest['source_sha256'])
    root=Path(root)
    return verify(plan,manifest['replay_receipt_sha256'],main=root/'main-independent-admission.json',
        main_sha256=refs['main-independent-admission.json'],byte=root/'main-byte-admission.json',byte_sha256=refs['main-byte-admission.json'],
        target=root/'teacher-target-admission.json',target_sha256=refs['teacher-target-admission.json'],
        collection_binding=root/'teacher-collection-binding.json',collection_binding_sha256=refs['teacher-collection-binding.json'])
