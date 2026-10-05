"""Synthetic receipt controls only; no training or experiment acceptance."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def code(monkeypatch):
    root=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module('rbf_final_priority_admission')


def example(module):
    seqs=[f'{i:04d}' for i in range(46)];model='1'*64;checkpoint='2'*64
    configuration=dict(state=dict(candidate_protocol='rbf-all-class-top64-v1'),limits=dict(residual_partition_version=1),
        backend=module.BACKEND,method='rbf',allocation='teacher')
    sources={'runtime.py':'3'*64};mainsha='4'*64;bytesha='5'*64
    origin=dict(configuration=configuration,final_main_acceptance_sha256=mainsha,final_main_byte_admission_sha256=bytesha,
        checkpoint=dict(sha256=checkpoint),final_refit_model_sha256=model,cache_manifest=dict(sha256='6'*64))
    recipe=hashlib.sha256(json.dumps(origin,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    c=dict(seed=1337,fixture=False,final_model_sha256=model,checkpoint_sha256=checkpoint,original_plan=origin,
        expected_runtime_sources=sources,main_replay_admission_sha256=mainsha,main_byte_admission_sha256=bytesha,
        task_id='teacher',recipe_sha256=recipe,registered_artifacts={'software-only':{}},labels=10,GT_read=False,test_read=False)
    plan=dict(fixture=False,configuration=configuration,model_binding=dict(seed=1337,model_sha256=model,
        checkpoint_sha256=checkpoint,dataset='spd',fit_split='train',frozen_cache_identity={'fixture':True}),
        protocol=dict(dataset='spd',split='train'),expected_sequences=seqs,expected_events=7445,
        source_sha256=sources,cache_sha256='6'*64,scorer_signature='7'*64)
    m=dict(kind=module.MAIN_KIND,driver_freeze_sha256=module.MAIN_FREEZE,seed=1337,task_id='main',recipe_sha256='8'*64,
        byte_admission_sha256=bytesha,completed_sequences=46,completed_events=7445,final_model_sha256=model,
        original_model_sha256='9'*64,old_model_full_forest_acceptance_inherited=False,checkpoint_sha256=checkpoint,
        atol=1e-8,rtol=1e-8,capacity_undecided_component_decisions=1,
        conditional_full_legal_action_search_and_declared_gap_verified=False,
        full_model_posterior_optimum_regret_independently_computed=False)
    for key in ('strict_declared_support_partition_coverage','padded_float64_mass_arithmetic_verified',
        'output_class_recovery_provenance_verified','all_raw_factor_and_causal_commits_verified',
        'all_decisions_search_or_capacity_undecided_semantics_verified','all_203_feature_recipe_values_independently_verified',
        'all_scorer_contexts_bound_to_original_causal_raw_history'):m[key]=True
    b=dict(kind=module.BYTE_KIND,seed=1337,task_id='main',recipe_sha256=m['recipe_sha256'],method='rbf',
        final_model_sha256=model,final_checkpoint_sha256=checkpoint,all_registered_bytes_verified=True,
        all_46_sequences_7445_events_and_final_model_factor_nodes_verified=True,
        sequences=[dict(sequence_id=s,events=162 if i<39 else 161) for i,s in enumerate(seqs)])
    # 39*162 + 7*161 = 7445, solely a metadata boundary fixture.
    t=dict(kind=module.TARGET_KIND,source_freeze_sha256=module.TARGET_FREEZE,
        unchanged_target_driver_freeze_sha256=module.ORIGINAL_NUMERICAL_FREEZE,seed=1337,
        final_model_sha256=model,checkpoint_sha256=checkpoint,completed_sequences=seqs,completed_events=7445,
        configuration=configuration,source_sha256=sources,teacher_replay_receipt_sha256='a'*64,
        main_replay_admission_sha256=mainsha,main_byte_admission_sha256=bytesha,task_id=c['task_id'],
        recipe_sha256=recipe,registered_artifacts=c['registered_artifacts'],scorer_signature=plan['scorer_signature'],
        frozen_cache_identity=plan['model_binding']['frozen_cache_identity'],labels=10,GT_read=False,test_read=False,
        target_atol=1e-8,target_rtol=1e-8,fresh_atol=1e-8,fresh_rtol=1e-8,max_target_error=0.,max_fresh_state_error=0.,
        capacity_undecided_component_decisions=2,capacity_undecided_counted_as_optimal=False,
        exact_identity_loss_or_posterior_optimum_target_claimed=False,strict_pipeline_isolated_selection=False,
        main_prerequisite=dict(kind='rbf_final_refit_main_prerequisite_for_capacity_teacher_v1',main_prerequisite_verified=True,
            teacher_runtime_or_targets_admitted=False,main_task_id='main',seed=1337,final_model_sha256=model,
            main_acceptance_sha256=mainsha,byte_admission_sha256=bytesha,
            sequence_proof_sha256={s:'b'*64 for s in seqs}))
    for key in ('full_real_teacher_target_admission','all_causal_features_and_counterfactual_targets_verified',
        'all_18_features_and_signed_targets_independently_recomputed','all_probe_operations_and_charged_steps_reconstructed',
        'all_203_feature_values_and_first_arrival_histories_verified','full_fresh_conditional_states_verified'):t[key]=True
    return plan,'a'*64,m,b,t,c,dict(main_sha256=mainsha,byte_sha256=bytesha)


def test_metadata_gate_preserves_capacity_and_scope(code):
    p,r,m,b,t,c,hashes=example(code)
    code.validate(p,r,m,b,t,c,**hashes)
    assert t['capacity_undecided_counted_as_optimal'] is False
    assert t['strict_pipeline_isolated_selection'] is False


@pytest.mark.parametrize('mutation',['old-kind','old-model','wrong-checkpoint','changed-source','partial-events',
    'wrong-main','new-configuration','modified-plan','relaxed-tolerance','NaN-error','wrong-seed',
    'capacity-optimum','isolated-claim','test-input','missing-target-scope'])
def test_mixed_or_incomplete_admission_is_refused(code,mutation):
    p,r,m,b,t,c,hashes=copy.deepcopy(example(code))
    if mutation=='old-kind':t['kind']='rbf_capacity_undecided_teacher_full_independent_model_bound_target_admission_v1'
    elif mutation=='old-model':t['final_model_sha256']=m['original_model_sha256']
    elif mutation=='wrong-checkpoint':b['final_checkpoint_sha256']='wrong'
    elif mutation=='changed-source':t['source_sha256']={'changed.py':'c'*64}
    elif mutation=='partial-events':b['sequences'][-1]['events']-=1
    elif mutation=='wrong-main':t['main_prerequisite']['main_task_id']='different'
    elif mutation=='new-configuration':t['configuration']={'method':'topk'}
    elif mutation=='modified-plan':c['original_plan']['world_size']=8
    elif mutation=='relaxed-tolerance':t['target_atol']=1e-4
    elif mutation=='NaN-error':t['max_target_error']=float('nan')
    elif mutation=='wrong-seed':t['seed']=2027
    elif mutation=='capacity-optimum':m['conditional_full_legal_action_search_and_declared_gap_verified']=True
    elif mutation=='isolated-claim':t['strict_pipeline_isolated_selection']=True
    elif mutation=='test-input':t['test_read']=True
    elif mutation=='missing-target-scope':t['all_18_features_and_signed_targets_independently_recomputed']=False
    with pytest.raises(ValueError):code.validate(p,r,m,b,t,c,**hashes)


def test_two_training_only_modules_do_not_relax_runtime_binding(code):
    actual={'runtime.py':'a'*64};consumer=dict(actual,**{name:'b'*64 for name in code.TRAINING_ONLY})
    roles=code.verify_teacher_sources(actual,consumer)
    assert set(roles['training_only_sources'])==code.TRAINING_ONLY
    assert roles['teacher_runtime_sources']==actual
    with pytest.raises(ValueError):code.verify_teacher_sources({'runtime.py':'c'*64},consumer)
    with pytest.raises(ValueError):code.verify_teacher_sources(actual,dict(consumer,unbound_runtime='b'*64))
    with pytest.raises(ValueError):code.verify_teacher_sources(consumer,consumer)


def test_exporter_revision_retains_stream_and_model_math(code):
    builder=importlib.import_module('prepare_final_refit_priority_export')
    original=(builder.PARENT/'candidate-v7'/builder.EXPORTER).read_text()
    rendered=builder.render_exporter(original)
    assert 'verify_teacher_sources' in rendered and "receipt['files']['collection-binding.json']" in rendered
    # Complete export payload/coverage/label loops are inherited verbatim;
    # only the three exact provenance callsites change.
    for start,end in [('    groups = {}','    from transvision.models.event_track_v2x.exclusive_priority_admission import verify\n'),
                      ('    verified_target =','def main():')]:
        assert original[original.index(start):original.index(end)]==rendered[rendered.index(start):rendered.index(end)]
    trainer=builder.PARENT/'candidate-v7'/builder.TRAINER
    assert hashlib.sha256(trainer.read_bytes()).hexdigest()==builder.TRAINER_SHA
    with pytest.raises(AssertionError):builder.render_exporter(rendered)
