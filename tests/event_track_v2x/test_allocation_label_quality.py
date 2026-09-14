import copy

import pytest

from tools.event_track_v2x import audit_allocation_label_quality as tool


def step(targets, *, selected=0, feature_marker=0.):
    rows=[]
    for i,y in enumerate(targets):
        x=[1.]+[0.]*(len(tool.FEATURES)-1);x[7]=feature_marker
        rows.append(dict(component=i,features=x,target=y,model_bound_before=.5,model_bound_after=.5-y,
            charged_steps=1,operation=dict(kind='prefix_refinement',requested_steps=1)))
    return dict(component=selected,kind='prefix_refinement',charged_search_steps=1,
        allocation_training=dict(candidates=rows,feature_recipe=tool.RECIPE,target_recipe=tool.TARGET,
            behavior='weighted_eta_deterministic',labels_are_model_not_true_risk=True,future_or_gt_inputs=False))


def event(steps, *, key='0', executions=None, cache_hits=0):
    rows=sum(len(s['allocation_training']['candidates']) for s in steps)
    return dict(tracking=dict(sequence_id='s',event_id=key,beam_recovery_allocation=True,
        training_trace_only=True,offline_counterfactual_probes=True,
        allocation_trace_field='recovery_allocation_trace',priority_changes_only_extra_recovery_order=True,
        recovery_allocation_trace=steps,teacher_probe_executions=rows-cache_hits if executions is None else executions,
        teacher_probe_cache_hits=cache_hits,priority_feature_rows=rows))


def test_group_vs_row_weighting_and_exact_feature_alias_floor():
    p=tool.Profile()
    p.add(('s','0'),step([.2]))
    p.add(('s','0'),step([0.,-.2]))
    r=p.result()
    assert r['counts']['rows']==3 and r['counts']['groups']==2
    assert r['counts']['ranking_signal_groups']==1
    assert r['events_with_ranking_signal']==1
    assert r['zero_predictor_equal_group_mse']==pytest.approx(.03)
    # Weights per group: (1, .5, .5) / 2, weighted mean .05.
    assert r['empirical_exact_feature_equal_group_mse_floor']==pytest.approx(.0275)
    assert r['exact_numeric_feature_vectors']==1
    assert r['rows_in_conflicting_feature_vectors']==3
    assert r['counts']['positive_opportunity_groups']==0


def test_selected_behavior_opportunity_is_not_group_spread():
    p=tool.Profile();p.add(('s','0'),step([-.2,.1],selected=1));p.add(('s','1'),step([-.2,.1],selected=0))
    r=p.result()
    assert r['counts']['ranking_signal_groups']==2 and r['counts']['positive_opportunity_groups']==1
    assert r['one_step_opportunity_quantiles']['maximum']==pytest.approx(.3)
    assert r['one_step_positive_opportunity_group_fraction']==.5


def test_nonzero_constant_targets_do_not_supply_ranking_signal():
    p=tool.Profile();p.add(('s','0'),step([.1,.1,.1]));r=p.result()
    assert r['counts']['positive']==3 and r['counts']['ranking_signal_groups']==0
    assert r['empirical_exact_feature_equal_group_mse_floor']==0


def test_near_zero_is_descriptive_not_a_filter():
    p=tool.Profile();p.add(('s','0'),step([1e-14,-1e-14,0.]));r=p.result()
    assert r['counts']['rows']==3 and r['near_zero_row_fraction']==1
    assert r['counts']['ranking_signal_groups']==0
    assert r['zero_predictor_equal_group_mse']>0


def test_signed_zero_features_are_same_numeric_input():
    p=tool.Profile();p.add(('s','0'),step([.1],feature_marker=-0.));p.add(('s','1'),step([-.1],feature_marker=0.))
    r=p.result()
    assert r['exact_numeric_feature_vectors']==1 and r['excess_feature_repetitions']==1


def test_distinct_features_do_not_create_alias_floor():
    p=tool.Profile();p.add(('s','0'),step([.1],feature_marker=1.));p.add(('s','1'),step([-.1],feature_marker=2.))
    assert p.result()['empirical_exact_feature_equal_group_mse_floor']==0


def test_event_candidate_and_probe_counters_have_same_grain():
    local,total=tool.Profile(),tool.Profile()
    value=tool.profile_events([event([step([0.,.1]),step([.1])],cache_hits=2),event([],key='1')],[local,total])
    assert value['events']==2 and value['candidate_rows']==3
    assert value['probe_executions']==1 and value['probe_cache_hits']==2
    assert local.result()==total.result()


@pytest.mark.parametrize('error',['future','true_risk','recipe','target_recipe','duplicate_component','wrong_selected',
    'nan_target','bad_target','wrong_arithmetic','dimension','nan_feature','bad_weight','bad_before','bad_after',
    'negative_cost','excess_cost','zero_request','bool_cost','wrong_operation','wrong_committed_cost','empty'])
def test_invalid_group_rejected(error):
    s=step([0.,.1]);g=s['allocation_training'];r=g['candidates'][0]
    if error=='future':g['future_or_gt_inputs']=True
    elif error=='true_risk':g['labels_are_model_not_true_risk']=False
    elif error=='recipe':g['feature_recipe']='other'
    elif error=='target_recipe':g['target_recipe']='other'
    elif error=='duplicate_component':g['candidates'][1]['component']=0
    elif error=='wrong_selected':s['component']=4
    elif error=='nan_target':r['target']=float('nan')
    elif error=='bad_target':r['target']=2.
    elif error=='wrong_arithmetic':r['target']=.2
    elif error=='dimension':r['features'].pop()
    elif error=='nan_feature':r['features'][1]=float('inf')
    elif error=='bad_weight':r['features'][0]=1.1
    elif error=='bad_before':r['model_bound_before']=1.2
    elif error=='bad_after':r['model_bound_after']=-.1
    elif error=='negative_cost':r['charged_steps']=-1
    elif error=='excess_cost':r['charged_steps']=2
    elif error=='zero_request':r['operation']['requested_steps']=0
    elif error=='bool_cost':r['charged_steps']=True
    elif error=='wrong_operation':s['kind']='other'
    elif error=='wrong_committed_cost':s['charged_search_steps']=2
    else:g['candidates']=[]
    with pytest.raises(ValueError):tool.Profile().add(('s','0'),s)


@pytest.mark.parametrize('error',['duplicate_event','not_teacher','not_beam','wrong_trace','wrong_execution_count',
    'negative_cache_count','wrong_feature_count'])
def test_invalid_event_or_counter_rejected(error):
    e=event([step([.1])]);a=e['tracking'];rows=[e]
    if error=='duplicate_event':rows.append(copy.deepcopy(e))
    elif error=='not_teacher':a['training_trace_only']=False
    elif error=='not_beam':a['beam_recovery_allocation']=False
    elif error=='wrong_trace':a['allocation_trace_field']='allocation_trace'
    elif error=='wrong_execution_count':a['teacher_probe_executions']=2
    elif error=='negative_cache_count':a['teacher_probe_cache_hits']=-1
    else:a['priority_feature_rows']=0
    with pytest.raises(ValueError):tool.profile_events(rows,[tool.Profile()])


def test_empty_profile_is_missing_signal_not_nan():
    r=tool.Profile().result()
    assert r['near_zero_row_fraction'] is None and r['absolute_target_quantiles'] is None
    assert r['empirical_exact_feature_equal_group_mse_floor'] is None
