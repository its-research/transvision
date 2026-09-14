import pytest

from tools.event_track_v2x.diagnose_class_bound_failure import failure_context
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_class_bound_beam import PersistentClassBoundBeamConfig,PersistentClassBoundBeamTracker
from transvision.models.event_track_v2x.persistent_slot_bound_beam import PersistentSlotBoundBeamConfig,PersistentSlotBoundBeamTracker
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig,PersistentBeamTracker
from test_forest_tracking import observation
from test_persistent_component_tracking import step


def test_only_known_ranking_exception_frames_are_read():
    assert failure_context(ValueError('arbitrary error')) is None


def test_numeric_context_survives_normal_rollback_without_instrumentation(tmp_path):
    t=PersistentClassBoundBeamTracker(tmp_path/'failure.db',sequence_id='0003',
        config=PersistentClassBoundBeamConfig(state=ForestTrackingConfig(active_limit=1),max_batch_frontier=1))
    raw=[observation('a'),observation('b',source=1)]
    step(t,raw,[((-1,0.),),((-1,0.),)])
    before=tuple(t.db.iterdump())
    with pytest.raises(ValueError,match='frontier capacity') as error:
        step(t,[observation('c',frame='later',state_us=1_200_000)],
             [((-1,0.),(0,0.),(1,0.))],time=1_300_000,event='later')
    context=failure_context(error.value)
    assert context['start_depth']==2 and context['component_nodes']==3
    assert context['current_prefix_depth']==2 and context['next_choice_count']==3
    assert context['new_component_source_frame_counts']==[dict(source=0,frame='later',n=1)]
    assert context['ranking_operations']>0
    assert context['predecessor_widths']==[1,1]
    assert context['traceback_state_read_after_normal_rollback']
    assert tuple(t.db.iterdump())==before and t.n==2
    t.close()


def test_late_state_failure_exposes_counts_not_raw_payloads_and_rolls_back(tmp_path):
    t=PersistentBeamTracker(tmp_path/'states.db',sequence_id='0003',
        config=PersistentBeamConfig(state=ForestTrackingConfig(active_limit=1,max_replay_operations=3)))
    step(t,[observation('a')],[((-1,0.),)])
    before=tuple(t.db.iterdump())
    new=[observation('b',frame='b',source=1,state_us=900_000,arrival_us=1_200_000),
         observation('c',frame='c',source=0,state_us=800_000,arrival_us=1_200_000)]
    with pytest.raises(ValueError,match='branch-state work cap') as error:
        step(t,new,[((-1,-100.),(0,0.)),((-1,-100.),(1,0.))],time=1_300_000,event='late')
    context=failure_context(error.value)
    assert context['failure_stage']=='branch_state_replay' and context['late_raw_replay_active']
    assert context['state_updates']==4 and context['state_update_cap']==3
    assert context['pending_root_prefixes']==2 and not context['full_raw_payloads_recorded']
    assert tuple(t.db.iterdump())==before and t.n==1
    t.close()


def test_slot_work_failure_context_includes_rejected_charge_and_actual_component(tmp_path):
    t=PersistentSlotBoundBeamTracker(tmp_path/'slot.db',sequence_id='0003',
        config=PersistentSlotBoundBeamConfig(state=ForestTrackingConfig(active_limit=1),max_ranking_operations=120))
    step(t,[observation('a')],[((-1,0.),)])
    before=tuple(t.db.iterdump())
    raw=[observation(str(i),source=1,frame='later',index=i,state_us=1_200_000) for i in range(8)]
    with pytest.raises(ValueError,match='ranking work capacity') as error:
        step(t,raw,[((-1,0.),(0,0.))]*8,time=1_300_000,event='later')
    context=failure_context(error.value)
    assert context['source_frame_exclusion_in_ranking_bound']
    assert context['component'] is not None
    assert context['new_component_source_frame_counts']==[dict(source=1,frame='later',n=8)]
    assert context['current_assignment_rows']==8
    assert context['ranking_operations']<=context['ranking_operation_cap']==120
    assert context['ranking_operations']+context['rejected_ranking_work_amount']>120
    assert context['rejected_ranking_work_kind'] in ('assignment_matrix_cells','assignment_dual_cells')
    assert tuple(t.db.iterdump())==before and t.n==1
    t.close()
