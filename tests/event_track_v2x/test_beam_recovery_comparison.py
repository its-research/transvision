"""Evidence and control validity, not synthetic claims of model performance."""
import copy
from dataclasses import asdict
import json

import pytest

from tools.event_track_v2x import compare_beam_recovery_controls as tool
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_configuration
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file


@pytest.fixture
def controls():
    reports=[];metrics=[]
    for index,backend in enumerate((tool.OFF,tool.ON)):
        plan=dict(kind='train_sequence_inference_diagnostic_plan_v1',backend=backend,
            configuration=asdict(diagnostic_configuration(backend)),cohort_mode='fixture-only',
            complete_input_train_cohort_verified=False,class_scope=['car'],cache_sha256='cache',
            identity_checkpoint_sha256='checkpoint',scorer_signature='same',selected_schedule=['same'],
            source_sha256={tool.BEAM_SOURCE:'beam','other-model':'same'},runtime=dict(pid=100+index,host='same'))
        reports.append(dict(backend=backend,plan=plan,events=[dict(sequence_id='s',frame_id='f',
            reference_us=10,decision_us=11,factor_rows_sha256='same',
            output_payload_sha256=str(index),output_ids_sha256=str(index))],
            fallback_events=0,fallback_unreported_events=0,factor_stream_sha256='same',
            receipt_sha256=str(index),predictions_sha256=str(index),source_directory='/fixture/'+backend))
        metrics.append(dict(primary=dict(HOTA=.2+.1*index),gt_manifest_sha256='gt',
            protocol={'split':'train_development'},runtime={'native':'same'},evaluator_sha256='same'))
    node=copy.deepcopy(reports[0]);node['backend']='node_beam';node['plan']['backend']='node_beam'
    node['plan']['configuration']=asdict(diagnostic_configuration('node_beam'))
    return reports,metrics,node


def test_toggle_only_complete_comparison_preserves_resource_and_evidence_limits(controls):
    result=tool.assemble(*controls,allow_fixture=True)
    assert result['primary_delta_on_minus_off']['HOTA']==pytest.approx(.1)
    assert result['disabled_predictions_byte_identical_to_original']
    assert result['actual_complete_factor_stream_identical'] and result['shared_node_beam_algorithm']
    assert result['extra_recovery_compute_included'] and result['changed_output_frames']==1
    assert not any(result[k] for k in ('physically_equal_resources_claimed','strong_baseline_comparison_complete',
        'validation','full_official_train','three_seed_comparison','statistical_inference','paper_eligible'))
    reports,metrics,node=controls
    assert result==tool.assemble(reports[::-1],metrics[::-1],node,allow_fixture=True)


def test_fixture_cannot_be_relabelled_as_real(controls):
    with pytest.raises(ValueError,match='provenance'):tool.assemble(*controls)


def test_only_pinned_offline_inventory_addition_is_recorded_without_rewriting(controls):
    reports,metrics,node=controls
    path,digest=next(iter(tool.REVIEWED_NONINFERENCE_ADDITIONS.items()))
    reports[0]['plan']['source_sha256'][path]=digest
    before=copy.deepcopy(controls)
    result=tool.assemble(*controls,allow_fixture=True)
    assert result['reviewed_noninference_source_additions']=={tool.OFF:{path:digest},tool.ON:{}}
    assert not result['source_inventories_byte_identical']
    assert result['shared_bound_source_hashes_identical']
    assert not result['historical_receipts_modified']
    assert controls==before
    assert result==tool.assemble(reports[::-1],metrics[::-1],node,allow_fixture=True)


@pytest.mark.parametrize('error',['unknown_addition','wrong_addition_hash','common_changed','missing_dependency'])
def test_inventory_normalization_does_not_hide_dependency_changes(controls,error):
    reports,metrics,node=controls
    path,digest=next(iter(tool.REVIEWED_NONINFERENCE_ADDITIONS.items()))
    sources=reports[0]['plan']['source_sha256']
    if error=='unknown_addition':sources['new_model.py']='new'
    elif error=='wrong_addition_hash':sources[path]='changed'
    elif error=='common_changed':
        sources[path]=digest;reports[1]['plan']['source_sha256'][path]='changed'
    else:sources.pop('other-model')
    with pytest.raises(ValueError,match='source'):
        tool.assemble(reports,metrics,node,allow_fixture=True)


@pytest.mark.parametrize('error',['missing','duplicate','switch','budget','hidden_knob','source','actual_factor',
    'factor_digest','clock','fallback','missing_fallback','gt','protocol','metric_runtime','evaluator',
    'node_backend','node_prediction','node_configuration','node_model','node_input','same_process'])
def test_hidden_changes_and_failed_original_reproduction_rejected(controls,error):
    reports,metrics,node=copy.deepcopy(controls);plan=reports[1]['plan']
    if error=='missing':reports.pop()
    elif error=='duplicate':reports[1]=reports[0]
    elif error=='switch':plan['configuration']['enable_recovery']=False
    elif error=='budget':plan['configuration']['recovery_budget']+=1
    elif error=='hidden_knob':plan['hidden_knob']=True
    elif error=='source':plan['source_sha256']['other-model']='changed'
    elif error=='actual_factor':reports[1]['events'][0]['factor_rows_sha256']='changed'
    elif error=='factor_digest':reports[1]['factor_stream_sha256']='changed'
    elif error=='clock':reports[1]['events'][0]['decision_us']+=1
    elif error=='fallback':reports[1]['fallback_events']=1
    elif error=='missing_fallback':reports[1]['fallback_unreported_events']=1
    elif error=='gt':metrics[1]['gt_manifest_sha256']='changed'
    elif error=='protocol':metrics[1]['protocol']['split']='val'
    elif error=='metric_runtime':metrics[1]['runtime']['native']='changed'
    elif error=='evaluator':metrics[1]['evaluator_sha256']='changed'
    elif error=='node_backend':node['backend']='joint_beam'
    elif error=='node_prediction':node['predictions_sha256']='changed'
    elif error=='node_configuration':node['plan']['configuration']['state']['birth_score']=.999
    elif error=='node_model':node['plan']['source_sha256'][tool.BEAM_SOURCE]='changed'
    elif error=='node_input':node['plan']['identity_checkpoint_sha256']='changed'
    else:plan['runtime']['pid']=100
    with pytest.raises(ValueError):tool.assemble(reports,metrics,node,allow_fixture=True)


@pytest.mark.parametrize('error',[None,'switch','budget','trace','support','negative_work','disabled_work'])
def test_resource_audit_is_read_from_actual_output_and_rejects_conflicts(tmp_path,controls,error):
    report=copy.deepcopy(controls[0][0 if error=='disabled_work' else 1])
    enabled=report['backend']==tool.ON
    report.update(source_directory=str(tmp_path),latency_seconds_p50_p95_p99_max=[1.,1.,1.,1.],
                  elapsed_seconds=1.,process_peak_rss_bytes=1024,database_bytes=100)
    flags=dict(frontier_completion_enabled=enabled,coverage_aware_proposal_admission=enabled,
        beam_recovery_backbone_enabled=True,additional_beam_recovery_enabled=enabled,
        irreversible_beam_enabled=not enabled)
    (tmp_path/'receipt.json').write_bytes(canonical(flags))
    (tmp_path/'development-inference-receipt.json').write_bytes(canonical(
        dict(replay_receipt_sha256=sha_file(tmp_path/'receipt.json'))))
    report['receipt_sha256']=sha_file(tmp_path/'development-inference-receipt.json')
    row=dict(recovery_enabled=enabled,recovery_search_steps=1,recovery_allocation_trace=[dict(charged_search_steps=1)],
        components=[dict(recovery_events=[],complete_raw_support_retained=True,recovery_cover_restarted_from_root=True)],
        recovery_extra_state_updates=2)
    if error=='switch':row['recovery_enabled']=False
    elif error=='budget':row['recovery_search_steps']=257
    elif error=='trace':row['recovery_allocation_trace'][0]['charged_search_steps']=0
    elif error=='support':row['components'][0]['complete_raw_support_retained']=False
    elif error=='negative_work':row['recovery_extra_state_updates']=-1
    (tmp_path/'tracking.jsonl').write_bytes(canonical({'tracking':row})+b'\n')
    if error:
        with pytest.raises(ValueError):tool.read_work(report)
    else:
        result=tool.read_work(report)
        assert result['recovery_steps']==1 and result['extra_state_updates']==2 and result['cover_restarts']==1
        assert not result['exclusive_host'] and not result['timing_repeated']
