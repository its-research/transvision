"""Control-matrix tests are synthetic; they never certify real experiments."""
from dataclasses import asdict
import copy
import hashlib
import json

import pytest

from tools.event_track_v2x import compare_train_risk_controls as tool
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_configuration


@pytest.fixture
def matrix():
    reports,metrics=[],[]
    old=next(s for s in tool.DRIVERS if s!=tool.CONTROL_DRIVER)
    for admission,backend in enumerate(tool.BACKENDS):
        for disable,threshold in enumerate(tool.THRESHOLDS):
            ordinal=len(reports)
            plan=dict(kind='train_sequence_inference_diagnostic_plan_v1',
                configuration=asdict(diagnostic_configuration(backend,threshold)),
                cache_sha256='cache',cooperative_metadata_sha256='metadata',identity_checkpoint_sha256='checkpoint',
                identity_seed=1337,scorer_signature='frozen',cohort_mode='fixture-only',
                selected_schedule=[dict(sequence_id='s',vehicle_frame='f',infrastructure_frame='i',box_reference_timestamp_us=10)],
                class_scope=['car'],thread_environment={'threads':'one'},source_scope='declared',
                source_sha256={tool.DRIVER:tool.CONTROL_DRIVER if disable else old,
                    'transvision/models/event_track_v2x/forest_tracking.py':'model',
                    'tools/event_track_v2x/run_persistent_forest_v2.py':'shared-replay'},
                runtime=dict(pid=100+ordinal),conditional_bayes_without_threshold_fallback=bool(disable),
                max_model_regret_override=threshold if disable else None,complete_input_train_cohort_verified=False)
            reports.append(dict(backend=backend,plan=plan,factor_stream_sha256='same',
                events=[dict(sequence_id='s',frame_id='f',reference_us=10,decision_us=11,factor_rows_sha256='same',
                    output_payload_sha256=str(ordinal),output_ids_sha256=str(ordinal))],
                fallback_events=0,fallback_unreported_events=0,receipt_sha256=f'receipt-{ordinal}',
                source_directory=f'/synthetic/{ordinal}'))
            metrics.append(dict(primary={'HOTA':.1+.2*admission+.1*disable+.05*admission*disable},
                gt_manifest_sha256='gt',protocol={'split':'train_development'},runtime={'native':'pinned'},
                evaluator_sha256=tool.METRIC_EVALUATOR))
    prior=dict(kind='train_inference_comparison_diagnostic_v1',analysis_complete=True,
        status='partial_due_to_failed_runs',actual_factors_identical=True,complete_backend_comparison=False,
        runs=[dict(backend=r['backend'],receipt_sha256=r['receipt_sha256']) for r in reports[::2]],
        failed_runs=[dict(backend='joint_beam',source_directory='/synthetic/failure',failure_receipt_sha256='failed',
            failure=dict(completed_frames=1,scheduled_frames=2,error='capacity'))])
    return reports,metrics,prior


def test_complete_factorial_controls_preserve_failures_and_limit_claims(matrix):
    result=tool.assemble(*matrix,allow_fixture=True)
    assert result['control_matrix_complete'] and result['cohort_mode']=='fixture-only'
    assert result['actual_complete_factor_stream_identical'] and result['model_sources_identical']
    assert len(result['contrasts'])==4 and result['descriptive_factor_interaction']['HOTA']==pytest.approx(.05)
    assert result['contrasts'][0]['primary_delta_variant_minus_reference']['HOTA']==pytest.approx(.1)
    assert result['contrasts'][1]['primary_delta_variant_minus_reference']['HOTA']==pytest.approx(.15)
    assert result['prior_failed_backends'][0]['backend']=='joint_beam'
    assert not any(result[k] for k in ('strong_baseline_comparison_complete','recovery_only_ablation',
        'physically_equal_resources_claimed','validation','paper_eligible','statistical_inference','metrics_recomputed'))


def test_fixture_cannot_be_relabelled_by_formal_comparison(matrix):
    with pytest.raises(ValueError,match='provenance'): tool.assemble(*matrix)


@pytest.mark.parametrize('error',['missing','duplicate','factor_digest','actual_factor','clock','cohort','scorer',
    'model_source','driver','source_inventory','birth_score','budget','coverage_version','threshold',
    'fallback_flag','fallback_observed','fallback_unknown','gt','protocol','runtime','evaluator',
    'failed_omitted','prior_replaced','extra_plan_knob','inference_environment','default_label'])
def test_unintended_control_changes_or_failed_baseline_omission_are_rejected(matrix,error):
    reports,metrics,prior=copy.deepcopy(matrix);plan=reports[1]['plan']
    if error=='missing': reports.pop()
    elif error=='duplicate': reports[-1]=copy.deepcopy(reports[0])
    elif error=='factor_digest': reports[1]['factor_stream_sha256']='different'
    elif error=='actual_factor': reports[1]['events'][0]['factor_rows_sha256']='different'
    elif error=='clock': reports[1]['events'][0]['decision_us']=12
    elif error=='cohort': plan['selected_schedule'][0]['vehicle_frame']='different'
    elif error=='scorer': plan['scorer_signature']='different'
    elif error=='model_source': plan['source_sha256']['transvision/models/event_track_v2x/forest_tracking.py']='different'
    elif error=='driver': plan['source_sha256'][tool.DRIVER]='unreviewed'
    elif error=='source_inventory': plan['source_sha256']['new-model.py']='different'
    elif error=='birth_score': plan['configuration']['state']['birth_score']=.4
    elif error=='budget': plan['configuration']['state']['expansion_budget']+=1
    elif error=='coverage_version': reports[2]['plan']['configuration']['coverage_admission_version']=2
    elif error=='threshold': plan['configuration']['state']['max_model_regret']=.9
    elif error=='fallback_flag': plan['conditional_bayes_without_threshold_fallback']=False
    elif error=='fallback_observed': reports[1]['fallback_events']=1
    elif error=='fallback_unknown': reports[1]['fallback_unreported_events']=1
    elif error=='gt': metrics[1]['gt_manifest_sha256']='different'
    elif error=='protocol': metrics[1]['protocol']['split']='val'
    elif error=='runtime': metrics[1]['runtime']['native']='different'
    elif error=='evaluator': metrics[1]['evaluator_sha256']='different'
    elif error=='failed_omitted': prior['failed_runs']=[]
    elif error=='extra_plan_knob': plan['allocation_policy_signature']='unadvertised'
    elif error=='inference_environment': plan['runtime']['torch']='different'
    elif error=='default_label': reports[0]['plan']['conditional_bayes_without_threshold_fallback']=True
    else: prior['runs'][0]['receipt_sha256']='another-run'
    with pytest.raises(ValueError): tool.assemble(reports,metrics,prior,allow_fixture=True)


def test_reordered_input_cells_do_not_change_named_contrasts(matrix):
    reports,metrics,prior=matrix
    expected=tool.assemble(*matrix,allow_fixture=True)
    actual=tool.assemble(list(reversed(reports)),list(reversed(metrics)),prior,allow_fixture=True)
    assert expected['contrasts']==actual['contrasts']
    assert expected['descriptive_factor_interaction']==actual['descriptive_factor_interaction']


def test_unknown_third_knob_is_not_silently_dropped(matrix):
    reports,metrics,prior=matrix
    reports[1]['plan']['configuration']['unadvertised_switch']=True
    with pytest.raises(ValueError,match='third configuration factor'):
        tool.assemble(reports,metrics,prior,allow_fixture=True)


@pytest.mark.parametrize('changed',[None,'predictions.jsonl','tracking.jsonl','receipt.json'])
def test_metric_binding_accepts_genuinely_empty_predictions_but_not_resealed_wrong_inputs(tmp_path,monkeypatch,changed):
    """The native engines are tested separately; this fixture tests binding only."""
    def save(path,value):
        raw=tool.canonical(value);path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    receipt=dict(tracking_sha256='tracking',plan_sha256='plan')
    receipt_sha=save(tmp_path/'receipt.json',receipt)
    outer_sha=save(tmp_path/'development-inference-receipt.json',dict(replay_receipt_sha256=receipt_sha))
    report=dict(source_directory=str(tmp_path),receipt_sha256=outer_sha,backend=tool.BACKENDS[0],
        events=[{}],total_output_boxes=0,predictions_sha256='predictions',
        fallback_events=0,fallback_unreported_events=0,high_bound_events=0)
    inputs={str(tmp_path/name):digest for name,digest in (
        ('predictions.jsonl','predictions'),('tracking.jsonl','tracking'),('plan.json','plan'),
        ('development-inference-receipt.json',outer_sha),('receipt.json',receipt_sha))}
    labels={k:{'car':0.} for k in ('amota','amotp','mota','ids','frag','fp','fn')}
    summary={k:0. for k in ('HOTA','AssA','DetA','IDF1')}
    value=dict(kind='spd_train_sequence_development_metrics_v1',status='complete',backend=report['backend'],
        validation=False,paper_eligible=False,full_official_train=False,inference_receipt_sha256=outer_sha,
        evaluator_sha256=tool.METRIC_EVALUATOR,protocol=dict(split='train_development',evaluated_classes=['car']),
        counts=dict(frames=1,sequences=1,predictions_per_class={}),input_sha256=inputs,runtime={'fixture':True},
        metrics=dict(nuscenes={'label_metrics':labels},trackeval={'car':{'summary':summary}}),
        identity_event_diagnostics={'counts':dict(fallback_events=0,fallback_unreported_events=0,high_model_bound_events=0)},
        gt_manifest_sha256='gt')
    if changed: value['input_sha256'][str(tmp_path/changed)]='different-input'
    path=tmp_path/'metrics.json';digest=save(path,value)
    checked=[];monkeypatch.setattr(tool,'validate_runtime',lambda v:checked.append(v))
    if changed:
        with pytest.raises(ValueError,match='bind'): tool.read_metrics(path,digest,report)
        assert checked==[]
    else:
        result=tool.read_metrics(path,digest,report)
        assert result['primary']['HOTA']==0. and checked==[{'fixture':True}]
