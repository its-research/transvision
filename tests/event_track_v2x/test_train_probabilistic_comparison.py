"""Fixed control provenance and numerical audit failures, not performance claims."""
import copy
from dataclasses import asdict

import numpy as np
import pytest

from tools.event_track_v2x import compare_train_probabilistic_controls as tool
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from transvision.models.event_track_v2x.jpda_marginals import lbp_jpda
from transvision.models.event_track_v2x.resource_sweep import configuration


@pytest.fixture
def controls():
    plan=dict(kind='train_sequence_inference_diagnostic_plan_v1',backend=tool.REFERENCE,
        configuration=asdict(configuration(dict(backend=tool.REFERENCE))),
        cache_sha256='cache',cooperative_metadata_sha256='metadata',identity_checkpoint_sha256='checkpoint',
        identity_seed=1337,scorer_signature='same',selected_sequence='s',selected_schedule=['s'],
        class_scope=['car'],cohort_mode='fixture-only',complete_input_train_cohort_verified=False,
        thread_environment={'threads':'one'},source_scope='same',source_sha256={'model':'same'},
        runtime=dict(pid=100,host='same'))
    reference=dict(backend=tool.REFERENCE,plan=plan,factor_stream_sha256='same',
        events=[dict(sequence_id='s',frame_id='f',reference_us=10,decision_us=11,factor_rows_sha256='same')],
        receipt_sha256='reference',source_directory='/fixture/reference')
    metric=dict(primary={'HOTA':.2,'IDF1':.3},gt_manifest_sha256='gt',protocol={'split':'train_development'},
                runtime={'native':'same'},evaluator_sha256='same')
    reports=[];metrics=[]
    for i,b in enumerate(tool.producer.BACKENDS):
        r=copy.deepcopy(reference);r.update(backend=b,receipt_sha256=b,source_directory='/fixture/'+b)
        r['plan'].update(backend=b,producer=tool.producer.PRODUCER,
            configuration=asdict(configuration(dict(backend=b))),same_state_time_protocol_as_recoverable=False,
            recovery_only_ablation=False,reproduced_public_method=False)
        r['plan']['runtime']['pid']=101+i
        r['plan']['source_sha256'][tool.OWN]=sha_file(tool.ROOT/tool.OWN)
        m=copy.deepcopy(metric);m['primary']['HOTA']+=.01*i
        reports.append(r);metrics.append(m)
    return reports,metrics,reference,metric


def test_all_three_fixed_controls_keep_adaptation_and_resource_limits(controls):
    result=tool.assemble(*controls,allow_fixture=True)
    assert result['actual_complete_factor_stream_identical']
    assert result['updater_only_difference_between_probabilistic_controls']
    assert result['primary_delta_variant_minus_reference']['pkf']['HOTA']==pytest.approx(.02)
    assert all(result[k] is False for k in ('same_state_time_protocol_as_beam','recovery_only_ablation',
        'reproduced_public_method','physically_equal_resources_claimed','validation','paper_eligible'))
    r,m,b,bm=controls
    assert result==tool.assemble(r[::-1],m[::-1],b,bm,allow_fixture=True)
    with pytest.raises(ValueError,match='provenance'):tool.assemble(*controls)


@pytest.mark.parametrize('error',['missing','duplicate','factor','event_factor','clock','scorer','schedule',
    'state','algorithm','decoder','budget','source','new_source','producer','gt','metric_runtime','metric_evaluator',
    'runtime','same_process','extra_knob','recovery_claim','reference_enabled'])
def test_noncomparable_control_or_missing_cell_is_rejected(controls,error):
    reports,metrics,reference,rm=controls;p=reports[0]['plan']
    if error=='missing':reports.pop()
    elif error=='duplicate':reports[1]=copy.deepcopy(reports[0])
    elif error=='factor':reports[0]['factor_stream_sha256']='changed'
    elif error=='event_factor':reports[0]['events'][0]['factor_rows_sha256']='changed'
    elif error=='clock':reports[0]['events'][0]['decision_us']+=1
    elif error=='scorer':p['scorer_signature']='changed'
    elif error=='schedule':p['selected_schedule']=['different']
    elif error=='state':p['configuration']['state']['birth_score']=.7
    elif error=='algorithm':p['configuration']['association_algorithm']='exact'
    elif error=='decoder':p['configuration']['anchor_decoder']='marginal-bayes'
    elif error=='budget':p['configuration']['inference']['max_iterations']+=1
    elif error=='source':p['source_sha256']['model']='changed'
    elif error=='new_source':p['source_sha256']['new']='unreviewed'
    elif error=='producer':p['producer']='other'
    elif error=='gt':metrics[0]['gt_manifest_sha256']='changed'
    elif error=='metric_runtime':metrics[0]['runtime']['native']='changed'
    elif error=='metric_evaluator':metrics[0]['evaluator_sha256']='changed'
    elif error=='runtime':p['runtime']['host']='changed'
    elif error=='same_process':p['runtime']['pid']=100
    elif error=='extra_knob':p['undocumented']=True
    elif error=='recovery_claim':p['recovery_only_ablation']=True
    else:reference['plan']['configuration']['enable_recovery']=True
    with pytest.raises(ValueError):tool.assemble(reports,metrics,reference,rm,allow_fixture=True)


@pytest.fixture
def scan():
    f=LogAssociationFactors([[1.,2.],[3.,4.]],[0.,0.],[-1.,-2.])
    return dict(conditioned_track_roots=[0,1],indices=[2,3],log_pair=f.log_pair,
        allowed=f.allowed,log_birth=f.log_right_unmatched,factors_sha256=f.digest(),
        marginals=asdict(lbp_jpda(f)),anchors=[dict(index=2,root=0),dict(index=3,root=1)])


def test_real_lbp_scan_normalization_audit_is_not_an_exact_posterior_claim(scan):
    config=asdict(configuration(dict(backend='jpda_ci')))
    result=tool.audit_scan(scan,config)
    assert result['message_updates']>0 and result['max_row_or_column_error']<=1e-10+1e-12
    assert scan['marginals']['log_partition'] is None


@pytest.mark.parametrize('error',['factor_hash','algorithm','partition','negative','nonfinite','unnormalized',
    'shape','stopping','negative_work','excess_work','duplicate_anchor','unknown_root'])
def test_forged_scan_normalization_solver_or_identity_claims_rejected(scan,error):
    q=scan['marginals'];config=asdict(configuration(dict(backend='jpda_ci')))
    if error=='factor_hash':scan['factors_sha256']='changed'
    elif error=='algorithm':q['algorithm']='exact'
    elif error=='partition':q['log_partition']=0.
    elif error in ('negative','nonfinite','unnormalized'):
        q['pair']=[list(row) for row in q['pair']]
        q['pair'][0][0]={'negative':-.1,'nonfinite':float('nan'),'unnormalized':.9}[error]
    elif error=='shape':q['left_unmatched']=[1.,2.,3.]
    elif error=='stopping':q['log_message_residual']=1.
    elif error=='negative_work':q['iterations']=-1
    elif error=='excess_work':q['message_updates']=config['inference']['max_message_updates']+1
    elif error=='duplicate_anchor':scan['anchors'][1]['root']=0
    else:scan['anchors'][1]['root']=999
    with pytest.raises(ValueError):tool.audit_scan(scan,config)


def test_empty_left_scan_audits_as_births_without_iterations():
    f=LogAssociationFactors(np.zeros((0,2)),[],[-1.,-2.],np.zeros((0,2),dtype=bool))
    scan=dict(conditioned_track_roots=[],indices=[0,1],log_pair=f.log_pair,allowed=f.allowed,
        log_birth=f.log_right_unmatched,factors_sha256=f.digest(),marginals=asdict(lbp_jpda(f)),
        anchors=[dict(index=0,root=0),dict(index=1,root=1)])
    result=tool.audit_scan(scan,asdict(configuration(dict(backend='pkf'))))
    assert result['iterations']==result['message_updates']==result['max_row_or_column_error']==0
