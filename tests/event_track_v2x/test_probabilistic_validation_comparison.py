"""Synthetic report objects test aggregation, never substitute for real val."""
import copy
from dataclasses import asdict

import pytest

from tools.event_track_v2x import compare_probabilistic_validation as tool


@pytest.fixture
def cohort():
    config={r:asdict(tool.PersistentProbabilisticConfig(update_rule=r)) for r in tool.evaluation.RULES}
    plan=dict(configuration_by_update_rule=config,source_sha256={'fixture':'source'},jobs=[],checkpoints=[])
    cells=[]
    for i,s in enumerate(tool.evaluation.SEEDS):
        cp=dict(seed=s,sha256=f'checkpoint-{s}',model_sha256=f'model-{s}',scorer_signature=f'scorer-{s}')
        plan['checkpoints'].append(cp)
        for j,r in enumerate(tool.evaluation.RULES):
            path=f'/fixture/seed-{s}-{r}'
            plan['jobs'].append(dict(seed=s,update_rule=r,output=path,arguments=dict(
                cache_sha256='cache',schedule_sha256='schedule',checkpoint_sha256=cp['sha256'],update_rule=r)))
            report=dict(seed=s,update_rule=r,configuration=config[r],run_directory=path,
                checkpoint_sha256=cp['sha256'],scorer_signature=cp['scorer_signature'],
                cache_sha256='cache',schedule_sha256='schedule',inference_source_sha256=plan['source_sha256'],
                protocol={'fixture':True},ground_truth_manifest_sha256='gt-manifest',ground_truth_sha256='gt',
                inference_runtime=dict(pid=100+len(cells),threads=1),factor_stream_sha256=f'factors-{s}',
                paper_eligible=False,fair_resources_verified=False,reproduced_public_method=False,
                same_state_time_protocol_as_recoverable=False,validation_checkpoint_selection=False,
                validation_parameter_search=False,same_input_selection_as_legacy_source_ablation=False)
            primary={m:.2+.001*i+.01*j for m in tool.METRICS}
            cells.append(dict(report=copy.deepcopy(report),primary=primary,
                evaluator_runtime={'fixture':'native'},sequences={f'{k:04d}':{m:primary[m] for m in ('HOTA','AssA','DetA','IDF1')} for k in range(21)}))
    return plan,cells


def test_complete_nine_cells_use_all_seeds_without_best_seed_choice(cohort):
    p,c=cohort;result=tool.assemble(p,c)
    hota=result['three_seed_descriptive']['jpda-ci']['HOTA']
    assert hota['mean']==pytest.approx(.201)
    assert hota['sample_sd']==pytest.approx(.001)
    assert set(hota['by_seed'])=={'1337','2027','3407'}
    assert result==tool.assemble(p,c[::-1])
    assert result['paired_delta_from_jpda_ci']['pkf']['2027']['primary']['HOTA']==pytest.approx(.02)
    assert result['metric_direction']['FP']=='lower'
    assert result['metric_direction']['HOTA']=='higher'
    assert result['sample_sd_is_not_confidence_interval']
    for k in ('best_seed_selected','hard_identity_invariance_verified','fair_resources_verified',
              'full_paper_comparison_completed','paper_eligible'):
        assert result[k] is False


@pytest.mark.parametrize('error',['missing','duplicate','missing_job','duplicate_job','model_duplicate',
    'checkpoint','scorer','source','cache','schedule','configuration','plan_config','path','runtime',
    'metric_runtime','protocol','gt','sequence','missing_metric','nan','factor','favourable_seed',
    'resource_claim','selection_claim','legacy_input_claim','job_rule','job_checkpoint'])
def test_incomplete_or_noncomparable_cells_are_rejected(cohort,error):
    p,c=copy.deepcopy(cohort);r=c[0]['report']
    if error=='missing':c.pop()
    elif error=='duplicate':c[0]=copy.deepcopy(c[1])
    elif error=='missing_job':p['jobs'].pop()
    elif error=='duplicate_job':p['jobs'][0]=copy.deepcopy(p['jobs'][1])
    elif error=='model_duplicate':p['checkpoints'][0]['model_sha256']=p['checkpoints'][1]['model_sha256']
    elif error=='checkpoint':r['checkpoint_sha256']='different'
    elif error=='scorer':r['scorer_signature']='different'
    elif error=='source':r['inference_source_sha256']={}
    elif error=='cache':r['cache_sha256']='different'
    elif error=='schedule':r['schedule_sha256']='different'
    elif error=='configuration':r['configuration']['state']['birth_score']=.4
    elif error=='plan_config':p['configuration_by_update_rule']['pkf']['max_scan_tracks']+=1
    elif error=='path':r['run_directory']+='/other'
    elif error=='runtime':r['inference_runtime']['threads']=2
    elif error=='metric_runtime':c[0]['evaluator_runtime']['fixture']='other'
    elif error=='protocol':r['protocol']={'legacy':True}
    elif error=='gt':r['ground_truth_sha256']='other'
    elif error=='sequence':c[0]['sequences'].pop('0000')
    elif error=='missing_metric':c[0]['primary'].pop('FN')
    elif error=='nan':c[0]['primary']['HOTA']=float('nan')
    elif error=='factor':r['factor_stream_sha256']='different'
    elif error=='favourable_seed':r['seed']=2027
    elif error=='resource_claim':r['fair_resources_verified']=True
    elif error=='selection_claim':r['validation_checkpoint_selection']=True
    elif error=='legacy_input_claim':r['same_input_selection_as_legacy_source_ablation']=True
    elif error=='job_rule':p['jobs'][0]['arguments']['update_rule']='pkf'
    elif error=='job_checkpoint':p['jobs'][0]['arguments']['checkpoint_sha256']='other'
    with pytest.raises(ValueError):tool.assemble(p,c)
