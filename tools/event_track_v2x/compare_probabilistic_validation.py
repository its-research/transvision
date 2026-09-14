#!/usr/bin/env python3
"""All-nine fixed SPD val cells or no comparison; descriptive three-seed summary.

Only the three declared probabilistic adaptations are compared. No best-seed
selection, failed-cell omission, public-method or equal-resource claims.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x import evaluate_probabilistic_validation as evaluation
from tools.event_track_v2x import evaluate_source_ablation_v2 as native
from transvision.models.event_track_v2x.persistent_probabilistic_tracking import PersistentProbabilisticConfig

METRICS=dict(native.PRIMARY_METRICS,FP=('nuscenes','fp'),FN=('nuscenes','fn'))
LOWER={'AMOTP_m','FP','FN','IDS','Frag'}


def primary_vector(primary):
    return {name:native._finite(primary[group][key]) for name,(group,key) in METRICS.items()}


def load_report(path, expected_sha):
    path=Path(path).absolute()
    if native.sha(path)!=expected_sha:raise ValueError('metric report identity differs')
    report=native.read_json(path)
    if report.get('kind')!=evaluation.KIND or report.get('status')!='complete':
        raise ValueError('complete probabilistic evaluation report required')
    bound=evaluation.inspect_run(report['run_directory'],report['inference_receipt_sha256'],
        report['inference_audit_path'],report['inference_audit_sha256'])
    plan,receipt=bound['plan'],bound['receipt']
    for name,expected in (('seed',plan['checkpoint_seed']),('update_rule',plan['configuration']['update_rule']),
            ('configuration',plan['configuration']),('inference_runtime',plan['runtime']),
            ('cache_sha256',plan['cache_sha256']),('schedule_sha256',plan['schedule_sha256']),
            ('checkpoint_sha256',plan['checkpoint_sha256']),('scorer_signature',plan['scorer_signature']),
            ('inference_source_sha256',plan['source_sha256']),('predictions_sha256',receipt['predictions_sha256']),
            ('factor_stream_sha256',bound['audit']['factor_stream_sha256'])):
        if report.get(name)!=expected:raise ValueError('evaluation-to-inference binding differs: '+name)
    expected_protocol=evaluation.protocol(native.load_adapter())
    if (report.get('protocol')!=expected_protocol or report.get('protocol_sha256')!=hashlib.sha256(native.canonical(expected_protocol)).hexdigest()
            or report.get('ground_truth_manifest_sha256')!=native.GT_MANIFEST_SHA256
            or report.get('ground_truth_sha256')!=native.GT_SHA256
            or report.get('coverage',{}).get('frames')!=3316 or report['coverage'].get('sequences')!=21):
        raise ValueError('fixed full-val car metric protocol differs')
    payloads={}
    if set(report.get('files',{}))!={'metrics.json','runtime.json','golden-cases.json'}:
        raise ValueError('complete metric, runtime and golden-case artifacts required')
    evidence=dict(bound['evidence'])
    for name,record in report['files'].items():
        p=evaluation.child(path.parent,name)
        if native.evidence(p)!=record:raise ValueError('metric artifact changed')
        payloads[name]=native.read_json(p);evidence[str(p)]=record
    native.validate_runtime(payloads['runtime.json'])
    if (payloads['golden-cases.json'].get('passed') is not True
            or native.car_metrics(payloads['metrics.json'])!=report['primary_car']):
        raise ValueError('metric summary differs or golden cases failed')
    expected_inputs=dict(bound['evidence'])
    for p in (Path(evaluation.__file__),Path(native.__file__),native.ADAPTER_PATH,
              evaluation.child(report['ground_truth_directory'],'manifest.json'),
              evaluation.child(report['ground_truth_directory'],'ground-truth.jsonl')):
        expected_inputs[str(p.absolute())]=native.evidence(p)
    if report.get('input_evidence')!=expected_inputs:raise ValueError('evaluation input inventory differs')
    evidence.update(expected_inputs);evidence[str(path)]=native.evidence(path)
    evaluation.unchanged(evidence)
    return dict(report=report,primary=primary_vector(report['primary_car']),
        sequences=native.sequence_vectors(payloads['metrics.json']),
        evaluator_runtime=payloads['runtime.json'],report_sha256=expected_sha,evidence=evidence)


def assemble(campaign, cells):
    expected={(s,r) for s in evaluation.SEEDS for r in evaluation.RULES}
    keys=[(c['report']['seed'],c['report']['update_rule']) for c in cells]
    jobs=campaign.get('jobs',[])
    job_keys=[(j['seed'],j['update_rule']) for j in jobs]
    if len(cells)!=9 or set(keys)!=expected or len(jobs)!=9 or set(job_keys)!=expected:
        raise ValueError('all nine distinct frozen seed/method cells required; no failure omission')
    checkpoints={c['seed']:c for c in campaign['checkpoints']}
    if (len(campaign['checkpoints'])!=3 or set(checkpoints)!=set(evaluation.SEEDS)
            or len({c['model_sha256'] for c in checkpoints.values()})!=3):
        raise ValueError('three distinct frozen trained identity models required')
    jobs={key:job for key,job in zip(job_keys,jobs)}
    mapping={key:cell for key,cell in zip(keys,cells)}
    reference=cells[0]; first=reference['report']
    runtime={k:v for k,v in first['inference_runtime'].items() if k!='pid'}
    for key,cell in mapping.items():
        r=cell['report'];job=jobs[key];cp=checkpoints[key[0]]
        config=asdict(PersistentProbabilisticConfig(update_rule=key[1]))
        if (r['configuration']!=config or campaign['configuration_by_update_rule'].get(key[1])!=config
                or r['run_directory']!=job['output'] or r['checkpoint_sha256']!=cp['sha256']
                or r['scorer_signature']!=cp['scorer_signature']
                or r['inference_source_sha256']!=campaign['source_sha256']
                or r['cache_sha256']!=job['arguments']['cache_sha256']
                or r['schedule_sha256']!=job['arguments']['schedule_sha256']
                or job['arguments']['update_rule']!=key[1]
                or job['arguments']['checkpoint_sha256']!=cp['sha256']):
            raise ValueError('result differs from fixed campaign input, configuration or model')
        if (any(r[k]!=first[k] for k in ('cache_sha256','schedule_sha256','protocol',
                'ground_truth_manifest_sha256','ground_truth_sha256'))
                or {k:v for k,v in r['inference_runtime'].items() if k!='pid'}!=runtime
                or cell['evaluator_runtime']!=reference['evaluator_runtime']
                or set(cell['sequences'])!=set(reference['sequences'])):
            raise ValueError('common input cohort or inference/evaluator environment differs')
        if set(cell['primary'])!=set(METRICS):raise ValueError('complete primary metrics required')
        for value in cell['primary'].values():native._finite(value)
        for flag in ('paper_eligible','fair_resources_verified','reproduced_public_method',
                     'same_state_time_protocol_as_recoverable','validation_checkpoint_selection','validation_parameter_search',
                     'same_input_selection_as_legacy_source_ablation'):
            if r.get(flag) is not False:raise ValueError('adaptation or selection boundary changed')
    for seed in evaluation.SEEDS:
        if len({mapping[(seed,r)]['report']['factor_stream_sha256'] for r in evaluation.RULES})!=1:
            raise ValueError('actual factors differ across methods within a seed')
    descriptive={}
    for rule in evaluation.RULES:
        descriptive[rule]={}
        for metric in METRICS:
            values={str(s):mapping[(s,rule)]['primary'][metric] for s in evaluation.SEEDS}
            descriptive[rule][metric]=dict(mean=statistics.fmean(values.values()),sample_sd=statistics.stdev(values.values()),
                minimum=min(values.values()),maximum=max(values.values()),by_seed=values)
    deltas={rule:{str(seed):{
        'primary':{m:mapping[(seed,rule)]['primary'][m]-mapping[(seed,'jpda-ci')]['primary'][m] for m in METRICS},
        'per_sequence':{sid:{m:mapping[(seed,rule)]['sequences'][sid][m]-mapping[(seed,'jpda-ci')]['sequences'][sid][m]
            for m in ('HOTA','AssA','DetA','IDF1')} for sid in sorted(reference['sequences'])}}
        for seed in evaluation.SEEDS} for rule in ('jpda-kalman','pkf')}
    return dict(kind='fixed_probabilistic_three_seed_full_val_comparison_v1',status='complete',
        frames_per_run=3316,sequences_per_run=21,completed_cells=9,three_seed_descriptive=descriptive,
        paired_delta_from_jpda_ci=deltas,metric_direction={m:'lower' if m in LOWER else 'higher' for m in METRICS},
        actual_factors_identical_within_each_seed=True,best_seed_selected=False,
        sample_sd_is_not_confidence_interval=True,hard_identity_invariance_verified=False,
        same_input_selection_as_legacy_source_ablation=False,same_state_time_protocol_as_recoverable=False,
        public_methods_reproduced=False,fair_resources_verified=False,
        full_paper_comparison_completed=False,paper_eligible=False)


def compare(campaign_path,campaign_sha,reports,output):
    campaign_path=Path(campaign_path).absolute();output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new non-symlink comparison output required')
    if native.sha(campaign_path)!=campaign_sha:raise ValueError('frozen campaign identity differs')
    campaign=native.read_json(campaign_path)
    if (campaign.get('kind')!='fixed_probabilistic_full_spd_val_campaign_preflight_v1'
            or campaign.get('source_sha256')!=evaluation.inference_sources()):
        raise ValueError('current frozen full-val campaign required')
    cells=[load_report(path,sha) for path,sha in reports]
    result=assemble(campaign,cells)
    evidence={str(campaign_path):native.evidence(campaign_path),str(Path(__file__)):native.evidence(__file__)}
    for cell in cells:
        for p,value in cell['evidence'].items():
            if p in evidence and evidence[p]!=value:raise ValueError('conflicting shared evidence')
            evidence[p]=value
    evaluation.unchanged(evidence)
    result.update(campaign_sha256=campaign_sha,report_sha256={str(c['report']['seed'])+'/'+c['report']['update_rule']:c['report_sha256'] for c in cells},
                  input_evidence=evidence)
    output.mkdir();native.write_json(output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('status','completed_cells','three_seed_descriptive','paper_eligible')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--campaign-sha256',required=True)
    parser.add_argument('--report',nargs=2,action='append',required=True,metavar=('REPORT','SHA256'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.campaign,args.campaign_sha256,args.report,args.output)
