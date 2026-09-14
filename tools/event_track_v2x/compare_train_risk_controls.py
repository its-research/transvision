#!/usr/bin/env python3
"""Sealed 2x2 TRAIN controls: capacity admission by decision fallback policy.

Requires four complete inference/metric pairs and the original comparison that
retains the failed joint-beam baseline. Only the reviewed CLI parameter patch
may differ in sources; all model sources and actual factor streams must agree.
This is one-sequence development evidence, not a resource-matched paper result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.audit_train_inference_comparison import inspect, inspect_failure
from tools.event_track_v2x.evaluate_source_ablation_v2 import car_metrics, metric_vector, validate_runtime, _finite
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.forest_training_data import _new_json

BACKENDS=('component_completion','component_covered_completion')
THRESHOLDS=(.05,1.)
DRIVER='tools/event_track_v2x/run_train_inference_diagnostic.py'
DRIVERS={
    '726da37720f5f769d08894a0617ef550d688fce0a9f83d2593fd22ecc4a856b0',
    '4e04f9bf53f2d8d05cc3f47ef391fd08bf811341b65edf13ed090e5027685966',
}
CONTROL_DRIVER='4e04f9bf53f2d8d05cc3f47ef391fd08bf811341b65edf13ed090e5027685966'
METRIC_EVALUATOR='7a4a3f4e30596e28383bbf336e45a2ce7c4e696c1b8d012479f0dcf988dd68f9'


def read(path,digest):
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('regular evidence file required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('comparison input identity differs')
    return json.loads(raw)


def cell(report):
    plan=report['plan'];backend=report['backend'];threshold=plan['configuration']['state']['max_model_regret']
    if backend not in BACKENDS or type(threshold) not in (int,float) or threshold not in THRESHOLDS:
        raise ValueError('only the predeclared capacity and fallback controls are allowed')
    return backend,threshold


def cell_name(key):
    return key[0]+('/threshold-fallback' if key[1]==.05 else '/conditional-bayes')


def read_metrics(path,digest,report):
    value=read(path,digest);replay=Path(report['source_directory'])
    outer=read(replay/'development-inference-receipt.json',report['receipt_sha256'])
    receipt=read(replay/'receipt.json',outer['replay_receipt_sha256'])
    if (value.get('kind')!='spd_train_sequence_development_metrics_v1' or value.get('status')!='complete'
            or value.get('backend')!=report['backend'] or value.get('validation') is not False
            or value.get('paper_eligible') is not False or value.get('full_official_train') is not False
            or value.get('inference_receipt_sha256')!=report['receipt_sha256']
            or value.get('evaluator_sha256')!=METRIC_EVALUATOR
            or value['protocol']['split']!='train_development'
            or value['protocol']['evaluated_classes']!=['car']
            or value['counts']['frames']!=len(report['events']) or value['counts']['sequences']!=1
            or set(value['counts']['predictions_per_class'])-{'car'}
            or value['counts']['predictions_per_class'].get('car',0)!=report['total_output_boxes']):
        raise ValueError('complete matching car train metrics required')
    expected={replay/'predictions.jsonl':report['predictions_sha256'],
        replay/'tracking.jsonl':receipt['tracking_sha256'],replay/'plan.json':receipt['plan_sha256'],
        replay/'development-inference-receipt.json':report['receipt_sha256'],
        replay/'receipt.json':outer['replay_receipt_sha256']}
    if any(value['input_sha256'].get(str(p))!=h for p,h in expected.items()):
        raise ValueError('metrics do not bind the supplied inference inputs')
    validate_runtime(value['runtime'])
    primary=car_metrics(value['metrics']);vector=metric_vector(primary)
    vector.update(FP=_finite(primary['nuscenes']['fp']),FN=_finite(primary['nuscenes']['fn']))
    diagnostic=value['identity_event_diagnostics']
    for key in ('fallback_events','fallback_unreported_events','high_model_bound_events'):
        source_key='high_bound_events' if key=='high_model_bound_events' else key
        if diagnostic['counts'][key]!=report[source_key]: raise ValueError('metric and inference audits disagree')
    return dict(primary=vector,protocol=value['protocol'],runtime=value['runtime'],
        gt_manifest_sha256=value['gt_manifest_sha256'],metric_sha256=digest,metric_path=str(Path(path).absolute()),
        evaluator_sha256=value['evaluator_sha256'],identity_event_counts=diagnostic['counts'])


def assemble(reports,metrics,prior,*,allow_fixture=False):
    if len(reports)!=4 or len(metrics)!=4: raise ValueError('all four predeclared control cells are required')
    cells={cell(r):(r,m) for r,m in zip(reports,metrics)}
    if set(cells)!={(b,t) for b in BACKENDS for t in THRESHOLDS}:
        raise ValueError('all four distinct predeclared control cells are required')
    first,first_metric=cells[(BACKENDS[0],.05)]
    shared_keys=('cache_sha256','cooperative_metadata_sha256','identity_checkpoint_sha256',
        'identity_seed','scorer_signature','cohort_mode','selected_schedule','class_scope',
        'thread_environment','source_scope')
    reference_sources=first['plan']['source_sha256']
    if DRIVER not in reference_sources or len(reference_sources)<2: raise ValueError('model and driver sources required')
    normalized=[];stable_plans=[];runtimes=[];source_differences={}
    for key,(report,metric) in cells.items():
        plan=report['plan']
        if (plan.get('kind')!='train_sequence_inference_diagnostic_plan_v1' or plan['class_scope']!=['car']
                or not allow_fixture and (plan['cohort_mode']!='real-train-development'
                    or plan.get('complete_input_train_cohort_verified') is not True)):
            raise ValueError('real complete train provenance required; no fixture relabelling')
        if any(plan[k]!=first['plan'][k] for k in shared_keys): raise ValueError('control cohort or scoring inputs differ')
        stable=copy.deepcopy(plan)
        for name in ('backend','configuration','source_sha256','runtime',
                     'max_model_regret_override','conditional_bayes_without_threshold_fallback'):
            stable.pop(name,None)
        stable_plans.append(stable)
        runtimes.append({k:v for k,v in plan['runtime'].items() if k!='pid'})
        if (report['factor_stream_sha256']!=first['factor_stream_sha256']
                or [(e['sequence_id'],e['frame_id'],e['reference_us'],e['decision_us'],e['factor_rows_sha256']) for e in report['events']]
                !=[(e['sequence_id'],e['frame_id'],e['reference_us'],e['decision_us'],e['factor_rows_sha256']) for e in first['events']]):
            raise ValueError('actual complete factor stream or event clocks differ')
        sources=plan['source_sha256']
        if (set(sources)!=set(reference_sources) or sources[DRIVER] not in DRIVERS
                or any(sources[p]!=reference_sources[p] for p in sources if p!=DRIVER)):
            raise ValueError('unreviewed driver or model source drift')
        source_differences[cell_name(key)]=[p for p in sources if sources[p]!=reference_sources[p]]
        config=copy.deepcopy(plan['configuration']);config['state'].pop('max_model_regret')
        if key[0]=='component_covered_completion':
            if type(config.get('coverage_admission_version')) is not int or config.pop('coverage_admission_version')!=1:
                raise ValueError('only coverage admission version one is declared')
        normalized.append(config)
        if key[1]==1. and (plan.get('conditional_bayes_without_threshold_fallback') is not True
                or plan.get('max_model_regret_override')!=1. or sources[DRIVER]!=CONTROL_DRIVER
                or report['fallback_unreported_events'] or report['fallback_events']):
            raise ValueError('declared conditional Bayes control still uses or omits fallback')
        if key[1]==.05 and (plan.get('max_model_regret_override') not in (None,.05)
                or plan.get('conditional_bayes_without_threshold_fallback',False) is not False):
            raise ValueError('default threshold control labels conflict with actual configuration')
        if any(metric[k]!=first_metric[k] for k in ('gt_manifest_sha256','protocol','runtime','evaluator_sha256')):
            raise ValueError('ground truth or metric protocol differs')
    if any(c!=normalized[0] for c in normalized[1:]):
        raise ValueError('a third configuration factor changed besides admission and fallback')
    if any(p!=stable_plans[0] for p in stable_plans[1:]) or any(r!=runtimes[0] for r in runtimes[1:]):
        raise ValueError('unadvertised plan or inference environment drift')
    if (prior.get('kind')!='train_inference_comparison_diagnostic_v1'
            or prior.get('analysis_complete') is not True or prior.get('status')!='partial_due_to_failed_runs'
            or prior.get('actual_factors_identical') is not True or len(prior['runs'])!=2
            or prior.get('complete_backend_comparison') is not False
            or [f['backend'] for f in prior['failed_runs']]!=['joint_beam']
            or {(r['backend'],r['receipt_sha256']) for r in prior['runs']}
                !={(b,cells[(b,.05)][0]['receipt_sha256']) for b in BACKENDS}):
        raise ValueError('original failed baseline context cannot be omitted or replaced')
    edges=[((b,.05),(b,1.),'fallback') for b in BACKENDS]
    edges += [((BACKENDS[0],t),(BACKENDS[1],t),'admission') for t in THRESHOLDS]
    contrasts=[]
    for left,right,factor in edges:
        a,am=cells[left];b,bm=cells[right]
        contrasts.append(dict(factor=factor,reference=cell_name(left),variant=cell_name(right),
            primary_delta_variant_minus_reference={k:bm['primary'][k]-am['primary'][k] for k in am['primary']},
            changed_output_frames=sum(x['output_payload_sha256']!=y['output_payload_sha256'] for x,y in zip(a['events'],b['events'])),
            changed_output_id_frames=sum(x['output_ids_sha256']!=y['output_ids_sha256'] for x,y in zip(a['events'],b['events']))))
    interaction={k:(cells[(BACKENDS[1],1.)][1]['primary'][k]-cells[(BACKENDS[1],.05)][1]['primary'][k])
        -(cells[(BACKENDS[0],1.)][1]['primary'][k]-cells[(BACKENDS[0],.05)][1]['primary'][k]) for k in first_metric['primary']}
    return dict(kind='train_admission_fallback_controls_v1',status='complete',control_matrix_complete=True,
        cohort_mode=first['plan']['cohort_mode'],frames=len(first['events']),
        cells={cell_name(key):dict(inference_receipt_sha256=r['receipt_sha256'],source_directory=r['source_directory'],
            configuration=r['plan']['configuration'],**m) for key,(r,m) in cells.items()},
        contrasts=contrasts,descriptive_factor_interaction=interaction,source_differences=source_differences,
        actual_complete_factor_stream_identical=True,model_sources_identical=True,
        distinct_reported_process_ids=len({r['plan']['runtime']['pid'] for r in reports})==4,
        gt_and_metrics_used_for_prediction=False,metrics_recomputed=False,
        prior_failed_backends=[dict(backend=f['backend'],failure_receipt_sha256=f['failure_receipt_sha256'],
            source_directory=f['source_directory'],completed_frames=f['failure']['completed_frames'],
            scheduled_frames=f['failure']['scheduled_frames'],reason=f['failure']['error']) for f in prior['failed_runs']],
        strong_baseline_comparison_complete=False,recovery_only_ablation=False,
        physically_equal_resources_claimed=False,timing_repeated=False,statistical_inference=False,
        full_official_train=False,validation=False,three_seed_comparison=False,paper_eligible=False)


def compare(references,prior_path,prior_sha256,output):
    if len(references)!=4: raise ValueError('four complete run, receipt, metric and metric-hash references required')
    evidence={Path(prior_path).absolute():prior_sha256,Path(__file__):sha_file(Path(__file__))}
    prior=read(prior_path,prior_sha256);reports=[];metrics=[]
    for replay,receipt_sha,metric_path,metric_sha in references:
        report=inspect(replay,receipt_sha)
        metric=read_metrics(metric_path,metric_sha,report)
        reports.append(report);metrics.append(metric)
        evidence[contained_file(replay,'development-inference-receipt.json')]=receipt_sha
        evidence[Path(metric_path).absolute()]=metric_sha
        raw=read(metric_path,metric_sha)
        for path,digest in raw['input_sha256'].items():
            previous=evidence.setdefault(Path(path),digest)
            if previous!=digest: raise ValueError('conflicting evidence identity across metrics')
        outer=read(Path(replay)/'development-inference-receipt.json',receipt_sha)
        receipt=read(Path(replay)/'receipt.json',outer['replay_receipt_sha256'])
        for head in receipt['sequence_heads'].values():
            evidence[contained_file(replay,head['database'])]=head['database_sha256']
    result=assemble(reports,metrics,prior)
    for failure in prior['failed_runs']:
        current=inspect_failure(failure['source_directory'],failure['failure_receipt_sha256'])
        if current['captured_artifact_sha256']!=failure['captured_artifact_sha256']:
            raise ValueError('original failed baseline artifacts changed')
        for name,digest in current['captured_artifact_sha256'].items():
            evidence[Path(current['source_directory'])/name]=digest
    if any(sha_file(path)!=digest for path,digest in evidence.items()): raise ValueError('comparison evidence changed')
    result['input_sha256']={str(p):h for p,h in evidence.items()}
    output=_directory(output);_new_json(output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','contrasts','descriptive_factor_interaction','paper_eligible')},sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',nargs=4,required=True,metavar=('REPLAY','RECEIPT_SHA','METRICS','METRICS_SHA'))
    parser.add_argument('--prior-comparison',type=Path,required=True);parser.add_argument('--prior-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.run,args.prior_comparison,args.prior_sha256,args.output)
