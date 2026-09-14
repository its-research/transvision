#!/usr/bin/env python3
"""Prediction-only train diagnostics; output identity counts are NOT ID switches.

All supplied runs must share their complete sequence and frozen factor stream.
No GT, tracking metric, future input, tuning, or publication is performed.
An optional sealed teacher is compared byte-for-byte to ordinary inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from tools.event_track_v2x.run_train_inference_diagnostic import BACKENDS, ADDITIONAL_BACKENDS


def inspect(directory, receipt_sha256, *, teacher=False, allow_fixture=False):
    directory=Path(directory)
    name='receipt.json' if teacher else 'development-inference-receipt.json'
    path=contained_file(directory,name)
    if sha_file(path)!=receipt_sha256:
        raise ValueError('supplied receipt identity differs')
    outer=json.loads(path.read_bytes())
    receipt_path=contained_file(directory,'receipt.json')
    if not teacher and (outer['kind']!='train_sequence_inference_diagnostic_v1'
            or outer['status']!='complete' or sha_file(receipt_path)!=outer['replay_receipt_sha256']):
        raise ValueError('final inference diagnostic receipt required')
    receipt=json.loads(receipt_path.read_bytes())
    if (receipt['status']!='complete' or receipt['cache_split']!='train'
            or receipt['allocation_teacher'] is not teacher or receipt['learned_identity_enabled'] is not True
            or receipt['completed_frames']!=receipt['scheduled_frames']):
        raise ValueError('complete learned train replay in the declared mode required')
    for filename,key in (('plan.json','plan_sha256'),('tracking.jsonl','tracking_sha256'),
                         ('predictions.jsonl','predictions_sha256'),('frame-timings.jsonl','frame_timings_sha256')):
        if sha_file(contained_file(directory,filename))!=receipt[key]:
            raise ValueError('replay artifact identity differs: '+filename)
    plan=json.loads((directory/'plan.json').read_bytes())
    if plan.get('class_scope')!=['car']:
        raise ValueError('explicit car scope required')
    if not teacher and (outer['plan_sha256']!=receipt['plan_sha256']
            or outer['backend']!=plan['backend'] or outer['cohort_mode']!=plan['cohort_mode']
            or not allow_fixture and (plan['cohort_mode']!='real-train-development'
                or outer['complete_selected_sequence_verified'] is not True)):
        raise ValueError('real train diagnostic provenance required; fixture cannot be relabelled')
    for head in receipt['sequence_heads'].values():
        if sha_file(contained_file(directory,head['database']))!=head['database_sha256']:
            raise ValueError('result database changed')
    events,factors,seen,previous=[],hashlib.sha256(),Counter(),set()
    with (directory/'tracking.jsonl').open('rb') as audits, (directory/'predictions.jsonl').open('rb') as predictions:
        for line in audits:
            audit=json.loads(line)['tracking']
            encoded=predictions.readline()
            if not encoded:
                raise ValueError('prediction stream shorter than audit')
            prediction=json.loads(encoded)
            key=audit['sequence_id'],audit['event_id']
            if key!=(prediction['sequence_id'],prediction['frame_id']):
                raise ValueError('prediction and audit event differ')
            if hashlib.sha256(canonical({k:v for k,v in prediction.items() if k!='commit_sha256'})).hexdigest()!=prediction['commit_sha256']:
                raise ValueError('prediction commit digest differs')
            ids=[p['track_id'] for p in prediction['predictions']]
            if len(set(ids))!=len(ids) or any(p['class_label']!='car' for p in prediction['predictions']):
                raise ValueError('duplicate prediction ID or noncar output')
            current=set(ids)
            bound=audit.get('model_regret_upper')
            if bound is not None and (not math.isfinite(bound) or not 0<=bound<=1):
                raise ValueError('invalid model bound')
            events.append(dict(sequence_id=key[0],frame_id=key[1],reference_us=prediction['box_reference_timestamp_us'],
                decision_us=prediction['decision_timestamp_us'],output_count=len(ids),
                factor_rows_sha256=audit['factor_rows_sha256'],
                new_output_track_ids=len(current-set(seen)),lost_from_previous_output_ids=len(previous-current),
                model_regret_upper=bound,fallback=audit.get('global_fallback_used'),
                output_payload_sha256=hashlib.sha256(canonical(prediction['predictions'])).hexdigest(),
                output_ids_sha256=hashlib.sha256(canonical(sorted(ids))).hexdigest(),
                search_steps=audit.get('search_steps')))
            seen.update(ids);previous=current
            factors.update(canonical([*key,audit['factor_rows_sha256']])+b'\n')
        if predictions.readline():
            raise ValueError('prediction stream longer than audit')
    keys=[(e['sequence_id'],e['frame_id'],e['reference_us']) for e in events]
    if len(events)!=receipt['completed_frames'] or len(set(keys))!=len(keys):
        raise ValueError('replay event coverage differs')
    if not teacher and keys!=[(r['sequence_id'],r['vehicle_frame'],r['box_reference_timestamp_us']) for r in plan['selected_schedule']]:
        raise ValueError('diagnostic omitted or changed scheduled frames')
    timings=[json.loads(line) for line in (directory/'frame-timings.jsonl').read_bytes().splitlines()]
    if [(r['sequence_id'],r['frame_id'],r['box_reference_timestamp_us']) for r in timings]!=keys:
        raise ValueError('timing coverage differs')
    seconds=[r['step_seconds'] for r in timings]
    if any(not math.isfinite(v) or v<0 for v in seconds):
        raise ValueError('invalid replay latency')
    quantiles=np.quantile(seconds,[.5,.95,.99,1.]).tolist()
    if quantiles!=receipt['latency_seconds_p50_p95_p99_max']:
        raise ValueError('latency summary differs')
    if sha_file(path)!=receipt_sha256 or any(sha_file(directory/filename)!=receipt[key] for filename,key in
            (('plan.json','plan_sha256'),('tracking.jsonl','tracking_sha256'),('predictions.jsonl','predictions_sha256'),
             ('frame-timings.jsonl','frame_timings_sha256'))):
        raise ValueError('replay artifact changed during analysis')
    return dict(backend='teacher' if teacher else plan['backend'],plan=plan,events=events,
        receipt_sha256=receipt_sha256,source_directory=str(directory.absolute()),factor_stream_sha256=factors.hexdigest(),
        predictions_sha256=receipt['predictions_sha256'],unique_output_track_ids=len(seen),
        total_output_boxes=sum(seen.values()),single_event_output_ids=sum(n==1 for n in seen.values()),
        fallback_events=sum(e['fallback'] is True for e in events),
        fallback_unreported_events=sum(e['fallback'] is None for e in events),
        high_bound_events=sum(e['model_regret_upper'] is not None and e['model_regret_upper']>=.99 for e in events),
        bound_unreported_events=sum(e['model_regret_upper'] is None for e in events),
        latency_seconds_p50_p95_p99_max=quantiles,elapsed_seconds=receipt['elapsed_seconds'],
        process_peak_rss_bytes=receipt['process_peak_rss_bytes'],database_bytes=receipt['database_bytes'])


def inspect_failure(directory, failure_sha256, *, allow_fixture=False):
    directory=Path(directory)
    path=contained_file(directory,'failure.json')
    if sha_file(path)!=failure_sha256: raise ValueError('failure receipt identity differs')
    failure=json.loads(path.read_bytes())
    plan_path=contained_file(directory,'plan.json')
    if (failure['status']!='failed' or failure['partial_outputs_not_final_results'] is not True
            or failure['completed_frames']>=failure['scheduled_frames']
            or sha_file(plan_path)!=failure['plan_sha256']
            or (directory/'development-inference-receipt.json').exists() or (directory/'receipt.json').exists()):
        raise ValueError('a terminal incomplete inference failure is required')
    plan=json.loads(plan_path.read_bytes())
    if plan['backend'] not in BACKENDS+ADDITIONAL_BACKENDS or not allow_fixture and plan['cohort_mode']!='real-train-development':
        raise ValueError('declared train inference failure required')
    prefix=[]
    for line in contained_file(directory,'tracking.jsonl').read_bytes().splitlines():
        audit=json.loads(line)['tracking']
        prefix.append((audit['sequence_id'],audit['event_id'],audit['factor_rows_sha256']))
    if len(prefix)!=failure['completed_frames']: raise ValueError('failed prefix coverage differs')
    artifacts={name:sha_file(contained_file(directory,name)) for name in
        ('plan.json','failure.json','tracking.jsonl','predictions.jsonl','frame-timings.jsonl')}
    artifacts.update({p.name:sha_file(contained_file(directory,p.name)) for p in directory.glob('sequence-*.sqlite')})
    return dict(backend=plan['backend'],plan=plan,failure=failure,completed_prefix_factors=prefix,
        captured_artifact_sha256=artifacts,source_directory=str(directory.absolute()),
        failure_receipt_sha256=failure_sha256,metrics_not_computed=True)


def compare(runs, output, *, teacher=None, failed_runs=(), allow_fixture=False):
    if len(runs)<2 or len(runs)>len(BACKENDS):
        raise ValueError('two or three predeclared inference runs required')
    reports=[inspect(path,digest,allow_fixture=allow_fixture) for path,digest in runs]
    names=[r['backend'] for r in reports]
    if len(set(names))!=len(names) or any(name not in BACKENDS for name in names):
        raise ValueError('distinct known inference backends required')
    def cohort(report):
        p=report['plan']
        return (p['cache_sha256'],p['cooperative_metadata_sha256'],p['identity_checkpoint_sha256'],
            p['configuration']['state'],p['cohort_mode'],p['selected_schedule'],p['source_sha256'],report['factor_stream_sha256'])
    if any(cohort(r)!=cohort(reports[0]) for r in reports[1:]):
        raise ValueError('inference runs differ in cohort, checkpoint, state model or actual factor stream')
    failures=[inspect_failure(path,digest,allow_fixture=allow_fixture) for path,digest in failed_runs]
    all_names=names+[r['backend'] for r in failures]
    if len(set(all_names))!=len(all_names) or set(all_names)!=set(BACKENDS):
        raise ValueError('every predeclared backend must have one completed or failed run; no silent omissions')
    for failed in failures:
        p,reference=failed['plan'],reports[0]['plan']
        if any(p[key]!=reference[key] for key in ('cache_sha256','cooperative_metadata_sha256',
                'identity_checkpoint_sha256','cohort_mode','selected_schedule','source_sha256')) or p['configuration']['state']!=reference['configuration']['state']:
            raise ValueError('failed backend has a different experiment cohort')
        expected=[(e['sequence_id'],e['frame_id'],e['factor_rows_sha256']) for e in reports[0]['events']]
        if failed['completed_prefix_factors']!=expected[:failed['failure']['completed_frames']]:
            raise ValueError('failed backend completed prefix used different actual factors')
        failed['completed_prefix_actual_factors_match']=True
    runtimes=[r['plan']['runtime'] for r in reports]
    fresh_pids=len({r['pid'] for r in runtimes})==len(runtimes)
    same_runtime=all({k:v for k,v in r.items() if k!='pid'}=={k:v for k,v in runtimes[0].items() if k!='pid'} for r in runtimes[1:])
    equivalence=None
    if teacher is not None:
        old=inspect(*teacher,teacher=True,allow_fixture=allow_fixture)
        baseline=next((r for r in reports if r['backend']=='component_completion'),None)
        if baseline is None or (old['factor_stream_sha256']!=baseline['factor_stream_sha256']
                or old['plan']['identity_checkpoint_sha256']!=baseline['plan']['identity_checkpoint_sha256']
                or old['plan']['configuration']!=baseline['plan']['configuration']):
            raise ValueError('teacher and ordinary inference inputs differ')
        equivalence=dict(teacher_receipt_sha256=teacher[1],
            teacher_predictions_sha256=old['predictions_sha256'],
            ordinary_predictions_sha256=baseline['predictions_sha256'],
            byte_identical=old['predictions_sha256']==baseline['predictions_sha256'])
    reference=reports[0]
    comparisons=[]
    for report in reports[1:]:
        comparisons.append(dict(reference=reference['backend'],backend=report['backend'],
            changed_output_frames=sum(a['output_payload_sha256']!=b['output_payload_sha256'] for a,b in zip(reference['events'],report['events'])),
            changed_output_id_frames=sum(a['output_ids_sha256']!=b['output_ids_sha256'] for a,b in zip(reference['events'],report['events']))))
    result=dict(kind='train_inference_comparison_diagnostic_v1',
        status='partial_due_to_failed_runs' if failures else 'complete',analysis_complete=True,
        cohort_mode=reference['plan']['cohort_mode'],frames=len(reference['events']),
        runs=reports,comparisons=comparisons,teacher_probe_equivalence=equivalence,
        failed_runs=failures,complete_backend_comparison=not failures,predeclared_backends=BACKENDS,
        actual_factors_identical=True,three_seed_comparison=False,real_tracking_evaluation=False,
        distinct_reported_process_ids=fresh_pids,same_reported_runtime=same_runtime,
        identity_counts_are_not_id_switches=True,physically_equal_resources_claimed=False,
        timing_repeated=False,paper_eligible=False,analyzer_sha256=sha_file(Path(__file__)))
    output=_directory(output)
    _new_json(output/'comparison.json',result)
    print(json.dumps(dict({k:v for k,v in result.items() if k not in ('runs','failed_runs')},
        failed_backends=[dict(backend=f['backend'],completed_frames=f['failure']['completed_frames'],
            reason=f['failure']['error']) for f in failures]),sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',nargs=2,metavar=('DIRECTORY','RECEIPT_SHA256'),required=True)
    parser.add_argument('--teacher',nargs=2,metavar=('DIRECTORY','RECEIPT_SHA256'))
    parser.add_argument('--failed-run',action='append',nargs=2,metavar=('DIRECTORY','FAILURE_SHA256'),default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compare(args.run,args.output,teacher=args.teacher,failed_runs=args.failed_run)
