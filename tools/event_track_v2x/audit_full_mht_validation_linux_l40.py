#!/usr/bin/env python3
"""Separate Linux L40S full official SPD-val wrapper around the independent scan-MHT ledger audit.

The existing subset audit is unchanged. This entry point additionally requires
the frozen full schedule, K=4 configuration and one of the three trained seeds.
No GT, Torch, live tracker, parameter training or public upload is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x import audit_mht_validation as ledger

common, native = ledger.common, ledger.native
KIND = 'scan_mht_full_spd_val_audit_linux_l40_v1'
CACHE_SHA = '66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8'
SCHEDULE_SHA = '2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a'
SPLIT_SHA = '4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3'
CONFIG_SHA = '80df622ed0671807f8102bb907c0bb53360991038870b7f3dac066c2f2201f7f'
CHECKPOINTS = {
    1337:'70609d3a7bf330d0d56b92150d0f97db511dfe4545c07cc193e993d0b77a0e96',
    2027:'0d0289fc153ea66b5e022c4cc1dd02f1d4f0e552599a9d4f01e924e5dda0a33f',
    3407:'96d1db472bef678112de2187046c7d8002df35bfe0dfeb153c746948d59f5a36',
}
RUNTIME = dict(device='cpu', numpy='1.26.4', torch='2.6.0+cu124', scipy='1.14.1',sqlite_version='3.45.1',
    assignment_binary_sha256='bb5e2231cdf4381bc3e141e956e37e8ee74902f308f5dca5db458abdd06edb26',
    torch_threads=1,torch_interop_threads=1,exclusive_host=False,same_latency_or_memory_verified=False)
THREAD_ENV = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',
    PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')


def check_headers(final,receipt,plan):
    """Metadata checks only; actual full coverage is proved by the ledger audit."""
    seed = plan.get('checkpoint_seed')
    if (receipt.get('kind') != 'scan_mht_scheduled_replay_v1' or receipt.get('status') != 'complete'
            or receipt.get('scheduled_frames') != 3316 or receipt.get('completed_frames') != 3316
            or receipt.get('cache_split') != 'val' or receipt.get('cache_sha256') != CACHE_SHA
            or len(receipt.get('sequence_heads',{})) != 21
            or any(final.get(k) != v for k,v in receipt.items())
            or final.get('full_official_validation_schedule_completed') is not True):
        raise ValueError('complete official 3316-frame 21-sequence MHT receipt required')
    required_false = ('geometry_development_baseline','gt_model_inputs','parameter_training',
                      'test_payloads_read','reproduced_public_method','paper_eligible',
                      'same_latency_or_memory_verified','exclusive_host')
    if (any(receipt.get(k) is not False for k in required_false)
            or receipt.get('learned_identity_enabled') is not True or receipt.get('global_scan_mht') is not True
            or receipt.get('source_arrival_policy') != 'scheduled_pair_snapshot_at_reference_plus_100ms'):
        raise ValueError('learned non-training car MHT execution declaration required')
    if (plan.get('kind') != 'scan_mht_replay_plan_v1' or plan.get('class_scope') != ['car']
            or plan.get('cache_split') != 'val' or plan.get('cache_sha256') != CACHE_SHA
            or plan.get('split_sha256') != SPLIT_SHA or plan.get('schedule_sha256') != SCHEDULE_SHA
            or plan.get('full_official_schedule_verified') is not True or plan.get('scheduled_frames') != 3316
            or type(seed) is not int or seed not in CHECKPOINTS
            or plan.get('checkpoint_sha256') != CHECKPOINTS[seed]
            or final.get('checkpoint_sha256') != CHECKPOINTS[seed]
            or ledger.digest(plan.get('configuration')) != CONFIG_SHA):
        raise ValueError('frozen K=4 car plan, official schedule and declared trained seed required')
    if (any(plan.get(k) is not False for k in ('geometry_baseline','gt_model_inputs','parameter_training',
            'validation_parameter_search','validation_checkpoint_selection','paper_eligible',
            'same_latency_or_memory_verified','reproduced_public_method'))
            or plan.get('val_seen_during_research') is not True):
        raise ValueError('frozen validation selection boundary differs')
    runtime = plan.get('runtime',{})
    if any(runtime.get(k) != v for k,v in RUNTIME.items()) or runtime.get('thread_environment') != THREAD_ENV:
        raise ValueError('frozen inference runtime or assignment solver binary differs')


def bind_run(run, final_sha):
    run = Path(run).absolute(); evidence = {}
    def bind(path,expected=None):
        path = Path(path).absolute()
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('non-symlink evidence required')
        observed = native.evidence(path)
        if expected is not None and observed['sha256'] != expected:
            raise ValueError('full MHT evidence identity differs: '+path.name)
        evidence[str(path)] = observed
        return path
    final = native.read_json(bind(common.child(run,'full-validation-receipt.json'),final_sha))
    receipt = native.read_json(bind(common.child(run,'receipt.json')))
    plan = native.read_json(bind(common.child(run,'plan.json'),receipt['plan_sha256']))
    check_headers(final,receipt,plan)
    inputs = plan.get('input_file_sha256',{})
    def unique(expected,name):
        matches = [Path(p) for p,h in inputs.items() if h == expected]
        if len(matches) != 1 or matches[0].name != name:
            raise ValueError('one bound '+name+' required')
        return bind(matches[0],expected)
    schedule_path = unique(SCHEDULE_SHA,'schedule.json')
    cache_path = unique(CACHE_SHA,'manifest.json')
    checkpoint_path = unique(CHECKPOINTS[plan['checkpoint_seed']],'checkpoint.json')
    checkpoint = native.read_json(checkpoint_path)
    weights = common.child(checkpoint_path.parent,checkpoint['weights']['path'])
    if (set(inputs) != {str(p) for p in (schedule_path,cache_path,checkpoint_path,weights)}
            or inputs[str(weights)] != checkpoint['weights']['sha256']
            or checkpoint.get('full_official_train') is not True or checkpoint.get('data_split') != 'train'
            or checkpoint.get('seed') != plan['checkpoint_seed'] or checkpoint.get('labels_in_model_inputs') is not False):
        raise ValueError('exact frozen training checkpoint/input inventory required')
    bind(weights,checkpoint['weights']['sha256'])
    schedule = native.read_json(schedule_path); rows = schedule['frames']; cache = native.read_json(cache_path)
    if (schedule.get('kind') != 'spd_official_validation_prediction_schedule_v1'
            or schedule.get('contains_ground_truth') is not False or schedule.get('contains_system_error_offset') is not False
            or schedule.get('split_sha256') != SPLIT_SHA or len(rows) != 3316
            or len({r['sequence_id'] for r in rows}) != 21 or ledger.digest(rows) != plan['schedule_rows_sha256']
            or cache.get('split') != 'val' or cache.get('frame_count') != 7189
            or set(cache.get('sequences',[])) != {r['sequence_id'] for r in rows}
            or set(receipt['sequence_heads']) != {r['sequence_id'] for r in rows}
            or sum(h['frames'] for h in receipt['sequence_heads'].values()) != 3316):
        raise ValueError('exact official schedule/cache coverage required')
    if plan.get('source_sha256') != ledger.inference_sources():
        raise ValueError('complete current MHT inference source inventory required')
    for name,sha in plan['source_sha256'].items(): bind(common.child(ROOT,name),sha)
    for p in (Path(__file__),Path(ledger.__file__),Path(common.__file__),Path(native.__file__)): bind(p)
    for name,key in (('predictions.jsonl','predictions_sha256'),('tracking.jsonl','tracking_sha256'),
                     ('frame-timings.jsonl','frame_timings_sha256')):
        bind(common.child(run,name),receipt[key])
    for head in receipt['sequence_heads'].values(): bind(common.child(run,head['database']),head['database_sha256'])
    common.unchanged(evidence)
    return dict(run=run,final=final,receipt=receipt,plan=plan,rows=rows,evidence=evidence)


def audit(run, final_sha, output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new non-symlink full audit output required')
    bound = bind_run(run,final_sha); run = bound['run']; receipt = bound['receipt']
    result = ledger.audit_replay(run,native.sha(run/'receipt.json'),bound['rows'])
    if result['frames'] != 3316 or len(result['sequences']) != 21:
        raise ValueError('independent ledger did not cover full official validation')
    import numpy as np
    for list_key,receipt_key in (('latencies','latency_seconds_p50_p95_p99_max'),
            ('frame_latencies','frame_latency_seconds_p50_p95_p99_max')):
        values = [x for s in result['sequences'].values() for x in s[list_key]]
        quantiles = np.quantile(values,[.5,.95,.99,1.]).tolist()
        if quantiles != receipt[receipt_key]: raise ValueError('recomputed latency quantiles differ')
        result[receipt_key] = quantiles
    result['input_evidence'].update(bound['evidence'])
    common.unchanged(result['input_evidence'])
    result.update(kind=KIND,full_validation_coverage_verified=True,source_directory=str(run),
        inference_receipt_sha256=final_sha,seed=bound['plan']['checkpoint_seed'],runtime=bound['plan']['runtime'],
        database_bytes=receipt['database_bytes'],process_peak_rss_bytes=receipt['process_peak_rss_bytes'],
        sequence_heads=receipt['sequence_heads'],test_payloads_read=False,parameter_training=False)
    output.mkdir(); native.write_json(output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','seed','frames','observations','full_validation_coverage_verified')}))
    return result


def inspect_audited_run(run, final_sha, audit_path, audit_sha):
    """Bind a completed full audit before the separate evaluator may read GT."""
    bound = bind_run(run,final_sha); path = Path(audit_path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('non-symlink independent audit required')
    actual = native.evidence(path)
    if actual['sha256'] != audit_sha: raise ValueError('independent full MHT audit identity differs')
    value = native.read_json(path)
    if (value.get('kind') != KIND or value.get('status') != 'complete'
            or value.get('inference_receipt_sha256') != final_sha
            or value.get('replay_receipt_sha256') != native.sha(bound['run']/'receipt.json')
            or value.get('source_directory') != str(bound['run'])
            or value.get('configuration') != bound['plan']['configuration']
            or value.get('seed') != bound['plan']['checkpoint_seed'] or value.get('runtime') != bound['plan']['runtime']
            or value.get('full_validation_coverage_verified') is not True or value.get('frames') != 3316
            or value.get('sequence_heads') != bound['receipt']['sequence_heads']
            or set(value.get('sequences',{})) != set(bound['receipt']['sequence_heads'])
            or value.get('input_evidence') != bound['evidence']):
        raise ValueError('complete matching full MHT audit required; subset audit is insufficient')
    if (any(s.get('frames') != value['sequence_heads'][key]['frames']
            or s.get('prediction_sha256') != value['sequence_heads'][key]['prediction_sha256']
            for key,s in value['sequences'].items())
            or sum(s.get('observations',-1) for s in value['sequences'].values()) != value.get('observations')):
        raise ValueError('independent sequence coverage or head differs')
    for key in ('raw_factor_ledger_reconstructed','retained_alias_weights_reconstructed',
                'one_to_one_and_irreversible_predecessors_verified','published_ID_lifecycle_reconstructed'):
        if value.get(key) is not True: raise ValueError('required independent identity check missing')
    for key in ('ground_truth_read','tracking_metrics_computed','full_top_k_optimality_recomputed',
                'conditional_mass_bounds_recomputed','gaussian_state_numerics_recomputed',
                'fair_resources_verified','paper_eligible','test_payloads_read','parameter_training'):
        if value.get(key) is not False: raise ValueError('independent audit scope or claim differs')
    bound['evidence'][str(path)] = actual
    common.unchanged(bound['evidence']); bound['audit'] = value
    return bound


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); audit(args.run,args.receipt_sha256,args.output)
