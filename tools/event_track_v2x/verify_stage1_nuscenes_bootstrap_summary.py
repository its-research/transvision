"""Independently read full bootstrap bytes and recompute every reported summary.

Uses Python fsum, sorted order statistics and sample variance. It verifies the
durable statistics output, not the underlying matching engine or new replay.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def original(path,digest):
    p=Path(path)
    if not p.is_file() or any(x.is_symlink() for x in [p,*p.parents]) or sha(p)!=digest:
        raise ValueError('exact ordinary frozen statistics input required')
    return p


def percentile(values,p):
    ordered=sorted(float(x) for x in values)
    if not ordered or not all(math.isfinite(x) for x in ordered):raise ValueError('finite complete percentile population required')
    position=(len(ordered)-1)*p;left=math.floor(position);right=math.ceil(position)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)


def verify(root,receipt_sha):
    root=Path(root);receipt_path=original(root/'bootstrap-receipt.json',receipt_sha);r=json.loads(receipt_path.read_bytes())
    values_path=original(root/'all13-complete-bootstrap-draw-metrics.npy',r['actual_bootstrap_values_sha256'])
    plan_path=original(root/'plan.json',r['plan_sha256']);plan=json.loads(plan_path.read_bytes())
    for path,digest in r['full_input_sha256'].items():original(path,digest)
    if not r['all13_complete_10000_sequence_draw_metrics_present'] or not plan['no_finite_replicate_deletion'] or plan['seed']!=1337 or plan['draws']!=10000:
        raise ValueError('full original statistics contract required')
    values=np.load(values_path,allow_pickle=False);metrics=r['nuScenes_bootstrap_metric_scope'];ids=list(r['per_run'])
    # JSON sorted keys are not the array axis: authoritative run order lives in
    # the admitted cache campaign, itself bound by the full input inventory.
    campaign_paths=[Path(p) for p in r['full_input_sha256'] if Path(p).name=='campaign-receipt.json']
    if len(campaign_paths)!=1:raise ValueError('unique actual cache admission required')
    campaign=json.loads(campaign_paths[0].read_bytes());ids=[x['run_id'] for x in campaign['results']]
    if values.shape!=(13,10000,15) or len(metrics)!=15 or len(set(ids))!=13:raise ValueError('all complete raw draws and original axes required')
    maximum=0.;checks=0;intervals=0
    def close(x,y):
        nonlocal maximum,checks
        if not math.isfinite(float(x)) or not math.isfinite(float(y)) or abs(x-y)>1e-12:
            raise ValueError('independent complete summary differs at fixed 1e-12')
        maximum=max(maximum,abs(x-y));checks+=1
    def interval(v,reported):
        nonlocal intervals
        finite=sum(math.isfinite(float(x)) for x in v)
        if finite!=10000:
            if reported['available'] or reported['finite_replicates']!=finite or reported['total_replicates']!=10000:raise ValueError('undefined draw silently removed')
        else:
            if not reported['available'] or reported['total_replicates']!=10000:raise ValueError('complete interval omitted')
            close(percentile(v,.025),reported['low']);close(percentile(v,.975),reported['high'])
        intervals+=1
    model_draws={}
    for model in ['M0','M1','M2','M3','M4']:
        reported=r['models'][model];runs=[x for x in ids if x.startswith(model+'-')]
        if reported['run_ids']!=runs or reported['original_run_count']!=len(runs):raise ValueError('model seed axis differs')
        ix=[ids.index(x) for x in runs]
        model_draws[model]=[[math.fsum(float(values[i,k,j]) for i in ix)/len(ix) for j in range(15)] for k in range(10000)]
        qualified=all(r['per_run'][x]['strict_replay_pass'] for x in runs)
        if reported['all_strict_replay_pass']!=qualified:raise ValueError('historical failure qualification changed')
        for j,name in enumerate(metrics):
            full=[r['per_run'][x]['full_native_metrics'][name] for x in runs];mean=math.fsum(full)/len(full)
            close(mean,reported['raw_descriptive_metrics'][name]['mean'])
            if len(full)>1:close(math.sqrt(math.fsum((x-mean)**2 for x in full)/(len(full)-1)),reported['raw_descriptive_metrics'][name]['sample_std'])
            elif reported['raw_descriptive_metrics'][name]['sample_std'] is not None:raise ValueError('single deterministic run has fabricated SD')
            if qualified:interval([x[j] for x in model_draws[model]],reported['qualified_10000_sequence_intervals'][name])
        if not qualified and reported['qualified_10000_sequence_intervals'] is not None:raise ValueError('failed model has qualified inference')
    if r['per_run']['M3-seed-2027']['strict_replay_pass'] is not False:raise ValueError('M3 failure not preserved')
    comparisons=r['qualified_strict_replay_paired_comparisons']
    if set(comparisons)!={'M1-minus-M0','M2-minus-M0','M4-minus-M0'}:raise ValueError('qualified comparison population differs')
    for model in ['M1','M2','M4']:
        for j,name in enumerate(metrics):
            delta=[x[j]-y[j] for x,y in zip(model_draws[model],model_draws['M0'])]
            interval(delta,comparisons[model+'-minus-M0'][name])
    for path,digest in r['full_input_sha256'].items():original(path,digest)
    original(values_path,r['actual_bootstrap_values_sha256']);original(receipt_path,receipt_sha)
    return dict(kind='rbf_stage1_all13_full_bootstrap_independent_statistics_output_readback_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all13_complete_draw_arrays_read_back=True,all_descriptive_and_qualified_interval_summaries_independently_recomputed=True,scalar_checks=checks,intervals_checked=intervals,maximum_difference=maximum,absolute_tolerance=1e-12,shape=list(values.shape),nonfinite_actual_draws=int((~np.isfinite(values)).sum()),original_receipt_sha256=receipt_sha,values_sha256=r['actual_bootstrap_values_sha256'],verifier_sha256=sha(__file__),M3_original_failure_preserved=True,M1_single_run_sample_std_null=True,independent_matching_engine_or_resampled_engine_reexecution=False,full_K_or_paper_acceptance=False,goal_status='active')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=verify(a.root,a.receipt_sha256)
    with Path(a.output).open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True))
