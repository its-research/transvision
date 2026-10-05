"""Independent GPU candidate readback; no full-cohort or performance promotion."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha

EXECUTOR = R/'source-freezes/rbf-batched-complete-sequence-GPU-measurement-v1-20261004'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def measurement_gate(document,rank,events):
    assert document['kind']=='rbf_real_replay_per_device_memory_measurement_v1'
    assert document['GPU_uuid']==rank['gpu_uuid'] and document['device']=='cuda:'+str(rank['rank'])
    assert document['sampler_finished'] is True and not document['sampler_errors']
    assert document['replay_failure'] is None and document['completed_events']==events
    assert document['TF32_matmul'] is document['TF32_cudnn'] is False
    assert document['target_independently_accepted'] is False
    assert document['isolated_performance_or_paper_acceptance'] is False
    assert document['target_percent']==[75,80] and document['device_usage_includes_all_processes'] is True
    elapsed=document['elapsed_seconds_including_observer']
    assert type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0
    assert math.isclose(document['events_per_second_with_observer'],events/elapsed,rel_tol=1e-12)
    samples=document['samples'];assert len(samples)>=2
    times=[];ratios=[];totals=set()
    for sample in samples:
        t=sample['elapsed_seconds'];assert math.isfinite(t) and 0<=t<=elapsed
        free=sample['device_free_bytes'];total=sample['device_total_bytes']
        assert type(free) is int and type(total) is int and 0<=free<=total and total>0
        totals.add(total);percent=100*(total-free)/total
        assert math.isclose(percent,sample['device_used_percent'],abs_tol=1e-10)
        allocated=sample['process_tensor_allocated_bytes'];reserved=sample['process_allocator_reserved_bytes']
        assert type(allocated) is int and type(reserved) is int and 0<=allocated<=reserved<=total
        times.append(t);ratios.append(percent)
    assert len(totals)==1 and all(b>=a for a,b in zip(times,times[1:]))
    for key in ('peak_process_tensor_allocated_bytes','peak_process_allocator_reserved_bytes'):
        assert type(document[key]) is int and 0<=document[key]<=next(iter(totals))
    assert document['peak_process_tensor_allocated_bytes']>=max(s['process_tensor_allocated_bytes'] for s in samples)
    assert document['peak_process_allocator_reserved_bytes']>=max(s['process_allocator_reserved_bytes'] for s in samples)
    gaps=[times[0],elapsed-times[-1],*(b-a for a,b in zip(times,times[1:]))]
    return dict(GPU_uuid=document['GPU_uuid'],samples=len(samples),minimum_percent=min(ratios),maximum_percent=max(ratios),
        arithmetic_mean_percent=sum(ratios)/len(ratios),maximum_observation_gap_seconds=max(gaps),
        all_observed_samples_in_target=all(75<=v<=80 for v in ratios),
        continuous_occupancy_between_samples_proven=False,
        reported_events_per_second_with_observer=events/elapsed,
        isolated_throughput_or_all_GPU_target_accepted=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task-id',required=True);args=parser.parse_args()
    assert len(args.task_id)==32 and all(c in '0123456789abcdef' for c in args.task_id)
    own=Path(__file__).resolve().parent
    freeze=json.loads((own/'source-freeze.json').read_bytes())
    for name,spec in freeze['sources'].items():assert sha(own/name)==spec['sha256']
    for ref in freeze['references']:assert sha(ref['path'])==ref['sha256']
    helper=module('batch_safe_bytes',Path(freeze['safe_byte_helper']['path']))
    legacy=module('batch_source_helpers',Path(freeze['source_helper']['path']))
    prep=json.loads((EXECUTOR/'preparation.json').read_bytes())
    for name,spec in prep['sources'].items():assert sha(EXECUTOR/name)==spec['sha256']
    from clearml import Task
    task=Task.get_task(task_id=args.task_id);params=task.get_parameters();plan=json.loads(params['General/plan'])
    assert type(plan['world_size']) is int and plan['world_size'] in (4,8)
    assert plan==dict(prep['plan'],world_size=plan['world_size'])
    recipe=hashlib.sha256(canonical(plan)).hexdigest();assert params['General/recipe_sha256']==recipe
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest()==plan['bootstrap_sha256']
    status=str(task.status)
    if status not in ('completed','failed'):
        print(json.dumps(dict(task_id=task.id,status=status,ETA='unknown',readback_started=False)));return
    root=R/'artifacts/rbf-batched-GPU-complete-sequence-independent-v1-20261004'/task.id
    root.mkdir(parents=True,exist_ok=True)
    result=root/('independent-candidate-receipt.json' if status=='completed' else 'independent-failure-bytes.json')
    if result.exists():
        prior=json.loads(result.read_bytes());assert prior['task_id']==task.id and prior['recipe_sha256']==recipe
        for key,spec in prior['artifacts'].items():
            assert task.artifacts[key].hash==spec['sha256'] and task.artifacts[key].size==spec['bytes']
        print(json.dumps(dict(existing_receipt=str(result),not_repeated=True)));return
    keys={'receipt','exclusive-source-manifest',*(f'replay-rank{i}' for i in range(plan['world_size']))}
    assert 'receipt' in task.artifacts and set(task.artifacts)<=keys
    artifacts={k:helper.read_artifact(task,k,root/(k+('.json' if k in ('receipt','exclusive-source-manifest') else '.tar.gz'))) for k in sorted(task.artifacts)}
    report=json.loads((root/'receipt.json').read_bytes());reported=dict(report['plan']);reported.pop('cache_relative_root',None)
    assert reported==plan and report['task_id']==task.id
    assert report['kind']=='rbf_batched_complete_sequence_GPU_measurement_candidate_v1'
    assert report['GPU_memory_target_admitted'] is report['production_promotion_allowed'] is False
    if status=='failed':
        new(result,dict(task_id=task.id,recipe_sha256=recipe,artifacts=artifacts,failure=report['failure'],automatic_retry_allowed=False,accepted=False))
        register(result,'rbf-batched-GPU-candidate-failure-bytes');return
    assert set(artifacts)==keys and report['failure'] is None and report['all_selected_complete_sequences_completed'] is True
    assert report['all_46_sequences_7445_events_completed'] is False
    source=json.loads((root/'exclusive-source-manifest.json').read_bytes())
    for a,b in (('patches','exclusive_patches'),('source_replacements','source_replacements'),('original_source','source'),('configuration','configuration'),('bootstrap_sha256','bootstrap_sha256')):assert source[a]==plan[b]
    ranks=report['ranks'];assert sorted(r['rank'] for r in ranks)==list(range(plan['world_size']))
    assert len({r['gpu_uuid'] for r in ranks})==plan['world_size']
    index=json.loads(legacy.INDEX.read_bytes());assert sha(legacy.INDEX)==plan['numeric_reference_admission_sha256']
    entry=next(e for e in index['seeds'] if e['seed']==2027)
    for k in ('training_byte_proof','prediction_byte_proof','numeric_completion'):assert sha(entry[k])==entry[k+'_sha256']
    checkpoint_path=Path(entry['training_byte_proof']).parent/'checkpoint';assert sha(checkpoint_path)==plan['checkpoint']['sha256']
    checkpoint=json.loads(checkpoint_path.read_bytes());assert checkpoint['model_sha256']==plan['final_refit_model_sha256']
    schedule_path=R/'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed2027/events.json';assert sha(schedule_path)==plan['events']['sha256']
    schedule=json.loads(schedule_path.read_bytes());sequences=sorted(schedule['origin_us_by_sequence'])[:plan['world_size']]
    groups={s:[e for e in schedule['events'] if e['sequence_id']==s] for s in sequences}
    expected={s:[] for s in sequences};proof=json.loads(Path(entry['prediction_byte_proof']).read_bytes())
    for asset in plan['forward_outputs']:
        assert asset==proof['artifacts'][asset['key']]
        path=Path(entry['prediction_byte_proof']).parent/(asset['key']+'.jsonl');assert sha(path)==asset['sha256']
        for line in path.open():
            row=json.loads(line)
            if row['sequence_id'] in expected:expected[row['sequence_id']].append(row)
    sources=legacy.expected_sources(plan)
    sys.path.insert(0,str(legacy.CPU));cache_oracle=module('batch_cache203',legacy.CPU/'final_cache203.py')
    cache=cache_oracle.CacheAdmission(2027,checkpoint_path)
    fresh=module('batch_fresh',legacy.FRESH);assert sha(legacy.FRESH)==legacy.FRESH_SHA
    outcomes=[]
    for rank in range(plan['world_size']):
        rr=next(r for r in ranks if r['rank']==rank);sequence=sequences[rank];events=groups[sequence]
        assert rr['world_size']==plan['world_size'] and rr['all_sequences_completed'] is True
        assert rr['TF32_matmul'] is rr['TF32_cudnn'] is False
        assert len(rr['sequences'])==1 and rr['events_committed']==len(events)
        assert rr['sequences'][0]['sequence_id']==sequence
        helper.unpack(root/f'replay-rank{rank}.tar.gz',root/f'rank{rank}-unpack')
        directory=root/f'rank{rank}-unpack/rank-{rank}'/sequence
        receipt=json.loads((directory/'receipt.json').read_bytes());rp=json.loads((directory/'plan.json').read_bytes())
        assert rp['configuration']==plan['configuration'] and rp['fixture'] is False and rp['source_sha256']==sources
        assert rp['model_binding']['checkpoint_sha256']==plan['checkpoint']['sha256'] and rp['model_binding']['model_sha256']==checkpoint['model_sha256']
        assert rp['events_sha256']==hashlib.sha256(canonical(events)).hexdigest()
        assert receipt['completed_events']==len(events) and receipt['completed_sequences']==[sequence]
        for name,digest in receipt['files'].items():assert sha(directory/name)==digest
        dbinfo=receipt['databases'][sequence];dbpath=(directory/dbinfo['path']).resolve()
        assert dbpath.is_relative_to(directory.resolve()) and sha(dbpath)==dbinfo['sha256']
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            commits=db.execute('SELECT event_id,prediction,audit FROM events ORDER BY ordinal').fetchall()
            assert [c[0] for c in commits]==[e['event_id'] for e in events]
            with (directory/'predictions.jsonl').open('rb') as pf,(directory/'audit.jsonl').open('rb') as af:
                for (_,p,a),pl,al in zip(commits,pf,af,strict=True):assert p==pl.rstrip(b'\n') and a==al.rstrip(b'\n')
            count=0;maximum=0.
            for i,node,raw,digest in db.execute('SELECT i,node_id,raw,sha FROM observations ORDER BY i'):
                assert i==count and hashlib.sha256(raw).hexdigest()==digest
                ref=expected[sequence][i];assert ref['row']==i and ref['node_id']==node
                factors=db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
                assert [p for p,w in factors]==[-1]+ref['context_indices'][:-1]
                for (_,actual),value in zip(factors,helper.normalized_factors(ref['logits']),strict=True):
                    assert math.isfinite(actual) and abs(actual-value)<=1e-4+1e-4*abs(value)
                    maximum=max(maximum,abs(actual-value))
                count+=1
            assert count==len(expected[sequence])
        def progress(value):print(json.dumps(dict(rank=rank,progress=value,ETA='unknown')),flush=True)
        features=cache_oracle.verify_database(dbpath,dbinfo['sha256'],cache)
        state=fresh.verify_database(dbpath,dbinfo['sha256'],progress=progress)
        assert features['events']==state['events']==len(events) and features['rows']==count
        measured=measurement_gate(json.loads((directory/'GPU-runtime-measurement.json').read_bytes()),rr,len(events))
        outcome=dict(rank=rank,sequence=sequence,events=len(events),rows=count,NN_max_abs_error=maximum,raw_cache203=features,fresh_state=state,measurement=measured)
        new(root/f'rank-{rank}-independent.json',outcome);outcomes.append(outcome)
    new(result,dict(kind='rbf_batched_GPU_selected_complete_sequences_independent_v1',task_id=task.id,recipe_sha256=recipe,artifacts=artifacts,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),outcomes=outcomes,
        full_cohort_or_three_seed_acceptance=False,structure_action_independent_acceptance_pending=True,
        GPU_memory_target_admitted=False,production_promotion_allowed=False,paper_performance_complete=False))
    register(result,'rbf-batched-GPU-selected-complete-sequence-independent-readback')
    print(json.dumps(dict(receipt=str(result))))


if __name__=='__main__':main()
