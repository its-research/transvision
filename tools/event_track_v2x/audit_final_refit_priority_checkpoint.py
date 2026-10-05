"""Independently audit final priority data serialization and selected MLP weights.

This is a local numerical checkpoint audit. It does not certify cloud byte
readback, optimizer trajectory, learned tracking, resource cost or paper metrics.
No training model, optimizer, solver or producer feature function is imported.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from rbf_nested_seen_val_v2_common import R,new,register,sha

CONSUMER=R/'source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005'
CONSUMER_FREEZE_SHA='e358113abf6c23bab8cd2bb27f4a2df0d9586ff9ceda88e64679f6b87071b961'
RECIPE='causal-component-search-state-priority-v1'
TARGET='signed_weighted_model_decision_bound_reduction_per_charged_step_v1'
FEATURES=('loss_weight','eta_upper','conditional_risk','optimization_gap','model_regret_upper','active_entropy',
    'top_two_log_gap','log_nodes','log_loss_nodes','log_active','log_frontier','frontier_min_depth_fraction',
    'frontier_max_depth_fraction','remaining_budget_fraction','proposal_operation','log_operation_steps','log_prefix_nodes','weighted_eta')
ATOL=RTOL=1e-8


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def contained(root,name):
    require(isinstance(name,str) and bool(name),'nonempty relative file required')
    relative=Path(name);root=Path(root).absolute();path=root/relative
    require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix()==name,'canonical relative file required')
    require(path.is_file() and path.resolve().is_relative_to(root.resolve())
        and not any(p.is_symlink() for p in (path,*path.parents)),'contained nonsymlink file required')
    return path


def read_json(root,name,expected=None):
    path=contained(root,name)
    if expected is not None:require(sha(path)==expected,'input hash changed: '+name)
    return json.loads(path.read_bytes())


def holdout_split(sequences):
    require(len(sequences)>=2 and len(set(sequences))==len(sequences),'distinct train sequences required')
    ordered=sorted(sequences,key=lambda scene:(hashlib.sha256(('priority-holdout-v1:'+scene).encode()).hexdigest(),scene))
    held=set(ordered[:max(1,len(sequences)//5)])
    return sorted(set(sequences)-held),sorted(held)


def weights(path,hidden=32):
    with np.load(path,allow_pickle=False) as archive:
        require(len(archive.files)==4 and set(archive.files)=={'w0','w1','w2','w3'},'four distinct weight arrays required')
        arrays=tuple(archive['w'+str(i)] for i in range(4))
    shapes=((hidden,18),(hidden,),(1,hidden),(1,))
    for value,shape in zip(arrays,shapes):
        require(value.dtype.str=='<f8' and value.shape==shape and np.isfinite(value).all(),'finite float64 fixed-architecture weights required')
    digest=hashlib.sha256(canonical([RECIPE,FEATURES,TARGET,[x.shape for x in arrays]]))
    for value in arrays:digest.update(value.tobytes(order='C'))
    return arrays,digest.hexdigest()


def scores(arrays,x):
    x=np.asarray(x,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==18 and np.isfinite(x).all(),'finite 18-feature rows required')
    a,b,c,d=arrays
    with np.errstate(over='raise',invalid='raise'):
        hidden=np.tanh(np.dot(x,a.T)+b)
        result=np.tanh(np.dot(hidden,c.T)+d).reshape(-1)
    require(np.isfinite(result).all() and np.all(np.abs(result)<=1),'finite bounded priority output required')
    return result


def group_arrays(value):
    require(set(value)=={'features','targets'},'only model feature and target fields allowed')
    x=np.asarray(value['features'],dtype=np.float64);y=np.asarray(value['targets'],dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==18 and 1<=len(x)<=4096 and y.shape==(len(x),), 'invalid group shape')
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.abs(y)<=1+1e-12),'finite signed targets required')
    return x,y


def verify_export(data,manifest,cohort,receipt,arrays,held,*,expected_events,expected_sequences,expected_labels,progress=lambda value:None):
    """Compare every serialized group to the already independently admitted audit.

    Target/causal correctness comes from the separately bound full teacher
    oracle; this reader proves exact export coverage and checkpoint scores.
    """
    records=manifest['shards'];sequences=[r['sequence_id'] for r in records]
    require(sequences==sorted(expected_sequences) and len(set(sequences))==len(sequences),'complete unique ordered sequence shards required')
    require(len({r['path'] for r in records})==len(records),'distinct shard files required')
    for item in records:
        require(type(item['groups']) is int and item['groups']>=0 and type(item['rows']) is int and item['rows']>=0,'valid counts required')
    audit=contained(cohort,'audit.jsonl');require(sha(audit)==receipt['files']['audit.jsonl'],'admitted raw audit differs')
    event_path=contained(cohort,'events.json');require(sha(event_path)==receipt['files']['events.json'],'admitted event schedule differs')
    events=json.loads(event_path.read_bytes());require(len(events)==expected_events,'full event schedule required')
    seen=set();counts={s:dict(groups=0,rows=0,events=0) for s in sequences};losses={'fit':[],'held':[]};frames=labels=0
    started=time.monotonic()
    with ExitStack() as stack:
        streams={r['sequence_id']:stack.enter_context(contained(data,r['path']).open('rb')) for r in records}
        for item in records:require(sha(contained(data,item['path']))==item['sha256'],'serialized training shard changed')
        with audit.open('rb') as source:
            for line in source:
                row=json.loads(line);scene=row['sequence_id'];event_id=row['event_id']
                require(frames<len(events) and scene in counts and (scene,event_id) not in seen,'unexpected or duplicate teacher event')
                require((scene,event_id)==(events[frames]['sequence_id'],events[frames]['event_id']),'original event order differs')
                require(row['kind']=='persistent_exclusive_completion_teacher_raw_probe_witness_v1'
                    and row['training_trace_only'] is row['offline_counterfactual_probes'] is True,'raw offline teacher audit required')
                seen.add((scene,event_id));counts[scene]['events']+=1;frames+=1
                for operation in row['allocation_trace']:
                    record=operation['allocation_training']
                    require(record['feature_recipe']==RECIPE and record['target_recipe']==TARGET
                        and record['future_or_gt_inputs'] is False and record['labels_are_model_not_true_risk'] is True,'teacher feature/target scope differs')
                    expected={'features':[c['features'] for c in record['candidates']], 'targets':[c['target'] for c in record['candidates']]}
                    raw=streams[scene].readline();require(bool(raw),'export omitted a teacher group')
                    x,y=group_arrays(json.loads(raw));ex,ey=group_arrays(expected)
                    require(x.shape==ex.shape and np.array_equal(x,ex) and np.array_equal(y,ey),'export altered or reordered teacher features/targets')
                    prediction=scores(arrays,x)
                    squared=(prediction-y)**2
                    loss=math.fsum(map(float,squared))/len(squared)
                    require(math.isfinite(loss),'finite independent group loss required')
                    losses['held' if scene in held else 'fit'].append(loss)
                    counts[scene]['groups']+=1;counts[scene]['rows']+=len(y);labels+=len(y)
                if frames%100==0:
                    elapsed=time.monotonic()-started
                    progress(dict(completed_events=frames,total_events=expected_events,completed_labels=labels,
                        elapsed_seconds=elapsed,ETA_seconds=(expected_events-frames)*elapsed/frames,
                        ETA_scope='current serialization and fixed-weight audit only; event work varies'))
        for item in records:
            require(streams[item['sequence_id']].read()==b'','export contains extra groups')
            count=counts[item['sequence_id']]
            require(count['groups']==item['groups'] and count['rows']==item['rows'],'shard coverage differs')
            require(sha(contained(data,item['path']))==item['sha256'],'shard mutated during audit')
    require(frames==expected_events and labels==expected_labels and set(s for s,_ in seen)==set(expected_sequences),'full teacher export coverage required')
    require(sha(audit)==receipt['files']['audit.jsonl'] and sha(event_path)==receipt['files']['events.json'],'teacher inputs mutated')
    require(losses['fit'] and losses['held'],'both fit and holdout require groups')
    return dict(events=frames,labels=labels,sequences=counts,fit_groups=len(losses['fit']),holdout_groups=len(losses['held']),
        final_checkpoint_fit_mse=math.fsum(losses['fit'])/len(losses['fit']),
        final_checkpoint_holdout_mse=math.fsum(losses['held'])/len(losses['held']),
        exact_export_features_and_targets_verified=True,all_export_groups_independently_scored=True)


def verify_selection(epochs,checkpoint,summary):
    require(len(epochs)==10,'all ten epochs required')
    for i,row in enumerate(epochs,1):
        require(type(row['epoch']) is int and row['epoch']==i,'contiguous epochs required')
        require(row['fit_groups']==summary['fit_groups'] and row['holdout_groups']==summary['holdout_groups'],'per-epoch group coverage differs')
        require(row['official_validation_or_test_read'] is False,'test/val selection forbidden')
        for key in ('training_mse','train_sequence_holdout_mse'):require(math.isfinite(row[key]) and row[key]>=0,'finite loss required')
    best=min(range(10),key=lambda i:epochs[i]['train_sequence_holdout_mse'])
    require(checkpoint['selection']=='minimum_train_sequence_holdout_mse_earliest_tie'
        and type(checkpoint['selected_epoch']) is int and checkpoint['selected_epoch']==best+1,'minimum holdout with earliest exact tie required')
    require(checkpoint['selected_train_holdout_mse']==epochs[best]['train_sequence_holdout_mse'],'selected stored loss differs')
    recomputed=summary['final_checkpoint_holdout_mse'];stored=checkpoint['selected_train_holdout_mse']
    require(math.isclose(recomputed,stored,abs_tol=ATOL,rel_tol=RTOL),'fixed checkpoint does not reproduce selected holdout loss')
    return dict(selected_epoch=best+1,stored_holdout_mse=stored,independent_holdout_mse=recomputed,
        absolute_error=abs(recomputed-stored),atol=ATOL,rtol=RTOL)


def load_helper(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','teacher-cohort','teacher-journal','published-prerequisite','runtime-admission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();own=Path(__file__).resolve().parent
    freeze=read_json(own,'source-freeze.json')
    for name,entry in freeze['sources'].items():require(sha(own/name)==entry['sha256'],'independent audit source changed')
    require(sha(CONSUMER/'source-freeze.json')==CONSUMER_FREEZE_SHA,'qualified final consumer required')
    consumer_freeze=read_json(CONSUMER,'source-freeze.json')
    for name,entry in consumer_freeze['sources'].items():require(sha(CONSUMER/name)==entry['sha256'],'qualified source bytes changed')
    bridge=load_helper('final_priority_checkpoint_provenance',CONSUMER/'run_final_refit_priority_after_targets.py')
    source,_=bridge.source_gate(CONSUMER)
    gate=load_helper('final_priority_checkpoint_metadata',source/'transvision/models/event_track_v2x/exclusive_priority_admission.py')
    run=args.run.absolute();data=run/'data';fit=run/'fit'
    completion=read_json(run,'completion.json');run_plan=read_json(run,'plan.json')
    require(all(completion[k]==v for k,v in run_plan.items()),'completion does not match original bridge plan')
    require(completion['status']=='completed_pending_independent_checkpoint_admission'
        and completion['source_preparation_sha256']==bridge.PREPARATION_SHA
        and completion['bridge_sha256']==sha(CONSUMER/'run_final_refit_priority_after_targets.py'),'qualified completed final bridge required')
    manifest=read_json(data,'manifest.json',completion['training_manifest_sha256'])
    require(manifest['kind']=='exclusive_component_priority_counterfactual_train_v1'
        and manifest['solver_backend']=='exclusive_root_partition_regions_v1' and manifest['split']=='train'
        and manifest['fixture'] is manifest['future_or_gt_inputs'] is manifest['paper_eligible'] is False
        and manifest['labels_are_model_not_true_risk'] is True and manifest['full_source_first_arrival_trace'] is False,
        'real original-paired model-only train export required')
    gate.verify_exported(data,manifest)
    teacher=read_json(args.teacher_cohort,'receipt.json',completion['teacher_receipt_sha256'])
    binding=read_json(args.teacher_cohort,'collection-binding.json',teacher['files']['collection-binding.json'])
    require(binding==read_json(data,'teacher-collection-binding.json'),'exported collection binding differs')
    require(manifest['replay_receipt_sha256']==completion['teacher_receipt_sha256'],'exported teacher receipt differs')
    plan=read_json(args.teacher_cohort,'plan.json',teacher['files']['plan.json'])
    require(plan==manifest['independent_admission_plan'],'export uses a different teacher plan')
    target=read_json(data,'teacher-target-admission.json',completion['teacher_target_sha256'])
    main_proof=read_json(data,'main-independent-admission.json');byte=read_json(data,'main-byte-admission.json')
    require(completion['teacher_task_id']==target['task_id'] and completion['main_task_id']==main_proof['task_id'],'completed run used different upstream tasks')
    require(sha(args.published_prerequisite)==target['published_prerequisite_sha256'],'published main prerequisite changed')
    require(sha(args.runtime_admission)==completion['runtime_admission_sha256'],'runtime admission changed')
    jobs=[j for j in json.loads(args.teacher_journal.read_bytes())['jobs'] if j['task_id']==target['task_id']];require(len(jobs)==1,'one teacher attempt required')
    from clearml import Task
    lookup=lambda task_id:Task.get_task(task_id=task_id)
    bridge.live_provenance(lookup,jobs[0],main_proof,byte,target,binding,args.published_prerequisite)
    bridge.runtime_gate(args.runtime_admission,lookup)
    seed=completion['seed'];require(seed==manifest['upstream_identity_seed']==target['seed'] and type(seed) is int,'matching seed required')
    checkpoint=read_json(fit/str(seed),'checkpoint.json',completion['checkpoint_sha256'])
    weights_path=contained(fit/str(seed),'weights.npz')
    require(sha(weights_path)==checkpoint['weights_sha256']==completion['weights_sha256'],'weights changed')
    train_plan=read_json(fit,'plan.json',checkpoint['plan_sha256']);receipt=read_json(fit,'receipt.json')
    require(receipt['kind']=='component_priority_fit_receipt_v1' and receipt['status']=='complete'
        and receipt['seeds']==[dict(seed=seed,checkpoint_sha256=completion['checkpoint_sha256'],policy_signature=checkpoint['policy_signature'])],'complete matching fit receipt required')
    require(train_plan['seeds']==[seed] and train_plan['epochs']==10 and train_plan['hidden']==32 and train_plan['learning_rate']==.001,'frozen fitting hyperparameters required')
    require(train_plan['manifest_sha256']==checkpoint['training_manifest_sha256']==completion['training_manifest_sha256'],'training manifest binding differs')
    require(checkpoint['kind']=='exclusive_component_priority_model_progress_checkpoint_v1' and checkpoint['seed']==seed
        and checkpoint['split']=='train' and checkpoint['feature_recipe']==manifest['feature_recipe']==RECIPE
        and checkpoint['feature_names']==manifest['feature_names']==list(FEATURES)
        and checkpoint['target_recipe']==manifest['target_recipe']==TARGET,'checkpoint feature recipe differs')
    require(checkpoint['binding']==manifest['binding'] and checkpoint['source_sha256']==train_plan['source_sha256']==manifest['source_sha256'],'runtime binding changed')
    require(checkpoint['full_official_train_trace'] is manifest['full_official_train_trace'] is True
        and checkpoint['head_fit_sequence_isolated'] is True and checkpoint['head_fit_includes_holdout'] is False
        and checkpoint['official_validation_or_test_used_for_selection'] is False,'train-only head selection required')
    for obj in (checkpoint,train_plan,completion):require(obj['strict_pipeline_isolated_selection'] is obj['paper_eligible'] is False,'do not claim pipeline isolation or paper performance')
    sequences=plan['expected_sequences'];require(len(sequences)==46 and plan['expected_events']==7445,'complete original train scope required')
    fit_sequences,held=holdout_split(sequences)
    require(train_plan['fit_sequences']==checkpoint['fit_sequences']==fit_sequences and train_plan['holdout_sequences']==checkpoint['holdout_sequences']==held,'deterministic sequence split differs')
    require(train_plan['objective']=='equal_group_mean_squared_one_operation_model_progress'
        and train_plan['selection']==checkpoint['selection']=='minimum_train_sequence_holdout_mse_earliest_tie','frozen objective/selection required')
    arrays,signature=weights(weights_path)
    require(signature==checkpoint['policy_signature'] and signature!=checkpoint['initial_policy_signature'],'policy signature differs or weights did not change')
    output=args.output.absolute();require(output.is_relative_to(R/'artifacts') and not output.exists(),'new experiment output required')
    output.mkdir(parents=True,exist_ok=False)
    input_hashes={str(p):sha(p) for p in (run/'completion.json',data/'manifest.json',fit/'plan.json',fit/'receipt.json',
        fit/str(seed)/'checkpoint.json',fit/str(seed)/'weights.npz',fit/str(seed)/'epochs.jsonl')}
    try:
        summary=verify_export(data,manifest,args.teacher_cohort,teacher,arrays,set(held),expected_events=7445,
            expected_sequences=sequences,expected_labels=target['labels'],progress=lambda x:print(json.dumps(x),flush=True))
        epochs=[json.loads(line) for line in contained(fit/str(seed),'epochs.jsonl').read_bytes().splitlines()]
        selection=verify_selection(epochs,checkpoint,summary)
        for p,digest in input_hashes.items():require(sha(p)==digest,'input changed during numerical audit')
        proof=output/'local-numeric-audit.json'
        new(proof,dict(kind='rbf_final_priority_local_full_export_and_selected_checkpoint_numeric_audit_v1',
            seed=seed,teacher_task_id=target['task_id'],main_task_id=main_proof['task_id'],input_hashes=input_hashes,
            source_freeze_sha256=sha(own/'source-freeze.json'),consumer_freeze_sha256=CONSUMER_FREEZE_SHA,
            policy_signature=signature,export=summary,selection=selection,
            producer_model_or_optimizer_imported=False,optimizer_trajectory_independently_recomputed=False,
            cloud_fit_artifacts_independently_read_back=False,learned_replay_accepted=False,
            strict_pipeline_isolated_selection=False,full_Stage2_complete=False,paper_performance_complete=False))
        register(proof,'rbf-final-priority-local-checkpoint-numeric-audit');print(json.dumps(dict(receipt=str(proof))),flush=True)
    except BaseException as error:
        failure=output/'failure.json';new(failure,dict(error_type=type(error).__name__,message=str(error),automatic_retry=False,accepted=False));register(failure,'rbf-final-priority-checkpoint-audit-failure');raise


if __name__=='__main__':main()
