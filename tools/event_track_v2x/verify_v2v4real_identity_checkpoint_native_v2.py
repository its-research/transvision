#!/usr/bin/env python3
"""Independent native checkpoint provenance and every-row float64 NN audit.

Uses the unchanged, SHA-bound NumPy reference from the SPD numerical audit.
No trainer, Torch, production feature/identity/motion modules are imported.
This checks supplied local bytes; separate cloud readback remains required.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np

REFERENCE_SHA='cb86ff0ef14d78f75bc15de78683e7d2888310ae995e6447f70cbf41f15d0cf4'
HEADS=('cross_source','temporal')
ATOL=RTOL=1e-4
FIT=dict(epochs=10,batch_size=64,hidden=128,heads=4,dropout=.1,learning_rate=.0003,
    weight_decay=.0001,gradient_clip=5.,geometry_weight=1.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def contained(root,name):
    p=Path(name);root=Path(root).absolute()
    if (not isinstance(name,str) or not name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name):
        raise ValueError('canonical contained payload required')
    target=root/p
    if not target.is_file() or any(p.is_symlink() for p in (target,*target.parents)):
        raise ValueError('ordinary contained payload required')
    target.resolve().relative_to(root.resolve());return target


def read(path,digest,pins):
    p=Path(path).absolute()
    if not p.is_file() or any(v.is_symlink() for v in (p,*p.parents)) or sha(p)!=digest:
        raise ValueError('supplied input byte identity differs')
    pins[str(p)]=digest;return json.loads(p.read_bytes())


def new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def model_digest(weights):
    tensors={}
    for name,value in sorted(weights.items()):
        a=np.ascontiguousarray(value,dtype='<f4')
        tensors[name]=hashlib.sha256(str((tuple(a.shape),'torch.float32')).encode()+a.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def phase_score(phase,records,world,*,training):
    """Recompute two head macro means and visit counts from complete history."""
    rows=phase['sequences']
    if [r['sequence_id'] for r in rows]!=[r['sequence_id'] for r in records]:
        raise ValueError('selection/refit phase sequence support differs')
    for row,record in zip(rows,records):
        if row['supervised_rows']!=record['supervised_rows']:raise ValueError('phase skipped admitted supervised rows')
        for head in HEADS:
            x=row[head]
            if (type(x['samples']) is not int or not 0<=x['samples']<=record['supervised_rows']
                    or not math.isfinite(x['loss_sum']) or x['loss_sum']<0 or (x['samples']==0 and x['loss_sum']!=0)):
                raise ValueError('invalid named identity head loss/count')
    means={};counts={}
    for h in HEADS:
        values=[r[h]['loss_sum']/r[h]['samples'] for r in rows if r[h]['samples']]
        if not values:raise ValueError('phase lacks informative '+h)
        means[h]=math.fsum(values)/len(values);counts[h]=len(values)
    score=math.fsum(means.values())
    if means!=phase['head_macro_means'] or counts!=phase['informative_sequences'] or score!=phase['selection_score']:
        raise ValueError('checkpoint selection macro criterion differs')
    total=sum(r['supervised_rows'] for r in records)
    expected_batches=sum((r['supervised_rows']+63)//64 for r in records)
    coverage=phase['rank_coverage']
    expected_local=[sum((r['supervised_rows']//64)*(64//world)+max(0,(r['supervised_rows']%64+world-1-rank)//world)
        for r in records) for rank in range(world)]
    if (phase['supervised_rows']!=total or phase['global_batches']!=expected_batches
            or len(coverage)!=world or [r['rank'] for r in coverage]!=list(range(world))
            or [r['local_rows'] for r in coverage]!=expected_local or any(r['batches']!=expected_batches for r in coverage)
            or len({r['model_sha256'] for r in coverage})!=1
            or (training and not 1<=phase['optimizer_steps']<=expected_batches)
            or (not training and phase['optimizer_steps']!=0)
            or not math.isfinite(phase['maximum_preclip_gradient_norm'])):
        raise ValueError('native phase optimizer/rank/row coverage differs')
    return score,coverage[0]['model_sha256']


def provenance(args,reference,pins):
    data=Path(args.data).absolute();training=Path(args.training).absolute()
    manifest=read(contained(data,'manifest.json'),args.manifest_sha256,pins)
    admitted=read(args.row_admission,args.row_admission_sha256,pins)
    partition=read(args.partition,args.partition_sha256,pins)
    receipt=read(contained(training,'receipt.json'),args.receipt_sha256,pins)
    ck=read(contained(training,'final/checkpoint.json'),receipt['checkpoint_sha256'],pins)
    plan=read(contained(training,'plan-before-updates.json'),receipt['plan_sha256'],pins)
    if (args.seed not in (1337,2027,3407) or admitted['seed']!=args.seed or admitted['sequences']!=32
            or admitted['official_test_labels_read'] is not False
            or admitted['kind']!='v2v4real_identity_rows_independent_native_target_v2_admission'
            or not all(admitted[k] is True for k in ('all_original_cache_nodes_and_prediction_only_features_verified',
                'all_causal_parent_contexts_independently_verified','all_native_temporal_cross_source_partial_targets_independently_verified'))
            or admitted['input_sha256'].get(str(data/'manifest.json'))!=args.manifest_sha256):
        raise ValueError('full seed-bound native independent row proof required')
    records=manifest['shards'];sequences=manifest['sequences'];p=manifest['provenance']
    if (manifest['kind']!='persistent_forest_train_rows_v1' or manifest['split']!='train'
            or manifest['scheduled_frames']!=7105 or manifest['sealed_source_frames']!=14210
            or manifest['ingested_source_frames']!=14210 or manifest['source_unavailable']!=0
            or manifest['labels_in_model_inputs'] is not False or manifest['gt_boxes_or_ids_in_shards'] is not False
            or p['dataset']!='v2v4real' or p['seed']!=args.seed or p['full_official_train_verified'] is not True
            or p['official_test_labels_read'] is not False or p['native_class_scope']!=['vehicle']
            or p['network_class_slot_mapping']!={'car':'vehicle'} or p['original_nested_partition_sha256']!=args.partition_sha256
            or partition['official_split']!='train' or partition['official_test_used_for_partition'] is not False
            or p['sequence_to_name_group']!=partition['sequence_to_name_group'] or sequences!=sorted(set(sequences))
            or len(sequences)!=32 or [r['sequence_id'] for r in records]!=sequences
            or set(sequences)!=set(partition['sequence_to_name_group']) or len({r['path'] for r in records})!=32
            or sum(r['nodes'] for r in records)!=admitted['nodes']
            or [(r['sequence_id'],r['nodes'],r['reason_counts']) for r in records]!=[
                (r['sequence_id'],r['nodes'],r['reason_counts']) for r in admitted['results']]):
        raise ValueError('full native train manifest/independent row scope differs')
    fit=set(partition['identity_fit']);holdout=set(partition['identity_selection'])
    if not fit or not holdout or fit&holdout:raise ValueError('original disjoint fit/selection groups required')
    roles={s:'identity_fit' if g in fit else 'identity_selection' if g in holdout else None
        for s,g in partition['sequence_to_name_group'].items()}
    if None in roles.values() or roles!=p['sequence_identity_roles']:raise ValueError('original roles differ')
    groups={role:[r for r in records if roles[r['sequence_id']]==role] for role in ('identity_fit','identity_selection')}
    if (admitted.get('all_prediction_context_fields_byte_identical_to_admitted_rows') is not True
            or admitted.get('target_and_byte_checks_exact') is not True
            or p.get('native_target_contract')!='author_native_mapped_id_one_explicit_association_v2'
            or ck.get('native_target_contract')!=p['native_target_contract']
            or plan.get('native_target_contract')!=p['native_target_contract']
            or plan.get('kind')!='v2v4real_native_joint_identity_selection_full_refit_plan_v2'):
        raise ValueError('corrected native v2 checkpoint/training source targets required')
    for field in ('cross_source_positive_parent_edges','temporal_positive_parent_edges'):
        byscene={r['sequence_id']:r[field] for r in admitted['results']}
        if admitted.get(field,0)<=0 or sum(byscene.values())!=admitted[field]:raise ValueError('real positive '+field+' required')
        for role,rr in groups.items():
            if any(r.get(field)!=byscene[r['sequence_id']] for r in rr) or sum(r[field] for r in rr)<=0:
                raise ValueError('both train roles need nonzero independent '+field)
    world=ck['distributed_world_size'];runtime=receipt['runtime']
    if (world not in (4,8) or len(runtime)!=world or [r['rank'] for r in runtime]!=list(range(world))
            or len({r['gpu_uuid'] for r in runtime})!=world or len({r['host'] for r in runtime})!=1
            or any(r['world_size']!=world or r['gpu_uuid']=='unavailable' or 'L40' in r['gpu_name']
                or r['tf32_matmul'] is not False or r['tf32_cudnn'] is not False for r in runtime)):
        raise ValueError('four/eight distinct eligible CUDA rank runtime required')
    if (receipt['kind']!='v2v4real_native_joint_identity_selection_full_refit_candidate_v1'
            or receipt['completed'] is not True or receipt['seed']!=args.seed or ck['seed']!=args.seed
            or ck['kind']!='persistent_forest_row_identity_checkpoint_v1' or ck['dataset']!='v2v4real'
            or ck['architecture']!=dict(hidden=128,heads=4,dropout=.1) or ck['data_split']!='train'
            or ck['dataset_sha256']!=args.manifest_sha256 or ck['row_protocol']!=manifest['row_protocol']
            or ck['frozen_cache_identity']!=manifest['frozen_cache_identity'] or ck['geometry_weight']!=1.
            or ck['labels_in_model_inputs'] is not False or ck['local_partial_label_objective'] is not True
            or ck['validation_or_test_selection'] is not False or ck['fixture'] is not False
            or ck['native_class_scope']!=['vehicle'] or ck['network_class_slot_mapping']!={'car':'vehicle'}
            or ck['strict_pipeline_isolated_selection'] is not False or ck['train_test_overlap_disclosed'] is not True
            or ck['plan_sha256']!=receipt['plan_sha256'] or ck['source_sha256']!=plan['source_sha256']
            or plan['fit_config']!=FIT or plan['seed']!=args.seed or plan['manifest_sha256']!=args.manifest_sha256
            or plan['row_admission_sha256']!=args.row_admission_sha256 or plan['partition_sha256']!=args.partition_sha256
            or plan['official_test_labels_read'] is not False or plan['train_test_overlap_disclosed'] is not True
            or plan['physical_session_independence_claimed'] is not False or plan['strict_pipeline_isolated_selection'] is not False
            or plan['upstream_sequence_isolated']!=partition['upstream_sequence_isolated']
            or plan['rank_runtime']!=runtime or ck['partition']!=dict(fit=sequences,holdout=[])):
        raise ValueError('native training/checkpoint/frozen recipe binding differs')
    expected_selection=dict(fit=[r['sequence_id'] for r in groups['identity_fit']],holdout=[r['sequence_id'] for r in groups['identity_selection']])
    if ck['selection_partition']!=expected_selection or plan['fit_sequences']!=expected_selection['fit'] or plan['holdout_sequences']!=expected_selection['holdout']:
        raise ValueError('frozen original selection sources differ')
    for name,h in plan['source_sha256'].items():
        read_source=contained(args.source_root,name)
        if sha(read_source)!=h:raise ValueError('actual frozen training/scoring source differs')
        pins[str(read_source)]=h
    selected=read(contained(training,'selection/selected-epoch.json'),ck['selection_receipt_sha256'],pins)
    history=selected['history']
    if selected['official_test_read'] is not False or [r['epoch'] for r in history]!=list(range(1,11)):
        raise ValueError('complete ten-epoch original train selection history required')
    scores=[]
    for entry in history:
        e=entry['epoch'];path=contained(training,f'selection/epoch-{e:04d}/epoch.json')
        if json.loads(path.read_bytes())!=entry:raise ValueError('original epoch history bytes disagree')
        pins[str(path)]=sha(path)
        _,fit_model=phase_score(entry['fit'],groups['identity_fit'],world,training=True)
        score,holdout_model=phase_score(entry['holdout'],groups['identity_selection'],world,training=False)
        wp=contained(training,f'selection/epoch-{e:04d}/weights.pt');pins[str(wp)]=sha(wp)
        if model_digest(reference.weights(wp))!=entry['model_sha256'] or fit_model!=holdout_model or fit_model!=entry['model_sha256']:
            raise ValueError('selection checkpoint/phase actual model bytes differ')
        scores.append((score,e))
    epoch=min(scores)[1]
    if selected['selected_epoch']!=epoch or ck['selected_epoch']!=epoch or receipt['selected_epoch']!=epoch:
        raise ValueError('frozen earliest minimum train holdout epoch differs')
    refit=receipt['refit_history']
    if [r['epoch'] for r in refit]!=list(range(1,epoch+1)):raise ValueError('selected full-train epoch count differs')
    for entry in refit:
        path=contained(training,f"final/epoch-{entry['epoch']:04d}.json")
        if json.loads(path.read_bytes())!=entry:raise ValueError('original full-refit epoch metadata differs')
        pins[str(path)]=sha(path);_,digest=phase_score(entry['fit'],records,world,training=True)
        if digest!=entry['model_sha256']:raise ValueError('refit rank/model history differs')
    weights_path=contained(training,'final/'+ck['weights']['path'])
    if sha(weights_path)!=ck['weights']['sha256'] or receipt['weights_sha256']!=sha(weights_path):raise ValueError('final weights bytes differ')
    pins[str(weights_path)]=sha(weights_path);weights=reference.weights(weights_path)
    initial=contained(training,'initial-weights.pt');pins[str(initial)]=sha(initial)
    if (sha(initial)!=receipt['initial_weights_sha256'] or model_digest(reference.weights(initial))!=ck['initial_model_sha256']
            or model_digest(weights)!=ck['model_sha256'] or ck['model_sha256']==ck['initial_model_sha256']
            or refit[-1]['model_sha256']!=ck['model_sha256']):raise ValueError('actual initial/final/last epoch weights differ')
    for record in records:
        path=contained(data,record['path'])
        if sha(path)!=record['sha256']:raise ValueError('full admitted input shard bytes differ')
        pins[str(path)]=record['sha256']
    return data,training,manifest,receipt,ck,weights


def verify(args):
    ref=Path(args.reference).absolute()
    if sha(ref)!=REFERENCE_SHA or any(p.is_symlink() for p in (ref,*ref.parents)):
        raise ValueError('unchanged independent NumPy reference required')
    spec=importlib.util.spec_from_file_location('unchanged_joint_float64_reference',ref)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    pins={str(ref):REFERENCE_SHA};out=Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh ordinary independent output required')
    data,training,manifest,receipt,ck,weights=provenance(args,reference,pins)
    out.mkdir(parents=True);done=failed=0;maximum=0.;results=[];started=time.monotonic();world=ck['distributed_world_size']
    new(out/'plan.json',dict(kind='v2v4real_full_train_identity_independent_float64_plan_v1',seed=args.seed,
        receipt_sha256=args.receipt_sha256,checkpoint_sha256=receipt['checkpoint_sha256'],weights_sha256=ck['weights']['sha256'],
        row_admission_sha256=args.row_admission_sha256,reference_sha256=REFERENCE_SHA,atol=ATOL,rtol=RTOL,
        actual_cloud_registered_bytes_independently_read=False,all_rows_selected_without_targets=True,no_Torch_or_production_NN_imports=True))
    try:
        forward=receipt['all_row_forward']
        if len(forward)!=world or [r['rank'] for r in forward]!=list(range(world)):raise ValueError('full forward rank scope differs')
        with (out/'failed-rows.jsonl').open('x') as failures:
            for rank,rank_proof in enumerate(forward):
                records=manifest['shards'][rank::world];path=contained(training,rank_proof['path'])
                if (path.name!=f'predictions-rank{rank}.jsonl' or sha(path)!=rank_proof['sha256']
                        or rank_proof['model_sha256']!=ck['model_sha256'] or rank_proof['world_size']!=world
                        or rank_proof['target_masks_used_to_select_forward_rows'] is not False
                        or rank_proof['rows']!=sum(r['nodes'] for r in records)
                        or rank_proof['sequences']!=[dict(sequence_id=r['sequence_id'],rows=r['nodes'],source_shard_sha256=r['sha256']) for r in records]):
                    raise ValueError('exact all-row forward artifact/rank coverage differs')
                pins[str(path)]=rank_proof['sha256']
                with path.open() as stream:
                    for record in records:
                        with np.load(contained(data,record['path']),allow_pickle=False) as payload:
                            a={k:payload[k] for k in ('features','source','information_us','arrival_us','mean','covariance','state_us','lengths','contexts','decision_us','node_id')}
                        scene_max=0.;scene_failed=0
                        for start in range(0,record['nodes'],64):
                            indices=np.arange(start,min(start+64,record['nodes']),dtype=np.int64)
                            vectors=reference.reference(weights,a,indices)
                            for i,v in zip(indices,vectors):
                                row=json.loads(next(stream));length=int(a['lengths'][i])
                                if (set(row)!=set(('sequence_id','row','node_id','decision_us','context_indices','logits'))
                                        or type(row['row']) is not int or row['row']!=int(i) or row['sequence_id']!=record['sequence_id']
                                        or row['node_id']!=str(a['node_id'][i]) or row['decision_us']!=int(a['decision_us'][i])
                                        or row['context_indices']!=a['contexts'][i,:length].tolist()):raise ValueError('forward row identity/support differs')
                                got=np.asarray(row['logits'],dtype=np.float64)
                                if got.shape!=v.shape or not np.isfinite(got).all() or not np.isfinite(v).all():raise ValueError('finite full row logits required')
                                error=float(np.max(np.abs(got-v)));maximum=max(maximum,error);scene_max=max(scene_max,error)
                                if not np.allclose(got,v,atol=ATOL,rtol=RTOL):
                                    failed+=1;scene_failed+=1;failures.write(json.dumps(dict(sequence_id=record['sequence_id'],row=int(i),max_absolute_error=error,
                                        actual=got.tolist(),independent_float64=v.tolist()))+'\n')
                            done+=len(indices)
                        results.append(dict(sequence_id=record['sequence_id'],rows=record['nodes'],failed_rows=scene_failed,max_absolute_error=scene_max))
                        print(json.dumps(dict(stage='full_native_identity_independent_float64',seed=args.seed,completed_rows=done,
                            total_rows=sum(r['nodes'] for r in manifest['shards']),failed_rows=failed)),flush=True)
                    if stream.read().strip():raise ValueError('unexpected extra rank output rows')
        if len(results)!=32 or len({r['sequence_id'] for r in results})!=32 or done!=sum(r['nodes'] for r in manifest['shards']):
            raise ValueError('complete native 32-sequence numerical cohort required')
        for p,h in pins.items():
            if sha(p)!=h:raise ValueError('actual inputs changed during independent admission')
        proof=dict(kind='v2v4real_native_joint_identity_local_checkpoint_full_forward_independent_v1',seed=args.seed,
            full_independent_numeric_pass=failed==0,rows=done,sequences=results,numeric_failed_rows=failed,max_absolute_error=maximum,
            selected_epoch=ck['selected_epoch'],checkpoint_sha256=receipt['checkpoint_sha256'],weights_sha256=ck['weights']['sha256'],
            manifest_sha256=args.manifest_sha256,row_admission_sha256=args.row_admission_sha256,partition_sha256=args.partition_sha256,
            selection_metadata_macro_and_original_roles_independently_verified=True,all_actual_initial_epoch_final_weights_read=True,
            unchanged_reference_sha256=REFERENCE_SHA,source_sha256=sha(__file__),input_sha256=pins,atol=ATOL,rtol=RTOL,
            actual_cloud_registered_bytes_independently_read=False,train_test_overlap_disclosed=True,physical_session_independence_claimed=False,
            full_online_RBF_accepted=False,paper_metrics_complete=False,seconds=time.monotonic()-started)
        new(out/'acceptance-receipt.json',proof)
        if failed:raise ValueError('retain native numerical failures at original tolerance')
    except BaseException as e:
        new(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),completed_rows=done,numeric_failed_rows=failed,
            full_independent_numeric_pass=False,partials_and_failures_preserved=True,source_sha256=sha(__file__)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[1337,2027,3407],required=True)
    for name in ('data','manifest-sha256','row-admission','row-admission-sha256','partition','partition-sha256',
            'training','receipt-sha256','source-root','reference','output'):p.add_argument('--'+name,required=True)
    verify(p.parse_args())
