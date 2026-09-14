#!/usr/bin/env python3
"""Separate, evaluator-only car GT and native metrics for a complete train sequence.

Never labels a train result as validation. Loads only hash-pinned trusted local
train conversion bytes, selects raw cooperative labels by the sealed schedule,
and reuses the unchanged official metric adapter. No GT reaches a predictor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import itertools
import math
from pathlib import Path
import pickle
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tools/event_track_v2x/evaluate_source_ablation_v2.py'
SOURCE_SHA='a9e80719682477c3f1a4ab6030d906d20f848d2584578e3fb8dfe45c6f399356'
TRAIN_CACHE_SHA='1137740ecdf2aca7372536998ac485585ca89f4792bf68e287fa75d2e07a6fa0'
VEHICLE_INFOS_SHA='d4b328e4e83d73b09d6961aec05747c13647ead92b576d6605da3a8f2d2577bb'
EVENT_SOURCE=ROOT/'tools/event_track_v2x/analyze_mechanism_events_v2.py'
EVENT_SOURCE_SHA='e3ad276c1443538f5b7b6b99dce2d7908295335ede9d143a9249f5294cd48963'


def sha(path):
    with Path(path).open('rb') as stream:
        value=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''): value.update(block)
    return value.hexdigest()


def read(path,digest):
    if sha(path)!=digest: raise ValueError('input identity differs: '+str(path))
    return json.loads(Path(path).read_bytes())


def evaluator():
    if sha(SOURCE)!=SOURCE_SHA: raise ValueError('sealed evaluator dependency changed')
    spec=importlib.util.spec_from_file_location('sealed_car_metric_dependency',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.load_adapter()


def new_directory(path):
    path=Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('new nonsymlink evaluator output required')
    path.mkdir()
    return path


def timestamp(value):
    if isinstance(value,bool) or isinstance(value,str) and not value.isdigit():
        raise ValueError('integer microsecond timestamp required')
    result=int(value)
    if result<0 or not isinstance(value,str) and result!=value:
        raise ValueError('integer microsecond timestamp required')
    return result


def identity_events(adapter,gt,predictions,tracking_path):
    """Reuse strict GT overlap and censored first-ID disagreement diagnostics.

    Regret bounds and internal recoveries are reported alongside observations,
    never interpreted as calibrated GT error probabilities or recovery success.
    """
    if sha(EVENT_SOURCE)!=EVENT_SOURCE_SHA: raise ValueError('sealed identity event dependency changed')
    spec=importlib.util.spec_from_file_location('sealed_train_identity_events',EVENT_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not gt or len(gt)!=len(predictions): raise ValueError('complete nonempty event input required')
    sequence=gt[0]['sequence_id'];identity=module.IdentityAudit(sequence)
    previous='0'*64;rows=[]
    counts=Counter(frames=0,duplicate_gt_frames=0,duplicate_excess_predictions=0,
        frames_with_duplicates=0,recovery_event_records=0,frames_with_recovery_records=0,
        recovery_unreported_components=0,recovery_unreported_frames=0,fallback_events=0,fallback_unreported_events=0,
        high_model_bound_events=0,model_bound_unreported_events=0)
    with Path(tracking_path).open('rb') as stream:
        for frame,pred,raw in itertools.zip_longest(gt,predictions,stream):
            if frame is None or pred is None or raw is None: raise ValueError('tracking audit event coverage differs')
            audit=json.loads(raw)['tracking']
            if (frame['sequence_id']!=sequence
                    or (audit['sequence_id'],audit['event_id'])!=(sequence,frame['frame_id'])
                    or audit['prediction_sha256']!=pred['commit_sha256']
                    or audit['previous_audit_sha256']!=previous):
                raise ValueError('tracking audit prediction or chain differs')
            previous=hashlib.sha256(adapter.canonical(audit)).hexdigest()
            bound=audit.get('model_regret_upper');fallback=audit.get('global_fallback_used')
            if bound is not None and (type(bound) not in (int,float) or not math.isfinite(bound) or not 0<=bound<=1):
                raise ValueError('invalid model regret bound')
            if fallback is not None and type(fallback) is not bool: raise ValueError('invalid fallback flag')
            labels,duplicates,ground=module.map_boxes(pred['predictions'],frame,adapter)
            unique={label['gt_id']:box['track_id'] for box,label in zip(pred['predictions'],labels) if label['status']=='unique'}
            before=identity.counts.copy()
            identity.step(frame['box_reference_timestamp_us'],[b['track_id'] for b in ground],unique)
            mapping=Counter(label['status'] for label in labels)
            counts.update({'prediction_'+key:value for key,value in mapping.items()})
            components=audit.get('components')
            if components is not None and type(components) is not list: raise ValueError('invalid component audit list')
            recoveries=0;unreported=0
            for component in components or []:
                events=component.get('recovery_events')
                if events is None:
                    unreported+=1
                elif type(events) is not list: raise ValueError('invalid recovery event list')
                else: recoveries+=len(events)
            counts.update(frames=1,duplicate_gt_frames=len(duplicates),
                duplicate_excess_predictions=sum(len(v)-1 for v in duplicates.values()),
                frames_with_duplicates=int(bool(duplicates)),recovery_event_records=recoveries,
                frames_with_recovery_records=int(recoveries>0),recovery_unreported_components=unreported,
                recovery_unreported_frames=int(components is None or unreported>0),
                fallback_events=int(fallback is True),fallback_unreported_events=int(fallback is None),
                high_model_bound_events=int(bound is not None and bound>=.99),
                model_bound_unreported_events=int(bound is None))
            rows.append(dict(sequence_id=sequence,frame_id=frame['frame_id'],
                reference_timestamp_us=frame['box_reference_timestamp_us'],mapping_counts=dict(mapping),
                roi_gt_objects=len(ground),identity_unique_gt_frames=len(unique),
                identity_unknown_gt_frames=len(ground)-len(unique),
                anchor_disagreement_gt_frames=identity.counts['anchor_disagreement_gt_frames']-before['anchor_disagreement_gt_frames'],
                duplicate_gt_frames=len(duplicates),model_regret_upper=bound,fallback=fallback,
                recovery_event_records=recoveries if components is not None and not unreported else None))
    episodes=identity.finish();counts.update(identity.counts)
    counts.update(anchor_disagreement_episodes=len(episodes),
        anchor_episodes_left_censored=sum(e['left_censored'] for e in episodes),
        anchor_episodes_right_censored=sum(e['right_censored'] for e in episodes),
        anchor_episodes_return_observed=sum(not e['right_censored'] for e in episodes))
    spans=[e['observed_error_span_seconds'] for e in episodes]
    return dict(kind='train_prediction_identity_event_diagnostic_v1',counts=dict(counts),frames=rows,episodes=episodes,
        observed_error_span_seconds_quantiles=None if not spans else np.quantile(spans,[.5,.95,.99,1.]).tolist(),
        quantile_order=['p50','p95','p99','max'],
        protocol={k:module.PROTOCOL[k] for k in ('iou_threshold','comparison','roi','mapping','duplicates',
            'identity_proxy','identity_censoring','identity_duration','initial_anchor','unknown_policy')},
        event_dependency_sha256=EVENT_SOURCE_SHA,gt_for_evaluation_only=True,
        internal_recovery_is_not_gt_identity_recovery=True,model_bound_is_not_gt_error_probability=True,
        gt_error_calibration_computed=False,causal_effect_identified=False)


def car_frame(adapter, annotations, info, row):
    sid,fid=row['sequence_id'],row['vehicle_frame']
    if info['scene_token']!=sid or timestamp(info['timestamp'])!=row['box_reference_timestamp_us']:
        raise ValueError('pose and prediction schedule differ')
    rot,trans,ego=adapter._pose(info)
    objects=[];seen=set()
    for annotation in annotations:
        if timestamp(annotation['veh_pointcloud_timestamp'])!=row['box_reference_timestamp_us']:
            raise ValueError('raw cooperative GT time differs')
        if annotation['type'] not in adapter.CLASS_MAP: raise ValueError('unknown cooperative category')
        if adapter.CLASS_MAP.get(annotation['type'])!='car': continue
        tid=annotation['track_id']
        if not isinstance(tid,str) or not tid or tid in seen:
            raise ValueError('invalid or duplicate car GT identity; no repair')
        seen.add(tid)
        location=np.array([annotation['3d_location'][k] for k in ('x','y','z')],dtype=float)
        dimensions=np.array([annotation['3d_dimensions'][k] for k in ('l','w','h')],dtype=float)
        heading=np.array([math.cos(annotation['rotation']),math.sin(annotation['rotation']),0.])@rot
        state=np.r_[location@rot+trans,dimensions,math.atan2(heading[1],heading[0]),0.,0.]
        if not np.isfinite(state).all() or np.any(dimensions<=0): raise ValueError('invalid car GT state')
        objects.append(dict(track_id=tid,class_label='car',mean=state.tolist(),
            annotation_token=annotation['token'],velocity_available=False))
    return dict(sequence_id=sid,frame_id=fid,box_reference_timestamp_us=row['box_reference_timestamp_us'],
        ego_translation_world=ego.tolist(),objects=sorted(objects,key=lambda box:box['track_id']))


def prepare(plan_path,plan_sha256,projection,vehicle_infos,target_audit,target_audit_sha256,output):
    module,adapter=evaluator()
    plan=read(plan_path,plan_sha256)
    if (plan.get('kind')!='train_sequence_inference_diagnostic_plan_v1'
            or plan.get('cohort_mode')!='real-train-development' or plan.get('class_scope')!=['car']
            or plan.get('complete_input_train_cohort_verified') is not True or plan['cache_sha256']!=TRAIN_CACHE_SHA):
        raise ValueError('sealed real-train inference plan required')
    audit=read(target_audit,target_audit_sha256)
    if audit['converted_train_sha256']['vehicle-side']!=VEHICLE_INFOS_SHA or sha(vehicle_infos)!=VEHICLE_INFOS_SHA:
        raise ValueError('untrusted conversion bytes; refusing pickle')
    metadata=Path(projection)/'cooperative/data_info.json'
    pairs=read(metadata,plan['cooperative_metadata_sha256'])
    if audit['cooperative_metadata_sha256']!=plan['cooperative_metadata_sha256'] or len(pairs)!=7445:
        raise ValueError('full train metadata identity differs')
    selected=[p for p in pairs if p['vehicle_sequence']==plan['selected_sequence']]
    if any(p['infrastructure_sequence']!=plan['selected_sequence'] for p in selected):
        raise ValueError('cooperative sequence mismatch')
    inventory={r['path']:r for r in audit['cooperative_label_inventory']}
    if len(inventory)!=len(audit['cooperative_label_inventory']): raise ValueError('duplicate label inventory')
    # Whitelisted, previously audited train-only local bytes. Annotation arrays
    # in this pickle are not used; raw cooperative GT is prepared separately.
    payload=Path(vehicle_infos).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=VEHICLE_INFOS_SHA:
        raise ValueError('conversion bytes changed before unpickling')
    infos=pickle.loads(payload)['infos'];del payload
    poses={str(info['token']):info for info in infos}
    if len(poses)!=len(infos): raise ValueError('duplicate train pose frame')
    rows=sorted([dict(sequence_id=p['vehicle_sequence'],vehicle_frame=p['vehicle_frame'],
        infrastructure_frame=p['infrastructure_frame'],box_reference_timestamp_us=timestamp(poses[p['vehicle_frame']]['timestamp']))
        for p in selected],key=lambda r:r['box_reference_timestamp_us'])
    if not rows or rows!=plan['selected_schedule'] or len(rows)!=plan['scheduled_frames']:
        raise ValueError('selected sequence is not complete or time alignment differs')
    frames,sources=[],{str(plan_path):plan_sha256,str(metadata):plan['cooperative_metadata_sha256'],
        str(vehicle_infos):VEHICLE_INFOS_SHA,str(target_audit):target_audit_sha256,str(SOURCE):SOURCE_SHA}
    for row in rows:
        if not re.fullmatch(r'\d{6}',row['vehicle_frame']): raise ValueError('unsafe cooperative frame name')
        relative='cooperative/label/'+row['vehicle_frame']+'.json'
        path=Path(projection)/relative
        record=inventory[relative]
        annotations=read(path,record['sha256'])
        if path.stat().st_size!=record['bytes']: raise ValueError('label size differs')
        sources[str(path)]=record['sha256']
        frames.append(car_frame(adapter,annotations,poses[row['vehicle_frame']],row))
    if any(sha(path)!=digest for path,digest in sources.items()): raise ValueError('GT source changed during preparation')
    output=new_directory(output)
    with (output/'ground-truth.jsonl').open('xb') as stream:
        for frame in frames: stream.write(adapter.canonical(frame)+b'\n')
    manifest=dict(kind='spd_train_sequence_evaluator_ground_truth_v1',selected_schedule=rows,
        source_sha256=sources,ground_truth_sha256=sha(output/'ground-truth.jsonl'),
        frames=len(frames),class_scope=['car'],gt_objects=sum(len(f['objects']) for f in frames),
        evaluator_only=True,contains_train_payload=True,contains_test_payload=False,
        full_official_train=False,validation=False,paper_eligible=False,preparer_sha256=sha(Path(__file__)))
    module.write_json(output/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','selected_schedule')},sort_keys=True))
    return manifest


def evaluate(ground_truth,manifest_sha256,replay,receipt_sha256,output):
    module,adapter=evaluator()
    manifest=read(Path(ground_truth)/'manifest.json',manifest_sha256)
    if (manifest['kind']!='spd_train_sequence_evaluator_ground_truth_v1' or manifest['class_scope']!=['car']
            or manifest.get('evaluator_only') is not True or manifest.get('contains_train_payload') is not True
            or manifest.get('contains_test_payload') is not False or manifest.get('validation') is not False):
        raise ValueError('separate car train evaluator GT required')
    replay=Path(replay)
    final=read(replay/'development-inference-receipt.json',receipt_sha256)
    receipt=read(replay/'receipt.json',final['replay_receipt_sha256'])
    plan=read(replay/'plan.json',receipt['plan_sha256'])
    if (final['kind']!='train_sequence_inference_diagnostic_v1' or final['status']!='complete'
            or final['plan_sha256']!=receipt['plan_sha256'] or receipt['status']!='complete'
            or receipt['cache_split']!='train' or final['cohort_mode']!='real-train-development'
            or final['complete_selected_sequence_verified'] is not True or receipt['allocation_teacher'] is not False
            or final.get('training_performed') is not False or final.get('offline_teacher_probes') is not False
            or receipt.get('learned_identity_enabled') is not True
            or plan.get('kind')!='train_sequence_inference_diagnostic_plan_v1'
            or plan.get('cache_sha256')!=TRAIN_CACHE_SHA or receipt.get('cache_sha256')!=TRAIN_CACHE_SHA
            or plan.get('complete_input_train_cohort_verified') is not True
            or final.get('backend')!=plan.get('backend') or final.get('selected_sequence')!=plan.get('selected_sequence')
            or final.get('completed_frames')!=receipt['completed_frames']
            or plan['class_scope']!=['car'] or plan['cohort_mode']!='real-train-development'
            or receipt['completed_frames']!=receipt['scheduled_frames'] or receipt['completed_frames']!=manifest['frames']
            or plan['selected_schedule']!=manifest['selected_schedule']):
        raise ValueError('completed matching train inference required')
    gt_path=Path(ground_truth)/'ground-truth.jsonl';pred_path=replay/'predictions.jsonl';tracking_path=replay/'tracking.jsonl'
    inputs={gt_path:manifest['ground_truth_sha256'],pred_path:receipt['predictions_sha256'],
        tracking_path:receipt['tracking_sha256'],Path(ground_truth)/'manifest.json':manifest_sha256,
        replay/'development-inference-receipt.json':receipt_sha256,replay/'receipt.json':final['replay_receipt_sha256'],
        replay/'plan.json':receipt['plan_sha256'],Path(__file__):sha(Path(__file__)),EVENT_SOURCE:EVENT_SOURCE_SHA}
    if any(sha(path)!=digest for path,digest in inputs.items()): raise ValueError('evaluation input or dependency changed')
    gt=[json.loads(line) for line in gt_path.read_bytes().splitlines()]
    predictions=[json.loads(line) for line in pred_path.read_bytes().splitlines()]
    schedule=[(r['sequence_id'],r['vehicle_frame'],r['box_reference_timestamp_us']) for r in plan['selected_schedule']]
    actual=[(r['sequence_id'],r['frame_id'],r['box_reference_timestamp_us']) for r in gt]
    if (len(gt)!=manifest['frames'] or actual!=schedule or len(set(actual))!=len(actual)
            or {r[0] for r in actual}!={plan['selected_sequence']}
            or any(box['class_label']!='car' for frame in gt for box in frame['objects'])
            or any(box['class_label']!='car' for frame in predictions for box in frame['predictions'])):
        raise ValueError('complete single-sequence car-only payload required')
    counts=adapter.validate_predictions(predictions,gt)
    events=identity_events(adapter,gt,predictions,tracking_path)
    runtime=adapter.runtime_evidence();module.validate_runtime(runtime)
    metrics=adapter.compute_metrics(gt,predictions,classes=('car',))
    if any(sha(path)!=digest for path,digest in inputs.items()):
        raise ValueError('evaluation inputs changed during metric computation')
    protocol=copy.deepcopy(adapter.PROTOCOL)
    protocol.update(kind='spd_train_sequence_development_car_protocol_v1',split='train_development',
        expected_sequences=1,expected_frames=len(gt),evaluated_classes=['car'],supplementary_classes=[])
    protocol['roi']['class_range_m']={'car':50.}
    result=dict(kind='spd_train_sequence_development_metrics_v1',status='complete',backend=plan['backend'],
        counts=counts,metrics=metrics,identity_event_diagnostics=events,protocol=protocol,runtime=runtime,
        input_sha256={str(path):digest for path,digest in inputs.items()},
        gt_manifest_sha256=manifest_sha256,inference_receipt_sha256=receipt_sha256,
        evaluator_sha256=sha(Path(__file__)),upstream_in_sample=True,validation=False,
        full_official_train=False,three_seed_comparison=False,paper_eligible=False)
    output=new_directory(output);module.write_json(output/'metrics.json',result)
    print(json.dumps(dict(status='complete',backend=plan['backend'],frames=len(gt),
        HOTA=metrics['trackeval']['car']['summary']['HOTA'],paper_eligible=False),sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name in ('plan','projection','vehicle-infos','target-audit','output'): prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--plan-sha256',required=True);prep.add_argument('--target-audit-sha256',required=True)
    run=sub.add_parser('evaluate')
    for name in ('ground-truth','replay','output'): run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--manifest-sha256',required=True);run.add_argument('--receipt-sha256',required=True)
    a=parser.parse_args()
    if a.mode=='prepare': prepare(a.plan,a.plan_sha256,a.projection,a.vehicle_infos,a.target_audit,a.target_audit_sha256,a.output)
    else: evaluate(a.ground_truth,a.manifest_sha256,a.replay,a.receipt_sha256,a.output)
