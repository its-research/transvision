#!/usr/bin/env python3
"""Audit completed task-scoped TRAIN inference from raw, prediction-only inputs.

Reconstruct each event's arrived pose and raw-node scope independently of the
online allocation helper. Historical component maps are read only up to that
event's recorded depth. No GT, inference rerun, model update or metric claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.audit_train_inference_comparison import inspect
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.recovery_task_scope import SCOPE_MODE, VerifiedEgoPoseTable
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.forest_training_data import _new_json


def near(actual,expected):
    return type(actual) in (float,int) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=0.,abs_tol=1e-12)


class EventAudit:
    """Independent event-time reference; never calls the online scope selector."""
    def __init__(self, db, cache, poses, plan):
        self.db,self.cache,self.poses,self.plan=db,cache,poses,plan
        self.config=plan['configuration'];self.sequence=plan['selected_sequence']
        self.receipts={};self.n=0;self.last_decision=-1;self.last_reference=-1
        self.raw={}
        for i,payload,checksum in db.execute('SELECT i,raw,sha FROM observations ORDER BY i'):
            if i!=len(self.raw) or hashlib.sha256(payload).hexdigest()!=checksum:
                raise ValueError('raw observation order or digest differs')
            value=json.loads(payload)
            if set(value)!={'sequence_id','node','detection_index','state_us','mean','covariance','score','features','source_cache_sha256'}:
                raise ValueError('unknown raw observation fields')
            if value['sequence_id']!=self.sequence or len(value['mean'])!=9 or not all(map(math.isfinite,value['mean'])):
                raise ValueError('invalid raw prediction-only state')
            self.raw[i]=value

    def step(self,audit,prediction):
        seq,frame=prediction['sequence_id'],prediction['frame_id']
        ref,decision=prediction['box_reference_timestamp_us'],prediction['decision_timestamp_us']
        if ((seq,frame)!=(self.sequence,audit['event_id']) or audit['sequence_id']!=seq
                or type(ref) is not int or type(decision) is not int
                or not self.last_reference<ref<=decision or decision<self.last_decision):
            raise ValueError('invalid audit event or causal clock')
        ing=audit['cache_ingestion']
        if (ing['gt_model_inputs'] is not False or ing['cache_manifest_sha256']!=self.cache.manifest_sha256
                or ing['class_scope']!=['car']):raise ValueError('cache input provenance differs')
        for d in ing['new_deliveries']:
            key=(d['side'],d['frame_id'])
            ck=(seq,*key)
            if (d['sequence_id']!=seq or key in self.receipts or ck not in self.cache.index
                    or type(d['arrival_us']) is not int or not self.last_decision<=d['arrival_us']<=decision):
                raise ValueError('duplicate, future or withheld source receipt')
            entry,meta=map(json.loads,self.cache.index[ck])
            if (d['frame_sha256']!=entry['frame_sha256'] or
                    max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us'])>d['arrival_us']):
                raise ValueError('source frame identity or availability differs')
            self.receipts[key]=d
        for duplicate in ing['duplicate_deliveries']:
            d=duplicate['receipt'];first=self.receipts.get((d['side'],d['frame_id']))
            if (first is None or d['sequence_id']!=seq or d['frame_sha256']!=first['frame_sha256']
                    or not first['arrival_us']<=d['arrival_us']<=decision
                    or duplicate['first_arrival_us']!=first['arrival_us']):
                raise ValueError('duplicate changed first arrival')
        n=audit['observation_count']
        if (type(n) is not int or not self.n<=n<=len(self.raw)
                or n-self.n!=audit['new_observations'] or n-self.n!=ing['new_observations']):
            raise ValueError('raw event coverage differs')
        for i in range(self.n,n):
            raw=self.raw[i];node=raw['node']
            side='vehicle-side' if node['source_id']==0 else 'infrastructure-side'
            first=self.receipts.get((side,node['frame_id']))
            if (node['source_id'] not in (0,1) or first is None
                    or raw['source_cache_sha256']!=first['frame_sha256']
                    or node['arrival_us']!=first['arrival_us']
                    or not 0<=raw['state_us']<=node['information_us']<=node['arrival_us']<=decision):
                raise ValueError('raw node not bound to an arrived source frame')
            meta=json.loads(self.cache.index[(seq,side,node['frame_id'])][1])
            if (raw['state_us']!=meta['box_reference_timestamp_us'] or
                    node['information_us']!=max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us'])):
                raise ValueError('raw state or information time differs from its source cache')
        choices=[]
        for (side,fid),d in self.receipts.items():
            if side!='vehicle-side':continue
            pose=self.poses[(seq,fid)]
            if pose['state_us']<=ref:
                choices.append(dict(pose,arrival_us=d['arrival_us']))
        pose=max(choices,key=lambda p:(p['state_us'],p['frame_id'])) if choices else None
        age=None if pose is None else ref-pose['state_us']
        fallback=('missing_arrived_pose' if pose is None else
                  'stale_arrived_pose' if age>self.config['state']['window_us'] else None)
        expected=dict(kind='arrived_vehicle_pose_recovery_allocation_v1',mode=SCOPE_MODE,
            pose_table_sha256=self.plan['ego_pose_table_sha256'],radius_m=50.,pose=pose,
            fallback=fallback,pose_age_us=age,motion_policy='hold_last_received_ego_position',
            raw_point_policy='independent_constant_velocity',decision_loss_scope_changed=False,
            input_detections_filtered=False,gt_model_inputs=False)
        if (audit['recovery_task_scope']!=expected or ing['recovery_task_scope']!=expected
                or audit['recovery_allocation_scope']!=SCOPE_MODE
                or audit['recovery_scope_changes_decision_loss'] is not False):
            raise ValueError('actual scope is not the latest causally available pose contract')
        lower=max(0,ref-self.config['state']['window_us'])
        indices=[i for i in range(n) if self.raw[i]['state_us']>=lower]
        if audit['decision_indices']!=indices:raise ValueError('global decoder scope changed')
        selected=set(indices);seen=set();counts={};original={};components={}
        for s in audit['components']:
            c,depth=s['component'],s['nodes']
            if c in components or type(depth) is not int or not 0<depth<=n:
                raise ValueError('invalid historical component')
            created=self.db.execute('SELECT created_us FROM component_catalog WHERE component=?',(c,)).fetchone()
            if created is None or created[0]>decision:
                raise ValueError('historical audit borrowed a future component merge')
            members=[g for g, in self.db.execute('SELECT global_i FROM component_members '
                     'WHERE component=? AND local_i<? ORDER BY local_i',(c,depth))]
            if len(members)!=depth or any(not 0<=g<n or g in seen for g in members):
                raise ValueError('future or overlapping historical component members')
            seen.update(members)
            scope=[i for i,g in enumerate(members) if g in selected]
            if s['decision_indices']!=scope or s['indices_count']!=len(scope):
                raise ValueError('component decoder scope changed')
            original[c]=len(scope)/len(indices) if indices else 0.
            if not near(s['weight'],original[c]):raise ValueError('global loss weight changed')
            count=0
            if fallback is None:
                ex,ey=pose['ego_translation_world'][:2]
                for i in scope:
                    raw=self.raw[members[i]];m=raw['mean'];dt=(ref-raw['state_us'])/1e6
                    count+=math.hypot(m[0]+dt*m[7]-ex,m[1]+dt*m[8]-ey)<50.
            counts[c]=None if fallback else count
            components[c]=s
        if seen!=set(range(n)):raise ValueError('component maps omit raw support')
        total=None if fallback else sum(counts.values())
        weights=original if fallback else {c:counts[c]/total if total else 0. for c in components}
        enabled=self.config['enable_recovery'];spent=Counter()
        if audit['recovery_enabled'] is not enabled:raise ValueError('recovery switch differs')
        for item in audit['recovery_allocation_trace']:
            c=item['component'];steps=item['charged_search_steps']
            if (not enabled or c not in weights or weights[c]<=0 or not near(item['priority_weight'],weights[c])
                    or type(steps) is not int or steps<0):
                raise ValueError('extra search used a wrong or zero task priority')
            spent[c]+=steps
        total_spent=sum(spent.values())
        if (audit['recovery_search_steps']!=total_spent or total_spent>self.config['recovery_budget']
                or not enabled and (total_spent or audit['recovery_allocation_trace'])):
            raise ValueError('charged search budget differs')
        for c,s in components.items():
            if enabled and (s['recovery_scope_raw_nodes']!=counts[c]
                    or not near(s['recovery_priority_weight'],weights[c]) or s['recovery_steps']!=spent[c]
                    or s['complete_raw_support_retained'] is not True):
                raise ValueError('component scope, search count or support audit differs')
        self.n,self.last_decision,self.last_reference=n,decision,ref
        return dict(sequence_id=seq,frame_id=frame,reference_us=ref,decision_us=decision,
            pose_frame=None if pose is None else pose['frame_id'],pose_age_us=age,fallback=fallback,
            recent_raw_nodes=len(indices),task_raw_nodes=total,extra_search_steps=total_spent,
            unused_extra_budget=self.config['recovery_budget']-total_spent if enabled else None,
            components=[dict(component=c,recent_raw_nodes=s['indices_count'],task_raw_nodes=counts[c],
                original_loss_weight=original[c],search_priority_weight=weights[c],extra_search_steps=spent[c])
                for c,s in components.items()])


def audit_run(replay,receipt_sha256,cache,pose_path,pose_sha256,output,*,allow_fixture=False):
    report=inspect(replay,receipt_sha256,allow_fixture=allow_fixture)
    plan=report['plan'];root=Path(report['source_directory'])
    if (report['backend'] not in ('beam_recovery','beam_recovery_disabled')
            or plan['configuration'].get('recovery_allocation_scope')!=SCOPE_MODE
            or plan.get('ego_pose_table_sha256')!=pose_sha256
            or plan['cache_sha256']!=cache.manifest_sha256):
        raise ValueError('complete, explicitly task-scoped cache-bound replay required')
    VerifiedEgoPoseTable(pose_path,pose_sha256,cache)
    poses={(r['sequence_id'],r['frame_id']):r for r in json.loads(Path(pose_path).read_bytes())['poses']}
    receipt=json.loads((root/'receipt.json').read_bytes())
    if (receipt['ego_pose_table_sha256']!=pose_sha256 or receipt['recovery_allocation_scope']!=SCOPE_MODE
            or len(receipt['sequence_heads'])!=1):raise ValueError('completed single-sequence task scope required')
    head=receipt['sequence_heads'][plan['selected_sequence']]
    database=contained_file(root,head['database'])
    outer=json.loads((root/'development-inference-receipt.json').read_bytes())
    evidence={Path(pose_path):pose_sha256,Path(__file__):sha_file(Path(__file__)),database:head['database_sha256'],
        cache.root/'manifest.json':cache.manifest_sha256,
        root/'development-inference-receipt.json':receipt_sha256,root/'receipt.json':outer['replay_receipt_sha256']}
    for name,key in (('plan.json','plan_sha256'),('tracking.jsonl','tracking_sha256'),('predictions.jsonl','predictions_sha256')):
        evidence[root/name]=receipt[key]
    if any(sha_file(p)!=h for p,h in evidence.items()):raise ValueError('scope evidence changed before audit')
    events=[]
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        verifier=EventAudit(db,cache,poses,plan)
        committed=db.execute('SELECT audit,prediction FROM events ORDER BY rowid')
        with (root/'tracking.jsonl').open('rb') as audits,(root/'predictions.jsonl').open('rb') as predictions:
            for stored_audit,stored_prediction in committed:
                a=json.loads(audits.readline())['tracking'];p=json.loads(predictions.readline())
                if a!=json.loads(stored_audit) or p!=json.loads(stored_prediction):
                    raise ValueError('file output differs from committed database event')
                events.append(verifier.step(a,p))
            if audits.readline() or predictions.readline():raise ValueError('extra uncommitted output')
        if verifier.n!=len(verifier.raw) or len(events)!=receipt['completed_frames']:
            raise ValueError('incomplete raw or event coverage')
    if any(sha_file(p)!=h for p,h in evidence.items()):raise ValueError('scope evidence changed during audit')
    counts=Counter(e['fallback'] or 'fresh_arrived_pose' for e in events)
    result=dict(kind='train_recovery_task_scope_audit_v1',status='complete',backend=report['backend'],
        frames=len(events),pose_status_counts=dict(counts),
        events_with_zero_task_raw_nodes=sum(e['task_raw_nodes']==0 for e in events),
        extra_search_steps=sum(e['extra_search_steps'] for e in events),
        independent_raw_scope_reconstruction=True,global_decoder_scope_and_weights_unchanged=True,
        no_future_pose_or_observation_used=True,all_historical_raw_nodes_in_component_maps=True,
        report_contains_GT=False,gt_model_inputs=False,parameter_training=False,inference_rerun=False,
        greedy_order_optimality_verified=False,tracking_metrics_computed=False,
        model_risk_is_not_metric_bound=True,validation=False,paper_eligible=False,
        cohort_mode=plan['cohort_mode'],input_sha256={str(p):h for p,h in evidence.items()},events=events)
    destination=_directory(output);_new_json(destination/'scope-audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','backend','frames','pose_status_counts','extra_search_steps','paper_eligible')},sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','cache','pose-table','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('receipt-sha256','cache-sha256','pose-table-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    audit_run(args.replay,args.receipt_sha256,VerifiedForestCache(args.cache,args.cache_sha256),
        args.pose_table,args.pose_table_sha256,args.output)
