#!/usr/bin/env python3
"""Explain a completed recovery control using the sealed native car ROI.

Evaluator-only inspection. No inference rerun, GT feedback, ROI change,
metric recomputation or claim that an internal recovery is ground-truth repair.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x import evaluate_train_inference_diagnostic as evaluation


def frame_key(frame):
    return frame['sequence_id'],frame['frame_id'],frame['box_reference_timestamp_us']


def deltas(adapter,ground,off,on):
    if not ground or not len(ground)==len(off)==len(on):raise ValueError('complete aligned frame lists required')
    changed=[];roi_changes=0;count=Counter()
    for frame,left,right in zip(ground,off,on):
        if not frame_key(frame)==frame_key(left)==frame_key(right):raise ValueError('frame identity differs')
        a={b['track_id']:b for b in left['predictions']};b={p['track_id']:p for p in right['predictions']}
        if len(a)!=len(left['predictions']) or len(b)!=len(right['predictions']):raise ValueError('duplicate prediction ID')
        ar={p['track_id']:p for p in adapter._roi(left['predictions'],frame['ego_translation_world'],'car')}
        br={p['track_id']:p for p in adapter._roi(right['predictions'],frame['ego_translation_world'],'car')}
        roi_changes+=ar!=br
        count.update(off_output_boxes=len(a),on_output_boxes=len(b),off_roi_boxes=len(ar),on_roi_boxes=len(br))
        if a==b:continue
        fields=Counter()
        for key in a.keys()&b.keys():
            fields.update(name for name in set(a[key])|set(b[key]) if a[key].get(name)!=b[key].get(name))
        changed.append(dict(sequence_id=frame['sequence_id'],frame_id=frame['frame_id'],
            added_output_ids=sorted(b.keys()-a.keys()),removed_output_ids=sorted(a.keys()-b.keys()),
            modified_fields=dict(fields),roi_prediction_records_identical=ar==br))
    return dict(frames=len(ground),changed_output_frames=len(changed),roi_changed_frames=roi_changes,
        all_roi_prediction_records_identical=roi_changes==0,counts=dict(count),changed_frames=changed)


def audit(comparison,comparison_sha256,output):
    value=evaluation.read(comparison,comparison_sha256)
    if (value.get('kind')!='train_beam_recovery_controls_v1' or value.get('status')!='complete'
            or value.get('actual_complete_factor_stream_identical') is not True
            or value.get('disabled_predictions_byte_identical_to_original') is not True):
        raise ValueError('complete bound recovery-toggle comparison required')
    evidence={Path(p):h for p,h in value['input_sha256'].items()}
    evidence.update({Path(comparison):comparison_sha256,Path(__file__):evaluation.sha(__file__),
                     Path(evaluation.__file__):evaluation.sha(evaluation.__file__)})
    if any(evaluation.sha(p)!=h for p,h in evidence.items()):raise ValueError('comparison evidence changed')
    gt_paths=[p for p in evidence if p.name=='ground-truth.jsonl']
    if len(gt_paths)!=1:raise ValueError('one bound evaluator ground-truth stream required')
    load=lambda p:[json.loads(line) for line in Path(p).read_bytes().splitlines()]
    ground=load(gt_paths[0])
    off=Path(value['cells']['beam_recovery_disabled']['source_directory'])
    on=Path(value['cells']['beam_recovery']['source_directory'])
    for root in (off,on):
        if root/'predictions.jsonl' not in evidence or root/'tracking.jsonl' not in evidence:
            raise ValueError('unbound prediction or tracking stream')
    module,adapter=evaluation.evaluator()
    evidence[Path(module.__file__)]=evaluation.sha(module.__file__)
    evidence[Path(adapter.__file__)]=evaluation.sha(adapter.__file__)
    left,right=load(off/'predictions.jsonl'),load(on/'predictions.jsonl')
    adapter.validate_predictions(left,ground);adapter.validate_predictions(right,ground)
    result=deltas(adapter,ground,left,right)
    if result['frames']!=value['frames'] or result['changed_output_frames']!=value['changed_output_frames']:
        raise ValueError('ROI audit and full comparison coverage differ')
    recovery=[]
    for line in (on/'tracking.jsonl').open('rb'):
        row=json.loads(line)['tracking']
        for component in row['components']:
            if component['recovery_events']:
                recovery.append(dict(sequence_id=row['sequence_id'],frame_id=row['event_id'],
                    component=component['component'],nodes=component['nodes'],
                    log_retained=component['log_retained'],backbone_log_retained=component['backbone_retained_log_mass'],
                    eta_upper=component['eta_upper'],internal_events=component['recovery_events']))
    if any(evaluation.sha(p)!=h for p,h in evidence.items()):raise ValueError('ROI evidence changed during audit')
    result.update(kind='train_beam_recovery_roi_audit_v1',status='complete',internal_recoveries=recovery,
        roi='sealed native car XY distance strictly less than 50m',
        input_sha256={str(p):h for p,h in evidence.items()},gt_used_for_evaluation_only=True,
        gt_boxes_or_ids_in_report=False,internal_recovery_is_not_gt_recovery=True,
        inference_modified=False,metrics_recomputed=False,validation=False,paper_eligible=False)
    destination=evaluation.new_directory(output);module.write_json(destination/'roi-audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','changed_output_frames','roi_changed_frames',
        'all_roi_prediction_records_identical','counts','paper_eligible')},sort_keys=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.comparison,a.comparison_sha256,a.output)
