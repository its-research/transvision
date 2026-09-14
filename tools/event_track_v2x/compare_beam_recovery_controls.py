#!/usr/bin/env python3
"""Complete TRAIN recovery-toggle comparison; extra recovery compute is explicit.

Requires both completed replays, both sealed native metric outputs, and the
original completed node-beam reference. No partial results, parameter changes,
GT inference inputs, equal-resource claims or replacement of failed baselines.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.compare_train_risk_controls import read, read_metrics
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.resource_sweep import validate_execution_modes
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.forest_training_data import _new_json

OFF='beam_recovery_disabled'
ON='beam_recovery'
BEAM_SOURCE='transvision/models/event_track_v2x/persistent_beam_tracking.py'
# One reviewed inventory-only addition between sealed SPD runs. This offline
# V2V4Real GT module is not imported by the SPD producer or its initializers.
# No generic "ignore new files" rule, modified dependency or receipt rewrite.
REVIEWED_NONINFERENCE_ADDITIONS = {
    'transvision/models/event_track_v2x/v2v4real_ground_truth.py':
        '5cf32441356298c5f28f7c31285ce4374ccc4f209c2a3af716352dcb3e25f71b',
}


def source_inventory_controls(reports):
    """Return common bindings and an explicit, hash-pinned addition audit."""
    maps={r['backend']:r['plan']['source_sha256'] for r in reports}
    common=set.intersection(*(set(s) for s in maps.values()))
    union=set.union(*(set(s) for s in maps.values()))
    if any(len({s[p] for s in maps.values()})!=1 for p in common):
        raise ValueError('shared control source changed')
    extra=union-common
    if any(p not in REVIEWED_NONINFERENCE_ADDITIONS or
           any(s[p]!=REVIEWED_NONINFERENCE_ADDITIONS[p] for s in maps.values() if p in s)
           for p in extra):
        raise ValueError('unreviewed control source inventory difference')
    shared={p:maps[OFF][p] for p in sorted(common)}
    added={b:{p:s[p] for p in sorted(extra) if p in s} for b,s in sorted(maps.items())}
    return shared,added


def event_identity(report):
    return [(e['sequence_id'],e['frame_id'],e['reference_us'],e['decision_us'],e['factor_rows_sha256'])
            for e in report['events']]


def assemble(reports,metrics,node,*,allow_fixture=False):
    if len(reports)!=2 or len(metrics)!=2 or {r['backend'] for r in reports}!={OFF,ON}:
        raise ValueError('both distinct complete recovery controls required')
    cells={r['backend']:(r,m) for r,m in zip(reports,metrics)}
    off,om=cells[OFF];on,nm=cells[ON]
    shared_sources,source_additions=source_inventory_controls(reports)
    normalized=[]
    for backend,(report,metric) in cells.items():
        plan=copy.deepcopy(report['plan'])
        if (plan.get('kind')!='train_sequence_inference_diagnostic_plan_v1'
                or plan['backend']!=backend or plan['class_scope']!=['car']
                or not allow_fixture and (plan['cohort_mode']!='real-train-development'
                    or plan.get('complete_input_train_cohort_verified') is not True)):
            raise ValueError('complete real train car provenance required')
        if plan['configuration'].pop('enable_recovery') is not (backend==ON):
            raise ValueError('backend and recovery switch differ')
        plan.pop('backend');plan['runtime'].pop('pid')
        plan['source_sha256']=shared_sources
        normalized.append(plan)
        if report['fallback_events'] or report['fallback_unreported_events']:
            raise ValueError('threshold fallback used or unreported')
        if (report['factor_stream_sha256']!=off['factor_stream_sha256']
                or event_identity(report)!=event_identity(off)):
            raise ValueError('actual complete factor stream or event clocks differ')
        if any(metric[k]!=om[k] for k in ('gt_manifest_sha256','protocol','runtime','evaluator_sha256')):
            raise ValueError('ground truth or metric protocol differs')
    if normalized[0]!=normalized[1]:
        raise ValueError('configuration, source, cohort or environment changed besides recovery switch')
    if off['plan']['runtime']['pid']==on['plan']['runtime']['pid']:
        raise ValueError('separate fresh replay processes required')
    node_plan=node['plan'];off_plan=off['plan']
    if node['backend']!='node_beam' or node_plan['backend']!='node_beam':
        raise ValueError('original completed node-beam reference required')
    if not allow_fixture and (node_plan['cohort_mode']!='real-train-development'
            or node_plan.get('complete_input_train_cohort_verified') is not True):
        raise ValueError('original node reference must be real train')
    for key in ('cache_sha256','identity_checkpoint_sha256','scorer_signature','selected_schedule','class_scope'):
        if node_plan[key]!=off_plan[key]:raise ValueError('original node reference inputs differ')
    base=copy.deepcopy(off_plan['configuration'])
    for key in ('enable_recovery','recovery_budget','recovery_completions_per_component'):base.pop(key)
    # New allocation-only control still must reproduce the exact node-beam
    # output when disabled. Old receipts predate this optional explicit field.
    scope_mode=base.pop('recovery_allocation_scope','all_recent')
    if scope_mode not in ('all_recent','arrived_vehicle_raw_xy50'):
        raise ValueError('unknown recovery allocation scope')
    if (base!=node_plan['configuration']
            or node_plan['source_sha256'][BEAM_SOURCE]!=off_plan['source_sha256'][BEAM_SOURCE]
            or node['factor_stream_sha256']!=off['factor_stream_sha256']
            or event_identity(node)!=event_identity(off)
            or node['predictions_sha256']!=off['predictions_sha256']):
        raise ValueError('disabled recovery does not reproduce the original node beam')
    return dict(kind='train_beam_recovery_controls_v1',status='complete',frames=len(off['events']),
        cells={b:dict(inference_receipt_sha256=r['receipt_sha256'],source_directory=r['source_directory'],
                     configuration=r['plan']['configuration'],**m) for b,(r,m) in cells.items()},
        primary_delta_on_minus_off={k:nm['primary'][k]-om['primary'][k] for k in om['primary']},
        changed_output_frames=sum(a['output_payload_sha256']!=b['output_payload_sha256']
                                  for a,b in zip(off['events'],on['events'])),
        changed_output_id_frames=sum(a['output_ids_sha256']!=b['output_ids_sha256']
                                     for a,b in zip(off['events'],on['events'])),
        original_node_reference=dict(source_directory=node['source_directory'],receipt_sha256=node['receipt_sha256']),
        disabled_predictions_byte_identical_to_original=True,actual_complete_factor_stream_identical=True,
        recovery_toggle_only_configuration_difference=True,shared_node_beam_algorithm=True,
        reviewed_noninference_source_additions=source_additions,
        source_inventories_byte_identical=off['plan']['source_sha256']==on['plan']['source_sha256'],
        shared_bound_source_hashes_identical=True,historical_receipts_modified=False,
        recovery_allocation_scope=scope_mode,
        extra_recovery_compute_included=True,physically_equal_resources_claimed=False,
        strong_baseline_comparison_complete=False,internal_recovery_is_not_gt_recovery=True,
        full_official_train=False,validation=False,three_seed_comparison=False,
        statistical_inference=False,paper_eligible=False)


def read_work(report):
    root=Path(report['source_directory']);plan=report['plan'];enabled=report['backend']==ON
    outer=read(root/'development-inference-receipt.json',report['receipt_sha256'])
    receipt=read(root/'receipt.json',outer['replay_receipt_sha256'])
    validate_execution_modes(receipt,dict(backend=report['backend'],configuration=plan['configuration']))
    totals=dict(recovery_steps=0,recovery_events=0,extra_state_updates=0,cover_restarts=0)
    with (root/'tracking.jsonl').open('rb') as stream:
        for line in stream:
            row=json.loads(line)['tracking']
            steps=row['recovery_search_steps']
            if (row['recovery_enabled'] is not enabled or type(steps) is not int
                    or not 0<=steps<=plan['configuration']['recovery_budget']):
                raise ValueError('recovery audit switch or budget differs')
            if not enabled and (steps or row['recovery_allocation_trace']):
                raise ValueError('disabled recovery executed extra search')
            if sum(t['charged_search_steps'] for t in row['recovery_allocation_trace'])!=steps:
                raise ValueError('recovery trace and charged work differ')
            totals['recovery_steps']+=steps
            for component in row['components']:
                totals['recovery_events']+=len(component['recovery_events'])
                if enabled:
                    if component['complete_raw_support_retained'] is not True:
                        raise ValueError('recovery audit does not retain complete support')
                    totals['cover_restarts']+=int(component['recovery_cover_restarted_from_root'])
            if enabled:
                extra=row['recovery_extra_state_updates']
                if type(extra) is not int or extra<0:raise ValueError('invalid extra state work')
                totals['extra_state_updates']+=extra
    return dict(**totals,latency_seconds_p50_p95_p99_max=report['latency_seconds_p50_p95_p99_max'],
        elapsed_seconds=report['elapsed_seconds'],process_peak_rss_bytes=report['process_peak_rss_bytes'],
        database_bytes=report['database_bytes'],timing_repeated=False,exclusive_host=False)


def compare(references,node_reference,output):
    current=diagnostic_sources()
    reports=[];metrics=[];evidence={ROOT/p:h for p,h in current.items()}
    for module in ('compare_beam_recovery_controls.py','compare_train_risk_controls.py',
                   'audit_train_inference_comparison.py'):
        path=ROOT/'tools/event_track_v2x'/module
        evidence[path]=sha_file(path)
    for replay,receipt_sha,metric_path,metric_sha in references:
        report=inspect(replay,receipt_sha);reports.append(report)
        metrics.append(read_metrics(metric_path,metric_sha,report))
        evidence[Path(metric_path).absolute()]=metric_sha
        for path,digest in read(metric_path,metric_sha)['input_sha256'].items():
            if evidence.setdefault(Path(path),digest)!=digest:raise ValueError('conflicting evidence identity')
    node=inspect(*node_reference)
    for report in reports:
        if any(current.get(p)!=h for p,h in report['plan']['source_sha256'].items()):
            raise ValueError('bound inference source no longer matches its sealed run')
    # Reject a direct reference to the sole reviewed addition anywhere in the
    # remaining bound producer files; fixed hashes and source review, not a
    # claim that static string matching proves arbitrary dynamic-import safety.
    for p in REVIEWED_NONINFERENCE_ADDITIONS:
        marker=Path(p).stem.encode()
        if any(marker in (ROOT/name).read_bytes() for name in current if name!=p):
            raise ValueError('reviewed offline module is referenced by inference-bound source')
    result=assemble(reports,metrics,node)
    result['work']={r['backend']:read_work(r) for r in reports}
    for report in reports+[node]:
        root=Path(report['source_directory']);outer=read(root/'development-inference-receipt.json',report['receipt_sha256'])
        receipt=read(root/'receipt.json',outer['replay_receipt_sha256'])
        evidence[root/'development-inference-receipt.json']=report['receipt_sha256']
        evidence[root/'receipt.json']=outer['replay_receipt_sha256']
        for filename,key in (('plan.json','plan_sha256'),('predictions.jsonl','predictions_sha256'),
                ('tracking.jsonl','tracking_sha256'),('frame-timings.jsonl','frame_timings_sha256')):
            evidence[contained_file(root,filename)]=receipt[key]
        for head in receipt['sequence_heads'].values():
            evidence[contained_file(root,head['database'])]=head['database_sha256']
    if diagnostic_sources()!=current or any(sha_file(path)!=digest for path,digest in evidence.items()):
        raise ValueError('comparison evidence changed')
    result['input_sha256']={str(p):h for p,h in evidence.items()}
    output=_directory(output);_new_json(output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','primary_delta_on_minus_off','paper_eligible')},sort_keys=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',nargs=4,required=True,metavar=('REPLAY','RECEIPT_SHA','METRICS','METRICS_SHA'))
    p.add_argument('--node-reference',nargs=2,required=True,metavar=('REPLAY','RECEIPT_SHA'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.run,a.node_reference,a.output)
