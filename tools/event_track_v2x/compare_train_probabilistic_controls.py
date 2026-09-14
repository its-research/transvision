#!/usr/bin/env python3
"""Complete fixed TRAIN JPDA/PKF controls with an immutable beam reference.

Three updaters are mandatory. Raw factors must agree; state/time semantics are
explicitly different from beam. No failed-run omission, tuning or GPU work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.event_track_v2x import run_train_probabilistic_diagnostic as producer
from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.compare_train_risk_controls import read, read_metrics
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from transvision.models.event_track_v2x.resource_sweep import configuration, validate_execution_modes

OWN = 'tools/event_track_v2x/run_train_probabilistic_diagnostic.py'
REFERENCE = 'beam_recovery_disabled'
COMMON = ('cache_sha256', 'cooperative_metadata_sha256', 'identity_checkpoint_sha256',
    'identity_seed', 'scorer_signature', 'selected_sequence', 'selected_schedule',
    'class_scope', 'cohort_mode', 'thread_environment', 'source_scope')


def events(report):
    return [(e['sequence_id'], e['frame_id'], e['reference_us'], e['decision_us'], e['factor_rows_sha256'])
            for e in report['events']]


def assemble(reports, metrics, reference, reference_metric, *, allow_fixture=False):
    if len(reports)!=3 or len(metrics)!=3 or {r['backend'] for r in reports}!=set(producer.BACKENDS):
        raise ValueError('all three distinct complete probabilistic controls required')
    if reference['backend']!=REFERENCE or reference['plan']['backend']!=REFERENCE:
        raise ValueError('completed disabled-recovery beam reference required')
    if reference['plan']['configuration'].get('enable_recovery') is not False:
        raise ValueError('reference recovery must be disabled')
    rp = reference['plan']
    expected_sources = dict(rp['source_sha256'], **{OWN: sha_file(ROOT/OWN)})
    normalized = []
    cells = {}
    for report, metric in [*zip(reports, metrics), (reference, reference_metric)]:
        p = report['plan']
        if (p.get('kind')!='train_sequence_inference_diagnostic_plan_v1' or p['class_scope']!=['car']
                or not allow_fixture and (p['cohort_mode']!='real-train-development'
                    or p.get('complete_input_train_cohort_verified') is not True)):
            raise ValueError('complete real-train car provenance required')
        if any(p[k]!=rp[k] for k in COMMON) or p['configuration']['state']!=rp['configuration']['state']:
            raise ValueError('shared raw input, scoring, lifecycle or state configuration differs')
        if report['factor_stream_sha256']!=reference['factor_stream_sha256'] or events(report)!=events(reference):
            raise ValueError('actual factor stream or causal event schedule differs')
        if any(metric[k]!=reference_metric[k] for k in ('gt_manifest_sha256','protocol','runtime','evaluator_sha256')):
            raise ValueError('metric ground truth, protocol or runtime differs')
        runtime = dict(p['runtime']);runtime.pop('pid')
        reference_runtime = dict(rp['runtime']);reference_runtime.pop('pid')
        if runtime!=reference_runtime:
            raise ValueError('inference environment differs')
        if report is reference:
            continue
        backend = report['backend']
        if (p.get('producer')!=producer.PRODUCER or p['backend']!=backend
                or p['configuration']!=asdict(configuration(dict(backend=backend)))
                or p['source_sha256']!=expected_sources):
            raise ValueError('fixed probabilistic producer, configuration and source inventory required')
        for key in ('same_state_time_protocol_as_recoverable','recovery_only_ablation','reproduced_public_method'):
            if p.get(key) is not False:
                raise ValueError('baseline adaptation limits must be explicit')
        stable = copy.deepcopy(p)
        stable.pop('backend');stable['configuration'].pop('update_rule');stable['runtime'].pop('pid')
        normalized.append(stable)
        cells[backend] = dict(source_directory=report['source_directory'],
            inference_receipt_sha256=report['receipt_sha256'], configuration=p['configuration'], **metric)
    if any(p!=normalized[0] for p in normalized[1:]):
        raise ValueError('updater is not the only probabilistic plan difference')
    if len({r['plan']['runtime']['pid'] for r in [*reports, reference]})!=4:
        raise ValueError('distinct fresh inference processes required')
    return dict(kind='train_probabilistic_controls_v1', status='complete', frames=len(reference['events']),
        cells=cells, reference=dict(backend=REFERENCE, source_directory=reference['source_directory'],
            inference_receipt_sha256=reference['receipt_sha256'], **reference_metric),
        primary_delta_variant_minus_reference={b:{k:m['primary'][k]-reference_metric['primary'][k]
            for k in reference_metric['primary']} for b,m in cells.items()},
        actual_complete_factor_stream_identical=True, factor_stream_sha256=reference['factor_stream_sha256'],
        updater_only_difference_between_probabilistic_controls=True,
        same_state_time_protocol_as_beam=False, recovery_only_ablation=False,
        reproduced_public_method=False, physically_equal_resources_claimed=False,
        frozen_default_lbp_joint_map=True, numerical_marginal_audit_is_not_posterior_error_bound=True,
        full_official_train=False, validation=False, three_seed_comparison=False,
        statistical_inference=False, paper_eligible=False)


def audit_scan(scan, config):
    roots, indices = scan['conditioned_track_roots'], scan['indices']
    n, m = len(roots), len(indices)
    factors = LogAssociationFactors(np.asarray(scan['log_pair']).reshape(n,m), np.zeros(n),
                                   scan['log_birth'], np.asarray(scan['allowed'],dtype=bool).reshape(n,m))
    q = scan['marginals']
    if (scan['factors_sha256']!=factors.digest() or q['factor_sha256']!=factors.digest()
            or q['algorithm']!='jpda-lbp-williams-lau-v1' or q['log_partition'] is not None):
        raise ValueError('scan factor identity or approximate algorithm differs')
    pair=np.asarray(q['pair'],dtype=float).reshape(n,m)
    left=np.asarray(q['left_unmatched'],dtype=float);right=np.asarray(q['right_unmatched'],dtype=float)
    if left.shape!=(n,) or right.shape!=(m,):
        raise ValueError('marginal dimensions differ')
    if any(not np.all(np.isfinite(a)) or np.any(a<0) or np.any(a>1+1e-12) for a in (pair,left,right)):
        raise ValueError('invalid marginal probabilities')
    if np.any(pair[~np.asarray(factors.allowed,dtype=bool).reshape(n,m)]!=0):
        raise ValueError('forbidden scan edge received probability')
    tolerance = config['inference']['tolerance']
    error=max(float(np.max(np.abs(pair.sum(axis=1)+left-1),initial=0)),
              float(np.max(np.abs(pair.sum(axis=0)+right-1),initial=0)))
    if error>tolerance+1e-12:
        raise ValueError('scan marginals violate row or column normalization')
    for key in ('log_message_residual','marginal_consistency_error'):
        if type(q[key]) not in (int,float) or not math.isfinite(q[key]) or not 0<=q[key]<=tolerance:
            raise ValueError('scan did not satisfy the declared LBP stopping rule')
    for key in ('iterations','message_updates','components'):
        if type(q[key]) is not int or q[key]<0:
            raise ValueError('invalid solver work count')
    if (q['message_updates']>config['inference']['max_message_updates']
            or q['iterations']>config['inference']['max_iterations']*q['components']):
        raise ValueError('recorded scan exceeds inference limits')
    anchors=scan['anchors']
    assigned=[a['root'] for a in anchors]
    if ([a['index'] for a in anchors]!=indices or len(set(assigned))!=len(assigned)
            or any(a['root']!=a['index'] and a['root'] not in roots for a in anchors)):
        raise ValueError('scan identity anchors violate one-to-one structure')
    return dict(iterations=q['iterations'],message_updates=q['message_updates'],
        max_row_or_column_error=error,log_message_residual=q['log_message_residual'])


def read_work(report):
    root=Path(report['source_directory']);config=report['plan']['configuration']
    outer=read(root/'development-inference-receipt.json',report['receipt_sha256'])
    receipt=read(root/'receipt.json',outer['replay_receipt_sha256'])
    validate_execution_modes(receipt,dict(backend=report['backend'],configuration=config))
    expected=dict(probabilistic_single_history_enabled=True, probabilistic_association_algorithm='lbp',
        probabilistic_update_rule=config['update_rule'],probabilistic_anchor_decoder='joint-map')
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('executed probabilistic backend differs')
    totals=Counter(frames=0,scans=0,iterations=0,message_updates=0,state_updates=0)
    maximum=0.;residual=0.
    with (root/'tracking.jsonl').open('rb') as stream:
        for line in stream:
            row=json.loads(line)['tracking']
            if (row['association_algorithm']!='lbp' or row['update_rule']!=config['update_rule']
                    or row['anchor_decoder']!='joint-map' or row['recovery_enabled'] is not False
                    or row['same_state_time_protocol_as_recoverable'] is not False
                    or row['full_history_posterior_bound'] is not None
                    or row['unmatched_mass_scales_detection_score'] is not False):
                raise ValueError('per-event execution or uncertainty claim differs')
            work=row['state_work_breakdown']
            if any(type(n) is not int or n<0 for n in work.values()) or sum(work.values())!=row['state_updates']:
                raise ValueError('state work breakdown differs')
            totals.update(frames=1,state_updates=row['state_updates'])
            for scan in row['conditional_scans']:
                actual=audit_scan(scan,config)
                totals.update(scans=1,iterations=actual['iterations'],message_updates=actual['message_updates'])
                maximum=max(maximum,actual['max_row_or_column_error']);residual=max(residual,actual['log_message_residual'])
    if totals['frames']!=len(report['events']):
        raise ValueError('work audit coverage differs')
    return dict(totals,max_row_or_column_error=maximum,max_log_message_residual=residual,
        elapsed_seconds=report['elapsed_seconds'],latency_seconds_p50_p95_p99_max=report['latency_seconds_p50_p95_p99_max'],
        process_peak_rss_bytes=report['process_peak_rss_bytes'],database_bytes=report['database_bytes'],
        state_work_is_not_cross_backend_flops=True,timing_repeated=False,exclusive_host=False)


def compare(references, beam_reference, output):
    current=producer.sources();reports=[];metrics=[]
    evidence={ROOT/p:h for p,h in current.items()}
    for name in ('compare_train_probabilistic_controls.py','compare_train_risk_controls.py',
                 'audit_train_inference_comparison.py'):
        path=ROOT/'tools/event_track_v2x'/name;evidence[path]=sha_file(path)
    for replay,receipt_sha,metric_path,metric_sha in [*references,beam_reference]:
        report=inspect(replay,receipt_sha);metric=read_metrics(metric_path,metric_sha,report)
        reports.append(report);metrics.append(metric)
        evidence[Path(metric_path).absolute()]=metric_sha
        for path,digest in read(metric_path,metric_sha)['input_sha256'].items():
            if evidence.setdefault(Path(path),digest)!=digest:raise ValueError('conflicting metric evidence')
        if any(current.get(p)!=h for p,h in report['plan']['source_sha256'].items()):
            raise ValueError('bound inference source changed since replay')
        outer=read(Path(replay)/'development-inference-receipt.json',receipt_sha)
        receipt=read(Path(replay)/'receipt.json',outer['replay_receipt_sha256'])
        evidence[contained_file(replay,'frame-timings.jsonl')]=receipt['frame_timings_sha256']
        for head in receipt['sequence_heads'].values():
            evidence[contained_file(replay,head['database'])]=head['database_sha256']
    result=assemble(reports[:-1],metrics[:-1],reports[-1],metrics[-1])
    result['work']={r['backend']:read_work(r) for r in reports[:-1]}
    if producer.sources()!=current or any(sha_file(p)!=h for p,h in evidence.items()):
        raise ValueError('comparison evidence changed')
    result['input_sha256']={str(p):h for p,h in evidence.items()}
    output=_directory(output);_new_json(output/'comparison.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','primary_delta_variant_minus_reference','paper_eligible')},sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',nargs=4,required=True)
    parser.add_argument('--beam-reference',nargs=4,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compare(args.run,args.beam_reference,args.output)
