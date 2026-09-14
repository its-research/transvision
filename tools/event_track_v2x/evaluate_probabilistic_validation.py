#!/usr/bin/env python3
"""Native-engine car evaluation for audited complete probabilistic SPD val runs.

No Torch import is required in this separate evaluator environment. The metric
engines/ROI remain sealed; the input-selection description is car-before-top64,
not the legacy all-class source-ablation protocol. No test or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import evaluate_source_ablation_v2 as native

KIND = 'probabilistic_car_full_validation_evaluation_v1'
COMMON_SHA = 'a9e80719682477c3f1a4ab6030d906d20f848d2584578e3fb8dfe45c6f399356'
RULES = ('jpda-ci','jpda-kalman','pkf')
SEEDS = (1337,2027,3407)
INPUT_SELECTION = 'class_index==0 and raw_score>=0.05, then descending raw_score/index top64 per source frame'


def protocol(adapter):
    result = native.protocol(adapter)
    result.update(kind='spd_native_probabilistic_car_protocol_v1', input_candidate_selection=INPUT_SELECTION)
    result.pop('source_ablation_reference')
    return result


def child(root, name):
    root = Path(root).absolute(); relative = Path(name)
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
            or '\\' in str(name)):
        raise ValueError('unsafe evidence relative path')
    path = root/relative
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
        raise ValueError('regular non-symlink evidence required')
    return path


def unchanged(evidence):
    if any(native.evidence(p) != value for p,value in evidence.items()):
        raise ValueError('evaluation evidence changed')


def inference_sources():
    # Same closure as the producer, without importing Torch into the evaluator.
    paths=list((ROOT/'transvision/models/event_track_v2x').glob('*.py'))
    paths += [ROOT/p for p in ('transvision/__init__.py','transvision/models/__init__.py',
        'transvision/register.py','transvision/version.py')]
    paths += [ROOT/'tools/event_track_v2x'/name for name in ('run_probabilistic_tracking_v2.py',
        'run_persistent_forest_v2.py','run_tracking_v2.py','train_forest_identity.py','prepare_forest_training.py')]
    return {p.relative_to(ROOT).as_posix():native.sha(p) for p in paths}


def inspect_run(run, receipt_sha, audit_path, audit_sha):
    run = Path(run).absolute(); evidence = {}
    def read(path, expected=None):
        path = Path(path).absolute(); observed = native.evidence(path)
        if expected is not None and observed['sha256'] != expected:
            raise ValueError('evaluation input identity differs: '+str(path))
        evidence[str(path)] = observed
        return native.read_json(path)
    final = read(child(run,'full-validation-receipt.json'),receipt_sha)
    audit = read(audit_path,audit_sha)
    receipt = read(child(run,'receipt.json'),audit.get('replay_receipt_sha256'))
    plan = read(child(run,'plan.json'),receipt['plan_sha256'])
    if (audit.get('kind')!='probabilistic_full_spd_val_audit_v1' or audit.get('status')!='complete'
            or audit.get('inference_receipt_sha256')!=receipt_sha or audit.get('source_directory')!=str(run)
            or audit.get('full_validation_coverage_verified') is not True
            or audit.get('frames')!=3316 or len(audit.get('sequence_heads',{}))!=21
            or any(final.get(k)!=v for k,v in receipt.items())
            or final.get('full_official_validation_schedule_completed') is not True
            or receipt.get('status')!='complete' or receipt.get('completed_frames')!=3316
            or receipt.get('scheduled_frames')!=3316 or receipt.get('cache_split')!='val'
            or receipt.get('allocation_teacher') is not False or receipt.get('learned_identity_enabled') is not True
            or plan.get('kind')!='probabilistic_single_history_spd_val_adaptation_plan_v1'
            or plan.get('class_scope')!=['car'] or plan.get('checkpoint_seed') not in SEEDS
            or plan.get('geometry_baseline') is not False or audit.get('configuration')!=plan.get('configuration')):
        raise ValueError('complete learned full-val run and its independent audit required')
    config = plan['configuration']
    if (config.get('association_algorithm')!='lbp' or config.get('anchor_decoder')!='joint-map'
            or config.get('update_rule') not in RULES or audit.get('update_rule')!=config['update_rule']
            or audit.get('seed')!=plan['checkpoint_seed'] or audit.get('runtime')!=plan.get('runtime')
            or final.get('checkpoint_sha256')!=plan.get('checkpoint_sha256')
            or any(receipt.get('probabilistic_'+key)!=config[key]
                   for key in ('association_algorithm','update_rule','anchor_decoder'))):
        raise ValueError('audited fixed probabilistic backend differs')
    expected = {str(child(run,name)):receipt[key] for name,key in (
        ('plan.json','plan_sha256'),('predictions.jsonl','predictions_sha256'),
        ('tracking.jsonl','tracking_sha256'),('frame-timings.jsonl','frame_timings_sha256'))}
    expected.update({str(child(run,'receipt.json')):audit['replay_receipt_sha256'],
                     str(child(run,'full-validation-receipt.json')):receipt_sha})
    for head in receipt['sequence_heads'].values():
        expected[str(child(run,head['database']))]=head['database_sha256']
    for name in ('audit_probabilistic_validation.py','compare_train_probabilistic_controls.py'):
        p = child(ROOT,'tools/event_track_v2x/'+name);expected[str(p)]=native.sha(p)
    if audit.get('input_sha256')!=expected:
        raise ValueError('independent audit input inventory differs')
    for p,sha in expected.items():
        actual=native.evidence(p)
        if actual['sha256']!=sha:raise ValueError('audited evidence changed: '+p)
        evidence[p]=actual
    if plan.get('source_sha256')!=inference_sources():
        raise ValueError('complete current inference source inventory required')
    for relative,sha in plan['source_sha256'].items():
        p=child(ROOT,relative);actual=native.evidence(p)
        if actual['sha256']!=sha:raise ValueError('inference source changed: '+relative)
        evidence[str(p)]=actual
    unchanged(evidence)
    return dict(plan=plan,receipt=receipt,audit=audit,evidence=evidence)


def evaluate(run, receipt_sha, audit_path, audit_sha, ground_truth, output):
    output,ground_truth=Path(output).absolute(),Path(ground_truth).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new non-symlink evaluation output required')
    if native.sha(native.__file__)!=COMMON_SHA:
        raise ValueError('sealed evaluator helper source differs')
    bound=inspect_run(run,receipt_sha,audit_path,audit_sha)
    evidence=bound['evidence'];plan=bound['plan'];receipt=bound['receipt']
    for p in (Path(__file__),Path(native.__file__),native.ADAPTER_PATH):
        evidence[str(p.absolute())]=native.evidence(p)
    for name,expected in (('manifest.json',native.GT_MANIFEST_SHA256),('ground-truth.jsonl',native.GT_SHA256)):
        p=child(ground_truth,name);observed=native.evidence(p)
        if observed['sha256']!=expected:raise ValueError('sealed official val GT required')
        evidence[str(p)]=observed
    adapter=native.load_adapter();runtime=adapter.runtime_evidence();native.validate_runtime(runtime)
    gold=adapter.golden_cases()
    if gold.get('passed') is not True:raise ValueError('native car golden cases failed')
    manifest,gt=adapter.load_ground_truth(ground_truth)
    predictions_path=child(run,'predictions.jsonl')
    predictions=[json.loads(line) for line in predictions_path.read_bytes().splitlines()]
    coverage=adapter.validate_predictions(predictions,gt)
    if (coverage['frames']!=3316 or coverage['sequences']!=21
            or set(coverage['predictions_per_class'])-{'car'}):
        raise ValueError('complete car-only prediction coverage required')
    metrics=adapter.compute_metrics(gt,predictions,classes=('car',))
    primary=native.car_metrics(metrics)
    unchanged(evidence)
    output.mkdir()
    for name,value in (('metrics.json',metrics),('runtime.json',runtime),('golden-cases.json',gold)):
        native.write_json(output/name,value)
    selected_protocol=protocol(adapter)
    report=dict(kind=KIND,status='complete',run_directory=str(Path(run).absolute()),
        inference_receipt_sha256=receipt_sha,inference_audit_sha256=audit_sha,
        inference_audit_path=str(Path(audit_path).absolute()),ground_truth_directory=str(ground_truth),
        plan_sha256=receipt['plan_sha256'],predictions_sha256=receipt['predictions_sha256'],
        seed=plan['checkpoint_seed'],update_rule=plan['configuration']['update_rule'],configuration=plan['configuration'],
        cache_sha256=plan['cache_sha256'],schedule_sha256=plan['schedule_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'],scorer_signature=plan['scorer_signature'],
        factor_stream_sha256=bound['audit']['factor_stream_sha256'],inference_runtime=plan['runtime'],
        inference_source_sha256=plan['source_sha256'],ground_truth_manifest_sha256=native.GT_MANIFEST_SHA256,
        ground_truth_sha256=manifest['ground_truth_sha256'],coverage=coverage,primary_car=primary,
        protocol=selected_protocol,protocol_sha256=hashlib.sha256(native.canonical(selected_protocol)).hexdigest(),
        files={p.name:native.evidence(p) for p in sorted(output.iterdir())},input_evidence=evidence,
        reporting_scope='car_only',validation_already_seen=True,test_payloads_read=False,
        validation_checkpoint_selection=False,validation_parameter_search=False,
        same_input_selection_as_legacy_source_ablation=False,reproduced_public_method=False,
        same_state_time_protocol_as_recoverable=False,fair_resources_verified=False,paper_eligible=False)
    native.write_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','seed','update_rule','coverage','primary_car')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','audit','ground-truth','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('receipt-sha256','audit-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    evaluate(args.run,args.receipt_sha256,args.audit,args.audit_sha256,args.ground_truth,args.output)
