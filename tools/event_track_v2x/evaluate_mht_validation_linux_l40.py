#!/usr/bin/env python3
"""Native car metrics for independently audited Linux L40S full official MHT SPD val.

The frozen GT and metric runtime remain isolated from inference. No Torch,
training, test set, tuning, public upload or retrospective ID rewrite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x import audit_full_mht_validation_linux_l40 as contract

native,common = contract.native,contract.common
KIND = 'scan_mht_car_full_validation_evaluation_linux_l40_v1'


def protocol(adapter):
    result = common.protocol(adapter)
    result['kind'] = 'spd_native_scan_mht_car_protocol_v1'
    return result


def evaluate(run,receipt_sha,audit_path,audit_sha,ground_truth,output):
    output,ground_truth = Path(output).absolute(),Path(ground_truth).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new non-symlink evaluation output required')
    if native.sha(native.__file__) != common.COMMON_SHA:
        raise ValueError('sealed evaluator helper source differs')
    bound = contract.inspect_audited_run(run,receipt_sha,audit_path,audit_sha)
    evidence,plan,receipt = bound['evidence'],bound['plan'],bound['receipt']
    for p in (Path(__file__),native.ADAPTER_PATH): evidence[str(p.absolute())] = native.evidence(p)
    for name,expected in (('manifest.json',native.GT_MANIFEST_SHA256),('ground-truth.jsonl',native.GT_SHA256)):
        p = common.child(ground_truth,name); observed = native.evidence(p)
        if observed['sha256'] != expected: raise ValueError('sealed official val GT required')
        evidence[str(p)] = observed
    adapter = native.load_adapter();runtime = adapter.runtime_evidence();native.validate_runtime(runtime)
    gold = adapter.golden_cases()
    if gold.get('passed') is not True: raise ValueError('native car golden cases failed')
    manifest,gt = adapter.load_ground_truth(ground_truth)
    with common.child(run,'predictions.jsonl').open('rb') as stream:
        predictions = [json.loads(line) for line in stream]
    coverage = adapter.validate_predictions(predictions,gt)
    if (coverage['frames'] != 3316 or coverage['sequences'] != 21
            or set(coverage['predictions_per_class'])-{'car'}):
        raise ValueError('complete car-only prediction coverage required')
    metrics = adapter.compute_metrics(gt,predictions,classes=('car',)); primary = native.car_metrics(metrics)
    common.unchanged(evidence)
    output.mkdir()
    for name,value in (('metrics.json',metrics),('runtime.json',runtime),('golden-cases.json',gold)):
        native.write_json(output/name,value)
    selected_protocol = protocol(adapter)
    report = dict(kind=KIND,status='complete',run_directory=str(Path(run).absolute()),
        inference_receipt_sha256=receipt_sha,inference_audit_sha256=audit_sha,
        inference_audit_path=str(Path(audit_path).absolute()),ground_truth_directory=str(ground_truth),
        plan_sha256=receipt['plan_sha256'],predictions_sha256=receipt['predictions_sha256'],
        seed=plan['checkpoint_seed'],width=plan['configuration']['state']['active_limit'],
        configuration=plan['configuration'],cache_sha256=plan['cache_sha256'],schedule_sha256=plan['schedule_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'],scorer_signature=plan['scorer_signature'],
        inference_runtime=plan['runtime'],inference_source_sha256=plan['source_sha256'],
        ground_truth_manifest_sha256=native.GT_MANIFEST_SHA256,ground_truth_sha256=manifest['ground_truth_sha256'],
        coverage=coverage,primary_car=primary,protocol=selected_protocol,protocol_sha256=contract.ledger.digest(selected_protocol),
        files={p.name:native.evidence(p) for p in sorted(output.iterdir())},input_evidence=evidence,
        reporting_scope='car_only',validation_already_seen=True,test_payloads_read=False,parameter_training=False,
        validation_checkpoint_selection=False,validation_parameter_search=False,
        same_input_selection_as_legacy_source_ablation=False,reproduced_public_method=False,
        fair_resources_verified=False,paper_eligible=False)
    native.write_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','seed','width','primary_car')}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','audit','ground-truth','output'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('receipt-sha256','audit-sha256'): p.add_argument('--'+name,required=True)
    args = p.parse_args(); evaluate(args.run,args.receipt_sha256,args.audit,args.audit_sha256,args.ground_truth,args.output)
