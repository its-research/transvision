#!/usr/bin/env python3
"""Full 13-configuration first/replay provenance audit of preserved raw streams.

Reuses SHA-bound output-only oracles, without importing any predictor. Strict
byte differences remain failures; parsing a valid chain does not repair them.
GT, metric engines, detector/cache/fusion reconstruction and reruns are separate.
"""
import argparse
import ast
import gzip
import hashlib
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import time
import numpy as np

SEEDS=(1337,2027,3407)
FILES=('predictions.jsonl','association.jsonl','diagnostics.jsonl.gz')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def ordinary(path):
    p=Path(path).absolute()
    if not p.is_file() or any(v.is_symlink() for v in (p,*p.parents)):raise ValueError('ordinary original evidence required')
    return p


def read(path,digest=None):
    p=ordinary(path)
    if digest is not None and sha(p)!=digest:raise ValueError('frozen evidence SHA differs')
    return json.loads(p.read_bytes())


def new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def oracles(args):
    path=ordinary(args.audit_source)
    if sha(path)!=args.audit_source_sha256:raise ValueError('original independent output oracle differs')
    tree=ast.parse(path.read_text())
    if any(isinstance(n,(ast.Import,ast.ImportFrom)) and
            any(v.name.startswith(('torch','transvision')) for v in n.names) for n in ast.walk(tree)):
        raise ValueError('output oracle must not import a predictor')
    spec=importlib.util.spec_from_file_location('unchanged_output_stream_oracle',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    birth=ordinary(args.m4_source)
    if sha(birth)!=args.m4_source_sha256:raise ValueError('original M4 output oracle differs')
    # Extract exactly the established standalone scan function. Producer/model
    # imports and top-level producer code are never executed.
    functions=[n for n in ast.parse(birth.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scan_birth_streams']
    if len(functions)!=1:raise ValueError('exact original M4 scan function required')
    environment=dict(hashlib=hashlib,gzip=gzip,zip_longest=zip_longest,np=np,FRAME_COUNT=3316,SEQUENCE_COUNT=21)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(birth),'exec'),environment)
    return audit,environment['scan_birth_streams'],hashlib.sha256(ast.dump(functions[0],include_attributes=False).encode()).hexdigest()


def verify(args):
    root=Path(args.root).absolute();out=Path(args.output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh output directory required')
    inputs=read(root/'inputs.json',args.inputs_sha256)
    materialization=read(root/'materialization-receipt.json',args.materialization_sha256)
    if materialization['all_bytes_match_registered_originals'] is not True or materialization['archive_sha256']!=inputs['archive_sha256']:
        raise ValueError('all original registered stream bytes must be materialized')
    manifest=read(inputs['manifest'],inputs['manifest_sha256']);records=manifest['members']
    if len(records)!=78 or len({r['member'] for r in records})!=78:raise ValueError('complete 78-member historical cohort required')
    pins={str(root/'inputs.json'):args.inputs_sha256,str(root/'materialization-receipt.json'):args.materialization_sha256,
        inputs['manifest']:inputs['manifest_sha256']}
    inventory={}
    for record in records:
        name=record['member'];p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name:raise ValueError('canonical original stream path required')
        path=ordinary(root/'streams'/name)
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('registered original stream byte identity changed')
        pins[str(path)]=record['sha256'];inventory[name]=record
    for name,record in inputs['original_metadata'].items():
        path=ordinary(root/'streams'/name)
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('original metadata changed')
        pins[str(path)]=record['sha256']
    audit,m4,m4_ast=oracles(args);pins[str(ordinary(args.audit_source))]=args.audit_source_sha256;pins[str(ordinary(args.m4_source))]=args.m4_source_sha256
    out.mkdir(parents=True);new(out/'plan.json',dict(kind='rbf_stage1_13_run_preserved_output_full_provenance_plan_v1',
        inputs_sha256=args.inputs_sha256,materialization_sha256=args.materialization_sha256,audit_source_sha256=args.audit_source_sha256,
        original_M4_scan_AST_sha256=m4_ast,strict_byte_parity_required=True,GT_read=False,predictor_imported_or_executed=False,
        full_fusion_or_metric_engines_independently_recomputed=False,stage1_qualified=False))
    results=[];failures=[];cohorts=set();parity=[];started=time.monotonic()
    try:
        for campaign in ('mechanism','birth'):
            first=root/'streams'/campaign/'first';replay=root/'streams'/campaign/'replay'
            plans={side:read(root/'streams'/campaign/side/'frozen-plan.json') for side in ('first','replay')}
            if plans['first']!=plans['replay']:raise ValueError('first/replay original plans differ')
            plan=plans['first'];plan_sha=sha(first/'frozen-plan.json')
            specs=([dict(run_id=f'M0-seed-{s}',mode='M0',seed=s,deterministic_control=False) for s in SEEDS]
                +[dict(run_id='M1-all-unmatched',mode='M1',seed=1337,deterministic_control=True)]
                +[dict(run_id=f'{m}-seed-{s}',mode=m,seed=s,deterministic_control=False) for m in ('M2','M3') for s in SEEDS]
                if campaign=='mechanism' else [dict(run_id=f'M4-seed-{s}',mode='M4',seed=s,deterministic_control=False) for s in SEEDS])
            if (plan['runs']!=specs or plan['official_validation_frames']!=3316 or plan['sequences']!=21
                    or plan['cache_sha256']!=audit.CACHE_SHA or plan['schedule_sha256']!=audit.SCHEDULE_SHA
                    or plan['config']['deadline_us']!=100000 or plan['device']!='cpu' or plan['cpu_threads']!=1
                    or any(plan[k] is not False for k in ('gt_model_inputs','test_payloads_read','val_parameter_fitting','seed_selection','paper_eligible'))):
                raise ValueError('original campaign input/freeze contract differs')
            summaries={side:read(root/'streams'/campaign/side/'summary.json') for side in ('first','replay')}
            for side,summary in summaries.items():
                if summary['plan_sha256']!=plan_sha or set(summary['runs'])!={s['run_id'] for s in specs} or summary['status']!='completed':
                    raise ValueError('full original campaign summary required')
            for spec in specs:
                name=spec['run_id'];differences=[]
                for file in FILES:
                    left=inventory[campaign+'/first/'+name+'/'+file];right=inventory[campaign+'/replay/'+name+'/'+file]
                    if (left['sha256'],left['bytes'])!=(right['sha256'],right['bytes']):differences.append(file)
                parity.append(dict(run_id=name,all_three_streams_byte_identical=not differences,failed_streams=differences))
                for side in ('first','replay'):
                    folder=root/'streams'/campaign/side;receipt=read(folder/name/'receipt.json')
                    if receipt!=summaries[side]['runs'][name] or any(receipt[k]!=v for k,v in spec.items()):
                        raise ValueError('original run receipt/summary identity differs')
                    for file,key in [('predictions.jsonl','predictions_sha256'),('association.jsonl','association_sha256'),('diagnostics.jsonl.gz','diagnostics_gzip_sha256')]:
                        if receipt[key]!=inventory[campaign+'/'+side+'/'+name+'/'+file]['sha256']:raise ValueError('original receipt stream identity differs')
                    try:
                        value=m4(folder,spec,plan_sha,audit) if campaign=='birth' else audit.scan_run(folder,spec,plan_sha,plan['config']['deadline_us'])
                        result=dict(campaign=campaign,execution=side,run_id=name,parsed_internal_output_provenance=True,result=value)
                        cohorts.add(value['input_order_sha256'])
                    except BaseException as e:
                        result=dict(campaign=campaign,execution=side,run_id=name,parsed_internal_output_provenance=False,
                            error_type=type(e).__name__,error=str(e));failures.append(result)
                    results.append(result);new(out/(campaign+'-'+side+'-'+name+'.json'),result)
                    print(json.dumps(dict(stage='original_stage1_full_output_provenance',completed_executions=len(results),total_executions=26,
                        run_id=name,execution=side,internal_provenance_pass=result['parsed_internal_output_provenance'],strict_run_parity_pass=not differences)),flush=True)
        association_checks=[]
        for mode in ('M2','M3','M4'):
            campaign='birth' if mode=='M4' else 'mechanism'
            for seed in SEEDS:
                for side in ('first','replay'):
                    baseline=inventory[f'mechanism/{side}/M0-seed-{seed}/association.jsonl'];target=inventory[f'{campaign}/{side}/{mode}-seed-{seed}/association.jsonl']
                    association_checks.append(dict(mode=mode,seed=seed,execution=side,original_M0_pairs_and_weights_byte_identical=
                        (baseline['sha256'],baseline['bytes'])==(target['sha256'],target['bytes'])))
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('original inputs changed during full output audit')
        proof=dict(kind='rbf_stage1_13_configuration_26_execution_full_original_output_provenance_v1',
            complete_original_cohort_scanned=len(results)==26,all_internal_output_provenance_passed=not failures,
            all_runs_same_input_availability_and_candidate_order=len(cohorts)==1 and not failures,
            input_order_sha256=sorted(cohorts),runs=results,internal_failures=failures,strict_replay_byte_parity=parity,
            all_13_strict_replay_byte_parity_passed=all(r['all_three_streams_byte_identical'] for r in parity),
            same_M0_pair_weight_checks=association_checks,all_M2_M3_M4_original_pair_weights_preserved=all(r['original_M0_pairs_and_weights_byte_identical'] for r in association_checks),
            parsed_frames=sum(r['result']['frames'] for r in results if r['parsed_internal_output_provenance']),
            original_strict_M3_seed2027_failure_preserved=True,GT_read=False,models_executed=False,
            independent_detector_fusion_or_metric_recomputation=False,full_identity_error_duration_complete=False,
            stage1_qualified=False,paper_metrics_complete=False,input_sha256=pins,source_sha256=sha(__file__),seconds=time.monotonic()-started)
        new(out/'provenance-receipt.json',proof)
    except BaseException as e:
        new(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),completed_executions=len(results),
            partials_and_failures_preserved=True,stage1_qualified=False,source_sha256=sha(__file__)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','inputs-sha256','materialization-sha256','audit-source','audit-source-sha256','m4-source','m4-source-sha256','output'):
        p.add_argument('--'+name,required=True)
    verify(p.parse_args())
