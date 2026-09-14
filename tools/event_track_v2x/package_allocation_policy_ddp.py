#!/usr/bin/env python3
"""Build a private priority-DDP package only from a verified full TRAIN teacher.

No upload or training. Re-export the sealed teacher in task-local temporary
storage and compare every derived shard before packaging. Only source files,
numeric model-progress groups and manifests enter the package, never teacher
traces, predictions, raw GT, detector data or existing model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tools.event_track_v2x.train_allocation_policy_ddp import audit_data,priority_ddp_sources
from tools.event_track_v2x.prepare_forest_training import TRAIN_CACHE_SHA256
from transvision.models.event_track_v2x.allocation_training import allocation_sources,export_training
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file,sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from tools.event_track_v2x.assemble_allocation_teachers import KIND as ASSEMBLED_KIND, verify_assembly

PAIR_SHA256='787b7e4dfa3fb9d97ba0a6a4a216a2990af990dc9226ceec01053e7d2e7da984'
KIND='component_priority_ddp_package_v1'


def read_bound(root,name,expected):
    path=contained_file(root,name)
    if sha_file(path)!=expected:raise ValueError('teacher evidence changed: '+name)
    return json.loads(path.read_bytes())


def check_teacher_contract(manifest,receipt,final,plan):
    if (final.get('full_official_train_trace_completed') is not True
            or {k:v for k,v in final.items() if k!='full_official_train_trace_completed'}!=receipt
            or receipt.get('status')!='complete' or receipt.get('allocation_teacher') is not True
            or receipt.get('cache_split')!='train' or receipt.get('completed_frames')!=7445
            or receipt.get('scheduled_frames')!=7445 or len(receipt.get('sequence_heads',{}))!=46
            or receipt.get('learned_identity_enabled') is not True):
        raise ValueError('complete learned 46-sequence 7445-event teacher required')
    if (plan.get('kind') not in ('component_allocation_teacher_full_train_plan_v1',ASSEMBLED_KIND)
            or plan.get('full_official_train_verified') is not True
            or plan.get('input_full_official_train_verified') is not True
            or plan.get('development_sequence') is not None or plan.get('scheduled_frames')!=7445
            or plan.get('class_scope')!=['car'] or plan.get('cache_sha256')!=TRAIN_CACHE_SHA256
            or plan.get('cooperative_metadata_sha256')!=PAIR_SHA256
            or plan.get('geometry_development') is not False
            or plan.get('allocation_training_binding')!=manifest['binding']
            or any(plan['source_sha256'].get(p)!=h for p,h in allocation_sources().items())
            or {r['sequence_id'] for r in manifest['shards']}!=set(receipt['sequence_heads'])):
        raise ValueError('teacher provenance or exported tracking binding differs')


def check_event_coverage(trace_path,pairs):
    expected=[(p['vehicle_sequence'],p['vehicle_frame']) for p in pairs]
    if (len(expected)!=7445 or len(set(expected))!=7445 or len({s for s,_ in expected})!=46
            or any(p['vehicle_sequence']!=p['infrastructure_sequence'] for p in pairs)):
        raise ValueError('full distinct train cooperative schedule required')
    actual=[]
    with Path(trace_path).open('rb') as stream:
        for line in stream:
            row=json.loads(line)['tracking']
            if row.get('training_trace_only') is not True or row.get('offline_counterfactual_probes') is not True:
                raise ValueError('nonteacher trace event cannot enter training package')
            actual.append((row['sequence_id'],row['event_id']))
    if len(actual)!=len(expected) or len(set(actual))!=len(actual) or set(actual)!=set(expected):
        raise ValueError('teacher event coverage differs from frozen full train schedule')


def source_inventory():
    paths=[ROOT/'transvision'/n for n in ('__init__.py','register.py','version.py')]
    paths.append(ROOT/'transvision/models/__init__.py')
    paths.extend(sorted((ROOT/'transvision/models/event_track_v2x').glob('*.py')))
    paths.extend(ROOT/'tools/event_track_v2x'/n for n in ('__init__.py','prepare_forest_training.py',
        'train_forest_identity.py','train_forest_identity_ddp.py','train_allocation_policy_ddp.py',
        'run_clearml_allocation_policy_ddp.py'))
    records=[]
    for path in paths:
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('ordinary nonsymlink source files required')
        records.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha_file(path)))
    if set(priority_ddp_sources())-{r['path'] for r in records}:
        raise ValueError('training dependency missing from priority source package')
    return records


def write_archive(root,records,path,max_bytes):
    if sum(r['bytes'] for r in records)>max_bytes:raise ValueError('package exceeds declared unpacked size cap')
    with tarfile.open(path,'x:gz') as archive:
        for record in records:
            source=contained_file(root,record['path'])
            if source.stat().st_size!=record['bytes'] or sha_file(source)!=record['sha256']:
                raise ValueError('package input changed before archive')
            archive.add(source,arcname=record['path'],recursive=False)
            if sha_file(source)!=record['sha256']:raise ValueError('package input changed during archive')


def pack(data,manifest_sha256,teacher,teacher_final_sha256,cooperative_metadata,output):
    data,teacher,output=Path(data),Path(teacher),Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('new nonsymlink package output required')
    manifest,fit,held,statistics=audit_data(data,manifest_sha256)
    receipt=read_bound(teacher,'receipt.json',manifest['replay_receipt_sha256'])
    final=read_bound(teacher,'full-train-teacher-receipt.json',teacher_final_sha256)
    plan=read_bound(teacher,'plan.json',receipt['plan_sha256'])
    check_teacher_contract(manifest,receipt,final,plan)
    pairs_path=Path(cooperative_metadata)
    if pairs_path.is_symlink() or sha_file(pairs_path)!=PAIR_SHA256:
        raise ValueError('frozen full train cooperative metadata required')
    trace=contained_file(teacher,'tracking.jsonl')
    if sha_file(trace)!=receipt['tracking_sha256']:raise ValueError('teacher trace changed')
    check_event_coverage(trace,json.loads(pairs_path.read_bytes()))
    if plan['kind']==ASSEMBLED_KIND:
        verify_assembly(teacher,plan,receipt,pairs_path)
    # Existing exporter checks every numerical target against its counterfactual
    # arithmetic and preserves exact group membership, including negative labels.
    with tempfile.TemporaryDirectory(prefix='rbf-priority-package-audit-') as temporary:
        repeated=export_training(teacher,manifest['replay_receipt_sha256'],Path(temporary)/'derived')
        if repeated!=manifest:raise ValueError('provided derived data differ from the sealed teacher re-export')
    bound=priority_ddp_sources();sources=source_inventory()
    records=[dict(path='manifest.json',bytes=(data/'manifest.json').stat().st_size,sha256=manifest_sha256)]
    records.extend(dict(path=r['path'],bytes=contained_file(data,r['path']).stat().st_size,sha256=r['sha256'])
                   for r in manifest['shards'])
    evidence={pairs_path:PAIR_SHA256,trace:receipt['tracking_sha256'],
        teacher/'receipt.json':manifest['replay_receipt_sha256'],teacher/'plan.json':receipt['plan_sha256'],
        teacher/'full-train-teacher-receipt.json':teacher_final_sha256}
    for head in receipt['sequence_heads'].values():
        evidence[contained_file(teacher,head['database'])]=head['database_sha256']
    if any(sha_file(p)!=h for p,h in evidence.items()):raise ValueError('teacher evidence changed during package preflight')
    output.mkdir()
    try:
        write_archive(ROOT,sources,output/'source.tar.gz',16*1024**2)
        write_archive(data,records,output/'priority-groups.tar.gz',8*1024**3)
        if (bound!=priority_ddp_sources() or any(sha_file(p)!=h for p,h in evidence.items())
                or any(sha_file(ROOT/r['path'])!=r['sha256'] for r in sources)
                or any(sha_file(data/r['path'])!=r['sha256'] for r in records)):
            raise ValueError('package inputs changed during construction')
        result=dict(kind=KIND,training_manifest_sha256=manifest_sha256,split='train',class_scope=['car'],
            full_official_train_trace=True,teacher_final_sha256=teacher_final_sha256,
            teacher_receipt_sha256=manifest['replay_receipt_sha256'],teacher_plan_sha256=receipt['plan_sha256'],
            teacher_tracking_sha256=receipt['tracking_sha256'],teacher_frames=7445,teacher_sequences=46,
            binding=manifest['binding'],statistics=statistics,fit_sequences=sorted(r['sequence_id'] for r in fit),
            holdout_sequences=sorted(r['sequence_id'] for r in held),source_sha256=bound,source_inventory=sources,
            data_inventory=records,raw_GT_included=False,teacher_traces_included=False,
            predictions_included=False,original_detection_stream_included=False,trained_models_included=False,
            strict_pipeline_isolated_selection=False,final_full_train_refit=False,paper_eligible=False,
            package_tool_sha256=sha_file(__file__),
            artifacts=[dict(path=n,bytes=(output/n).stat().st_size,sha256=sha_file(output/n))
                       for n in ('source.tar.gz','priority-groups.tar.gz')])
        _new_json(output/'package.json',result)
        print(json.dumps(dict(package_sha256=sha_file(output/'package.json'),artifacts=result['artifacts'],
                             source_files=len(sources),training_submitted=False),sort_keys=True))
        return result
    except BaseException as error:
        _new_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,
            partial_package_not_for_upload=True,training_submitted=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','teacher','cooperative-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--teacher-final-sha256',required=True)
    a=p.parse_args();pack(a.data,a.manifest_sha256,a.teacher,a.teacher_final_sha256,a.cooperative_metadata,a.output)
