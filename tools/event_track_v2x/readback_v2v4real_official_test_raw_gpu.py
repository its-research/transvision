#!/usr/bin/env python3
"""Independent cloud-byte, raw-schema and full-source-frame coverage readback.

This does not independently reproduce detector numerics, 9D motion/covariance,
tracking or official metrics. Those downstream gates remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import re
import tarfile
import numpy as np
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as export
from tools.event_track_v2x import merge_v2v4real_official_test_raw_shards as merger
from tools.event_track_v2x.run_v2v4real_official_test_gpu_shards import download

ARTIFACTS=('native-features','native-feature-manifest','native-feature-index','projection-manifest',
           'split-admission','raw-merge-receipt.json','raw-export-receipt.json','projection-frames','execution-receipt','runtime-dependencies')


def verify_runtime_dependency_receipt(path,source,offline_manifest=None):
    """Validate the source-bound runtime evidence separately from predictions."""
    from tools.event_track_v2x import v2v4real_official_test_runtime as runtime
    row=source['sources']['tools/event_track_v2x/v2v4real_official_test_runtime.py']
    if export.sha(runtime.__file__)!=row['sha256']:
        raise ValueError('runtime checker policy differs from exact executed source')
    value=json.loads(Path(path).read_bytes())
    if (value.get('kind')!='v2v4real_native_runtime_dependency_attestation_v1'
            or value.get('same_interpreter_as_rank_children') is not True
            or value.get('model_or_inference_source_changed') is not False
            or value.get('raw_package_manager_output_persisted') is not False
            or value.get('experiment_accepted') is not False):
        raise ValueError('native runtime evidence scope differs')
    if (value['before']['interpreter']!=value['after']['interpreter']
            or not value['after']['interpreter'].startswith('/')):
        raise ValueError('native runtime interpreter identity differs')
    runtime.require_runtime(value['after'])
    packages=value.get('system_packages',[])
    if len({x['package'] for x in packages})!=len(packages):
        raise ValueError('duplicate OS package attestation')
    for package in packages:
        if (set(package)!={'package','version','architecture'}
                or package['package'] not in runtime.SYSTEM_PACKAGES.values()
                or not re.fullmatch(r'[A-Za-z0-9.+:~_-]+',package['version'])
                or package['architecture'] not in ('amd64','arm64')):
            raise ValueError('unrecognized native OS package attestation')
    offline=value.get('offline_libraries',[])
    if len(offline)>1:raise ValueError('ambiguous offline runtime library evidence')
    if offline:
        if offline_manifest is None:raise ValueError('independent offline library manifest required')
        export.asset(offline_manifest,runtime.OFFLINE_MANIFEST_SHA)
        inventory=json.loads(Path(offline_manifest).read_bytes())['inventory'];record=offline[0]
        if (record.get('task_id')!=runtime.OFFLINE_TASK
                or record.get('archive_sha256')!=runtime.OFFLINE_ARCHIVE_SHA
                or record.get('manifest_sha256')!=runtime.OFFLINE_MANIFEST_SHA
                or record.get('inventory')!=inventory
                or record.get('libc_and_loader_excluded') is not True
                or record.get('prior_SPD_runtime_acceptance_inherited') is not False):
            raise ValueError('offline runtime byte provenance differs')
    return dict(runtime_artifact_sha256=export.sha(path),
        fixed_Python_versions_and_import_results_verified=True,
        same_rank_interpreter_verified=True,system_packages=packages,offline_libraries=offline,
        independent_detector_numeric_accepted=False)


def verify(directory):
    d=Path(directory);manifest=json.loads((d/'native-feature-manifest').read_bytes())
    receipt=json.loads((d/'raw-export-receipt.json').read_bytes());merged=json.loads((d/'raw-merge-receipt.json').read_bytes())
    execution=json.loads((d/'execution-receipt').read_bytes())
    export.asset(d/'projection-frames',export.FRAME_SHA)
    export.asset(d/'projection-manifest',export.PROJECTION_SHA);export.asset(d/'split-admission',export.ADMISSION_SHA)
    records=[json.loads(line) for line in (d/'projection-frames').read_bytes().splitlines()]
    export.validate_scope(json.loads((d/'projection-manifest').read_bytes()),json.loads((d/'split-admission').read_bytes()),records)
    if manifest.get('kind')!=export.KIND or set(manifest)!=merger.RAW_MANIFEST_FIELDS|{'projection_manifest','split_admission'}:
        raise ValueError('complete official-test raw envelope required')
    seed=manifest['checkpoint_seed']
    if (seed not in export.CHECKPOINTS or manifest['checkpoint_sha256']!=export.CHECKPOINTS[seed]
            or manifest['gt_read'] is not False or manifest['paper_metric'] is not False
            or manifest['matching_candidate_rule']!='raw_score_ge_0.05_stable_top64_before_nms'
            or manifest['velocity_exported'] is not False or manifest['covariance_exported'] is not False
            or receipt['raw_manifest_sha256']!=export.sha(d/'native-feature-manifest')
            or receipt['raw_archive_sha256']!=export.sha(d/'native-features')
            or receipt['merge_receipt_sha256']!=export.sha(d/'raw-merge-receipt.json')
            or merged['raw_archive_sha256']!=receipt['raw_archive_sha256']
            or execution['raw_archive_sha256']!=receipt['raw_archive_sha256']):
        raise ValueError('raw checkpoint/provenance bytes differ')
    raw=(d/'native-feature-index').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['rows_sha256']:raise ValueError('raw index bytes differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    expected={(r['sequence_id'],r['frame_ordinal'],'vehicle-side' if r['is_ego'] else 'infrastructure-side'):r for r in records}
    if len(expected)!=3986 or len(rows)!=3986 or len({merger.identity(r) for r in rows})!=3986 or {merger.identity(r) for r in rows}!=set(expected):
        raise ValueError('original full-source coverage differs')
    for row in rows:
        original=expected[merger.identity(row)]
        if row['frame_id']!=original['frame_key'] or row['source_to_world']!=original['source_to_world'] or row['pcd_sha256']!=original['pcd_sha256']:
            raise ValueError('raw source row binding differs')
    names=['anchors.npy','manifest.json','predictions.jsonl',*(r['payload'] for r in rows)]
    if len(set(names))!=len(names):raise ValueError('duplicate raw payload path')
    valid=0
    with tarfile.open(d/'native-features') as stream:
        members=stream.getmembers()
        if [m.name for m in members]!=names or any(not m.isfile() for m in members):raise ValueError('raw archive inventory differs')
        if stream.extractfile('manifest.json').read()!=(d/'native-feature-manifest').read_bytes() or stream.extractfile('predictions.jsonl').read()!=raw:
            raise ValueError('raw archive metadata differs')
        anchors=stream.extractfile('anchors.npy').read()
        if hashlib.sha256(anchors).hexdigest()!=manifest['anchor_file_sha256']:raise ValueError('anchor file differs')
        for row in rows:valid+=merger.validate_payload(stream.extractfile(row['payload']).read(),row)
    if valid!=manifest['appearance_valid_candidates'] or sum(r['candidates'] for r in rows)!=manifest['total_candidates']:
        raise ValueError('raw candidate totals differ')
    return dict(checkpoint_seed=seed,source_rows=3986,paired_frames=1993,sequences=9,
                empty_source_rows=sum(r['candidates']==0 for r in rows),candidates=manifest['total_candidates'],appearance_valid_candidates=valid,
                independent_byte_schema_coverage_accepted=True,independent_detector_numeric_accepted=False,
                native_nine_state_cache_admitted=False,tracking_accepted=False,paper_metric_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-id',required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from clearml import Task
    task=Task.get_task(task_id=a.task_id)
    if str(task.status)!='completed':raise ValueError('completed candidate required; do not restart live task')
    source=json.loads((a.source/'source-manifest.json').read_bytes());params=task.get_parameters()
    if (hashlib.sha256(task.data.script.diff.encode()).hexdigest()!=source['bootstrap_sha256']
            or params.get('General/execution_bundle_sha256')!=source['execution_bundle_sha256']):
        raise ValueError('cloud execution source identity differs')
    a.output.mkdir(parents=True,exist_ok=False);inputs=[]
    for name in ARTIFACTS:
        item=task.artifacts[name];inputs.append(download(task,name,item.hash,a.output/name))
    runtime_value=json.loads((a.output/'runtime-dependencies').read_bytes());offline_manifest=None
    if runtime_value.get('offline_libraries'):
        from tools.event_track_v2x import v2v4real_official_test_runtime as runtime
        offline_manifest=a.output/'offline-libraries-independent-manifest.json'
        inputs.append(download(Task.get_task(task_id=runtime.OFFLINE_TASK),'manifest',runtime.OFFLINE_MANIFEST_SHA,offline_manifest))
    runtime_acceptance=verify_runtime_dependency_receipt(a.output/'runtime-dependencies',source,offline_manifest)
    result=verify(a.output)
    result['runtime_dependency_evidence']=runtime_acceptance
    if params.get('General/seed')!=str(result['checkpoint_seed']):raise ValueError('task seed differs')
    result.update(kind='v2v4real_official_test_raw_independent_cloud_byte_coverage_v1',task_id=task.id,
        bootstrap_sha256=source['bootstrap_sha256'],cloud_artifacts=inputs,checker_sha256=export.sha(__file__),
        input_sha256={name:export.sha(a.output/name) for name in ARTIFACTS})
    path=a.output/'independent-byte-schema-coverage-receipt.json';path.write_bytes(export.canonical(result))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from rbf_nested_seen_val_v2_common import register
    register(path,result['kind']);print(json.dumps(result))


if __name__=='__main__':main()
