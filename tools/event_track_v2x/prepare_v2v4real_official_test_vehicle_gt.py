#!/usr/bin/env python3
"""Evaluator-only native vehicle GT from the three pinned original test ZIPs.

No detector, fitting, threshold selection, prediction or tracking evaluation.
Each part is compared with the existing fixed native numerical GT oracle.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.extract_v2v4real_archive import extract_volume
from tools.event_track_v2x.prepare_v2v4real_ground_truth import prepare
from tools.event_track_v2x.v2v4real_gt_oracle import SOURCE_HASHES,COMMIT,verified_sources
from transvision.models.event_track_v2x.detection_cache_v2 import canonical,sha_file
from transvision.models.event_track_v2x.v2v4real_ground_truth import load_vehicle_ground_truth
from tools.event_track_v2x.merge_v2v4real_vehicle_ground_truth import pinned

FRAME_SHA='5662c02a999eac83e8523f4cd0f483721fd49b2f5cfd19ea8bdf7471af42d61a'
ARCHIVES={'test_01.zip':'1c380c6759ee83dda79b36d2f583a960875afbae934b73866dc74178b64909ef',
          'test_02.zip':'b7bb9eba40dfbeea05bc2e9f867607426fddbb2083440590ff65460fbee2c119',
          'test_03.zip':'214e20700554fba38e50e1ec456be7726835b42c913ac892ec2266d00db9fe30'}


def coverage(frames,source_rows):
    """Check every original pair/ego identity, including frames with no GT."""
    expected={}
    for r in source_rows:
        if r['is_ego'] is True:
            key=(r['sequence_id'],r['frame_key'],r['frame_ordinal'])
            if key in expected:raise ValueError('duplicate original ego row')
            expected[key]=r['cav_id']
    keys=[(r['sequence_id'],r['frame_key'],r['frame_ordinal']) for r in frames]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('original GT frame coverage differs')
    if any(r['ego_cav']!=expected[k] for r,k in zip(frames,keys)):raise ValueError('original ego reference differs')


def part(name,archives,release,release_sha,source_rows,oracle,root):
    root=Path(root);archive=pinned(Path(archives)/name,ARCHIVES[name])
    volume=root/'volumes'/Path(name).stem
    receipt=extract_volume(archive,release,release_sha,volume)
    receipt_sha=sha_file(volume/'receipt.json')
    sequences={x.split('/')[0] for x in (r['path'] for r in receipt['files'])}
    egos={r['sequence_id']:r['cav_id'] for r in source_rows if r['is_ego'] is True and r['sequence_id'] in sequences}
    if set(egos)!=sequences:raise ValueError('original projection lacks volume egos')
    ego=root/'inputs'/(Path(name).stem+'-ego-agents.json');ego.write_bytes(canonical(egos))
    output=root/'parts'/Path(name).stem
    # Use a fresh interpreter so the oracle's private OpenCOOD modules never leak between parts.
    command=[sys.executable,str(ROOT/'tools/event_track_v2x/prepare_v2v4real_ground_truth.py'),
        '--volume',str(volume),'--receipt-sha256',receipt_sha,'--ego-agents',str(ego),
        '--oracle-root',str(oracle),'--evaluation-protocol','v2v4real-official-benchmark-vehicle-v1','--output',str(output)]
    with (root/'logs'/(Path(name).stem+'-GT.log')).open('xb') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    pinned(archive,ARCHIVES[name])
    return dict(archive=name,archive_sha256=ARCHIVES[name],volume_receipt_sha256=receipt_sha,
                manifest_path=str(output/'manifest.json'),manifest_sha256=sha_file(output/'manifest.json'),command=command)


def union(parts,source_rows,output):
    frames=[];manifests=[];audits=[];bindings=[];seen=set()
    for record in parts:
        root=Path(record['manifest_path']).parent
        m,rows=load_vehicle_ground_truth(root,expected_manifest_sha256=record['manifest_sha256'])
        if (m['split']!='official_test' or m['test_payloads_read'] is not True
                or m['reference_commit']!=COMMIT or m['reference_source_sha256']!=SOURCE_HASHES
                or m['oracle_all_ids_and_roi_match'] is not True or m['oracle_compared_frames']!=m['paired_frames']
                or m['oracle_corner_atol_m']!=1e-4 or not 0<=m['oracle_max_abs_corner_error_m']<=1e-4):
            raise ValueError('full native vehicle oracle comparison required')
        groups=set(m['ego_agents'])
        if groups&seen:raise ValueError('duplicate GT volume sequence')
        seen|=groups
        stable=('recipe','class_scope','evaluation_protocol','native_label_source','coordinate_frame','box_representation',
                'gt_range','source_local_roi','final_ego_roi','deduplication','identity_rule','reference_scope')
        if manifests and any(m[k]!=manifests[0][k] for k in stable):raise ValueError('mixed GT definitions')
        audit=[json.loads(x) for x in (root/'audit.jsonl').read_bytes().splitlines()]
        if len(audit)!=len(rows) or any((r['sequence_id'],r['frame_key'])!=(x['sequence_id'],x['frame_key']) for r,x in zip(rows,audit)):
            raise ValueError('GT audit coverage differs')
        frames.extend(rows);audits.extend(audit);manifests.append(m);bindings.append(record)
    coverage(frames,source_rows)
    if len(frames)!=1993 or len(source_rows)!=3986 or len(seen)!=9:raise ValueError('full official-test cohort required')
    frames.sort(key=lambda r:(r['sequence_id'],r['frame_ordinal']))
    audits.sort(key=lambda r:(r['sequence_id'],r['frame_key']))
    raw=b''.join(canonical(r)+b'\n' for r in frames);auditraw=b''.join(canonical(r)+b'\n' for r in audits)
    m=dict(manifests[0])
    for key in ('volume_receipt_sha256','ego_agents_sha256','runtime'):m.pop(key,None)
    counts=Counter()
    for x in manifests:counts.update(x['raw_annotation_counts'])
    m.update(ego_agents={s:v for x in manifests for s,v in x['ego_agents'].items()},paired_frames=len(frames),source_frames=3986,
        sequence_frames=dict(Counter(r['sequence_id'] for r in frames)),gt_objects=sum(len(r['objects']) for r in frames),
        raw_annotation_counts=dict(counts),frames_sha256=hashlib.sha256(raw).hexdigest(),audit_sha256=hashlib.sha256(auditraw).hexdigest(),
        oracle_compared_frames=len(frames),oracle_max_abs_corner_error_m=max(x['oracle_max_abs_corner_error_m'] for x in manifests),
        full_official_split_verified=True,part_bindings=bindings,projection_frames_sha256=FRAME_SHA,
        local_track_count=sum(x['local_track_count'] for x in manifests),
        local_tracks_with_multiple_mapped_ids=sum(x['local_tracks_with_multiple_mapped_ids'] for x in manifests),
        permitted_use='official_test_independent_evaluator_only',inference_input=False,parameter_training_performed=False,
        tracking_evaluation_performed=False,paper_eligible=False)
    m['part_source_sha256']={record['archive']:part['source_sha256'] for record,part in zip(bindings,manifests)}
    m['source_sha256']={'tools/event_track_v2x/prepare_v2v4real_official_test_vehicle_gt.py':sha_file(__file__)}
    for key in ('selected_native_annotation_counts','excluded_native_annotation_counts'):
        count=Counter()
        for x in manifests:count.update(x[key])
        m[key]=dict(count)
    output=Path(output);output.mkdir()
    (output/'frames.jsonl').write_bytes(raw);(output/'audit.jsonl').write_bytes(auditraw);(output/'manifest.json').write_bytes(canonical(m))
    load_vehicle_ground_truth(output,expected_manifest_sha256=sha_file(output/'manifest.json'))
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archives','release','source-frames','oracle','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();pinned(a.source_frames,FRAME_SHA);source_rows=[json.loads(x) for x in a.source_frames.read_bytes().splitlines()]
    verified_sources(a.oracle);release_sha=sha_file(a.release);source_sha=sha_file(__file__)
    a.output.mkdir(parents=True,exist_ok=False)
    for n in ('volumes','parts','inputs','logs'): (a.output/n).mkdir()
    started=time.monotonic();parts=[]
    try:
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(part,n,a.archives,a.release,release_sha,source_rows,a.oracle,a.output) for n in ARCHIVES]
            for job in as_completed(jobs):
                parts.append(job.result());elapsed=time.monotonic()-started
                print(json.dumps(dict(stage='official_test_vehicle_GT_parts',completed=len(parts),total=3,
                    ETA_seconds=elapsed/len(parts)*(3-len(parts)),eta_scope='GT preparation only; tracking ETA unknown')),flush=True)
        m=union(sorted(parts,key=lambda r:r['archive']),source_rows,a.output/'official-test-union')
        pinned(a.source_frames,FRAME_SHA);pinned(a.release,release_sha);pinned(__file__,source_sha)
        receipt=dict(kind='v2v4real_official_test_vehicle_GT_native_oracle_union_v1',GT_offline_only=True,
            official_test_used_for_fit=False,all_1993_frames_native_oracle_compared=True,manifest_sha256=sha_file(a.output/'official-test-union/manifest.json'),
            producer_sha256=source_sha,release_sha256=release_sha,source_frames_sha256=FRAME_SHA,parts=parts,
            max_corner_error_m=m['oracle_max_abs_corner_error_m'],elapsed_seconds=time.monotonic()-started,
            command=sys.argv,tracking_evaluation_complete=False,paper_metric_complete=False)
        path=a.output/'acceptance-receipt.json';path.write_bytes(canonical(receipt))
        sys.path.insert(0,str(ROOT/'tools/event_track_v2x'))
        from rbf_nested_seen_val_v2_common import register
        register(path,receipt['kind']);print(json.dumps(receipt),flush=True)
    except Exception as exc:
        (a.output/'failure.json').write_bytes(canonical(dict(exception_type=type(exc).__name__,parts=parts,
            official_test_used_for_fit=False,paper_metric_complete=False)))
        raise


if __name__=='__main__':main()
