#!/usr/bin/env python3
"""Merge complete, byte-bound official-test frame-pair shards; no metric claims."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as export
# Standalone execution schema; a parity test binds this to the cache consumer.
# Do not import the 9D-cache builder (and its training dependencies) on GPU workers.
RAW_MANIFEST_FIELDS={'anchor_array_sha256','anchor_file_sha256','anchor_generation','appearance_valid_candidates',
    'candidate_protocol','checkpoint_seed','checkpoint_sha256','cohort','covariance_exported',
    'downstream_covariance_velocity_and_identity_admission_pending','export_groups','feature_method','gt_read',
    'kind','matching_candidate_rule','official_test_read','paper_metric','partition_sha256',
    'prepared_projection_sha256','protocol_id','rows','rows_sha256','sequence_ids','split',
    'split_admission_sha256','state_layout','total_candidates','velocity_exported'}


def identity(row):
    return row['sequence_id'],row['frame_ordinal'],row['side']


def validate_payload(raw,row):
    if hashlib.sha256(raw).hexdigest()!=row['payload_sha256']:
        raise ValueError('payload bytes differ')
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        n=row['candidates']
        if type(n) is not int or not 0<=n<=64:raise ValueError('candidate count differs')
        shapes={'states_source_legacy7':(n,7),'raw_scores':(n,), 'class_indices':(n,),
                'appearance':(n,128),'appearance_valid':(n,)}
        if set(a.files)!=set(shapes) or any(a[k].shape!=v for k,v in shapes.items()):
            raise ValueError('raw payload schema differs')
        if (not all(np.isfinite(a[k]).all() for k in shapes) or a['appearance_valid'].dtype!=np.bool_
                or not np.issubdtype(a['class_indices'].dtype,np.integer)
                or np.any(a['class_indices']!=0) or np.any(a['states_source_legacy7'][:,3:6]<=0)
                or np.any(a['raw_scores']<.05) or np.any(a['raw_scores']>1)
                or np.any(np.diff(a['raw_scores'])>0)
                or np.any(a['appearance'][~a['appearance_valid']]!=0)
                or not np.allclose(np.linalg.norm(a['appearance'][a['appearance_valid']],axis=1),1,atol=1e-6,rtol=0)):
            raise ValueError('raw candidate contract differs')
        return int(a['appearance_valid'].sum())


def merge(shards,frames,output):
    """All original rows must occur exactly once before a complete envelope exists."""
    frames=export.asset(frames,export.FRAME_SHA)
    records=[json.loads(line) for line in frames.read_bytes().splitlines()]
    if len(records)!=3986:raise ValueError('complete original schedule required')
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh output required')
    if len(shards)<2:raise ValueError('multiple shards required')
    expected={}
    for r in export.shard_records(records):
        key=(r['sequence_id'],r['frame_ordinal'],'vehicle-side' if r['is_ego'] else 'infrastructure-side')
        if key in expected:raise ValueError('duplicate original source identity')
        expected[key]=r
    headers=[];seen_ranks=set();common=None
    variable={'kind','shard','rows','rows_sha256','total_candidates','appearance_valid_candidates'}
    for directory in map(Path,shards):
        mraw=(directory/'native-feature-manifest').read_bytes();m=json.loads(mraw)
        receipt=json.loads((directory/'raw-export-receipt.json').read_bytes())
        if m.get('kind')!=export.SHARD_KIND:raise ValueError('partial shard kind required')
        if set(m)!=RAW_MANIFEST_FIELDS|{'projection_manifest','split_admission','shard'}:
            raise ValueError('shard manifest schema differs')
        required={'protocol_id':export.PROTOCOL_ID,'split':'official_test','gt_read':False,
                  'paper_metric':False,'official_test_read':True,'velocity_exported':False,
                  'covariance_exported':False,'prepared_projection_sha256':export.PROJECTION_SHA,
                  'split_admission_sha256':export.ADMISSION_SHA,'partition_sha256':export.PARTITION_SHA,
                  'matching_candidate_rule':'raw_score_ge_0.05_stable_top64_before_nms'}
        if any(m.get(k)!=v for k,v in required.items()):raise ValueError('raw export contract differs')
        s=m['shard'];rank=s['rank'];world=s['world_size']
        if (world!=len(shards) or type(rank) is not int or not 0<=rank<world or rank in seen_ranks
                or s!={'rank':rank,'world_size':world,'rule':'sorted_original_frame_pair_ordinal_mod_world_size',
                      'full_source_rows':3986,'projection_frames_sha256':export.FRAME_SHA}):
            raise ValueError('missing, duplicate or incompatible shard rank')
        seen_ranks.add(rank)
        export.asset(directory/'projection-manifest',export.PROJECTION_SHA)
        export.asset(directory/'split-admission',export.ADMISSION_SHA)
        export.validate_scope(json.loads((directory/'projection-manifest').read_bytes()),
                              json.loads((directory/'split-admission').read_bytes()),records)
        seed=m['checkpoint_seed']
        if (seed not in export.CHECKPOINTS or m['checkpoint_sha256']!=export.CHECKPOINTS[seed]
                or receipt['raw_manifest_sha256']!=hashlib.sha256(mraw).hexdigest()
                or receipt['raw_archive_sha256']!=export.sha(directory/'native-features')
                or receipt['producer_sha256']!=export.sha(export.__file__)
                or receipt['source_archive_sha256']!=export.SOURCE_ARCHIVE_SHA
                or receipt['inference_core_sha256']!=export.CORE_SHA
                or receipt['shard']!={'rank':rank,'world_size':world}
                or receipt['TF32_enabled'] is not False):
            raise ValueError('source, checkpoint or shard receipt differs')
        fixed={k:v for k,v in m.items() if k not in variable}
        if common is None:common=fixed
        if fixed!=common:raise ValueError('mixed shard runtime contract')
        rowsraw=(directory/'native-feature-index').read_bytes()
        rows=[json.loads(line) for line in rowsraw.splitlines()]
        desired={identity(dict(sequence_id=r['sequence_id'],frame_ordinal=r['frame_ordinal'],
                               side='vehicle-side' if r['is_ego'] else 'infrastructure-side'))
                 for r in export.shard_records(records,rank,world)}
        if (hashlib.sha256(rowsraw).hexdigest()!=m['rows_sha256'] or len(rows)!=m['rows']
                or len({identity(r) for r in rows})!=len(rows) or {identity(r) for r in rows}!=desired):
            raise ValueError('shard schedule coverage differs')
        for row in rows:
            original=expected[identity(row)]
            if (row['frame_id']!=original['frame_key'] or row['pcd_sha256']!=original['pcd_sha256']
                    or row['source_to_world']!=original['source_to_world']):
                raise ValueError('source frame binding differs')
        inventory={n:export.sha(directory/n) for n in ('native-features','native-feature-manifest',
            'native-feature-index','projection-manifest','split-admission','raw-export-receipt.json')}
        headers.append((directory,mraw,m,receipt,rowsraw,rows,inventory))
    output.mkdir(parents=True);work=output/'raw';(work/'payloads').mkdir(parents=True)
    merged=[];valid=0;bindings=[]
    for directory,mraw,m,receipt,rowsraw,rows,inventory in headers:
        archive=directory/'native-features';before=export.sha(archive)
        names=['anchors.npy','manifest.json','predictions.jsonl',*(r['payload'] for r in rows)]
        if len(set(names))!=len(names) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('unsafe shard archive member')
        with tarfile.open(archive) as stream:
            members=stream.getmembers()
            if [p.name for p in members]!=names or any(not p.isfile() for p in members):
                raise ValueError('shard archive inventory differs')
            if stream.extractfile('manifest.json').read()!=mraw or stream.extractfile('predictions.jsonl').read()!=rowsraw:
                raise ValueError('archive envelope differs')
            anchors=stream.extractfile('anchors.npy').read()
            if hashlib.sha256(anchors).hexdigest()!=m['anchor_file_sha256']:raise ValueError('anchors differ')
            array=np.load(io.BytesIO(anchors),allow_pickle=False)
            if array.ndim!=4 or array.shape[-1]!=7 or not np.isfinite(array).all():raise ValueError('invalid anchors')
            anchor_path=work/'anchors.npy'
            if anchor_path.exists() and anchor_path.read_bytes()!=anchors:raise ValueError('mixed anchor arrays')
            if not anchor_path.exists():anchor_path.write_bytes(anchors)
            shard_valid=0
            for row in rows:
                raw=stream.extractfile(row['payload']).read();shard_valid+=validate_payload(raw,row)
                name=f'payloads/frame-{len(merged):06d}.npz';(work/name).write_bytes(raw)
                merged.append(dict(row,payload=name))
            if shard_valid!=m['appearance_valid_candidates'] or sum(r['candidates'] for r in rows)!=m['total_candidates']:
                raise ValueError('shard candidate totals differ')
            valid+=shard_valid
        if export.sha(archive)!=before:raise ValueError('shard changed during merge')
        bindings.append(dict(rank=m['shard']['rank'],archive_sha256=before,
                             manifest_sha256=hashlib.sha256(mraw).hexdigest(),actual_device=receipt['actual_device'],
                             input_sha256=inventory))
    merged.sort(key=identity)
    if len(merged)!=len(expected) or {identity(r) for r in merged}!=set(expected):raise ValueError('complete cohort coverage differs')
    manifest=dict(common,kind=export.KIND,rows=len(merged),
                  rows_sha256=hashlib.sha256(b''.join(export.canonical(r)+b'\n' for r in merged)).hexdigest(),
                  total_candidates=sum(r['candidates'] for r in merged),appearance_valid_candidates=valid)
    archive_sha=export.pack(work,output,manifest,merged,headers[0][0]/'projection-manifest',headers[0][0]/'split-admission')
    for directory,_,_,_,_,_,inventory in headers:
        for name,digest in inventory.items():export.asset(directory/name,digest)
    export.asset(frames,export.FRAME_SHA)
    receipt=dict(kind='v2v4real_official_test_raw_shard_merge_v1',shards=bindings,rows=len(merged),
                 raw_archive_sha256=archive_sha,raw_manifest_sha256=export.sha(output/'native-feature-manifest'),
                 producer_sha256=export.sha(export.__file__),merger_sha256=export.sha(__file__),
                 source_frames_including_empty_retained=True,independent_raw_acceptance=False,paper_metric=False)
    (output/'raw-merge-receipt.json').write_bytes(export.canonical(receipt))
    complete=dict(kind='v2v4real_official_test_raw_export_v1',split='official_test',gt_read=False,
        official_test_read=True,paper_metric=False,raw_manifest_sha256=receipt['raw_manifest_sha256'],
        raw_archive_sha256=archive_sha,rows_sha256=manifest['rows_sha256'],rows=len(merged),
        checkpoint_seed=manifest['checkpoint_seed'],checkpoint_sha256=manifest['checkpoint_sha256'],
        projection_manifest_sha256=export.PROJECTION_SHA,projection_frames_sha256=export.FRAME_SHA,
        split_admission_sha256=export.ADMISSION_SHA,source_archive_sha256=export.SOURCE_ARCHIVE_SHA,
        inference_core_sha256=export.CORE_SHA,producer_sha256=export.sha(export.__file__),
        merge_receipt_sha256=export.sha(output/'raw-merge-receipt.json'),execution_method='complete_frame_pair_shard_merge',
        source_frames_including_empty_retained=True,independent_raw_acceptance=False,formal_metrics_complete=False,
        statistical_independence_claimed=False,original_checkpoint_known_overlap_disclosed=True)
    (output/'raw-export-receipt.json').write_bytes(export.canonical(complete))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',action='append',required=True);p.add_argument('--frames',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(merge(a.shard,a.frames,a.output)))


if __name__=='__main__':main()
