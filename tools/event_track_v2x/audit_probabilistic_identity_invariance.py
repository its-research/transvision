#!/usr/bin/env python3
"""Check unchanged hard identity decisions separately from geometric metrics.

Reconstruct every historical anchor digest from the completed database prefix.
Hash conditional scans and non-state output records; never read GT payloads or
equate metric changes with changed identity decisions. No inference is rerun.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.compare_train_risk_controls import read
from tools.event_track_v2x.run_train_probabilistic_diagnostic import BACKENDS
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json


def fingerprint(anchors, audits, predictions):
    for i, (index, root) in enumerate(anchors):
        if (type(index) is not int or type(root) is not int or index!=i or not 0<=root<=i
                or anchors[root][1]!=root):
            raise ValueError('contiguous immutable root map required')
    scans=hashlib.sha256();history=hashlib.sha256();membership=hashlib.sha256()
    prefix=hashlib.sha256();cursor=0;events=[]
    for audit,prediction in itertools.zip_longest(audits,predictions):
        if audit is None or prediction is None:
            raise ValueError('complete paired audit and prediction streams required')
        key=(audit['sequence_id'],audit['event_id'])
        if key!=(prediction['sequence_id'],prediction['frame_id']):
            raise ValueError('audit and prediction event differ')
        n=audit['observation_count']
        if type(n) is not int or not cursor<=n<=len(anchors):
            raise ValueError('nondecreasing historical observation prefix required')
        while cursor<n:
            prefix.update(canonical(anchors[cursor])+b'\n');cursor+=1
        if prefix.hexdigest()!=audit['identity_anchor_sha256']:
            raise ValueError('historical anchor hash differs from its actual database prefix')
        scans.update(canonical([*key,audit['conditional_scans']])+b'\n')
        history.update(canonical([*key,audit['identity_anchor_sha256']])+b'\n')
        boxes=prediction['predictions']
        membership.update(canonical([*key,[{k:v for k,v in b.items() if k not in ('mean','covariance')}
                                          for b in boxes]])+b'\n')
        events.append(dict(sequence_id=key[0],frame_id=key[1],
            payload_sha256=hashlib.sha256(canonical(boxes)).hexdigest()))
    if not events or cursor!=len(anchors) or len({(r['sequence_id'],r['frame_id']) for r in events})!=len(events):
        raise ValueError('complete unique event and anchor coverage required')
    return dict(frames=len(events),observations=len(anchors),events=events,
        conditional_scan_stream_sha256=scans.hexdigest(),identity_anchor_stream_sha256=history.hexdigest(),
        non_state_prediction_stream_sha256=membership.hexdigest(),final_anchor_map_sha256=prefix.hexdigest(),
        every_historical_anchor_hash_reconstructed=True)


def audit(comparison,comparison_sha256,output):
    value=read(comparison,comparison_sha256)
    if (value.get('kind')!='train_probabilistic_controls_v1' or value.get('status')!='complete'
            or set(value['cells'])!=set(BACKENDS) or value.get('actual_complete_factor_stream_identical') is not True):
        raise ValueError('complete bound three-updater comparison required')
    evidence={Path(comparison).absolute():comparison_sha256,Path(__file__):sha_file(__file__)}
    results={}
    for backend in BACKENDS:
        cell=value['cells'][backend];root=Path(cell['source_directory'])
        report=inspect(root,cell['inference_receipt_sha256'])
        if report['backend']!=backend or len(report['events'])!=value['frames']:
            raise ValueError('comparison cell or complete replay coverage differs')
        outer=read(root/'development-inference-receipt.json',cell['inference_receipt_sha256'])
        receipt=read(root/'receipt.json',outer['replay_receipt_sha256'])
        if len(receipt['sequence_heads'])!=1:
            raise ValueError('one complete sequence required')
        head=receipt['sequence_heads'][report['plan']['selected_sequence']]
        paths={root/'development-inference-receipt.json':cell['inference_receipt_sha256'],
            root/'receipt.json':outer['replay_receipt_sha256'],root/'plan.json':receipt['plan_sha256'],
            root/'tracking.jsonl':receipt['tracking_sha256'],root/'predictions.jsonl':receipt['predictions_sha256'],
            root/'frame-timings.jsonl':receipt['frame_timings_sha256'],
            contained_file(root,head['database']):head['database_sha256']}
        if any(value['input_sha256'].get(str(p))!=h or sha_file(p)!=h for p,h in paths.items()):
            raise ValueError('inference evidence is not bound by the completed comparison')
        evidence.update(paths)
        with closing(sqlite3.connect(contained_file(root,head['database']).as_uri()+'?mode=ro',uri=True)) as db:
            db.execute('PRAGMA query_only=ON')
            anchors=list(db.execute('SELECT i,root FROM identity_anchors ORDER BY i'))
            if len(anchors)!=db.execute('SELECT count(*) FROM observations').fetchone()[0]:
                raise ValueError('raw observation and hard anchor coverage differs')
        with (root/'tracking.jsonl').open('rb') as rows,(root/'predictions.jsonl').open('rb') as predictions:
            results[backend]=fingerprint(anchors,(json.loads(r)['tracking'] for r in rows),map(json.loads,predictions))
    first=results[BACKENDS[0]]
    keys=lambda r:[(e['sequence_id'],e['frame_id']) for e in r['events']]
    if any(keys(r)!=keys(first) for r in results.values()):
        raise ValueError('updater event schedules differ')
    result=dict(kind='probabilistic_hard_identity_invariance_audit_v1',status='complete',frames=first['frames'],
        runs=results,all_conditional_scans_identical=len({r['conditional_scan_stream_sha256'] for r in results.values()})==1,
        all_historical_identity_maps_identical=len({r['identity_anchor_stream_sha256'] for r in results.values()})==1,
        output_ids_membership_scores_and_lifecycle_identical=len({r['non_state_prediction_stream_sha256'] for r in results.values()})==1,
        changed_state_payload_frames_relative_to_jpda_ci={b:sum(a['payload_sha256']!=c['payload_sha256']
            for a,c in zip(first['events'],r['events'])) for b,r in results.items()},
        database_prefix_identity_hashes_independently_reconstructed=True,
        GT_payloads_read=False,metrics_recomputed=False,inference_rerun=False,
        metric_identity_is_not_hard_anchor_identity=True,validation=False,paper_eligible=False)
    if any(sha_file(p)!=h for p,h in evidence.items()):
        raise ValueError('identity audit evidence changed')
    result['input_sha256']={str(p):h for p,h in evidence.items()}
    output=_directory(output);_new_json(output/'identity-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','input_sha256')},sort_keys=True))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--comparison-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.comparison,args.comparison_sha256,args.output)
