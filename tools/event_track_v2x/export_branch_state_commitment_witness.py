"""Export hash-bound branch projections for portable independent readback.

Reproducing a floating-point byte commitment requires the producer's NumPy/BLAS
runtime. These witnesses retain every projected value, so a different host can
verify the commitment exactly and independently check the numerical values.
No production tracker is imported, no states are recomputed from other branches,
and these witnesses alone grant no numerical or experimental acceptance.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from accept_optimized_branch_state_sequence import StoredProjection, bind_branch_hashes, bind_request, digest
from rbf_nested_seen_val_v2_common import new, sha


def export(database,expected_sha,output):
    assert sha(database)==expected_sha
    assert not output.exists() and not output.with_suffix('.receipt.json').exists()
    db=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    started=time.monotonic();count=branches=predictions=previous_n=0
    previous=['0'*64,'0'*64];cache=OrderedDict()
    try:
        meta={k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')}
        assert meta['schema']=='experimental_exclusive_batched_state_v1'
        def kernel(component):
            if component not in cache:
                cache[component]=StoredProjection(db,component)
                if len(cache)>32:cache.popitem(last=False)
            cache.move_to_end(component);return cache[component]
        with output.open('x') as stream:
            for ordinal,event,request,pb,ab in db.execute('SELECT * FROM events ORDER BY ordinal'):
                assert ordinal==count
                p,a=json.loads(pb),json.loads(ab)
                assert a['event_id']==event and p['sequence_id']==a['sequence_id']==meta['sequence_id']
                previous_n=bind_request(db,request,p,a,meta['config'],previous_n,explicit_scope=True)
                assert p['previous_commit_sha256']==previous[0] and a['previous_audit_sha256']==previous[1]
                assert digest({k:v for k,v in p.items() if k!='commit_sha256'})==p['commit_sha256']==a['prediction_sha256']
                previous=[p['commit_sha256'],digest(a)]
                _,values,_=bind_branch_hashes(a,p,kernel,meta['config']['state'])
                record=dict(ordinal=ordinal,event_id=event,reference_us=p['box_reference_timestamp_us'],
                    decision_us=p['decision_timestamp_us'],branches=[dict(component=key[0],handle=key[1],predictions=value) for key,value in values])
                stream.write(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                count+=1;branches+=len(values);predictions+=sum(len(v) for _,v in values)
                if count%10==0:
                    print(json.dumps(dict(stage='independent_branch_commitment_witness',completed_events=count,
                        total_events=meta['state']['events'],branches=branches,ETA='unknown; heterogeneous branch sizes')),flush=True)
        assert count==meta['state']['events'] and previous_n==meta['state']['n']
        assert previous==[meta['state']['prediction_sha256'],meta['state']['audit_sha256']]
        assert sha(database)==expected_sha
        receipt=output.with_suffix('.receipt.json')
        new(receipt,dict(kind='rbf_complete_branch_projection_commitment_witness_v1',
            database_sha256=expected_sha,witness_sha256=sha(output),witness_bytes=output.stat().st_size,
            source_sha256=sha(__file__),projection_checker_sha256=sha(Path(__file__).with_name('accept_optimized_branch_state_sequence.py')),
            events=count,branches=branches,branch_predictions=predictions,request_commitments=count,
            all_branch_commitments_reproduced=True,numpy=np.__version__,elapsed_seconds=time.monotonic()-started,
            fresh_history_numeric_acceptance=False,independent_cloud_readback=False,paper_performance_complete=False))
        return receipt
    finally:db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--database-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(dict(receipt=str(export(args.database,args.database_sha256,args.output)))),flush=True)


if __name__=='__main__':main()
