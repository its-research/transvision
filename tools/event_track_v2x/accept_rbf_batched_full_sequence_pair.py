"""Independent per-database arithmetic and output parity for one real sequence.

This cannot grant full-cohort, GPU-memory, same-resource or paper acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from rbf_nested_seen_val_v2_common import R, new, register, sha


def module(name, path, expected, dependencies=None):
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    dependencies = dependencies or {}
    previous = {key: sys.modules.get(key) for key in dependencies}
    try:
        sys.modules.update(dependencies)
        spec.loader.exec_module(result)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    pair=args.pair.resolve();output=args.output.resolve()
    assert pair.is_relative_to(R/'artifacts') and output.is_relative_to(R/'artifacts')
    completion=json.loads((pair/'process-completion.json').read_bytes())
    assert completion['exit_codes']=={'serial':0,'batched':0}
    launch=json.loads((pair/'launch.json').read_bytes());assert sha(pair/'input-binding.json')==launch['binding_sha256']
    binding=json.loads((pair/'input-binding.json').read_bytes())
    assert binding['seed']==2027 and binding['sequence_id']=='0000' and binding['expected_events']==195
    freeze=R/'source-freezes/rbf-batched-full-real-sequence-CPU-pair-v1-20261004/source-freeze.json'
    assert sha(freeze)=='260b6c0396965bc8274af97ffcf1e7dc94f82044242ededc4cc8d52c9b1da08d'
    assert json.loads(freeze.read_bytes())['input_binding_sha256']==sha(pair/'input-binding.json')
    schedule=Path(binding['events_path']);assert sha(schedule)==binding['events_sha256']
    events=[e for e in json.loads(schedule.read_bytes())['events'] if e['sequence_id']=='0000'];assert len(events)==195
    structure_dir=R/'source-freezes/rbf-independent-exclusive-forest-structure-recovery-v1-20261002'
    gate=R/'artifacts/rbf-independent-exclusive-forest-oracle-software-gate-v1-20261002/software-gate-v2.json'
    structure_hashes=json.loads(gate.read_bytes())['sources']
    for name, checksum in structure_hashes.items():assert sha(structure_dir/name)==checksum
    structure=module('batch_pair_structure',structure_dir/'oracle.py',structure_hashes['oracle.py'])
    causal=module('batch_pair_causal',structure_dir/'causal.py',structure_hashes['causal.py'],{'oracle':structure})
    fresh=module('batch_pair_fresh',R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py',
        '46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58')
    action=module('batch_pair_action',R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004/decoder_capacity.py',
        '2f05714726a48280f64bdb559a330e1fb3582cf33d58d51ffc7a7a2e82c3d6cc')
    references={}
    refroot=R/'artifacts/rbf-final-refit-all-row-independent-numeric-v1-20261004/seed2027'
    refproof=json.loads((refroot/'independent-byte-coverage-receipt.json').read_bytes())
    index_path=R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'
    assert sha(index_path)=='93e7e3849dfdce2814c9a22510575dade16efd7a2fd5c6569f5074f0e3550816'
    index=json.loads(index_path.read_bytes());entry=next(v for v in index['seeds'] if v['seed']==2027)
    assert index['atol']==index['rtol']==1e-4 and index['tolerance_changed'] is False
    assert entry['numeric_failures']==0 and sha(entry['numeric_completion'])==entry['numeric_completion_sha256']
    assert sha(refroot/'independent-byte-coverage-receipt.json')==entry['prediction_byte_proof_sha256']
    assert sha(pair/'checkpoint/weights.pt')==refproof['weights_sha256']
    assert sha(pair/'checkpoint/checkpoint.json')==binding['checkpoint_sha256']
    for key, record in refproof['artifacts'].items():
        if not key.startswith('predictions-rank'):continue
        path=refroot/(key+'.jsonl');assert sha(path)==record['sha256']
        for line in path.open():
            row=json.loads(line)
            if row['sequence_id']=='0000':
                assert row['row'] not in references;references[row['row']]=row
    assert references
    output.mkdir(exist_ok=False)
    reports={};predictions={};raw_observation_hashes={}
    try:
        for role in ('serial','batched'):
            replay=pair/(role+'-replay');receipt=json.loads((replay/'receipt.json').read_bytes())
            assert receipt['completed_events']==195 and receipt['completed_sequences']==['0000']
            for name, checksum in receipt['files'].items():assert sha(replay/name)==checksum
            plan=json.loads((replay/'plan.json').read_bytes())
            assert plan['configuration']==binding['configuration'] and plan['model_binding']['checkpoint_sha256']==binding['checkpoint_sha256']
            for name, checksum in binding['source_hashes'][role].items():assert sha(pair/(role+'-source')/name)==checksum
            dbinfo=receipt['databases']['0000'];dbpath=replay/dbinfo['path'];assert dbpath.parent==replay and sha(dbpath)==dbinfo['sha256']
            db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)
            try:
                assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                assert [r[0] for r in db.execute('SELECT event_id FROM events ORDER BY ordinal')]==[e['event_id'] for e in events]
                assert db.execute('SELECT count(*) FROM observations').fetchone()[0]==len(references)
                rawhash=hashlib.sha256();maximum=0.
                for i,node,raw in db.execute('SELECT i,node_id,raw FROM observations ORDER BY i'):
                    ref=references[i];assert node==ref['node_id'];rawhash.update(raw)
                    factors=db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
                    assert [p for p,w in factors]==[-1]+ref['context_indices'][:-1]
                    values=np.asarray(ref['logits'],dtype=np.float64);m=values.max();expected=values-(m+np.log(np.exp(values-m).sum()))
                    got=np.asarray([w for p,w in factors]);assert np.isfinite(got).all() and np.allclose(got,expected,atol=1e-4,rtol=1e-4)
                    maximum=max(maximum,float(np.max(np.abs(got-expected))))
                raw_observation_hashes[role]=rawhash.hexdigest()
            finally:db.close()
            def progress(value):
                print(json.dumps(dict(role=role,oracle_progress=value,whole_acceptance_ETA='unknown')),flush=True)
            result=dict(NN_max_absolute_error=maximum,structure=structure.audit_database(dbpath,dbinfo['sha256'],progress),
                causal=causal.verify_database(dbpath,dbinfo['sha256']),fresh_state=fresh.verify_database(dbpath,dbinfo['sha256'],progress),
                action=action.verify_database(dbpath,dbinfo['sha256'],progress))
            assert all(result[k]['events']==195 for k in ('structure','causal','fresh_state','action'))
            new(output/(role+'-independent.json'),result);reports[role]=result
            predictions[role]=[json.loads(line) for line in (replay/'predictions.jsonl').open()]
        assert raw_observation_hashes['serial']==raw_observation_hashes['batched']
        parity=[]
        for i,(left,right) in enumerate(zip(predictions['serial'],predictions['batched'],strict=True)):
            identity=('sequence_id','frame_id','box_reference_timestamp_us','decision_timestamp_us','coordinate_frame','state_layout')
            assert all(left[k]==right[k] for k in identity)
            a=left['predictions'];b=right['predictions']
            same=[(p['track_id'],p['class_label']) for p in a]==[(p['track_id'],p['class_label']) for p in b]
            numeric=same and all(np.allclose(p[k],q[k],atol=1e-8,rtol=1e-8) for p,q in zip(a,b) for k in ('mean','covariance','score'))
            parity.append(dict(event_ordinal=i+1,discrete_outputs_identical=same,continuous_outputs_match=numeric))
        assert len(parity)==195
        new(output/'output-comparison.json',parity)
        final=output/'independent-sequence-receipt.json'
        new(final,dict(kind='rbf_serial_batched_single_complete_sequence_independent_scope_v1',seed=2027,sequence='0000',events=195,
            reports={role:sha(output/(role+'-independent.json')) for role in reports},raw_observation_hashes=raw_observation_hashes,
            output_comparison_sha256=sha(output/'output-comparison.json'),all_discrete_outputs_identical=all(x['discrete_outputs_identical'] for x in parity),
            all_continuous_outputs_match=all(x['continuous_outputs_match'] for x in parity),NN_atol_rtol=1e-4,independent_arithmetic_atol_rtol=1e-8,
            full46_sequence_or_three_seed_or_GPU_acceptance=False,production_promotion_allowed=False,GPU_memory_target_admitted=False))
        register(final,'rbf-batched-single-complete-sequence-independent-scope');print(json.dumps(dict(receipt=str(final))),flush=True)
    except BaseException as error:
        failure=output/'failure.json';new(failure,dict(type=type(error).__name__,message=str(error),completed_roles=list(reports),accepted=False))
        register(failure,'rbf-batched-single-sequence-independent-failure');raise


if __name__=='__main__':main()
