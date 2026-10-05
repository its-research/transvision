"""Independent candidate/reference correspondence and fresh numerical audit.

No production tracker or candidate execution code is imported. Exact tables
and complete semantic event audits transfer the already admitted reference's
structural scope; the original frozen numerical oracle checks the new states.
This is a single sequence gate, never full-cohort or paper acceptance.
"""
import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from rbf_nested_seen_val_v2_common import R, new, register, sha

FRESH = R/'source-freezes/rbf-independent-fresh-branch-state-v3-normalized-admission-20261001/oracle.py'
FRESH_SHA = '46d5f6b474202961077e43f9c572f6ecfb0a98e992e9f4b19fcd37cccfce1c58'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind_request(db, stored, prediction, audit, config, previous_n, *, explicit_scope):
    """Reproduce input commitments, including the original default scope.

    The recorded serial call passed None; the candidate passes the very same
    resulting decision indices explicitly. Prove that default expansion before
    comparing semantic input identity, rather than dropping request hashes.
    """
    n = audit['observation_count']
    assert n-previous_n == audit['new_observations']
    raw = [json.loads(v) for v, in db.execute('SELECT raw FROM observations WHERE i>=? AND i<? ORDER BY i', (previous_n,n))]
    assert len(raw) == n-previous_n
    indices = audit['decision_indices']
    if not explicit_scope:
        cutoff = max(0, prediction['box_reference_timestamp_us']-config['state']['window_us'])
        default = [i for i, in db.execute('SELECT i FROM observations WHERE i<? AND state_us>=? ORDER BY i', (n,cutoff))]
        assert indices == default, 'default and recorded decision scope differ'
        indices = None
    arguments = [prediction['frame_id'], prediction['box_reference_timestamp_us'], prediction['decision_timestamp_us'],
        raw, audit['appended_rows'], audit['rescored_rows'], indices, audit['cache_ingestion'], audit['scorer_binding']]
    assert digest(arguments) == stored, 'event request commitment differs'
    return n


def semantic_audit(value, execution, caps):
    value = dict(value)
    assert value.pop('state_execution') == execution
    assert value.pop('execution_extra_cache_caps') == caps
    assert value.pop('state_execution_independently_accepted') is False
    value.pop('prediction_sha256')
    value.pop('previous_audit_sha256')
    return value


def compare_tables(left, right):
    definitions = [dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")) for db in (left, right)]
    assert definitions[0] == definitions[1], 'table schema changed'
    assert list(left.execute("SELECT name,sql FROM sqlite_master WHERE type='view' ORDER BY name")) == list(
        right.execute("SELECT name,sql FROM sqlite_master WHERE type='view' ORDER BY name")), 'view schema changed'
    counts = {}
    for name in sorted(definitions[0]):
        assert re.fullmatch(r'[a-z][a-z0-9_]*', name)
        if name in ('meta', 'events', 'states', 'component_summaries') or re.fullmatch(r'pc[0-9]+_states', name):
            continue
        columns = left.execute('PRAGMA table_info('+name+')').fetchall()
        primary = [i+1 for i, col in sorted(enumerate(columns), key=lambda v:v[1][5]) if col[5]]
        assert primary, 'explicit row identity required'
        query = 'SELECT * FROM '+name+' ORDER BY '+','.join(map(str, primary))
        count = 0
        for a, b in zip(left.execute(query), right.execute(query), strict=True):
            assert a == b, 'structural/raw table differs: '+name
            count += 1
        counts[name] = count
    return counts


class StoredProjection:
    """Bind reported branch hashes to persisted states without tracker imports.

    This is deliberately not the numerical oracle: it verifies the serialized
    output commitment. The separate frozen fresh-history oracle checks states
    from observations using independent information-equation solves.
    """
    def __init__(self, db, component):
        assert type(component) is int and component >= 0
        self.db, self.prefix = db, f'pc{component}_'
        self.rows = {i: json.loads(raw) for i, raw in db.execute(f'SELECT i,raw FROM {self.prefix}observations')}
        self.prefixes = {r[0]: r for r in db.execute(
            f'SELECT h,parent,depth,choice,root,previous_root,sha FROM {self.prefix}prefixes')}

    def outputs(self, handle, reference, config, sequence, decision, nodes):
        path, seen = [], set()
        while handle:
            assert handle not in seen, 'prefix cycle'
            seen.add(handle)
            record = self.prefixes[handle]
            path.append(record)
            handle = record[1]
        assert len(path) == nodes, 'branch depth differs from event component'
        groups, labels, latest = {}, [], {}
        for h, parent, depth, choice, root, previous, prefix_sha in reversed(path):
            i = len(labels)
            assert depth == i+1 and -1 <= choice < i
            assert root == (i if choice == -1 else labels[choice]) and previous == latest.get(root)
            row = self.rows[i]
            assert row['node']['information_us'] <= row['node']['arrival_us'] <= decision
            labels.append(root); latest[root] = h
            groups.setdefault(root, []).append(row)
        result = []
        for root, rows in sorted(groups.items()):
            payload = self.db.execute(f'SELECT payload FROM {self.prefix}states WHERE h=?', (latest[root],)).fetchone()
            assert payload is not None, 'branch state missing'
            state = json.loads(payload[0])
            score = max(r['score'] * config['survival_per_second'] ** (max(0, reference-r['state_us'])/1e6) for r in rows)
            if state['max_score'] < config['birth_score'] or score < config['prune_score'] or max(0, reference-state['last_us']) > config['max_age_us']:
                continue
            dt = (reference-state['last_us'])/1e6
            transition = np.eye(9); transition[0, 7] = transition[1, 8] = dt
            mean = transition @ np.asarray(state['mean'], dtype=np.float64)
            covariance = transition @ np.asarray(state['covariance'], dtype=np.float64) @ transition.T + np.eye(9)*config['process_noise']*abs(dt)
            assert np.isfinite(mean).all() and np.isfinite(covariance).all()
            origin = self.rows[root]
            result.append(dict(track_id=sequence+':'+digest(['forest-birth', origin['node']['node_id']])[:24],
                class_label=('car', 'bicycle', 'pedestrian')[int(np.argmax(origin['features'][138:141]))],
                mean=mean.tolist(), covariance=covariance.tolist(), score=score))
        return result


def compare_predictions(first, second):
    maximum, count = 0., 0
    for first_state, second_state in zip(first, second, strict=True):
        x, y = dict(first_state), dict(second_state)
        for key in ('mean', 'covariance'):
            aa, bb = np.asarray(x.pop(key)), np.asarray(y.pop(key))
            assert aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
            assert np.allclose(aa, bb, atol=1e-8, rtol=1e-8), 'branch/output state numerical mismatch'
            maximum = max(maximum, float(np.max(np.abs(aa-bb))))
        assert x == y, 'output identity/score/provenance changed'
        count += 1
    return count, maximum


def bind_branch_hashes(audit, prediction, kernel, config):
    """Remove only independently reproduced float-dependent commitments."""
    originals = audit['components']
    audit = dict(audit)
    audit['components'] = []
    branch_outputs, chosen_outputs = [], []
    reference, decision = prediction['box_reference_timestamp_us'], prediction['decision_timestamp_us']
    for original in originals:
        component = dict(original); k = kernel(component['component'])
        last_time = max((k.rows[i]['state_us'] for i in range(component['nodes'])), default=None)
        expired = last_time is not None and reference-last_time > config['max_age_us']
        assert expired == component['expired_output_only'], 'component expiry differs'
        component['branches'] = []
        handles = [b['handle'] for b in original['branches']]
        assert handles == sorted(set([b['handle'] for b in original['active']] + [original['output_handle']]))
        for original_branch in original['branches']:
            branch = dict(original_branch); handle = branch['handle']
            assert k.prefixes[handle][2] == component['nodes']
            assert branch['sha256'] == k.prefixes[handle][6], 'branch prefix commitment differs'
            values = [] if expired else k.outputs(handle, reference, config, prediction['sequence_id'], decision, component['nodes'])
            assert branch.pop('state_sha256') == digest(values), 'branch state commitment differs'
            branch_outputs.append(((component['component'], handle), values))
            component['branches'].append(branch)
            if handle == component['output_handle']:
                chosen_outputs.extend(values)
        audit['components'].append(component)
    assert len({p['track_id'] for p in chosen_outputs}) == len(chosen_outputs)
    assert sorted(chosen_outputs, key=lambda p:p['track_id']) == sorted(prediction['predictions'], key=lambda p:p['track_id']), 'chosen projections differ from committed output'
    return audit, branch_outputs, chosen_outputs


def compare_events(left, right, execution, config):
    caps = dict(SQL_templates=256, ancestor_pairs_per_call=config['prefix_cache_entries'],
                raw_observations=config['prefix_cache_entries'])
    previous = [['0'*64, '0'*64] for _ in range(2)]
    caches = [OrderedDict(), OrderedDict()]
    last_summaries = [{}, {}]
    observation_counts = [0, 0]
    def kernel(side, component):
        cache = caches[side]
        if component not in cache:
            cache[component] = StoredProjection((left, right)[side], component)
            if len(cache) > 32:
                cache.popitem(last=False)
        cache.move_to_end(component)
        return cache[component]
    count, maximum, predictions, branches, branch_predictions = 0, 0., 0, 0, 0
    branch_maximum = 0.
    for rows in zip(left.execute('SELECT * FROM events ORDER BY ordinal'),
                    right.execute('SELECT * FROM events ORDER BY ordinal'), strict=True):
        assert rows[0][:2] == rows[1][:2], 'event identity changed'
        values = []
        for side, row in enumerate(rows):
            p, a = json.loads(row[3]), json.loads(row[4])
            observation_counts[side] = bind_request((left,right)[side],row[2],p,a,config,observation_counts[side],explicit_scope=bool(side))
            assert p['previous_commit_sha256'] == previous[side][0]
            assert a['previous_audit_sha256'] == previous[side][1]
            assert digest({k:v for k,v in p.items() if k!='commit_sha256'}) == p['commit_sha256'] == a['prediction_sha256']
            previous[side] = [p['commit_sha256'], digest(a)]
            for component in a['components']:
                last_summaries[side][component['component']] = component
            a, outputs, _ = bind_branch_hashes(a, p, lambda c:kernel(side,c), config['state'])
            values.append((p, a, outputs))
        a, b = (dict(v[1]) for v in values)
        a.pop('prediction_sha256'); a.pop('previous_audit_sha256')
        assert a == semantic_audit(b, execution, caps), 'full semantic audit differs'
        a, b = (dict(v[0]) for v in values)
        first, second = a.pop('predictions'), b.pop('predictions')
        for value in (a, b):
            value.pop('commit_sha256'); value.pop('previous_commit_sha256')
        assert a == b, 'output causal identity changed'
        total, difference = compare_predictions(first, second)
        predictions += total; maximum = max(maximum, difference)
        for (key_a, output_a), (key_b, output_b) in zip(values[0][2], values[1][2], strict=True):
            assert key_a == key_b, 'branch identity changed'
            total, difference = compare_predictions(output_a, output_b)
            branches += 1; branch_predictions += total
            branch_maximum = max(branch_maximum, difference)
        count += 1
        if count % 10 == 0:
            print(json.dumps(dict(stage='independent_commitment_and_audit_correspondence',completed_events=count,
                branch_commitments_checked=2*branches,ETA='unknown; component sizes vary')),flush=True)
    for side, db in enumerate((left, right)):
        actual = {c:json.loads(v) for c,v in db.execute('SELECT component,payload FROM component_summaries')}
        assert actual == last_summaries[side], 'stored component summary differs from final recorded component audit'
    return dict(events=count, predictions=predictions, max_output_state_abs_difference=maximum,
                branch_commitments_independently_bound=2*branches, branch_prediction_pairs_checked=branch_predictions,
                max_branch_projection_abs_difference=branch_maximum, stored_component_summaries_bound=True,
                request_commitments_independently_bound=2*count, original_default_decision_scope_exactly_preserved=True,
                final_commit_hashes=previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args();root=args.candidate;output=args.output
    completion=json.loads((root/'process-completion.json').read_bytes())
    assert completion['observed_exec_exit_code']==0
    assert sha(root/'candidate-check.json')==completion['candidate_receipt_sha256']
    candidate=json.loads((root/'candidate-check.json').read_bytes())
    assert candidate['events']==195 and candidate['complete_reference_sequence_checked'] is True
    assert candidate['numerical_atol']==candidate['numerical_rtol']==1e-8
    input_path=root/'input-binding.json';assert sha(input_path)==candidate['input_binding_sha256']
    binding=json.loads(input_path.read_bytes());reference=candidate['recorded_reference']
    proof_path=Path(reference['acceptance']);assert sha(proof_path)==reference['sha256']
    proof=json.loads(proof_path.read_bytes())
    assert proof['events']==195 and proof['sequence']==binding['sequence']=='0000'
    assert proof['all_discrete_outputs_identical'] is proof['all_continuous_outputs_match'] is True
    for role,digest_ in proof['reports'].items():assert sha(proof_path.parent/(role+'-independent.json'))==digest_
    dbpath=root/'candidate.sqlite';expected=candidate['database_sha256']['candidate']
    assert sha(dbpath)==expected and sha(reference['database'])==reference['database_sha256']
    assert sha(FRESH)==FRESH_SHA
    output.mkdir(parents=True,exist_ok=False)
    left=sqlite3.connect(Path(reference['database']).as_uri()+'?mode=ro',uri=True)
    right=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        assert right.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        a,b=({k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')} for db in (left,right))
        assert a.pop('schema')=='persistent_exclusive_completion_component_identity_v1'
        assert b.pop('schema')=='experimental_exclusive_batched_state_v1'
        execution=b.pop('state_execution')
        assert execution==dict(recipe='independent-root-wave-float64-state-candidate-v1',device='cpu',max_batch=64)
        state_a,state_b=a.pop('state'),b.pop('state');assert a==b
        for value in (state_a,state_b):
            value.pop('audit_sha256');value.pop('prediction_sha256')
        assert state_a==state_b
        tables=compare_tables(left,right)
        event_result=compare_events(left,right,execution,a['config'])
        assert event_result['events']==195
        for index,db in enumerate((left,right)):
            state=json.loads(db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
            assert [state['prediction_sha256'],state['audit_sha256']]==event_result['final_commit_hashes'][index]
        new(output/'structural-and-event-correspondence.json',dict(tables=tables,**event_result))
    except BaseException as error:
        new(output/'failure.json',dict(type=type(error).__name__,message=str(error),accepted=False));raise
    finally:
        left.close();right.close()
    try:
        spec=importlib.util.spec_from_file_location('frozen_independent_new_state_check',FRESH)
        fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
        def progress(value):print(json.dumps(dict(stage='optimized_candidate_fresh_state',**value)),flush=True)
        numeric=fresh.verify_database(dbpath,expected,progress)
        assert numeric['events']==195 and numeric['states']==candidate['materialized_branch_states_checked']
        assert numeric['stored_states_and_chosen_outputs_checked'] is True
        new(output/'fresh-state-independent.json',numeric)
        receipt=output/'acceptance.json'
        new(receipt,dict(kind='rbf_optimized_branch_state_single_sequence_independent_correspondence_and_fresh_state_v3',
            candidate_database_sha256=expected,reference_acceptance_sha256=reference['sha256'],
            source_sha256=sha(__file__),fresh_oracle_sha256=FRESH_SHA,
            correspondence_sha256=sha(output/'structural-and-event-correspondence.json'),
            fresh_state_sha256=sha(output/'fresh-state-independent.json'),events=195,
            states=numeric['states'],predictions=numeric['predictions'],max_fresh_state_abs_error=numeric['max_abs_error'],
            branch_commitments_independently_bound=event_result['branch_commitments_independently_bound'],
            request_commitments_independently_bound=event_result['request_commitments_independently_bound'],
            max_branch_projection_abs_difference=event_result['max_branch_projection_abs_difference'],
            complete_audit_and_raw_structural_table_correspondence=True,atol=1e-8,rtol=1e-8,
            actual_GPU_execution=False,full_cohort_accepted=False,production_promotion_allowed=False,paper_performance_complete=False))
        register(receipt,'rbf-optimized-state-single-sequence-independent-acceptance')
        print(json.dumps(dict(receipt=str(receipt))),flush=True)
    except BaseException as error:
        new(output/'failure.json',dict(type=type(error).__name__,message=str(error),accepted=False));raise


if __name__=='__main__':main()
