#!/usr/bin/env python3
"""Read-only independent identity ledger audit for sealed scan-MHT replays.

No live tracker, assignment solver, Torch, GT or state update is imported.
Reconstruct retained identity-class weights from raw factors, one-to-one scan
constraints and the preceding retained beam. This is NOT an independent
large-scene Top-K optimality proof or a Gaussian state-numerics check.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import hashlib
from itertools import groupby, zip_longest
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x import evaluate_probabilistic_validation as common

native = common.native


def digest(value):
    return hashlib.sha256(native.canonical(value)).hexdigest()


def inference_sources():
    paths = ['run_mht_tracking_v2.py', 'persistent_mht_tracking.py', 'ranked_partial_assignment.py']
    return dict(common.inference_sources(), **{
        'tools/event_track_v2x/'+name: native.sha(ROOT/'tools/event_track_v2x'/name) for name in paths})


def log_total(values):
    values = list(values)
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('nonempty finite alias factors required')
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(v-maximum) for v in values))


def same_weight(a, b):
    if not (math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError('independently reconstructed identity-class weight differs')


@dataclass(frozen=True)
class Branch:
    roots: tuple
    sha: str
    weight: float


class Ledger:
    def __init__(self, db, sequence, config):
        self.db, self.sequence, self.config = db, sequence, config
        self.observations, self.rows, self.slots = [], [], {}
        self.factor_ledger = hashlib.sha256()
        self.beam = {0: Branch((), digest(['persistent-forest-root', sequence]), 0.)}
        self.previous_decision = -1

    def append(self, audit, decision):
        start = len(self.rows); stop = audit['observation_count']
        if type(stop) is not int or stop < start or audit['new_observations'] != stop-start:
            raise ValueError('observation prefix coverage differs')
        appended = []
        for i in range(start, stop):
            record = self.db.execute('SELECT node_id,source,frame,state_us,arrival_us,score,raw,sha '
                                     'FROM observations WHERE i=?', (i,)).fetchone()
            if record is None:
                raise ValueError('missing raw observation')
            node, source, frame, state, arrival, score, raw, sha = record
            value = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != sha or value['sequence_id'] != self.sequence
                    or value['node']['node_id'] != node or value['node']['source_id'] != source
                    or value['node']['frame_id'] != frame or value['state_us'] != state
                    or value['node']['arrival_us'] != arrival or value['score'] != score
                    or not self.previous_decision < arrival <= decision
                    or max(state, value['node']['information_us']) > arrival):
                raise ValueError('raw observation identity or causal arrival differs')
            row = self.db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p', (i,)).fetchall()
            if (not row or row[0][0] != -1 or any(type(p) is not int or not -1 <= p < i
                                                or not math.isfinite(w) for p,w in row)):
                raise ValueError('illegal raw factor row')
            self.observations.append((node, source, frame, state, arrival, score))
            self.rows.append(dict(row)); self.slots.setdefault((source,frame), []).append(i)
            appended.append([list(pair) for pair in row])
            self.factor_ledger.update(native.canonical([i,sha,row])+b'\n')
        # Producer validates appended rows in its canonical parent order.
        if ([[list(p) for p in sorted(row)] for row in audit['appended_rows']] != appended
                or audit['factor_rows_sha256'] != self.factor_ledger.hexdigest() or audit['rescored_rows']):
            raise ValueError('published raw factor ledger differs')
        self.previous_decision = decision
        return start, stop

    def factors(self, branch, indices, slot):
        start = indices[0]
        occupied = {branch.roots[i] for i in self.slots[slot] if i < start}
        groups = []
        for i in indices:
            group = {}
            for p,w in self.rows[i].items():
                if 0 <= p < start and branch.roots[p] not in occupied:
                    group.setdefault(branch.roots[p], []).append((p,w))
            groups.append(group)
        roots = sorted({r for group in groups for r in group})
        pair = [[log_total(w for _,w in group[r]) if r in group else 0. for group in groups] for r in roots]
        return groups, roots, digest(dict(pair=pair, left=[0.]*len(roots),
            right=[self.rows[i][-1] for i in indices], allowed=[[r in group for group in groups] for r in roots]))

    def extend(self, handle, start, stop, groups_by_parent, slot):
        tail, cursor = [], handle
        for depth in range(stop, start, -1):
            prefix = self.db.execute('SELECT parent,depth,choice,root,sha FROM prefixes WHERE h=?', (cursor,)).fetchone()
            if prefix is None or prefix[1] != depth:
                raise ValueError('identity prefix depth or predecessor differs')
            tail.append((cursor, prefix)); cursor = prefix[0]
        if cursor not in self.beam:
            raise ValueError('pruned predecessor resurrected')
        branch = self.beam[cursor]; roots = list(branch.roots); weight = branch.weight; sha = branch.sha
        if len(roots) != start:
            raise ValueError('retained predecessor depth differs')
        groups = groups_by_parent[cursor]
        occupied = {roots[i] for i in self.slots[slot] if i < start}
        for index, (h, (_,depth,choice,root,node_sha)) in enumerate(reversed(tail), start):
            group = groups[index-start]
            if choice == -1:
                expected_root, increment = index, self.rows[index][-1]
            elif type(choice) is int and 0 <= choice < start:
                expected_root = roots[choice]
                if expected_root not in group or choice != min(p for p,_ in group[expected_root]):
                    raise ValueError('noncanonical or forbidden raw parent alias')
                increment = log_total(w for _,w in group[expected_root])
            else:
                raise ValueError('parent inside current scan or outside history')
            if root != expected_root or root in occupied:
                raise ValueError('identity root or one-to-one source/frame constraint differs')
            sha = digest([sha,index,choice,root])
            if sha != node_sha:
                raise ValueError('identity prefix digest differs')
            weight = math.fsum([weight,increment])
            stored = self.db.execute('SELECT revision,value FROM weights WHERE h=?', (h,)).fetchone()
            if stored is None or stored[0] != 0:
                raise ValueError('missing immutable class-weight ledger')
            same_weight(stored[1],weight)
            occupied.add(root); roots.append(root)
        return Branch(tuple(roots),sha,weight)

    def scan(self, scan, selected, start):
        indices = scan['observation_indices']; slot = scan['source'],scan['frame']
        if (not indices or indices != list(range(start,start+len(indices)))
                or indices[-1] >= len(self.rows)
                or any(self.observations[i][1:3] != slot for i in indices)):
            raise ValueError('source scan omits, repeats or mixes observation indices')
        parents = scan['ranked_parent_scans']
        if (len(parents) != len(self.beam) or scan['predecessor_histories'] != len(self.beam)
                or {p['parent_handle'] for p in parents} != set(self.beam)):
            raise ValueError('conditioned predecessor beam differs')
        groups_by_parent = {}; generated = solves = 0
        for p in parents:
            h = p['parent_handle']; branch = self.beam[h]
            if p['parent_sha256'] != branch.sha:
                raise ValueError('conditioned predecessor digest differs')
            same_weight(p['parent_log_weight'],branch.weight)
            groups,roots,sha = self.factors(branch,indices,slot)
            if roots != p['conditioned_track_roots'] or sha != p['factor_sha256']:
                raise ValueError('independent conditional scan factors differ')
            n = p['ranked_hypotheses']; width = self.config['state']['active_limit']
            if (type(n) is not int or not 1 <= n <= width
                    or p['requested_k_reached'] is not (n == width)
                    or not (p['requested_k_reached'] or p['support_exhausted'])):
                raise ValueError('partial ranked support claimed as complete Top-K')
            for key,limit in [('assignment_solves','max_assignment_solves'),
                              ('peak_frontier','max_assignment_frontier'),
                              ('peak_matrix_cells','max_assignment_matrix_cells')]:
                if type(p[key]) is not int or not 0 <= p[key] <= self.config[limit]:
                    raise ValueError('assignment work or capacity exceeds bound')
            generated += n; solves += p['assignment_solves']; groups_by_parent[h] = groups
        if (generated != scan['generated_candidates'] or scan['kept'] != len(selected)
                or len(selected) != min(width,generated) or len(set(selected)) != len(selected)):
            raise ValueError('retained scan width or candidate count differs')
        next_beam = {h:self.extend(h,start,start+len(indices),groups_by_parent,slot) for h in selected}
        if len({b.roots for b in next_beam.values()}) != len(next_beam):
            raise ValueError('duplicate physical identity history')
        self.beam = next_beam
        return len(indices),len(parents),generated,solves

    def output(self, prediction, audit):
        active = audit['active']
        if len(active) != len(self.beam) or {r['handle'] for r in active} != set(self.beam):
            raise ValueError('published retained history inventory differs')
        for row in active:
            b = self.beam[row['handle']]
            if row['sha256'] != b.sha:
                raise ValueError('published retained history digest differs')
            same_weight(row['log_weight'],b.weight)
        ordered = sorted(active,key=lambda r:(-r['log_weight'],r['sha256']))
        if (active != ordered or audit['output_handle'] != ordered[0]['handle']
                or audit['output_sha256'] != ordered[0]['sha256']
                or audit['retained_history_sha256'] != digest([
                    [r['handle'],r['sha256'],r['log_weight']] for r in sorted(active,key=lambda r:r['handle'])])):
            raise ValueError('MAP of retained histories or beam digest differs')
        branches = audit['branches']
        if len(branches) != len(active) or {r['handle'] for r in branches} != set(self.beam):
            raise ValueError('conditional state branch inventory differs')
        for row in branches:
            if row['sha256'] != self.beam[row['handle']].sha:
                raise ValueError('state branch identity differs')
            if row['handle'] == audit['output_handle'] and row['state_sha256'] != digest(prediction['predictions']):
                raise ValueError('selected conditional state digest differs')
        roots = self.beam[audit['output_handle']].roots; groups = {}
        for i,root in enumerate(roots): groups.setdefault(root,[]).append(self.observations[i])
        expected = []; c = self.config['state']; reference = prediction['box_reference_timestamp_us']
        for root,members in sorted(groups.items()):
            if (max(r[5] for r in members) < c['birth_score']
                    or max(0,reference-max(r[3] for r in members)) > c['max_age_us']): continue
            score = max(r[5]*c['survival_per_second']**(max(0,reference-r[3])/1e6) for r in members)
            if score >= c['prune_score']:
                expected.append(dict(track_id=self.sequence+':'+digest(['forest-birth',self.observations[root][0]])[:24],
                                     class_label='car',score=score))
        actual = [{k:v for k,v in p.items() if k not in ('mean','covariance')} for p in prediction['predictions']]
        if actual != expected:
            raise ValueError('output IDs, raw-score lifecycle or class scope differs')


def inspect_sequence(path, sequence, head, records, config):
    """Records are (schedule row, prediction, tracking wrapper, frame timing)."""
    with closing(sqlite3.connect(Path(path).absolute().as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        meta = {k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')}
        if (db.execute('PRAGMA quick_check').fetchall() != [('ok',)]
                or meta['schema'] != 'persistent_global_scan_mht_v1' or meta['sequence_id'] != sequence
                or meta['config'] != config or meta['state']['revision'] != 0):
            raise ValueError('sealed MHT database schema or configuration differs')
        ledger = Ledger(db,sequence,config); counts = Counter(); previous = previous_audit = '0'*64
        factor_stream = hashlib.sha256(); latencies = []; frame_latencies = []
        stored = db.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal')
        for ordinal,(record,saved) in enumerate(zip_longest(records,stored)):
            if record is None or saved is None or any(v is None for v in record):
                raise ValueError('database and replay stream coverage differs')
            row,prediction,wrapper,timing = record; a = wrapper['tracking']; reference = row['box_reference_timestamp_us']
            key = sequence,row['vehicle_frame'],reference,reference+100000
            if type(reference) is not int or key[3] <= ledger.previous_decision:
                raise ValueError('nonmonotonic causal decision clock')
            actual_key = lambda v:(v['sequence_id'],v['frame_id'],v['box_reference_timestamp_us'],v['decision_timestamp_us'])
            if (row['sequence_id'] != sequence or actual_key(prediction) != key or actual_key(timing) != key
                    or (a['sequence_id'],a['event_id']) != key[:2] or saved[:2] != (ordinal,key[1])
                    or json.loads(saved[2]) != prediction or json.loads(saved[3]) != a):
                raise ValueError('database event, published stream or causal clock differs')
            if (prediction['previous_commit_sha256'] != previous or prediction['commit_sha256'] !=
                    digest({k:v for k,v in prediction.items() if k != 'commit_sha256'})
                    or a['prediction_sha256'] != prediction['commit_sha256'] or a['previous_audit_sha256'] != previous_audit
                    or a['configuration_sha256'] != digest(config)):
                raise ValueError('immutable output or audit chain differs')
            if (a['kind'] != meta['schema'] or a['recovery_enabled'] is not False
                    or a['archived_prefixes_used_for_recovery'] is not False or a['restored_ancestors']
                    or a['frontier_restarted_from_root'] or a['frontier'] or a['resource_limited']
                    or a['conditional_scan_bounds_are_not_full_history_bounds'] is not True
                    or a['decision'] != dict(policy='MAP_of_retained_global_histories',risk_bound=None,
                                             normalized_weights_are_complete_posterior=False)
                    or 'eta_upper' in a or 'log_partition_upper' in a):
                raise ValueError('irreversible MHT algorithm or risk declaration differs')
            start,stop = ledger.append(a,key[3]); cursor = start; generated = solves = 0; slots = set()
            for index,scan in enumerate(a['scans']):
                slot = scan['source'],scan['frame']
                if slot in slots: raise ValueError('interleaved repeated source scan')
                slots.add(slot)
                selected = ([r['parent_handle'] for r in a['scans'][index+1]['ranked_parent_scans']]
                            if index+1 < len(a['scans']) else [r['handle'] for r in a['active']])
                n,parents,g,s = ledger.scan(scan,selected,cursor)
                cursor += n; generated += g; solves += s; counts.update(scans=1,conditioned_parent_scans=parents)
            if (cursor != stop or generated != a['generated_candidates'] or solves != a['assignment_solves']
                    or generated > config['max_generated_candidates'] or solves > config['max_assignment_solves']):
                raise ValueError('event scan coverage or shared work budget differs')
            ledger.output(prediction,a)
            x,y = timing['step_seconds'],timing['frame_seconds']
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= y): raise ValueError('invalid latency')
            latencies.append(x); frame_latencies.append(y)
            previous,previous_audit = prediction['commit_sha256'],digest(a)
            factor_stream.update(native.canonical([sequence,key[1],a['factor_rows_sha256']])+b'\n')
            counts.update(frames=1,observations=stop-start,assignment_solves=solves,generated_candidates=generated,
                          output_boxes=len(prediction['predictions']))
        if (not counts['frames'] or counts['frames'] != head['frames'] or previous != head['prediction_sha256']
                or meta['state']['n'] != len(ledger.rows) or meta['state']['events'] != counts['frames']
                or meta['state']['prediction_sha256'] != previous or meta['state']['audit_sha256'] != previous_audit
                or meta['state']['active'] != sorted(ledger.beam) or meta['state']['frontier']
                or db.execute('SELECT count(*) FROM observations').fetchone()[0] != len(ledger.rows)):
            raise ValueError('final database, observation coverage or retained beam differs')
        return dict(counts,prediction_sha256=previous,audit_sha256=previous_audit,
                    factor_stream_sha256=factor_stream.hexdigest(),latencies=latencies,frame_latencies=frame_latencies)


def audit_replay(run, receipt_sha, rows):
    """Explicit subset-capable helper; it never certifies full validation coverage."""
    if not isinstance(rows,list) or not rows:
        raise ValueError('nonempty explicit prediction schedule required')
    run = Path(run).absolute(); evidence = {}
    def bind(path, expected=None):
        actual = native.evidence(path)
        if expected is not None and actual['sha256'] != expected: raise ValueError('sealed replay artifact differs')
        evidence[str(path)] = actual
        return path
    receipt = native.read_json(bind(common.child(run,'receipt.json'),receipt_sha))
    plan = native.read_json(bind(common.child(run,'plan.json'),receipt['plan_sha256']))
    if (receipt['kind'] != 'scan_mht_scheduled_replay_v1' or receipt['status'] != 'complete'
            or receipt['completed_frames'] != len(rows) or receipt['scheduled_frames'] != len(rows)
            or plan['kind'] != 'scan_mht_replay_plan_v1' or plan['class_scope'] != ['car']
            or plan['cache_split'] not in ('train','val') or receipt['cache_split'] != plan['cache_split']
            or plan['scheduled_frames'] != len(rows)
            or plan['schedule_rows_sha256'] != digest(rows) or plan['source_sha256'] != inference_sources()):
        raise ValueError('complete current-source MHT replay and exact schedule required')
    for p,sha in plan.get('input_file_sha256',{}).items(): bind(Path(p),sha)
    for p,sha in plan['source_sha256'].items(): bind(common.child(ROOT,p),sha)
    for p in (Path(__file__),Path(common.__file__),Path(native.__file__)): bind(p)
    paths = [bind(common.child(run,name),receipt[key]) for name,key in (
        ('predictions.jsonl','predictions_sha256'),('tracking.jsonl','tracking_sha256'),
        ('frame-timings.jsonl','frame_timings_sha256'))]
    sequences = {}; database_bytes = 0
    with paths[0].open('rb') as predictions,paths[1].open('rb') as audits,paths[2].open('rb') as timings:
        records = zip_longest(rows,map(json.loads,predictions),map(json.loads,audits),map(json.loads,timings))
        for sequence,group in groupby(records,key=lambda r:None if r[0] is None else r[0]['sequence_id']):
            if sequence in sequences or sequence not in receipt['sequence_heads']:
                raise ValueError('unknown, repeated or excess sequence block')
            head = receipt['sequence_heads'][sequence]
            path = bind(common.child(run,head['database']),head['database_sha256']); database_bytes += path.stat().st_size
            sequences[sequence] = inspect_sequence(path,sequence,head,group,plan['configuration'])
    if set(sequences) != set(receipt['sequence_heads']) or database_bytes != receipt['database_bytes']:
        raise ValueError('full replay database inventory differs')
    common.unchanged(evidence)
    return dict(kind='scan_mht_replay_identity_audit_v1',status='complete',sequences=sequences,
        frames=sum(v['frames'] for v in sequences.values()),observations=sum(v['observations'] for v in sequences.values()),
        input_evidence=evidence,replay_receipt_sha256=receipt_sha,configuration=plan['configuration'],
        raw_factor_ledger_reconstructed=True,retained_alias_weights_reconstructed=True,
        one_to_one_and_irreversible_predecessors_verified=True,published_ID_lifecycle_reconstructed=True,
        full_top_k_optimality_recomputed=False,conditional_mass_bounds_recomputed=False,
        gaussian_state_numerics_recomputed=False,ground_truth_read=False,full_validation_coverage_verified=False,
        tracking_metrics_computed=False,fair_resources_verified=False,paper_eligible=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Development replay identity audit; no full-validation promotion.')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--schedule-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if native.sha(args.schedule) != args.schedule_sha256: raise ValueError('schedule digest differs')
    rows = native.read_json(args.schedule)
    if args.output.exists() or any(p.is_symlink() for p in (args.output,*args.output.parents)):
        raise ValueError('new non-symlink audit output required')
    result = audit_replay(args.run,args.receipt_sha256,rows)
    if native.sha(args.schedule) != args.schedule_sha256: raise ValueError('schedule changed during audit')
    result['input_evidence'][str(args.schedule.absolute())] = native.evidence(args.schedule)
    args.output.mkdir(); native.write_json(args.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','frames','observations','full_validation_coverage_verified')}))
