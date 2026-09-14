#!/usr/bin/env python3
"""Fresh chosen-branch CI replay of EVERY event in a sealed train diagnostic.

No GT, optimizer, association rerun, cached branch state or final factor table.
Event-time factors come from the append/rescore log; raw observations and prefix
ancestry are immutable inputs. Equality certifies conditional state computation,
NOT identity correctness, probability calibration or tracking improvement.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import scipy

from tools.event_track_v2x.audit_train_inference_comparison import inspect
from transvision.models.event_track_v2x.allocation_training import _directory
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, contained_file, sha_file
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, RawIdentityDetection, replay_forest_states
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode, digest

BACKENDS = frozenset(('node_beam', 'joint_beam', 'class_bound_joint_beam', 'slot_bound_joint_beam',
    'sparse_slot_bound_joint_beam', 'reachable_slot_bound_joint_beam', 'beam_recovery', 'beam_recovery_disabled'))
FIELDS = ('track_id', 'class_label', 'mean', 'covariance', 'score')
# State computation closure, not the producer's search/registration modules.
# Thus archived old/new search backends can be compared with unchanged CI code.
STATE_SOURCES = tuple('transvision/models/event_track_v2x/' + name + '.py' for name in (
    'forest_tracking', 'identity_forest', 'recoverable_states', 'tracking_v2', 'fusion',
    'arrays', 'prediction_features', 'detection_cache_v2', 'hypothesis_bank'))
AUDIT_SOURCES = ('tools/event_track_v2x/audit_branch_state_replay.py',
    'tools/event_track_v2x/audit_train_inference_comparison.py')
ZERO = '0' * 64


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def read_only_tables(action, table, column, database, trigger):
    """Enforce the no-state-cache/no-final-potentials contract at SQLite level."""
    if action == sqlite3.SQLITE_READ:
        allowed = table in ('observations', 'component_catalog', 'component_members') or re.fullmatch(r'pc[1-9]\d*_prefixes', table)
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def prefix_choices(db, sequence, component, handle, factors, expected_sha):
    """Verify every parent, depth, root and chained SHA without kernel caches."""
    nonnegative(handle, 'output handle')
    entries = []
    for depth in range(len(factors.nodes), -1, -1):
        row = db.execute(f'SELECT parent,depth,choice,root,sha FROM pc{component}_prefixes WHERE h=?', (handle,)).fetchone()
        if row is None or row[1] != depth:
            raise ValueError('historical output prefix depth differs')
        if depth == 0:
            previous = digest(['persistent-forest-root', sequence])
            if handle != 0 or row[:4] != (None, 0, None, None) or row[4] != previous:
                raise ValueError('historical prefix root differs')
            break
        entries.append(row)
        handle = nonnegative(row[0], 'parent handle')
    entries.reverse()
    choices = tuple(row[2] for row in entries)
    roots = factors.roots(choices)
    for depth, (row, root) in enumerate(zip(entries, roots)):
        previous = digest([previous, depth, row[2], root])
        if row[3] != root or row[4] != previous:
            raise ValueError('historical prefix root or digest differs')
    if previous != expected_sha:
        raise ValueError('chosen output prefix digest differs')
    return choices


class FreshStateAudit:
    """Low-level fixture-capable audit. Production provenance is checked by run."""

    def __init__(self, db, sequence, configuration, *, max_node_visits=2_000_000, require_cache_receipts=False):
        if type(max_node_visits) is not int or max_node_visits < 1:
            raise ValueError('positive node visit cap required')
        self.db, self.sequence = db, sequence
        self.configuration_sha = digest(configuration)
        self.config = ForestTrackingConfig(**configuration['state'])
        self.cap, self.require_cache_receipts = max_node_visits, require_cache_receipts
        self.observations, self.rows, self.raw_hashes, self.events = [], [], [], []
        self.prediction_head = self.audit_head = ZERO
        self.reference = self.decision = -1
        self.visits = 0
        self.seen = set()

    def check(self, audit, prediction):
        # Failure makes this object unusable as a complete audit; no partial
        # success receipt is published by run(). The original DB stays read-only.
        sequence, event = audit['sequence_id'], audit['event_id']
        if sequence != self.sequence or (sequence, event) != (prediction['sequence_id'], prediction['frame_id']) or event in self.seen:
            raise ValueError('duplicate or mismatched sequence/event')
        reference = nonnegative(prediction['box_reference_timestamp_us'], 'reference time')
        decision = nonnegative(prediction['decision_timestamp_us'], 'decision time')
        if not self.reference < reference <= decision or decision < self.decision:
            raise ValueError('nonmonotonic reference or decision time')
        payload = {k: v for k, v in prediction.items() if k != 'commit_sha256'}
        if (prediction['commit_sha256'] != digest(payload) or prediction['previous_commit_sha256'] != self.prediction_head
                or audit['previous_audit_sha256'] != self.audit_head or audit['prediction_sha256'] != prediction['commit_sha256']):
            raise ValueError('audit or prediction commit chain differs')
        if (audit['configuration_sha256'] != self.configuration_sha or prediction['coordinate_frame'] != 'world'
                or prediction['state_layout'] != 'gravity_xyz_length_width_height_yaw_vxy'):
            raise ValueError('state configuration or output layout differs')
        old_n, n = len(self.observations), nonnegative(audit['observation_count'], 'observation count')
        if (n < old_n or nonnegative(audit['new_observations'], 'new observations') != n - old_n
                or len(audit['appended_rows']) != n - old_n):
            raise ValueError('append coverage differs')
        if self.visits + n > self.cap:
            raise ValueError('fresh replay node visit cap exceeded; partial events are not acceptance')
        raw_rows = self.db.execute('SELECT i,node_id,source,frame,detection_index,state_us,arrival_us,score,raw,sha '
            'FROM observations WHERE i>=? AND i<? ORDER BY i', (old_n, n)).fetchall()
        if [r[0] for r in raw_rows] != list(range(old_n, n)):
            raise ValueError('raw observation coverage differs')
        new = []
        last_arrival = self.observations[-1].node.arrival_us if old_n else -1
        for row in raw_rows:
            if hashlib.sha256(row[8]).hexdigest() != row[9]:
                raise ValueError('raw observation digest differs')
            values = json.loads(row[8])
            raw = RawIdentityDetection(**dict(values, node=IdentityNode(**values['node'])))
            node = raw.node
            if (raw.sequence_id != sequence or not max(last_arrival, self.decision) <= node.arrival_us <= decision
                    or row[1:8] != (node.node_id, node.source_id, node.frame_id, raw.detection_index, raw.state_us, node.arrival_us, raw.score)):
                raise ValueError('raw identity/time index differs or future/withheld input')
            last_arrival = node.arrival_us
            new.append(raw)
        ingestion = audit.get('cache_ingestion')
        if self.require_cache_receipts and (not isinstance(ingestion, dict) or ingestion.get('gt_model_inputs') is not False
                or ingestion.get('class_scope') != ['car']):
            raise ValueError('car-only prediction cache receipt required')
        if ingestion is not None:
            deliveries = ingestion['new_deliveries']
            receipt_map = {(d['side'], d['frame_id']): d for d in deliveries}
            if len(receipt_map) != len(deliveries) or any(d['sequence_id'] != sequence
                    or not self.decision <= d['arrival_us'] <= decision for d in deliveries):
                raise ValueError('invalid causal delivery receipt')
            for raw in new:
                side = 'vehicle-side' if raw.node.source_id == 0 else 'infrastructure-side'
                d = receipt_map.get((side, raw.node.frame_id))
                if d is None or (d['arrival_us'], d['frame_sha256']) != (raw.node.arrival_us, raw.source_cache_sha256):
                    raise ValueError('raw input not bound to arrived delivery')
        updates = audit['rescored_rows']
        if len({i for i, _ in updates}) != len(updates):
            raise ValueError('duplicate rescore row')
        for i, row in updates:
            if type(i) is not int or not 0 <= i < old_n or sorted(p for p, _ in row) != [p for p, _ in self.rows[i]]:
                raise ValueError('rescore changed support or addresses a nonhistorical row')
            self.rows[i] = row
        self.rows.extend(audit['appended_rows'])
        self.observations.extend(new)
        self.raw_hashes.extend(row[9] for row in raw_rows)
        factors = ForestFactors(tuple(o.node for o in self.observations), tuple(self.rows))
        self.rows = list(factors.rows)
        factor_hash = hashlib.sha256()
        for i, (sha, row) in enumerate(zip(self.raw_hashes, self.rows)):
            factor_hash.update(canonical([i, sha, row]) + b'\n')
        if factor_hash.hexdigest() != audit['factor_rows_sha256']:
            raise ValueError('event-time factor digest differs')
        covered, predicted, branch_digests, component_ids = set(), [], [], set()
        suppressed = expired = 0
        for summary in audit['components']:
            component, depth = summary['component'], summary['nodes']
            if (type(component) is not int or component < 1 or component in component_ids
                    or type(depth) is not int or not 1 <= depth <= n):
                raise ValueError('invalid or duplicate historical component')
            component_ids.add(component)
            catalog = self.db.execute('SELECT created_us FROM component_catalog WHERE component=?', (component,)).fetchone()
            if catalog is None or catalog[0] > decision:
                raise ValueError('component unavailable at event decision')
            mapping = self.db.execute('SELECT local_i,global_i FROM component_members WHERE component=? AND local_i<? '
                'ORDER BY local_i', (component, depth)).fetchall()
            members = [r[1] for r in mapping]
            if ([r[0] for r in mapping] != list(range(depth)) or len(set(members)) != depth
                    or any(type(i) is not int or not 0 <= i < n for i in members) or covered.intersection(members)):
                raise ValueError('historical component member coverage differs')
            covered.update(members)
            local = {g: i for i, g in enumerate(members)}
            if any(p != -1 and p not in local for g in members for p, _ in self.rows[g]):
                raise ValueError('factor parent outside historical component')
            observations = tuple(self.observations[i] for i in members)
            local_factors = ForestFactors(tuple(o.node for o in observations),
                tuple(tuple((-1 if p == -1 else local[p], w) for p, w in self.rows[g]) for g in members))
            choices = prefix_choices(self.db, sequence, component, summary['output_handle'], local_factors, summary['output_sha256'])
            fresh, hidden = replay_forest_states(sequence, observations, local_factors, choices, reference, self.config)
            states = [{k: p[k] for k in FIELDS} for p in fresh]
            actual_expired = reference - max(o.state_us for o in observations) > self.config.max_age_us
            if summary.get('expired_output_only') is not actual_expired:
                raise ValueError('component expiry differs from raw state times')
            branches = summary['branches']
            if len({b['handle'] for b in branches}) != len(branches):
                raise ValueError('duplicate branch state receipt')
            chosen = [b for b in branches if b['handle'] == summary['output_handle']]
            if len(chosen) != 1 or chosen[0]['state_sha256'] != digest(states):
                raise ValueError('fresh chosen-branch state digest differs')
            branch_digests.append([component, summary['output_sha256'], digest(states)])
            predicted.extend(states)
            suppressed += len(hidden)
            expired += actual_expired
        if covered != set(range(n)):
            raise ValueError('components omit raw observations at event time')
        predicted.sort(key=lambda p: p['track_id'])
        if len({p['track_id'] for p in predicted}) != len(predicted) or canonical(predicted) != canonical(prediction['predictions']):
            raise ValueError('fresh entire-frame prediction differs (including missing/extra IDs)')
        self.visits += n
        self.reference, self.decision = reference, decision
        self.prediction_head, self.audit_head = prediction['commit_sha256'], digest(audit)
        self.seen.add(event)
        result = dict(sequence_id=sequence, frame_id=event, observations=n, components=len(component_ids),
            output_boxes=len(predicted), suppressed_roots=suppressed, expired_components=expired,
            factor_rows_sha256=factor_hash.hexdigest(), chosen_branches_sha256=digest(branch_digests),
            fresh_output_sha256=digest(predicted), entire_frame_exact=True)
        self.events.append(result)
        return result


def run(directory, receipt_sha256, output, *, max_node_visits=2_000_000, allow_fixture=False):
    directory = Path(directory)
    verified = inspect(directory, receipt_sha256, allow_fixture=allow_fixture)
    plan = verified['plan']
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    if verified['backend'] not in BACKENDS or len(receipt['sequence_heads']) != 1 or receipt.get('gt_model_inputs') is not False:
        raise ValueError('one complete prediction-only CI branch sequence required; not JPDA/PKF')
    sequence, head = next(iter(receipt['sequence_heads'].items()))
    source_hashes = {p: sha_file(ROOT / p) for p in STATE_SOURCES + AUDIT_SOURCES}
    if any(plan['source_sha256'].get(p) != source_hashes[p] for p in STATE_SOURCES):
        raise ValueError('fresh state implementation differs from frozen producer')
    if not allow_fixture and (plan['runtime']['python'] != sys.version or plan['runtime']['executable'] != sys.executable
            or plan['runtime']['platform'] != platform.platform()
            or any(os.environ.get(k) != v for k, v in plan['thread_environment'].items())):
        raise ValueError('run fresh audit in the producer runtime with its pinned thread environment')
    artifacts = ('plan.json', 'receipt.json', 'development-inference-receipt.json', 'tracking.jsonl',
                 'predictions.jsonl', 'frame-timings.jsonl', head['database'])
    before = {p: sha_file(contained_file(directory, p)) for p in artifacts}
    db_path = contained_file(directory, head['database'])
    # Reject a mutable/WAL input, not merely its last checkpointed main DB.
    if any(Path(str(db_path) + suffix).exists() for suffix in ('-wal', '-journal')):
        raise ValueError('closed non-WAL replay database required')
    output = _directory(output)
    audit_plan = dict(kind='chosen_branch_fresh_state_audit_plan_v1', input_directory=str(directory.absolute()),
        input_sha256=before, source_sha256=source_hashes, max_node_visits=max_node_visits,
        cohort_mode=plan['cohort_mode'], backend=verified['backend'], all_scheduled_frames=len(verified['events']),
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__),
        no_GT_read=True, state_cache_read=False, final_factor_table_read=False, paper_eligible=False)
    _new_json(output / 'plan.json', audit_plan)
    started = time.monotonic()
    checker = None
    try:
        with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
            db.set_authorizer(read_only_tables)
            checker = FreshStateAudit(db, sequence, plan['configuration'], max_node_visits=max_node_visits,
                require_cache_receipts=not allow_fixture)
            with (directory / 'tracking.jsonl').open('rb') as audits, (directory / 'predictions.jsonl').open('rb') as predictions:
                for line in audits:
                    encoded = predictions.readline()
                    if not encoded:
                        raise ValueError('prediction stream ended before audit')
                    checker.check(json.loads(line)['tracking'], json.loads(encoded))
                    if len(checker.events) % 25 == 0:
                        print(json.dumps(dict(verified_frames=len(checker.events), node_visits=checker.visits)), flush=True)
                if predictions.readline():
                    raise ValueError('extra prediction event')
            if (len(checker.events) != len(verified['events']) or len(checker.events) != head['frames']
                    or checker.prediction_head != head['prediction_sha256']
                    or db.execute('SELECT count(*) FROM observations').fetchone()[0] != len(checker.observations)):
                raise ValueError('complete sequence/head/raw coverage differs')
        if (before != {p: sha_file(contained_file(directory, p)) for p in artifacts}
                or source_hashes != {p: sha_file(ROOT / p) for p in source_hashes}):
            raise ValueError('inputs or fresh replay source changed during audit')
        result = dict(kind='chosen_branch_fresh_state_audit_v1', status='complete',
            plan_sha256=sha_file(output / 'plan.json'), backend=verified['backend'], frames=len(checker.events),
            observations=len(checker.observations), node_visits=checker.visits,
            output_boxes=sum(e['output_boxes'] for e in checker.events),
            component_checks=sum(e['components'] for e in checker.events),
            expired_component_checks=sum(e['expired_components'] for e in checker.events),
            factor_stream_sha256=verified['factor_stream_sha256'], events=checker.events,
            elapsed_seconds=time.monotonic() - started, entire_sequence_exact=True,
            cached_states_read=False, GT_read=False, association_recomputed=False,
            unchosen_branches_audited=False, independent_numerical_CI_implementation=False,
            identity_correctness_verified=False, formal_probability_bound_verified=False,
            real_tracking_evaluation=False, paper_eligible=False)
        _new_json(output / 'audit.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'events'}, sort_keys=True), flush=True)
        return result
    except Exception as error:
        _new_json(output / 'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
            verified_frames=0 if checker is None else len(checker.events), entire_sequence_exact=False,
            partial_events_not_acceptance=True, plan_sha256=sha_file(output / 'plan.json')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-node-visits', type=int, default=2_000_000)
    args = parser.parse_args()
    run(args.replay, args.receipt_sha256, args.output, max_node_visits=args.max_node_visits)
