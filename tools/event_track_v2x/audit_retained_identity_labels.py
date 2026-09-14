#!/usr/bin/env python3
"""Offline partial-label errors inside retained identity classes, all events.

Derived TRAIN labels are used only AFTER frozen inference, never by a scorer or
decoder. This is not native HOTA, full-identity truth, a complete-posterior risk
bound, a proof that a correct class was pruned, or a calibrated real posterior.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.event_track_v2x import audit_branch_state_replay as state_audit
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import DATA_KIND, TrainingShard, _new_json, row_protocol
from transvision.models.event_track_v2x.identity_forest import ForestFactors, digest
from transvision.models.event_track_v2x.hypothesis_bank import logsumexp

UNITS = 840  # lcm(1,...,8): exact equal-row errors with the frozen eight-parent gate.
SOURCES = state_audit.STATE_SOURCES + state_audit.AUDIT_SOURCES + (
    'tools/event_track_v2x/audit_retained_identity_labels.py',
    'transvision/models/event_track_v2x/forest_training_data.py',
    'transvision/models/event_track_v2x/forest_row_context.py',
    'transvision/models/event_track_v2x/forest_supervision.py')


def class_weight(factors, choices):
    """Sum ALL equivalent parent paths, not just the canonical representative."""
    roots = factors.roots(choices)
    if len(choices) != len(factors.nodes):
        raise ValueError('complete class required')
    weight = 0.
    for i, parent in enumerate(choices):
        if parent == -1:
            increment = dict(factors.rows[i])[-1]
        else:
            aliases = [(p, w) for p, w in factors.rows[i] if p >= 0 and roots[p] == roots[i]]
            if parent != min(p for p, _ in aliases):
                raise ValueError('class representative must use the canonical minimum parent')
            increment = logsumexp(w for _, w in aliases)
        weight = math.fsum((weight, increment))
    return weight, roots


def relation_error_units(roots, relations):
    """One row is mean error on its KNOWN candidate relations; birth excluded."""
    result = 0
    for query, edges in relations:
        if not edges or len(edges) > 8 or UNITS % len(edges):
            raise ValueError('one to eight known parent relations per scored row required')
        wrong = sum((roots[query] == roots[parent]) != truth for parent, truth in edges)
        result += wrong * (UNITS // len(edges))
    return result


def calibration(roots, weights, relations):
    """Equal-edge Brier/ECE for q(.|retained); omitted classes are NOT covered."""
    if not roots or len(roots) != len(weights) or any(not math.isfinite(w) for w in weights):
        raise ValueError('finite retained class weights required')
    normalizer = logsumexp(weights)
    probabilities = [math.exp(w - normalizer) for w in weights]
    bins = [dict(count=0, probability_sum=0., positive_count=0) for _ in range(10)]
    terms = []
    for query, edges in relations:
        for parent, truth in edges:
            p = math.fsum(w for w, root in zip(probabilities, roots) if root[query] == root[parent])
            if not -1e-12 <= p <= 1 + 1e-12:
                raise ValueError('invalid conditional relation probability')
            p = min(1., max(0., p))  # Roundoff only; no smoothing or fitted clipping.
            b = bins[min(9, int(p * 10))]
            b['count'] += 1; b['probability_sum'] += p; b['positive_count'] += int(truth)
            terms.append((p - int(truth)) ** 2)
    return dict(known_edges=len(terms), brier_sum=math.fsum(terms), bins=bins)


def zero_totals():
    return dict(scored_rows=0, known_edges=0, chosen_error_units=0, MAP_error_units=0,
        best_retained_error_units=0, best_union_error_units=0, brier_sum=0.,
        bins=[dict(count=0, probability_sum=0., positive_count=0) for _ in range(10)])


def accumulate(total, item):
    for key in total:
        if key != 'bins': total[key] += item[key]
    for dest, source in zip(total['bins'], item['bins']):
        for key in dest: dest[key] += source[key]


def rates(total):
    result = dict(total)
    denominator = UNITS * total['scored_rows']
    for key in ('chosen', 'MAP', 'best_retained', 'best_union'):
        result[key + '_equal_row_error'] = total[key + '_error_units'] / denominator if denominator else None
    result['chosen_to_union_best_gap'] = ((total['chosen_error_units'] - total['best_union_error_units']) / denominator
                                         if denominator else None)
    n = total['known_edges']
    result['conditional_equal_edge_Brier'] = total['brier_sum'] / n if n else None
    result['conditional_equal_edge_ECE10'] = (math.fsum(abs(b['probability_sum'] - b['positive_count'])
        for b in total['bins']) / n if n else None)
    return result


def score_candidates(candidates, chosen, relations):
    """Oracle minimum is only over active classes UNION the actual chosen class."""
    retained = [c for c in candidates if c['retained']]
    if not retained or len({c['handle'] for c in candidates}) != len(candidates):
        raise ValueError('distinct classes and nonempty retained set required')
    by_handle = {c['handle']: c for c in candidates}
    if chosen not in by_handle:
        raise ValueError('actual selected class must be evaluated')
    errors = {c['handle']: relation_error_units(c['roots'], relations) for c in candidates}
    map_class = min(retained, key=lambda c: (-c['log_weight'], c['sha256']))
    best = min(retained, key=lambda c: (errors[c['handle']], c['sha256']))
    union = min(candidates, key=lambda c: (errors[c['handle']], c['sha256']))
    value = dict(scored_rows=len(relations), chosen_error_units=errors[chosen],
        MAP_error_units=errors[map_class['handle']], best_retained_error_units=errors[best['handle']],
        best_union_error_units=errors[union['handle']],
        **calibration([c['roots'] for c in retained], [c['log_weight'] for c in retained], relations))
    audit = dict(chosen_handle=chosen, MAP_handle=map_class['handle'], best_retained_handle=best['handle'],
        best_union_handle=union['handle'], chosen_in_retained=by_handle[chosen]['retained'],
        candidates=[dict(handle=c['handle'], sha256=c['sha256'], log_weight=c['log_weight'],
                         retained=c['retained'], error_units=errors[c['handle']]) for c in candidates])
    return value, audit


class LabelAudit:
    def __init__(self, checker, shard, *, max_prefix_visits=4_000_000):
        if type(max_prefix_visits) is not int or max_prefix_visits < 1:
            raise ValueError('positive prefix visit cap required')
        self.checker, self.shard, self.cap = checker, shard, max_prefix_visits
        self.targets = []; self.visits = 0
        self.totals = {name: zero_totals() for name in ('decision_scope', 'new_rows_only')}
        self.events = []

    def check(self, audit, prediction):
        old_n = len(self.checker.observations)
        # Pure prediction/raw-state verification happens before labels are read.
        self.checker.check(audit, prediction)
        n = len(self.checker.observations)
        if n > self.shard.record['nodes']:
            raise ValueError('replay exceeds label shard')
        for i in range(old_n, n):
            context, target = self.shard.example(i)
            if (digest(asdict(context.observations[-1])) != self.checker.raw_hashes[i]
                    or context.decision_us != prediction['decision_timestamp_us']
                    or tuple(p for p, _ in self.checker.rows[i]) != (-1, *context.indices[:-1])):
                raise ValueError('training-label raw identity, arrival context or support differs from frozen replay')
            self.targets.append(tuple((parent, truth) for parent, truth, known in
                zip(context.indices[:-1], target.positives[1:], target.known[1:]) if known))
        scope = audit['decision_indices']
        if (len(set(scope)) != len(scope) or any(type(i) is not int or not 0 <= i < n for i in scope)):
            raise ValueError('valid unique decision scope required')
        covered = set(); event_totals = {k: zero_totals() for k in self.totals}; summaries = []
        for summary in audit['components']:
            component, depth = summary['component'], summary['nodes']
            members = [r[0] for r in self.checker.db.execute('SELECT global_i FROM component_members '
                'WHERE component=? AND local_i<? ORDER BY local_i', (component, depth))]
            local = {g: i for i, g in enumerate(members)}
            indices = summary['decision_indices']
            if (len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < depth for i in indices)):
                raise ValueError('valid local decision indices required')
            actual = {members[i] for i in indices}
            if actual != set(scope).intersection(members) or covered.intersection(actual):
                raise ValueError('component decision scope differs from global scope')
            covered.update(actual)
            factors = ForestFactors(tuple(self.checker.observations[g].node for g in members),
                tuple(tuple((-1 if p == -1 else local[p], w) for p, w in self.checker.rows[g]) for g in members))
            active = summary['active']
            if not active or len({b['handle'] for b in active}) != len(active):
                raise ValueError('distinct nonempty retained classes required')
            candidates = []
            records = [dict(b, retained=True) for b in active]
            if summary['output_handle'] not in {b['handle'] for b in active}:
                records.append(dict(handle=summary['output_handle'], sha256=summary['output_sha256'], retained=False))
            if self.visits + depth * len(records) > self.cap:
                raise ValueError('retained prefix visit cap exceeded; no partial result acceptance')
            for record in records:
                choices = state_audit.prefix_choices(self.checker.db, self.checker.sequence, component,
                    record['handle'], factors, record['sha256'])
                weight, roots = class_weight(factors, choices)
                if record['retained'] and not math.isclose(weight, record['log_weight'], rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError('independently summed root-class weight differs')
                candidates.append(dict(record, log_weight=weight, roots=roots))
            self.visits += depth * len(records)
            # The decoder's root-label risk is a different loss from this audit's
            # known pair-relation error. Do not reinterpret its risk bound here.
            views = {}
            for name in self.totals:
                queried = sorted(actual if name == 'decision_scope' else actual.intersection(range(old_n, n)))
                relations = [(local[g], tuple((local[p], truth) for p, truth in self.targets[g]))
                             for g in queried if self.targets[g]]
                value, certificate = score_candidates(candidates, summary['output_handle'], relations)
                accumulate(event_totals[name], value)
                views[name] = dict(rates(value), **certificate)
            summaries.append(dict(component=component, nodes=depth, views=views))
        if covered != set(scope):
            raise ValueError('incomplete decision scope coverage')
        for name in self.totals: accumulate(self.totals[name], event_totals[name])
        event = dict(sequence_id=audit['sequence_id'], frame_id=audit['event_id'],
            observation_count=n, decision_nodes=len(scope), factor_rows_sha256=audit['factor_rows_sha256'],
            views={name: rates(value) for name, value in event_totals.items()}, components=summaries)
        self.events.append(event)
        return event


def run(directory, receipt_sha256, data, manifest_sha256, checkpoint, output, *,
        max_prefix_visits=4_000_000, allow_fixture=False):
    directory, data, checkpoint = Path(directory), Path(data), Path(checkpoint)
    verified = state_audit.inspect(directory, receipt_sha256, allow_fixture=allow_fixture)
    plan = verified['plan']; receipt = json.loads((directory/'receipt.json').read_bytes())
    if verified['backend'] not in state_audit.BACKENDS or len(receipt['sequence_heads']) != 1 or receipt.get('gt_model_inputs') is not False:
        raise ValueError('one complete prediction-only CI identity replay required')
    manifest_path = contained_file(data, 'manifest.json')
    if sha_file(manifest_path) != manifest_sha256 or sha_file(checkpoint) != plan['identity_checkpoint_sha256']:
        raise ValueError('training manifest or frozen checkpoint identity differs')
    manifest, model = json.loads(manifest_path.read_bytes()), json.loads(checkpoint.read_bytes())
    if (manifest['kind'] != DATA_KIND or manifest['split'] != 'train' or model['data_split'] != 'train'
            or model['dataset_sha256'] != manifest_sha256 or manifest['labels_in_model_inputs'] is not False
            or manifest['row_protocol'] != model['row_protocol'] or manifest['row_protocol']['parent_limit'] != 8
            or manifest['row_protocol'] != row_protocol(state_audit.ForestTrackingConfig(**plan['configuration']['state']))
            or manifest['cache_manifest_sha256'] != plan['cache_sha256']):
        raise ValueError('frozen car train label/feature/selection protocol differs')
    if not allow_fixture and (manifest['provenance'].get('full_official_train_verified') is not True
            or len(manifest['sequences']) != 46 or manifest['scheduled_frames'] != 7445):
        raise ValueError('sealed official train-derived labels required')
    sequence, head = next(iter(receipt['sequence_heads'].items()))
    records = [r for r in manifest['shards'] if r['sequence_id'] == sequence]
    if len(records) != 1: raise ValueError('unique selected sequence shard required')
    record = records[0]; shard_path = contained_file(data, record['path'])
    shard = TrainingShard(shard_path, record, 8)
    sources = {p: sha_file(ROOT/p) for p in SOURCES}
    required = state_audit.STATE_SOURCES + SOURCES[-3:]
    if any(plan['source_sha256'].get(p) != sources[p] for p in required):
        raise ValueError('raw state or label interpretation source differs from producer')
    if not allow_fixture and (plan['runtime']['python'] != sys.version or plan['runtime']['executable'] != sys.executable
            or plan['runtime']['platform'] != platform.platform()
            or any(os.environ.get(k) != v for k, v in plan['thread_environment'].items())):
        raise ValueError('producer runtime and pinned thread environment required for exact state replay')
    artifacts = {p: sha_file(contained_file(directory, p)) for p in ('plan.json', 'receipt.json',
        'development-inference-receipt.json', 'tracking.jsonl', 'predictions.jsonl', 'frame-timings.jsonl', head['database'])}
    bound = {manifest_path: manifest_sha256, shard_path: record['sha256'], checkpoint: plan['identity_checkpoint_sha256']}
    db_path = contained_file(directory, head['database'])
    if any(Path(str(db_path)+s).exists() for s in ('-wal', '-journal')): raise ValueError('closed non-WAL database required')
    output = state_audit._directory(output)
    _new_json(output/'plan.json', dict(kind='retained_partial_label_diagnostic_plan_v1', source_sha256=sources,
        replay_directory=str(directory.absolute()), replay_artifact_sha256=artifacts, label_manifest_sha256=manifest_sha256,
        label_shard_sha256=record['sha256'], checkpoint_sha256=plan['identity_checkpoint_sha256'],
        max_prefix_visits=max_prefix_visits, loss_units_per_row=UNITS, label_manifest_path=str(manifest_path.absolute()),
        runtime=dict(python=sys.version, executable=sys.executable, numpy=state_audit.np.__version__,
                     scipy=state_audit.scipy.__version__, thread_environment=plan['thread_environment']),
        cohort_mode=plan['cohort_mode'], no_inference_mutation=True, labels_used_only_posthoc=True,
        official_metric=False, paper_eligible=False))
    started = time.monotonic(); audit = None
    try:
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)) as db:
            db.set_authorizer(state_audit.read_only_tables)
            checker = state_audit.FreshStateAudit(db, sequence, plan['configuration'],
                                                require_cache_receipts=not allow_fixture)
            audit = LabelAudit(checker, shard, max_prefix_visits=max_prefix_visits)
            with (directory/'tracking.jsonl').open('rb') as lines, (directory/'predictions.jsonl').open('rb') as predictions:
                for line in lines:
                    encoded = predictions.readline()
                    if not encoded: raise ValueError('prediction stream shorter than audit')
                    audit.check(json.loads(line)['tracking'], json.loads(encoded))
                    if len(audit.events) % 25 == 0:
                        print(json.dumps(dict(verified_frames=len(audit.events), prefix_visits=audit.visits)), flush=True)
                if predictions.readline(): raise ValueError('extra prediction event')
            if (len(audit.events) != len(verified['events']) or len(audit.events) != head['frames']
                    or checker.prediction_head != head['prediction_sha256']
                    or len(checker.observations) != record['nodes']
                    or db.execute('SELECT count(*) FROM observations').fetchone()[0] != record['nodes']):
                raise ValueError('complete label/raw/event coverage differs')
        if (any(sha_file(p) != h for p, h in bound.items())
                or any(sha_file(contained_file(directory, p)) != h for p, h in artifacts.items())
                or any(sha_file(ROOT/p) != h for p, h in sources.items())):
            raise ValueError('sealed label diagnostic inputs or source changed')
        result = dict(kind='retained_partial_label_diagnostic_v1', status='complete',
            plan_sha256=sha_file(output/'plan.json'), backend=verified['backend'], frames=len(audit.events),
            observations=record['nodes'], raw_row_reason_counts=record['reason_counts'], prefix_visits=audit.visits,
            factor_stream_sha256=verified['factor_stream_sha256'], views={k: rates(v) for k, v in audit.totals.items()},
            events=audit.events, elapsed_seconds=time.monotonic()-started, loss_units_per_row=UNITS,
            repeated_history_rows_are_not_independent_samples=True, calibration_conditions_on_retained_set=True,
            calibration_accounts_for_omitted_mass=False, local_labels_prove_pruning=False,
            raw_GT_boxes_or_ids_read=False, derived_training_labels_read=True, labels_in_inference=False,
            native_tracking_metrics=False, complete_identity_truth=False, paper_eligible=False)
        _new_json(output/'diagnostic.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'events'}, sort_keys=True), flush=True)
        return result
    except Exception as error:
        _new_json(output/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
            verified_frames=0 if audit is None else len(audit.events), partial_events_not_acceptance=True,
            plan_sha256=sha_file(output/'plan.json')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('replay', 'data', 'checkpoint', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--max-prefix-visits', type=int, default=4_000_000)
    args = parser.parse_args()
    run(args.replay, args.receipt_sha256, args.data, args.manifest_sha256, args.checkpoint, args.output,
        max_prefix_visits=args.max_prefix_visits)
