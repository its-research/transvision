#!/usr/bin/env python3
"""Finite-sample label diagnostics for COMPLETE real-train teacher leaves.

No parameter fitting, GT access, row filtering or training export. One row is
a candidate at a recovery decision; a group is one decision, not an independent
scene. Exact-feature repetitions and conditional target variation are described,
not treated as independent samples or an out-of-sample Bayes error certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.event_track_v2x import assemble_allocation_teachers as assembly
from transvision.models.event_track_v2x.allocation_policy import FEATURES, RECIPE, TARGET
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json

TOLERANCE = 1e-12


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric teacher value required')
    return float(value)


def quantiles(values):
    return dict(zip(('minimum', 'p50', 'p90', 'p99', 'maximum'),
                    np.quantile(values, [0, .5, .9, .99, 1]).tolist())) if values else None


class Profile:
    def __init__(self):
        self.counts = Counter(); self.kinds = Counter(); self.features = {}
        self.abs_targets = []; self.ranges = []; self.opportunities = []
        self.zero_mse_sum = 0.; self.events = set(); self.signal_events = set()

    def add(self, event, step):
        group = step['allocation_training']; rows = group['candidates']
        if (group['feature_recipe'] != RECIPE or group['target_recipe'] != TARGET
                or group['labels_are_model_not_true_risk'] is not True
                or group['future_or_gt_inputs'] is not False
                or group['behavior'] != 'weighted_eta_deterministic'
                or not 1 <= len(rows) <= 4096):
            raise ValueError('bounded model-only teacher candidate group required')
        components = [r['component'] for r in rows]
        if (any(type(c) is not int or c < 0 for c in components)
                or len(set(components)) != len(rows) or step['component'] not in components):
            raise ValueError('distinct candidates and observed behavior selection required')
        values = []
        for row in rows:
            x = [finite(v) for v in row['features']]
            y, before, after = map(finite, (row['target'], row['model_bound_before'], row['model_bound_after']))
            charged = row['charged_steps']; requested = row['operation']['requested_steps']
            if (len(x) != len(FEATURES) or not 0 <= x[0] <= 1 or abs(y) > 1 + TOLERANCE
                    or not 0 <= before <= 1 or not 0 <= after <= 1
                    or type(charged) is not int or type(requested) is not int
                    or not 0 <= charged <= requested or requested < 1
                    or abs(y - x[0]*(before-after)/max(1, charged)) > TOLERANCE):
                raise ValueError('valid feature/target domain, cost and arithmetic required')
            if row['component'] == step['component'] and (
                    row['operation']['kind'] != step['kind'] or charged != step['charged_search_steps']):
                raise ValueError('teacher probe and committed selected operation differ')
            # Numeric float64 equality, including normalized signed zero. This
            # is not a claim that the unrecorded full solver state is identical.
            key = struct.pack('<' + 'd'*len(x), *(0. if v == 0 else v for v in x))
            cell = self.features.setdefault(key, [0, 0., 0., 0., y, y])
            weight = 1./len(rows); total = cell[1] + weight
            delta = y - cell[2]; mean = cell[2] + weight/total*delta
            cell[3] += weight*delta*(y-mean)
            cell[0] += 1; cell[1] = total; cell[2] = mean
            cell[4] = min(cell[4], y); cell[5] = max(cell[5], y)
            self.counts.update(rows=1, positive=int(y > TOLERANCE), negative=int(y < -TOLERANCE),
                near_zero=int(abs(y) <= TOLERANCE), zero_weight=int(x[0] == 0),
                unchanged_bound=int(abs(before-after) <= TOLERANCE), zero_charged_steps=int(charged == 0),
                bound_before_at_least_099=int(before >= .99))
            self.kinds[row['operation']['kind']] += 1
            self.abs_targets.append(abs(y)); values.append(y)
        spread = max(values)-min(values); chosen = values[components.index(step['component'])]
        opportunity = max(values)-chosen
        self.ranges.append(spread); self.opportunities.append(opportunity)
        self.zero_mse_sum += math.fsum(v*v for v in values)/len(values)
        self.counts.update(groups=1, single_candidate_groups=int(len(rows) == 1),
            all_near_zero_groups=int(all(abs(v) <= TOLERANCE for v in values)),
            ranking_signal_groups=int(len(rows) > 1 and spread > TOLERANCE),
            positive_opportunity_groups=int(opportunity > TOLERANCE))
        self.events.add(event)
        if len(rows) > 1 and spread > TOLERANCE:
            self.signal_events.add(event)

    def result(self):
        c = self.counts; rows, groups = c['rows'], c['groups']
        repeated = [v for v in self.features.values() if v[0] > 1]
        aliases = [v for v in repeated if v[5]-v[4] > TOLERANCE]
        return dict(counts=dict(c), events_with_groups=len(self.events),
            events_with_ranking_signal=len(self.signal_events),
            near_zero_row_fraction=c['near_zero']/rows if rows else None,
            ranking_signal_group_fraction=c['ranking_signal_groups']/groups if groups else None,
            one_step_positive_opportunity_group_fraction=c['positive_opportunity_groups']/groups if groups else None,
            operation_candidate_rows=dict(self.kinds), absolute_target_quantiles=quantiles(self.abs_targets),
            within_group_target_range_quantiles=quantiles(self.ranges),
            one_step_opportunity_quantiles=quantiles(self.opportunities),
            exact_numeric_feature_vectors=len(self.features), repeated_feature_vectors=len(repeated),
            excess_feature_repetitions=rows-len(self.features),
            feature_vectors_with_target_range_above_tolerance=len(aliases),
            rows_in_conflicting_feature_vectors=sum(v[0] for v in aliases),
            zero_predictor_equal_group_mse=self.zero_mse_sum/groups if groups else None,
            empirical_exact_feature_equal_group_mse_floor=math.fsum(max(0., v[3]) for v in self.features.values())/groups if groups else None,
            empirical_feature_floor_is_not_generalization_bound=True,
            repetitions_are_not_independent_samples=True,
            opportunity_is_model_one_step_not_native_metric_or_sequential_gain=True)


def profile_events(records, profiles):
    seen = set(); executions = hits = rows = 0
    for record in records:
        a = record['tracking']; event = (a['sequence_id'], a['event_id'])
        if (event in seen or a.get('beam_recovery_allocation') is not True
                or a.get('training_trace_only') is not True or a.get('offline_counterfactual_probes') is not True
                or a.get('allocation_trace_field') != 'recovery_allocation_trace'
                or a.get('priority_changes_only_extra_recovery_order') is not True):
            raise ValueError('unique frozen beam teacher event required')
        seen.add(event); event_rows = 0
        for step in a['recovery_allocation_trace']:
            for profile in profiles:
                profile.add(event, step)
            event_rows += len(step['allocation_training']['candidates'])
        actual = a['teacher_probe_executions']; cached = a['teacher_probe_cache_hits']
        if (any(type(v) is not int or v < 0 for v in (actual, cached))
                or actual+cached != event_rows or a['priority_feature_rows'] != event_rows):
            raise ValueError('probe execution/cache/feature counts differ from candidate rows')
        executions += actual; hits += cached; rows += event_rows
    return dict(events=len(seen), candidate_rows=rows, probe_executions=executions, probe_cache_hits=hits,
        probe_cache_hit_fraction=hits/rows if rows else None)


def audit(references, metadata, output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new non-symlink diagnostic output required')
    expected, pair_sha = assembly._pairs(metadata, False)
    if not references or len(references) > len(expected):
        raise ValueError('one or more complete train sequence leaves required')
    leaves = [assembly._leaf(p, h, expected, pair_sha, False) for p, h in references]
    if (len({r['scene'] for r in leaves}) != len(leaves)
            or any(assembly._common(r['plan']) != assembly._common(leaves[0]['plan']) for r in leaves)
            or any(r['plan']['configuration'].get('enable_recovery') is not True for r in leaves)):
        raise ValueError('distinct same-model same-configuration beam recovery leaves required')
    sources = {str(Path(__file__)): sha_file(__file__), str(Path(assembly.__file__)): sha_file(assembly.__file__)}
    total = Profile(); sequences = {}; input_sha = {str(Path(metadata).absolute()): pair_sha}
    for leaf in sorted(leaves, key=lambda r: r['scene']):
        local = Profile()
        with (leaf['path']/'tracking.jsonl').open('rb') as stream:
            counts = profile_events(map(json.loads, stream), [local, total])
        if counts['events'] != leaf['receipt']['completed_frames']:
            raise ValueError('diagnostic did not cover complete leaf')
        sequences[leaf['scene']] = dict(local.result(), **counts, receipt_sha256=leaf['input']['receipt_sha256'])
        input_sha.update({str(p): h for p, h in leaf['artifacts'].items()})
        print(json.dumps({'sequence': leaf['scene'], **counts,
                          'ranking_signal_groups': local.counts['ranking_signal_groups']}), flush=True)
    assembly._unchanged(leaves)
    if any(sha_file(Path(p)) != h for p, h in {**input_sha, **sources}.items()):
        raise ValueError('teacher diagnostic evidence changed')
    result = dict(kind='complete_leaf_allocation_label_quality_v1', status='complete',
        grain='candidate_within_recovery_decision_group_within_event_within_train_sequence',
        numerical_tolerance=TOLERANCE, feature_recipe=RECIPE, target_recipe=TARGET,
        completed_sequence_count=len(sequences), expected_train_sequences=len(expected),
        completed_events=sum(r['events'] for r in sequences.values()),
        expected_train_events=sum(map(len, expected.values())),
        missing_train_sequences=sorted(set(expected)-set(sequences)),
        complete_official_train_cohort=len(leaves) == len(expected),
        sequences=sequences, pooled_profile=total.result(),
        source_sha256=sources, input_sha256=input_sha,
        candidate_rows_filtered=False, parameter_training_performed=False,
        GT_or_validation_or_test_labels_read=False, native_tracking_improvement_verified=False,
        teacher_behavior_equivalence_verified=False, independent_sample_count_claimed=False, paper_eligible=False)
    output.mkdir(); _new_json(output/'quality.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'completed_sequence_count', 'completed_events', 'pooled_profile')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', nargs=2, action='append', required=True, metavar=('DIRECTORY', 'RECEIPT_SHA256'))
    parser.add_argument('--cooperative-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.teacher, args.cooperative_metadata, args.output)
