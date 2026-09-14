#!/usr/bin/env python3
"""Connected-star counterexample: history truncation vs current action risk.

This exact finite-model diagnostic is not a dataset result or a new tracker.
No GT, future observations, parameter training or external services are used.
It isolates an uninformative risk bound even when the partition bound is exact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.allocation_training import _directory, allocation_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.forest_tracking import RawIdentityDetection, ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import IdentityNode, ForestFactors
from transvision.models.event_track_v2x.completion_component_tracking import CompletionComponentTracker, PersistentCompletionConfig
from transvision.models.event_track_v2x.relaxed_identity_risk import action_risk_certificate


def star_model(bits, birth_ratio=.001):
    if type(bits) is not int or not 0 <= bits <= 32 or not 0 < birth_ratio < 1:
        raise ValueError('zero to 32 bits and positive birth ratio below one required')
    raw = []
    for index in range(bits+2):
        timestamp = 1_000_000+1000*index
        node = IdentityNode(f'node-{index}', index % 2, timestamp, timestamp+10, f'frame-{index}')
        raw.append(RawIdentityDetection('star', node, 0, timestamp,
            [0., 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9)*.2, .8, np.zeros(203), 'a'*64))
    rows = (((-1, 0.),),)+tuple(((-1, 0.), (0, 0.)) for _ in range(bits))
    rows += (((-1, math.log(birth_ratio)), (0, 0.)),)
    return tuple(raw), rows


def diagnose_case(directory, bits, *, retained=4, birth_ratio=.001):
    if type(retained) is not int or retained < 1 or retained > 64:
        raise ValueError('one to 64 explicit histories required')
    raw, rows = star_model(bits, birth_ratio)
    # Distinct source/frame slots make every parent configuration legal.
    assert len({(o.node.source_id, o.node.frame_id) for o in raw}) == len(raw)
    count = min(retained, 2**bits)
    config = PersistentCompletionConfig(state=ForestTrackingConfig(
        active_limit=count, expansion_budget=0, window_us=1))
    tracker = CompletionComponentTracker(Path(directory)/f'bits-{bits}.sqlite', sequence_id='star', config=config)
    reference = raw[-1].state_us
    first = tracker.step(raw, rows, frame_id='current', event_id='current', reference_us=reference,
                         decision_us=reference+10)
    if len(tracker.kernels) != 1 or first.audit['decision_indices'] != [bits+1]:
        raise ValueError('single connected component and current-node-only loss required')
    kernel = next(iter(tracker.kernels.values()))
    tracker.db.execute('SAVEPOINT exact_top_k_diagnostic')
    try:
        for number in range(count):
            choices = [-1]+[0 if number & (1 << i) else -1 for i in range(bits)]+[0]
            handle = 0
            for parent in choices:
                if parent not in kernel._choices(handle):
                    raise ValueError('counterexample proposed an illegal history')
                handle = kernel._child(handle, parent)
            kernel.seed_complete_action(handle, reference+10)
        active, weights, lower, upper, eta = kernel._mass()
        fallback = next(iter(first.audit['components']))['output_handle']
        action, candidate = kernel._decode(active, weights, lower, eta, (bits+1,), fallback, fallback_on_risk=False)
        safe_action, actual = kernel._decode(active, weights, lower, eta, (bits+1,), fallback, fallback_on_risk=True)
        chosen_root = kernel._prefix(action).root
        actual_root = kernel._prefix(safe_action).root
        match_probability = 1/(1+birth_ratio)
        def exact_risk(root):
            return 1-(match_probability if root == 0 else 1-match_probability if root == bits+1 else 0.)
        optimum = 1-match_probability
        exact_log_z = bits*math.log(2)+math.log1p(birth_ratio)
        exact_eta = 1-count*math.exp(-exact_log_z)
        if (len(active) != count or abs(lower-math.log(count)) > 1e-12
                or abs(upper-exact_log_z) > 1e-9 or abs(eta-exact_eta) > 1e-12
                or chosen_root != 0):
            raise ValueError('kernel mass or action disagrees with exact model')
        direct = action_risk_certificate(ForestFactors(tuple(o.node for o in raw), rows),
            kernel.parents(action), scope=(bits+1,))
        return dict(history_ambiguity_bits=bits, legal_histories=2**(bits+1), retained_histories=count,
            one_connected_support_component=True, current_loss_nodes=1,
            log_partition_exact=exact_log_z, log_partition_upper=upper,
            exact_omitted_history_mass=exact_eta, omitted_history_mass_upper=eta,
            current_match_probability=match_probability, candidate_root=chosen_root,
            candidate_exact_model_regret=exact_risk(chosen_root)-optimum,
            candidate_reported_regret_upper=candidate['risk_bound'],
            direct_current_action_certificate=direct,
            threshold_selected_root=actual_root, threshold_selected_model_regret=exact_risk(actual_root)-optimum,
            threshold_selected_regret_upper=actual['risk_bound'], risk_threshold=config.state.max_model_regret,
            proposal_seeding_is_diagnostic_not_budgeted_online_search=True)
    finally:
        tracker.db.execute('ROLLBACK TO exact_top_k_diagnostic')
        tracker.db.execute('RELEASE exact_top_k_diagnostic')
        tracker._restore_runtime()
        if tracker.meta['prediction_sha256'] != first.prediction['commit_sha256']:
            raise ValueError('diagnostic modified committed output')
        tracker.close()


def run(output):
    output = _directory(output)
    sources = allocation_sources()
    sources['tools/event_track_v2x/diagnose_history_truncation.py'] = sha_file(Path(__file__))
    risk_source = 'transvision/models/event_track_v2x/relaxed_identity_risk.py'
    sources[risk_source] = sha_file(ROOT/risk_source)
    try:
        cases = [diagnose_case(output, bits) for bits in (0, 1, 2, 4, 8, 12, 16, 24, 32)]
        if any(sha_file(ROOT/path) != digest for path, digest in sources.items()):
            raise ValueError('counterexample sources changed during execution')
        result = dict(kind='connected_history_truncation_counterexample_v1', status='complete',
            source_sha256=sources, cases=cases, finite_model_only=True, gt_or_future_inputs=False,
            real_tracking_evaluation=False, paper_eligible=False)
        _new_json(output/'counterexample.json', result)
        print(json.dumps(result, sort_keys=True))
        return result
    except BaseException as error:
        _new_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
