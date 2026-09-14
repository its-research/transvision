from copy import deepcopy
import itertools
import json
import math
import os
import subprocess
import sys

import pytest

from tools.event_track_v2x import audit_retained_identity_labels as tool
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV
from test_forest_training_data import prepared_rows
from test_train_inference_diagnostic import pair_file


def factors():
    return ForestFactors(tuple(IdentityNode(str(i), i % 2, i, i, str(i)) for i in range(4)),
        (((-1, -.2),), ((-1, -.3), (0, -.4)), ((-1, -.5), (0, -.6), (1, -.7)),
         ((-1, -.8), (0, -.9), (1, -1.), (2, -1.1))))


def test_class_weight_sums_parent_aliases_against_complete_parent_forest_enumeration():
    f = factors(); expected = {}; representatives = {}
    for choices in itertools.product(*(tuple(p for p, _ in r) for r in f.rows)):
        roots = f.roots(choices)
        weight = math.exp(math.fsum(dict(row)[p] for p, row in zip(choices, f.rows)))
        expected[roots] = expected.get(roots, 0.) + weight
        representatives[roots] = min(representatives.get(roots, choices), choices)
    for roots, choices in representatives.items():
        actual, checked = tool.class_weight(f, choices)
        assert checked == roots
        assert math.exp(actual) == pytest.approx(expected[roots], rel=1e-12)
    with pytest.raises(ValueError, match='minimum parent'):
        tool.class_weight(f, (-1, 0, 1, 2))


def candidate(handle, roots, log_weight, *, retained=True):
    return dict(handle=handle, roots=tuple(roots), log_weight=log_weight,
                sha256=str(handle).zfill(64), retained=retained)


def test_loss_is_root_partition_invariant_and_equal_rows_not_equal_edges():
    roots = (0, 0, 2, 0)
    relations = [(1, ((0, False),)), (3, ((0, True), (1, True), (2, False)))]
    assert tool.relation_error_units(roots, relations) == tool.UNITS
    assert tool.relation_error_units((7, 7, 9, 7), relations) == tool.UNITS
    value, _ = tool.score_candidates([candidate(1, roots, 0.)], 1, relations)
    assert tool.rates(value)['chosen_equal_row_error'] == .5
    assert tool.rates(value)['conditional_equal_edge_Brier'] == .25


def test_score_gap_can_exist_with_the_better_labeled_class_still_retained():
    choices = [candidate(1, (0, 1), math.log(.8)), candidate(2, (0, 0), math.log(.2))]
    relations = [(1, ((0, True),))]
    result, certificate = tool.score_candidates(choices, 1, relations)
    rate = tool.rates(result)
    assert certificate['best_retained_handle'] == 2 and certificate['MAP_handle'] == 1
    assert rate['chosen_equal_row_error'] == 1 and rate['best_retained_equal_row_error'] == 0
    assert rate['chosen_to_union_best_gap'] == 1
    assert rate['conditional_equal_edge_Brier'] == pytest.approx(.64)
    assert rate['conditional_equal_edge_ECE10'] == pytest.approx(.8)


def test_output_outside_retained_set_is_not_a_negative_oracle_regret():
    choices = [candidate(1, (0, 1), 0.), candidate(2, (0, 0), -2., retained=False)]
    result, certificate = tool.score_candidates(choices, 2, [(1, ((0, True),))])
    rate = tool.rates(result)
    assert not certificate['chosen_in_retained']
    assert rate['best_retained_equal_row_error'] == 1
    assert rate['best_union_equal_row_error'] == rate['chosen_equal_row_error'] == 0
    assert rate['chosen_to_union_best_gap'] == 0
    assert rate['conditional_equal_edge_Brier'] == 1


def test_unknown_and_birth_only_rows_do_not_become_zero_error_examples():
    value, _ = tool.score_candidates([candidate(1, (0,), 0.)], 1, [])
    rate = tool.rates(value)
    assert rate['scored_rows'] == rate['known_edges'] == 0
    assert rate['chosen_equal_row_error'] is rate['conditional_equal_edge_Brier'] is None
    assert rate['conditional_equal_edge_ECE10'] is None


def test_no_zero_label_error_does_not_prove_pruning_even_with_complete_support():
    # Earlier gating separated two births. A later node has two correctly
    # positive parents, but a one-parent forest cannot merge their old roots.
    f = ForestFactors(tuple(IdentityNode(str(i), i % 2, i, i, str(i)) for i in range(3)),
                      (((-1, 0.),), ((-1, 0.),), ((-1, 0.), (0, 0.), (1, 0.))))
    complete = []
    for h, choices in enumerate(itertools.product(*(tuple(p for p, _ in row) for row in f.rows))):
        w, roots = tool.class_weight(f, choices)
        complete.append(candidate(h, roots, w))
    assert len(complete) == 3
    result, _ = tool.score_candidates(complete, 0, [(2, ((0, True), (1, True)))])
    assert tool.rates(result)['best_retained_equal_row_error'] == .5
    # Every legal class was retained: increasing K cannot remove this error.


def test_aggregation_preserves_equal_row_denominator_and_fixed_calibration_bins():
    a, _ = tool.score_candidates([candidate(1, (0, 1), 0.)], 1, [(1, ((0, True),))])
    b, _ = tool.score_candidates([candidate(1, (0, 0, 0), 0.)], 1,
        [(1, ((0, True),)), (2, ((0, True), (1, True)))])
    total = tool.zero_totals()
    tool.accumulate(total, a); tool.accumulate(total, b)
    r = tool.rates(total)
    assert r['chosen_equal_row_error'] == pytest.approx(1/3)
    assert r['conditional_equal_edge_Brier'] == .25
    assert r['conditional_equal_edge_ECE10'] == .25
    assert r['bins'][0]['count'] == 1 and r['bins'][9]['count'] == 3


@pytest.fixture
def sealed(prepared_rows, tmp_path, monkeypatch):
    data, _, cache, rows = prepared_rows
    path, _ = pair_file(tmp_path, rows)
    fit = fit_dataset(data, tool.sha_file(data/'manifest.json'), tmp_path/'fit',
        config=FitConfig(epochs=1, batch_size=4, hidden=8, heads=2, dropout=0.), require_full_train=False)
    cp = fit['seeds'][0]
    checkpoint = tmp_path/'fit'/cp['checkpoint_manifest']
    for k, v in THREAD_ENV.items(): monkeypatch.setenv(k, v)
    replay = tmp_path/'replay'
    code = '''import sys
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from tools.event_track_v2x.run_train_inference_diagnostic import run
c,h,m,mh,p,ph,o,s=sys.argv[1:]
run(VerifiedForestCache(c,h),m,mh,p,ph,o,sequence=s,backend='reachable_slot_bound_joint_beam',allow_fixture=True)
'''
    process = subprocess.run([sys.executable, '-c', code, str(cache.root), cache.manifest_sha256,
        str(path), tool.sha_file(path), str(checkpoint.parent), cp['checkpoint_sha256'], str(replay), rows[0]['sequence_id']],
        cwd=tool.ROOT, env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    return replay, tool.sha_file(replay/'development-inference-receipt.json'), data, tool.sha_file(data/'manifest.json'), checkpoint


def test_actual_sealed_fresh_process_fixture_full_coverage_and_no_input_mutation(sealed, tmp_path):
    head = next(iter(json.loads((sealed[0]/'receipt.json').read_bytes())['sequence_heads'].values()))
    before = {p: tool.sha_file(p) for p in (sealed[0]/'predictions.jsonl', sealed[0]/head['database'], sealed[-1])}
    code = 'from tools.event_track_v2x.audit_retained_identity_labels import run; import sys; run(*sys.argv[1:],allow_fixture=True)'
    result = subprocess.run([sys.executable, '-c', code, *map(str, sealed), str(tmp_path/'audit')],
        cwd=tool.ROOT, env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path/'audit'/'diagnostic.json').read_bytes())
    assert report['frames'] == len(report['events']) == 1 and report['status'] == 'complete'
    assert report['derived_training_labels_read'] and not report['labels_in_inference']
    assert not report['paper_eligible'] and not report['complete_identity_truth']
    assert not report['local_labels_prove_pruning'] and report['calibration_conditions_on_retained_set']
    assert all(tool.sha_file(p) == h for p, h in before.items())
    with pytest.raises(ValueError, match='provenance'):
        tool.run(*sealed, tmp_path/'cannot-relabel')
    assert not (tmp_path/'cannot-relabel').exists()
    with pytest.raises(ValueError, match='cap'):
        tool.run(*sealed, tmp_path/'capped', max_prefix_visits=1, allow_fixture=True)
    assert not (tmp_path/'capped'/'diagnostic.json').exists()
    assert json.loads((tmp_path/'capped'/'failure.json').read_bytes())['partial_events_not_acceptance']


@pytest.mark.parametrize('fault', ['manifest_pin', 'shard', 'checkpoint', 'weight', 'scope'])
def test_changed_label_sources_weights_or_scope_rejected(sealed, tmp_path, monkeypatch, fault):
    args = list(sealed)
    if fault == 'manifest_pin': args[3] = 'f'*64
    elif fault == 'checkpoint': args[-1].write_bytes(args[-1].read_bytes()+b' ')
    elif fault == 'shard':
        m = json.loads((args[2]/'manifest.json').read_bytes())
        path = args[2]/m['shards'][0]['path']; path.write_bytes(path.read_bytes()+b'changed')
    else:
        original = tool.LabelAudit.check
        def corrupt(self, audit, prediction):
            audit = deepcopy(audit)
            if fault == 'weight': audit['components'][0]['active'][0]['log_weight'] += .1
            else: audit['components'][0]['decision_indices'] = []
            return original(self, audit, prediction)
        monkeypatch.setattr(tool.LabelAudit, 'check', corrupt)
    with pytest.raises(ValueError): tool.run(*args, tmp_path/'fault', allow_fixture=True)
    assert not (tmp_path/'fault'/'diagnostic.json').exists()
