import json
import math

import pytest

from tools.event_track_v2x import audit_node_beam_pruning as tool


def case():
    # The third observation shares a source/frame with the first. Linking both
    # to one root is forbidden. Evidence available later IN THIS SAME EVENT can
    # therefore reverse the first extension's ranking.
    rows = [[(-1, 0.)], [(-1, math.log(.4)), (0, math.log(.6))], [(-1, math.log(.01)), (1, 0.)]]
    slots = [(0, 'f'), (1, 'f'), (0, 'f')]
    initial = [tool.score_prefix((-1,), rows, slots, 'root')]
    return initial, rows, slots


def test_exact_same_predecessor_event_oracle_exposes_within_event_pruning():
    initial, rows, slots = case()
    value = tool.enumerate_suffix(initial, rows, slots, 1, (-1, -1, 1), cap=5)
    assert value['exact_complete_classes'] == 3 and value['exact_generated_prefixes'] == 5
    assert value['node_beam_generated_prefixes'] == 3 and value['target_exact_rank'] == 1
    assert value['target_log_weight'] == math.log(.4)
    assert value['node_pruning_stages'] == [
        dict(depth=2, generated_classes=2, retained_classes=1, target_prefix_generated=True,
             target_prefix_rank=2, target_prefix_retained=False),
        dict(depth=3, generated_classes=1, retained_classes=1, target_prefix_generated=False,
             target_prefix_rank=None, target_prefix_retained=False)]
    assert not value['complete_history_posterior']


def test_event_oracle_never_returns_partial_success_on_cap():
    initial, rows, slots = case()
    with pytest.raises(ValueError, match='cap exhausted'):
        tool.enumerate_suffix(initial, rows, slots, 1, (-1, -1, 1), cap=4)


def test_equivalent_parent_paths_are_summed_not_maxed_or_counted_as_distinct_classes():
    rows = [[(-1, 0.)], [(-1, 0.), (0, 0.)], [(-1, -3.), (0, math.log(2.)), (1, math.log(3.))]]
    options = dict(tool.transitions((-1, 0), rows, [(0, 'a'), (0, 'b'), (0, 'c')]))
    assert list(options) == [-1, 0]
    assert math.isclose(options[0], math.log(5.), rel_tol=1e-15)


def test_same_source_frame_cannot_share_identity_root():
    _, rows, slots = case()
    assert list(tool.transitions((-1, 0), rows, slots)) == [(-1, math.log(.01))]
    assert dict(tool.transitions((-1, -1), rows, slots))[1] == 0.
    with pytest.raises(ValueError, match='legal canonical'):
        tool.score_prefix((-1, 0, 1), rows, slots, 'root')


@pytest.mark.parametrize('bad', ['empty', 'duplicate', 'wrong-target', 'zero-width', 'zero-cap'])
def test_unverifiable_enumeration_contracts_rejected(bad):
    initial, rows, slots = case(); target = (-1, -1, 1); width = 1; cap = 100
    if bad == 'empty': initial = []
    elif bad == 'duplicate': initial += initial
    elif bad == 'wrong-target': target = (-1, 0, 1)
    elif bad == 'zero-width': width = 0
    else: cap = 0
    with pytest.raises(ValueError): tool.enumerate_suffix(initial, rows, slots, width, target, cap=cap)


def test_historical_factor_reconstruction_uses_no_later_rows(tmp_path):
    component = dict(component=1, nodes=1, active=[{'handle': 1}], merge_retained_beams_only=False)
    records = [
        dict(event_id='a', observation_count=1, appended_rows=[[[-1, -.1]]], rescored_rows=[], components=[component]),
        dict(event_id='b', observation_count=2, appended_rows=[[[-1, -.3], [0, -.4]]],
             rescored_rows=[[0, [[-1, -.2]]]], components=[dict(component, nodes=2)]),
        dict(event_id='c', observation_count=2, appended_rows=[], rescored_rows=[[0, [[-1, 100.]]]], components=[])]
    path = tmp_path / 'tracking.jsonl'
    path.write_text(''.join(json.dumps({'tracking': r}) + '\n' for r in records))
    event, current, previous, rows = tool.historical_event(path, 'b', 1)
    assert event['event_id'] == 'b' and previous['nodes'] == 1 and current['nodes'] == 2
    assert rows == [[[-1, -.2]], [[-1, -.3], [0, -.4]]]
    with pytest.raises(ValueError, match='prior retained'):
        tool.historical_event(path, 'a', 1)
    with pytest.raises(ValueError, match='absent'):
        tool.historical_event(path, 'missing', 1)


@pytest.mark.parametrize('bad', ['append', 'future-rescore', 'merge'])
def test_historical_contract_does_not_infer_missing_or_merged_beam(tmp_path, bad):
    component = dict(component=1, nodes=1, merge_retained_beams_only=False)
    a = dict(event_id='a', observation_count=1, appended_rows=[[[-1, 0.]]], rescored_rows=[], components=[component])
    b = dict(event_id='b', observation_count=2, appended_rows=[[[-1, 0.]]], rescored_rows=[], components=[dict(component)])
    if bad == 'append': b['observation_count'] = 3
    elif bad == 'future-rescore': b['rescored_rows'] = [[1, [[-1, 0.]]]]
    else: b['components'][0]['merge_retained_beams_only'] = True
    path = tmp_path / 'tracking.jsonl'; path.write_text(''.join(json.dumps({'tracking': r}) + '\n' for r in (a, b)))
    with pytest.raises(ValueError): tool.historical_event(path, 'b', 1)
