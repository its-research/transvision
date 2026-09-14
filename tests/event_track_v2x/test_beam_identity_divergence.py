import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from tools.event_track_v2x import audit_beam_identity_divergence as tool


def test_root_projection_preserves_equivalence_not_cross_database_handles():
    assert tool.root_partition([-1, 0, 1, -1]) == (0, 0, 0, 3)
    assert tool.restrict_partition((0, 0, 2, 0), [10, 11, 12, 13], [11, 12, 13]) == (0, 1, 0)


@pytest.mark.parametrize('bad', [[0], [-2], [-1, 1], [-1, True], [-1, .0]])
def test_illegal_identity_choices_rejected(bad):
    with pytest.raises(ValueError, match='backward'):
        tool.root_partition(bad)


@pytest.mark.parametrize('roots,members,old', [((0,), [1, 2], [1]), ((0, 0), [1, 1], [1]),
                                             ((0,), [1], [2]), ((0,), [1], [1, 1])])
def test_incomplete_or_nonhistorical_projection_rejected(roots, members, old):
    with pytest.raises(ValueError): tool.restrict_partition(roots, members, old)


@pytest.fixture
def history():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE component_members(component,local_i,global_i)')
    def component(c, members):
        db.execute(f'CREATE TABLE pc{c}_prefixes(h INTEGER PRIMARY KEY,parent,depth,choice)')
        db.execute(f'INSERT INTO pc{c}_prefixes VALUES(0,NULL,0,NULL)')
        db.executemany('INSERT INTO component_members VALUES(?,?,?)', [(c, i, g) for i, g in enumerate(members)])
    def branch(c, choices):
        parent = 0
        for depth, choice in enumerate(choices, 1):
            h = db.execute(f'SELECT max(h)+1 FROM pc{c}_prefixes').fetchone()[0]
            db.execute(f'INSERT INTO pc{c}_prefixes VALUES(?,?,?,?)', (h, parent, depth, choice)); parent = h
        return parent
    yield db, component, branch
    db.close()


@pytest.mark.parametrize('recovered', [False, True])
def test_history_certificate_distinguishes_current_extension_from_old_dropped_class(history, recovered):
    db, component, branch = history
    component(1, [0, 1, 2, 999])  # Future appended member must not enter old/current projection.
    old = branch(1, [-1, 0]); chosen = branch(1, [-1, -1 if recovered else 0, 0])
    previous = {1: dict(component=1, nodes=2, active=[dict(handle=old)], output_handle=old)}
    now = dict(component=1, nodes=3, output_handle=chosen, predecessors=[1],
               recovery_events=[dict(previous_component=1)] if recovered else [])
    certificate = tool.history_certificate(db, now, previous)
    assert certificate['restored_previous_dropped_output_class'] is recovered
    assert certificate['predecessors'][0]['historical_nodes'] == 2
    now['recovery_events'] = [] if recovered else [dict(previous_component=1)]
    with pytest.raises(ValueError, match='independent historical'):
        tool.history_certificate(db, now, previous)


def test_merge_projects_root_equivalence_to_both_predecessor_maps(history):
    db, component, branch = history
    component(1, [0, 2]); component(2, [1, 3]); component(3, [0, 1, 2, 3, 4, 999])
    a, b = branch(1, [-1, 0]), branch(2, [-1, 0])
    chosen = branch(3, [-1, -1, 0, 1, 0])
    previous = {c: dict(nodes=2, active=[dict(handle=h)], output_handle=h) for c, h in [(1, a), (2, b)]}
    now = dict(component=3, nodes=5, output_handle=chosen, predecessors=[1, 2], recovery_events=[])
    report = tool.history_certificate(db, now, previous)
    assert len(report['predecessors']) == 2 and not report['restored_previous_dropped_output_class']
    now['output_handle'] = branch(3, [-1, -1, -1, 1, 2])
    now['recovery_events'] = [dict(previous_component=1)]
    report = tool.history_certificate(db, now, previous)
    assert [r['outside_previous_retained_and_output'] for r in report['predecessors']] == [True, False]


def test_missing_prefix_or_predecessor_cannot_be_treated_as_recovery(history):
    db, component, branch = history; component(1, [0, 1]); h = branch(1, [-1, 0])
    with pytest.raises(ValueError, match='depth'):
        tool.choices_for(db, 1, h, 1)
    with pytest.raises(ValueError, match='missing historical'):
        tool.history_certificate(db, dict(component=1, nodes=2, output_handle=h,
                                          predecessors=[2], recovery_events=[]), {})


def test_first_intervention_uses_previous_event_not_later_mutated_summary(history, tmp_path):
    db, component, branch = history; component(1, [0, 1, 2, 3])
    old = branch(1, [-1, 0]); same_history = branch(1, [-1, 0, -1]); base = branch(1, [-1, 0, 0])
    weighted = lambda h, w: dict(handle=h, log_weight=w, sha256=str(h))
    first = dict(component=1, nodes=2, predecessors=[1], active=[weighted(old, -1.)],
                 output_handle=old, backbone_output_handle=old, recovery_events=[])
    second = dict(component=1, nodes=3, predecessors=[1], active=[weighted(same_history, -2.)],
        output_handle=same_history, backbone_output_handle=base, backbone_active=[weighted(base, -3.)],
        recovery_events=[], decision={'risk_bound': .9}, recovery_steps=5)
    # A later summary cannot retroactively redefine what was retained at f1.
    third = dict(second, active=[weighted(base, -4.)], output_handle=base, backbone_output_handle=base)
    path = tmp_path / 'tracking.jsonl'
    path.write_text(''.join(json.dumps(dict(tracking=dict(event_id=f'f{i}', sequence_id='s', components=[c]))) + '\n'
                            for i, c in enumerate((first, second, third))))
    result = tool.first_intervention(path, db)
    assert result['first']['frame_id'] == 'f1' and result['direct_intervention_count'] == 1
    assert not result['first']['chosen_in_current_backbone']
    assert result['first']['candidate_log_weight_difference'] == 1.
    certificate = result['first']['history_certificate']
    assert not certificate['restored_previous_dropped_output_class']
    assert certificate['predecessors'][0]['compatible_previous_retained_or_output_handles'] == [old]


def test_native_threshold_uses_first_maximum_not_an_arbitrary_or_fixed_threshold():
    assert tool.selected_index(dict(mota=[None, .8, .8, .5], confidence=[None, .2, .7, .9])) == (1, .2)
    for values in (dict(mota=[None], confidence=[None]), dict(mota=[.1], confidence=[None])):
        with pytest.raises(ValueError): tool.selected_index(values)


def test_switch_events_keep_miss_gap_and_never_expose_gt_ids():
    rows = [dict(type=t, gt_id='private-gt-id', pred_id=p, frame_id=f'f{i}', timestamp_us=i*100000,
                 distance_m=distance) for i, (t, p, distance) in enumerate(
                     [('MATCH', 'p', 0.), ('MISS', None, None), ('SWITCH', 'q', .5)])]
    events = tool.switches(rows)
    assert events[0]['previous_native_match_frame'] == 'f0'
    assert events[0]['elapsed_since_previous_native_match_seconds'] == .2
    assert 'private-gt-id' not in json.dumps(events)
    assert events[0]['gt_key_sha256'] == tool.key_hash('private-gt-id')
    with pytest.raises(ValueError, match='prior association'): tool.switches(rows[2:])
    bad = copy.deepcopy(rows); bad[2]['pred_id'] = 'p'
    with pytest.raises(ValueError, match='prior association'): tool.switches(bad)


def test_real_sealed_native_engine_switches_empty_frame_and_strict_distance(tmp_path):
    python = Path(os.environ.get('RBF_NATIVE_EVALUATOR_PYTHON',
        '/private/tmp/eventtrack-evaluator-v1.zT0Eh9/venv/bin/python'))
    if not python.exists(): pytest.skip('separate sealed native evaluation runtime unavailable')
    command = '''
import copy,json
from tools.event_track_v2x import audit_beam_identity_divergence as tool
_,adapter=tool.evaluation.evaluator()
ground=[];predictions=[]
def box(tid,x):return dict(track_id=tid,class_label='car',mean=[x,0.,0.,4.,2.,1.,0.,0.,0.],score=.5)
for i in range(5):
    header=dict(sequence_id='s',frame_id=f'f{i}',box_reference_timestamp_us=i*100000)
    ground.append(dict(header,ego_translation_world=[0.,0.,0.],objects=[] if i==1 else [box('private-gt',0.)]))
    predictions.append(dict(header,predictions=[] if i==1 else [box('p' if i==0 else 'q',2. if i==3 else 0.)]))
metrics=adapter.compute_metrics(ground,predictions,classes=('car',))
result,records=tool.analyze_native(adapter,ground,predictions,metrics)
assert result['native_event_dataframe_exact'] and result['native_curves_exact']
assert result['selected_native_metrics']['ids']==1
assert [r['frame_id'] for r in result['switch_events']]==['f2']
assert [r['type'] for r in records]==['MATCH','SWITCH','MISS','MATCH']
assert 'private-gt' not in json.dumps(result)
bad=copy.deepcopy(metrics);bad['nuscenes']['ids']+=1
try:tool.analyze_native(adapter,ground,predictions,bad)
except ValueError as error:assert 'headline' in str(error)
else:raise AssertionError('changed headline accepted')
bad=copy.deepcopy(metrics);bad['nuscenes_curves']['car']['confidence'][0]=0.
try:tool.analyze_native(adapter,ground,predictions,bad)
except ValueError as error:assert 'curves' in str(error)
else:raise AssertionError('changed curve accepted')
print(json.dumps(dict(actual_native_runtime=True,records=len(records),switches=len(result['switch_events']))))
'''
    run = subprocess.run([str(python), '-c', command], cwd=tool.ROOT, env=dict(os.environ,
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mpl')), capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout.strip().splitlines()[-1]) == dict(actual_native_runtime=True, records=4, switches=1)
