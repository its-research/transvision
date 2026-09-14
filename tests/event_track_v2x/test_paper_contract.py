import itertools
import json

import numpy as np
import pytest
from test_detection_cache_v2 import _sources

from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.paper_decision import conditional_action
from transvision.models.event_track_v2x.paper_protocol import LEGACY, PAPER, PaperProtocol, training_partition


def test_candidate_competition_is_preserved():
    scores = np.r_[np.full(64, .9), .8]
    classes = np.r_[np.full(64, 2, dtype=np.int64), 0]
    assert PaperProtocol('spd', 'train').select(scores, classes).tolist() == list(range(64))
    assert PaperProtocol('spd', 'train', LEGACY).select(scores, classes).tolist() == [64]


@pytest.mark.parametrize('dataset,split', [('spd', 'test'), ('spd', 'official_test'), ('v2v4real', 'val')])
def test_split_is_explicit(dataset, split):
    with pytest.raises(ValueError):
        PaperProtocol(dataset, split)


def test_official_test_cannot_train():
    with pytest.raises(ValueError):
        PaperProtocol('v2v4real', 'official_test').require_train()
    assert training_partition(['s1', 's2', 's3']) == training_partition(['s3', 's2', 's1'])


def histories(f):
    # Independent Cartesian enumeration, deliberately no factors.children().
    result = []
    for choices in itertools.product(*(tuple(p for p, _ in row) for row in f.rows)):
        roots, slots, valid = [], {}, True
        for i, parent in enumerate(choices):
            root = i if parent < 0 else roots[parent]
            slot = (f.nodes[i].source_id, f.nodes[i].frame_id)
            if slot in slots.setdefault(root, set()):
                valid = False
            slots[root].add(slot)
            roots.append(root)
        if valid:
            result.append((choices, roots))
    return result


def test_full_legal_action_matches_independent_exhaustive_oracle():
    rng = np.random.default_rng(1337)
    nodes = tuple(IdentityNode(str(i), i % 2, i, i, str(i // 2)) for i in range(5))
    f = ForestFactors(nodes, tuple(tuple((p, 0.) for p in range(-1, i)) for i in range(5)))
    all_actions = histories(f)
    for _ in range(30):
        subset = rng.choice(len(all_actions), 5, replace=False)
        active = [all_actions[i][0] for i in subset]
        weights = rng.dirichlet(np.ones(5))

        def loss(roots):
            return sum(w * sum(a != b for a, b in zip(roots, all_actions[i][1])) / 5 for i, w in zip(subset, weights))

        action, audit = conditional_action(f, active, np.log(weights), range(5), budget=10000)
        actual = next(roots for a, roots in all_actions if a == action)
        assert loss(actual) == pytest.approx(min(loss(roots) for _, roots in all_actions))
        assert audit['optimization_gap'] == pytest.approx(0.)


def test_selected_nonexplicit_action_reconstructs_actual_raw_states(tmp_path):
    from test_forest_tracking import observation

    from transvision.models.event_track_v2x.detection_cache_v2 import canonical
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig, replay_forest_states
    from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
    raw = tuple(observation(str(i), i * .2, source=i % 2, index=i // 2) for i in range(5))
    f = ForestFactors(tuple(o.node for o in raw), tuple(tuple((p, 0.) for p in range(-1, i)) for i in range(5)))
    all_actions = histories(f)
    rng = np.random.default_rng(1337)
    for _ in range(100):
        subset = rng.choice(len(all_actions), 5, replace=False)
        active = [all_actions[i][0] for i in subset]
        weights = np.log(rng.dirichlet(np.ones(5)))
        expected, audit = conditional_action(f, active, weights, range(5))
        if audit['selected_outside_active']:
            break
    else:
        raise AssertionError('fixture failed to exercise nonexplicit action')
    config = PaperForestTrackingConfig(decision_mode='all-legal-hamming', expansion_budget=0, max_model_regret=1.)
    tracker = PersistentForestTracker(tmp_path / 'state.db', sequence_id='0003', config=PersistentForestConfig(state=config))
    tracker.step(raw, f.rows, frame_id='first', reference_us=1_000_000, decision_us=1_100_000, event_id='first')
    handles = []
    for action in active:
        handle = 0
        for parent in action:
            handle = tracker._child(handle, parent)
        handles.append(handle)
    chosen, audit = tracker._decode(handles, weights, 0., .1, tuple(range(5)), handles[0])
    assert tracker.parents(chosen) == expected and audit['selected_outside_active']
    predictions = tracker._predict(chosen, 1_000_000)
    reference, _ = replay_forest_states('0003', raw, f, expected, 1_000_000, config)
    selected = [{k: s[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for s in reference]
    assert canonical(predictions) == canonical(selected)
    assert audit['action_materialization_steps'] == 5
    tracker.close()


def test_zero_budget_retains_legal_incumbent_and_gap():
    nodes = tuple(IdentityNode(str(i), i % 2, i, i, str(i)) for i in range(3))
    f = ForestFactors(nodes, (((-1, 0.), ), ((-1, 0.), (0, 0.)), ((-1, 0.), (0, 0.), (1, 0.))))
    action, audit = conditional_action(f, [(-1, 0, -1), (-1, -1, 1)], [0., 0.], range(3), budget=0)
    f.roots(action)
    assert audit['action_search_steps'] == 0
    assert audit['action_space'] == 'all_supported_legal_histories'


def test_paper_cache_stream_and_training_use_same_all_class_selection(tmp_path):
    from tools.event_track_v2x.build_detection_cache_v2 import build_cache
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache
    from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig as ForestTrackingConfig
    from transvision.models.event_track_v2x.forest_tracking import cache_detections
    from transvision.models.event_track_v2x.forest_training_data import row_protocol
    from transvision.models.event_track_v2x.paper_runtime import default_configuration, replay
    roots, calibration, inputs, output = _sources(tmp_path)
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    cache = VerifiedForestCache(output, sha)
    deliveries = []
    for key, (entry, meta) in cache.index.items():
        if key[0] == '0003':
            deliveries.append(dict(sequence_id=key[0], side=key[1], frame_id=key[2], arrival_us=1_100_000, frame_sha256=json.loads(entry)['frame_sha256']))
    event = dict(sequence_id='0003', frame_id='out0', reference_us=1_100_000, decision_us=1_100_000, event_id='first', deliveries=deliveries)
    protocol = PaperProtocol('spd', json.loads(cache.manifest_json)['split'])
    config = default_configuration('geometry')
    assert row_protocol(ForestTrackingConfig(**config['state']))['candidate_protocol'] == PAPER
    receipt = replay(cache, [event], tmp_path / 'paper-run', protocol=protocol, configuration=config, scorer=GeometryForestScorer(), model_binding={}, fixture=True)
    assert receipt['completed_events'] == 1 and receipt['fixture']
    audit = json.loads((tmp_path / 'paper-run/audit.jsonl').read_bytes())
    expected = sum(
        len(cache_detections(cache.load_arrived(CacheDelivery(**d), 1_100_000), arrival_us=1_100_000, decision_us=1_100_000, origin_us=0, candidate_protocol=PAPER))
        for d in deliveries)
    assert audit['new_observations'] == expected
    assert audit['cache_ingestion']['class_scope'] == ['car', 'bicycle', 'pedestrian']
