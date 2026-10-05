import numpy as np
import itertools
import tempfile
import unittest
from pathlib import Path

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig, RawIdentityDetection, replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig, PersistentComponentTracker
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from transvision.models.event_track_v2x.paper_decision import conditional_action


def observation(i):
    when = 1_000_000 + i * 10_000
    features = np.zeros(203)
    features[138] = 1.
    return RawIdentityDetection('capacity', IdentityNode(str(i), i % 2, when, when + 10, str(i)),
                                0, when, [i * .1, 0., 1., 4., 2., 1.5, 0., 0., 0.],
                                np.eye(9) * .2, .8, features, 'a' * 64)


def verify_capacity(tmp_path, component):
    state = PaperForestTrackingConfig(decision_mode='all-legal-hamming', expansion_budget=32,
                                     max_model_regret=1., window_us=1000)
    cls, config_cls = ((PersistentComponentTracker, PersistentComponentConfig) if component
                       else (PersistentForestTracker, PersistentForestConfig))
    config = config_cls(state=state, max_decision_nodes=2)
    tracker = cls(tmp_path / 'capacity.db', sequence_id='capacity', config=config)
    observations, rows, results = [], [], []
    try:
        for i in range(3):
            raw = observation(i)
            row = ((-1, 0.),) if i == 0 else ((-1, 0.), (i - 1, .3))
            observations.append(raw); rows.append(row)
            results.append(tracker.step([raw], [row], frame_id=str(i), reference_us=raw.state_us,
                                        decision_us=raw.node.arrival_us, event_id=str(i)))
        audit = results[-1].audit
        entry = audit['components'][0] if component else audit
        assert len(audit['components']) == 1 if component else True
        decision = entry['decision']
        assert decision['status'] == 'undecided' and decision['resource_limited']
        assert entry['resource_limited']
        assert decision['required_nodes'] == 3 and decision['available_nodes'] == 2
        assert decision['risk_bound'] == 1. and decision['conditional_risk'] is None
        assert not decision['action_search_complete'] and not decision['numeric_certificate']
        assert decision['action_search_steps'] == decision['action_materialization_steps'] == 0
        assert decision['action_space'] == 'all_supported_legal_histories'
        kernel = tracker.kernels[entry['component']] if component else tracker
        handle = entry['output_handle']
        parents = tuple(kernel._prefix(kernel.ancestor(handle, i + 1)).choice for i in range(3))
        factors = ForestFactors(tuple(o.node for o in observations), tuple(rows))
        factors.roots(parents)  # Independently reject illegal parent/slot assignments.
        expected, _ = replay_forest_states('capacity', tuple(observations), factors, parents,
                                            observations[-1].state_us, state)
        selected = [{k: s[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for s in expected]
        assert canonical(results[-1].prediction['predictions']) == canonical(selected)
        assert tracker.n == 3 and tracker.config.max_decision_nodes == 2
        assert tracker.db.execute('SELECT count(*) FROM observations').fetchone()[0] == 3
        assert tracker.db.execute('SELECT count(*) FROM events').fetchone()[0] == 3
        assert tracker.db.execute('SELECT prediction FROM events WHERE ordinal=0').fetchone()[0] == results[0].prediction_json
        assert 'status' not in ((results[1].audit['components'][0] if component else results[1].audit)['decision'])
        before = tracker.prefix_count if not component else kernel.prefix_count
        with unittest.TestCase().assertRaisesRegex(ValueError, 'complete legal history'):
            kernel._decode([handle], [0.], 0., .1, (2,), 0)
        assert kernel.prefix_count == before
    finally:
        tracker.close()


class CapacityTests(unittest.TestCase):
    def test_single_forest(self):
        with tempfile.TemporaryDirectory() as path:
            verify_capacity(Path(path), False)

    def test_actual_shared_component_forest(self):
        with tempfile.TemporaryDirectory() as path:
            verify_capacity(Path(path), True)

    def test_in_capacity_action_search_matches_independent_enumeration(self):
        nodes = tuple(IdentityNode(str(i), i % 2, i, i, str(i // 2)) for i in range(4))
        rows = tuple(tuple((p, 0.) for p in range(-1, i)) for i in range(4))
        factors = ForestFactors(nodes, rows)
        actions = []
        for parents in itertools.product(*(tuple(p for p, _ in row) for row in rows)):
            roots, slots, valid = [], {}, True
            for i, parent in enumerate(parents):
                root = i if parent < 0 else roots[parent]
                slot = (nodes[i].source_id, nodes[i].frame_id)
                if slot in slots.setdefault(root, set()):
                    valid = False; break
                slots[root].add(slot); roots.append(root)
            if valid:
                actions.append((parents, roots))
        rng = np.random.default_rng(1337)
        for _ in range(30):
            chosen = rng.choice(len(actions), 4, replace=False)
            weights = rng.dirichlet(np.ones(4))
            def loss(roots):
                return sum(w * sum(x != y for x, y in zip(roots, actions[i][1])) / 4
                           for i, w in zip(chosen, weights))
            action, audit = conditional_action(factors, [actions[i][0] for i in chosen],
                                               np.log(weights), range(4), budget=10000)
            actual = next(r for a, r in actions if a == action)
            self.assertAlmostEqual(loss(actual), min(loss(r) for _, r in actions))
            self.assertAlmostEqual(audit['optimization_gap'], 0.)


if __name__ == '__main__':
    unittest.main()
