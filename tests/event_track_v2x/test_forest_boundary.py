from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_boundary import ForestBoundaryArchive, checkpoint_tracker
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.forest_tracking import (
    CausalForestTracker, ForestTrackingConfig, replay_forest_states,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors, digest
from transvision.models.event_track_v2x.tracking_v2 import propagate
from test_forest_tracking import observation, step


def make_archive(*, component=False, cutoff=1_100_000, observations=None, **options):
    config = ForestTrackingConfig(**{**dict(component_mode=component,
        active_limit=1, expansion_budget=0, max_model_regret=1.), **options})
    tracker = CausalForestTracker(sequence_id='0003', start_us=0,
        config=config, scorer=GeometryForestScorer(birth_logit=-20.))
    raw = observations if observations is not None else (
        observation('a', 0., state_us=1_000_000),
        observation('b', 1., source=1, state_us=1_100_000),
        observation('c', 2., state_us=1_200_000))
    step(tracker, raw, reference=1_200_000)
    return tracker, checkpoint_tracker(tracker, cutoff_us=cutoff, source_cache_manifest_sha256='b'*64)


def complete_parents(factors):
    # Independent exhaustive root/slot legality check, not bank enumeration.
    result = []
    for parents in itertools.product(*(tuple(p for p, _ in row) for row in factors.rows)):
        roots, used, legal = [], set(), True
        for i, parent in enumerate(parents):
            root = i if parent < 0 else roots[parent]
            roots.append(root)
            slot = (root, factors.nodes[i].source_id, factors.nodes[i].frame_id)
            if slot in used:
                legal = False
                break
            used.add(slot)
        if legal:
            result.append(parents)
    return result


def extended(archive, new=(), support=None):
    rows = archive.factors.rows
    for i, _ in enumerate(new, len(rows)):
        parents = support[i-len(archive.factors.rows)] if support is not None else range(-1, i)
        rows += (tuple((p, -.2*abs(p)) for p in parents),)
    return ForestFactors(tuple(o.node for o in archive.observations+tuple(new)), rows)


def assert_full_replay(archive, parents, *, new=(), reference=1_600_000,
                       decision=1_700_000, factors=None, projection=None):
    factors = factors or extended(archive, new)
    result = archive.replay(factors=factors, parents=parents, new_observations=new,
        reference_us=reference, decision_us=decision, projection=projection)
    predictions, suppressed = replay_forest_states(archive.sequence_id,
        archive.observations+tuple(new), factors, parents, reference, archive.config)
    # Exact bytes, including IDs, score, birth age and branch state covariance.
    assert result.prediction_json == canonical(predictions)
    assert result.audit['suppressed'] == list(suppressed)
    assert (result.audit['state_updates_after_projection']+result.audit['projection_build_state_updates']
            <= result.audit['state_work_upper'])
    assert not result.audit['historical_output_rewritten']
    assert not result.audit['full_sequence_identity_handoff_completed']
    assert not result.audit['new_posterior_or_missing_mass_bound_computed']
    return result


@pytest.mark.parametrize('component', [False, True])
@pytest.mark.parametrize('cutoff', [0, 1_100_000, 1_200_000])
def test_roundtrip_all_legal_histories_including_unexpanded_and_detachment(tmp_path, component, cutoff):
    tracker, archive = make_archive(component=component, cutoff=cutoff)
    before = tracker.commits[0].prediction_json, tracker.commits[0].audit_json
    path = tmp_path / 'archive.json'
    assert archive.save(path) == archive.commit
    loaded = ForestBoundaryArchive.load(path, expected_archive_sha256=archive.commit)
    assert loaded == archive
    assert hashlib.sha256(path.read_bytes()).hexdigest() != archive.commit
    for parents in complete_parents(archive.factors):
        projection = loaded.project(parents)
        assert projection.log_weight == pytest.approx(sum(dict(r)[p] for r, p in zip(archive.factors.rows, parents)))
        assert not projection.active_product_member
        assert not projection.all_component_leaves_previously_discovered
        assert projection.recovered_group_ids
        assert_full_replay(loaded, parents, projection=projection)
    modified = loaded.payload()
    modified['observations'][0]['mean'] = [999]*9
    assert loaded.observations[0].mean[0] == 0.
    assert (tracker.commits[0].prediction_json, tracker.commits[0].audit_json) == before
    # Later tracker activity does not mutate the old archive or its support.
    step(tracker, reference=1_400_000, event='next')
    assert loaded == archive and archive.source_prediction_sha256 == tracker.commits[0].prediction['commit_sha256']


def test_conditional_carry_is_at_actual_observation_time_not_cut_or_old_output():
    tracker, archive = make_archive(cutoff=1_200_000)
    parents = (-1, 0, 1)
    projection = archive.project(parents)
    carry, = projection.carries
    assert carry.last_state_us == 1_100_000
    assert carry.last_state_us < archive.cutoff_us == archive.reference_us
    new = (observation('d', 3., source=1, state_us=1_450_000),)
    result = assert_full_replay(archive, parents+(2,), new=new, projection=projection)
    assert [m['mode'] for m in result.audit['modes']] == ['conditional_carry']
    assert result.audit['state_updates_after_projection'] == 2  # Separator c, new d.
    assert result.predictions[0]['birth_state_us'] == 1_000_000
    # The actual repository process covariance is not a semigroup: adding an
    # artificial propagation to the cut changes the final covariance.
    m, c = np.asarray(carry.mean), np.asarray(carry.covariance)
    direct = propagate(m, c, .5, archive.config.process_noise)
    cut = propagate(m, c, .1, archive.config.process_noise)
    twice = propagate(*cut, .4, archive.config.process_noise)
    assert not np.allclose(direct[1], twice[1], atol=1e-12, rtol=1e-12)
    assert tracker.commits[0].prediction['box_reference_timestamp_us'] == 1_200_000


@pytest.mark.parametrize('component', [False, True])
def test_recovery_never_reuses_or_averages_the_old_branch_carry(component):
    _, archive = make_archive(component=component, cutoff=1_200_000)
    merged, births = archive.project((-1, 0, 1)), archive.project((-1, -1, -1))
    assert merged.commit != births.commit
    assert merged.carries[0].mean != births.carries[0].mean
    new = (observation('new', 3., source=1, state_us=1_450_000),)
    a = assert_full_replay(archive, (-1, 0, 1, 2), new=new, projection=merged)
    b = assert_full_replay(archive, (-1, -1, -1, 2), new=new, projection=births)
    assert len(a.predictions) == 1 and len(b.predictions) == 3
    assert a.predictions[0]['mean'] != b.predictions[0]['mean']
    assert a.predictions[0]['track_id'] == b.predictions[0]['track_id']
    with pytest.raises(ValueError, match='projection'):
        archive.replay(factors=extended(archive, new), parents=(-1, -1, -1, 2),
            new_observations=new, reference_us=1_600_000, decision_us=1_700_000, projection=merged)


def test_information_state_time_separation_forces_interleaved_raw_replay():
    late = observation('late-image', -.5, source=1, state_us=950_000, arrival_us=1_250_000)
    late = replace(late, node=replace(late.node, information_us=1_150_000))
    _, archive = make_archive(cutoff=1_100_000, observations=(observation('archived'), late))
    assert archive.separator_indices == (1,)
    result = assert_full_replay(archive, (-1, 0))
    assert result.audit['modes'][0]['mode'] == 'raw_replay_late_or_interleaved'
    assert result.predictions[0]['birth_state_us'] == 950_000


@pytest.mark.parametrize('time', [800_000, 1_000_000, 1_100_000])
def test_new_late_evidence_replays_raw_with_original_track_id(time):
    _, archive = make_archive(cutoff=1_200_000)
    new = (observation('late', -.4, source=0, frame='new-source-frame',
                       state_us=time, arrival_us=1_450_000),)
    result = assert_full_replay(archive, (-1, 0, 1, 0), new=new)
    assert result.audit['modes'][0]['mode'] == 'raw_replay_late_or_interleaved'
    assert result.predictions[0]['track_id'] == archive.project((-1, 0, 1)).carries[0].track_id


def test_score_birth_and_age_not_reset_by_boundary_or_identity_mass():
    _, archive = make_archive(cutoff=1_200_000,
        observations=(observation('strong', score=.9), observation('weak', x=30., index=1, score=.2)))
    result = assert_full_replay(archive, (-1, -1))
    assert len(result.predictions) == 1 and len(result.audit['suppressed']) == 1
    assert result.predictions[0]['score'] == .9*.95**.6
    assert result.predictions[0]['birth_state_us'] == 1_000_000
    assert result.predictions[0]['last_update_us'] == 1_000_000
    aged = assert_full_replay(archive, (-1, -1), reference=4_000_000, decision=4_100_000)
    assert aged.predictions == []
    assert len(archive.factors.nodes) == 2 and len(archive.project((-1, -1)).carries) == 2


def test_absolute_old_weights_may_rescore_but_not_change_state_or_support():
    _, archive = make_archive()
    updated = ForestFactors(archive.factors.nodes,
        tuple(tuple((p, w+7.*i) for p, w in row) for i, row in enumerate(archive.factors.rows)))
    old = assert_full_replay(archive, (-1, 0, 1))
    new = assert_full_replay(archive, (-1, 0, 1), factors=updated)
    assert new.prediction_json == old.prediction_json
    assert new.factors_sha256 != old.factors_sha256
    assert new.audit['projection_sha256'] == old.audit['projection_sha256']


def test_equal_separator_labels_do_not_imply_equal_full_boundary_states():
    # No observation reaches the chosen cut, so both separator vectors are empty.
    # The retired identities/states still differ and cannot be merged by that key.
    _, archive = make_archive(cutoff=1_200_000, observations=(
        observation('a'), observation('b', 2., source=1, state_us=1_100_000)))
    merged, separate = archive.project((-1, 0)), archive.project((-1, -1))
    assert merged.separator_roots == separate.separator_roots == ()
    assert merged.roots != separate.roots
    assert len(merged.carries) == 1 and len(separate.carries) == 2


@pytest.mark.parametrize('kind', ['future', 'withheld', 'sequence', 'alias', 'support', 'parent', 'time'])
def test_reject_invalid_continuation_before_any_state_replay(monkeypatch, kind):
    _, archive = make_archive()
    new = observation('new', state_us=1_450_000, source=1)
    reference, decision, parents = 1_600_000, 1_700_000, (-1, 0, 1, 2)
    if kind == 'future':
        new = replace(new, node=replace(new.node, arrival_us=1_800_000))
    elif kind == 'withheld':
        new = observation('withheld', source=1, state_us=1_250_000)
    elif kind == 'sequence':
        new = replace(new, sequence_id='other')
    elif kind == 'alias':
        new = replace(new, detection_index=archive.observations[0].detection_index,
            node=replace(new.node, source_id=0, frame_id=archive.observations[0].node.frame_id))
    elif kind == 'parent':
        parents = (-1, 0, 1, 3)
    elif kind == 'time':
        reference = 1_100_000
    factors = extended(archive, (new,))
    if kind == 'support':
        factors = ForestFactors(factors.nodes, (factors.rows[0], ((-1, 0.),), *factors.rows[2:]))
    import transvision.models.event_track_v2x.forest_boundary as module
    monkeypatch.setattr(module, '_replay', lambda *a, **kw: pytest.fail('invalid input replayed'))
    before = archive.payload()
    with pytest.raises(ValueError):
        archive.replay(factors=factors, parents=parents, new_observations=(new,),
                       reference_us=reference, decision_us=decision)
    assert archive.payload() == before


@pytest.mark.parametrize('limit', ['nodes', 'work'])
def test_continuation_capacity_fails_before_projection(monkeypatch, limit):
    options = {'max_nodes': 1} if limit == 'nodes' else {'max_replay_operations': 2}
    _, archive = make_archive(cutoff=1_200_000, observations=(observation('a'),), **options)
    new = (observation('new', source=1, state_us=1_450_000),)
    import transvision.models.event_track_v2x.forest_boundary as module
    monkeypatch.setattr(module, '_replay', lambda *a, **kw: pytest.fail('capacity checked too late'))
    with pytest.raises(ValueError, match='cap exceeded before replay'):
        archive.replay(factors=extended(archive, new), parents=(-1, 0), new_observations=new,
                       reference_us=1_600_000, decision_us=1_700_000)


@pytest.mark.parametrize('kind', ['raw', 'support', 'config', 'live_support', 'live_action', 'scorer_signature'])
def test_checkpoint_rejects_mutated_tracker(kind):
    tracker, archive = make_archive()
    if kind == 'raw':
        tracker.observations = (replace(tracker.observations[0], score=.1), *tracker.observations[1:])
    elif kind == 'support':
        factors = tracker.bank.factors
        tracker.bank.factors = ForestFactors(factors.nodes, tuple(((-1, 0.),) for _ in factors.rows))
    elif kind == 'live_support':
        tracker.support = tuple((-1,) for _ in tracker.support)
    elif kind == 'live_action':
        tracker.output_parents = (-1, 0, 0)
    elif kind == 'scorer_signature':
        tracker.scorer_signature = 'changed'
    else:
        tracker.config = replace(tracker.config, process_noise=.2)
    with pytest.raises(ValueError, match='changed since'):
        checkpoint_tracker(tracker, cutoff_us=1_100_000)
    archive.validate()


@pytest.mark.parametrize('kind', ['unknown', 'tamper', 'noncanonical', 'wrong_hash', 'file_hash', 'byte_cap'])
def test_load_fails_closed(tmp_path, kind):
    _, archive = make_archive(component=True)
    path = tmp_path / 'archive.json'
    archive.save(path)
    expected, cap = archive.commit, 128*1024**2
    payload = json.loads(path.read_bytes())
    if kind == 'unknown':
        payload['unknown'] = 1
    elif kind == 'tamper':
        payload['observations'][0]['mean'][0] = 99.
    elif kind == 'wrong_hash':
        expected = '0'*64
    elif kind == 'file_hash':
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    elif kind == 'byte_cap':
        cap = 10
    path.write_bytes(canonical(payload)+(b'\n' if kind == 'noncanonical' else b''))
    with pytest.raises(ValueError):
        ForestBoundaryArchive.load(path, expected_archive_sha256=expected, max_bytes=cap)


def test_exclusive_save_byte_cap_and_symlink_guard(tmp_path):
    _, archive = make_archive()
    path = tmp_path / 'archive.json'
    with pytest.raises(ValueError, match='byte cap'):
        archive.save(path, max_bytes=10)
    assert not path.exists()
    archive.save(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        archive.save(path)
    assert path.read_bytes() == before
    alias = tmp_path / 'alias.json'
    alias.symlink_to(path)
    directory_alias = tmp_path / 'directory'
    directory_alias.symlink_to(tmp_path, target_is_directory=True)
    for target in (alias, directory_alias / 'archive.json'):
        with pytest.raises(ValueError):
            ForestBoundaryArchive.load(target, expected_archive_sha256=archive.commit)
    with pytest.raises(ValueError, match='symlinks'):
        archive.save(directory_alias / 'new.json')


@pytest.mark.parametrize('component', [False, True])
def test_empty_scene_checkpoint(component, tmp_path):
    _, archive = make_archive(component=component, observations=())
    projection = archive.project(())
    assert projection.carries == () and projection.log_weight == 0.
    assert_full_replay(archive, ())
    archive.save(tmp_path / 'empty.json')
    assert ForestBoundaryArchive.load(tmp_path / 'empty.json', expected_archive_sha256=archive.commit) == archive
