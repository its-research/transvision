from dataclasses import asdict, replace
import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_boundary_prior import (
    extend_boundary_prior, marginalize_boundary,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from test_forest_boundary import assert_full_replay, make_archive
from test_forest_tracking import observation


def independent_posterior(factors):
    weights, counts = {}, {}
    # Independent exhaustive product + identity exclusion, no oracle/bank code.
    for choices in itertools.product(*(row for row in factors.rows)):
        roots, occupied, legal = [], set(), True
        for i, (p, _) in enumerate(choices):
            r = i if p < 0 else roots[p]
            roots.append(r)
            slot = r, factors.nodes[i].source_id, factors.nodes[i].frame_id
            if slot in occupied:
                legal = False
                break
            occupied.add(slot)
        if legal:
            key = tuple(roots)
            weights.setdefault(key, []).append(math.exp(math.fsum(w for _, w in choices)))
            counts[key] = counts.get(key, 0)+1
    mass = {k: math.fsum(v) for k, v in weights.items()}
    return mass, counts


def assert_exact_extension(extension, factors):
    weights, counts = independent_posterior(factors)
    assert {a.roots for a in extension.atoms} == set(weights)
    for atom in extension.atoms:
        assert atom.log_weight == pytest.approx(math.log(weights[atom.roots]), abs=2e-12)
        assert atom.parent_history_count == counts[atom.roots]
        assert factors.roots(atom.representative_parents) == atom.roots
    assert extension.log_partition == pytest.approx(math.log(math.fsum(weights.values())), abs=2e-12)
    assert extension.parent_history_count == sum(counts.values())
    # Compare Bayes root-Hamming risk on the identical complete legal action set.
    if factors.nodes:
        normalizer = math.fsum(weights.values())
        for action in weights:
            expected = math.fsum(w*sum(a != b for a, b in zip(action, truth))/len(action)
                                for truth, w in weights.items())/normalizer
            actual = math.fsum(math.exp(a.log_weight-extension.log_partition)*
                              sum(x != y for x, y in zip(action, a.roots))/len(action)
                              for a in extension.atoms)
            assert actual == pytest.approx(expected, abs=2e-12)


def test_multiple_parent_histories_sum_identity_mass_instead_of_max_or_dedup():
    _, archive = make_archive()
    factors = ForestFactors(archive.factors.nodes,
        tuple(tuple((p, 0.) for p, _ in row) for row in archive.factors.rows))
    prior = marginalize_boundary(archive, factors=factors)
    marginal, = prior.components
    assert len(marginal.atoms) == 5 and marginal.parent_history_count == 6
    all_one, = [a for a in marginal.atoms if a.roots == (0, 0, 0)]
    assert all_one.parent_history_count == 2
    assert all_one.log_weight == pytest.approx(math.log(2.))
    assert math.exp(all_one.log_weight-prior.log_partition) == pytest.approx(1/3)
    assert math.exp(all_one.log_weight-prior.log_partition) != pytest.approx(1/5)
    assert all(not g.snapshot.active for g in archive.groups)
    # Raw data memberships and conditional CI state are identical within atom;
    # summation does not fuse states from DIFFERENT root partitions.
    first = assert_full_replay(archive, (-1, 0, 0))
    second = assert_full_replay(archive, (-1, 0, 1))
    assert first.prediction_json == second.prediction_json
    extension = extend_boundary_prior(prior, factors, decision_us=archive.decision_us)
    assert_exact_extension(extension, factors)
    assert extension.enumerated_quotient_leaves == 5 < extension.parent_history_count


@pytest.mark.parametrize('seed', range(12))
def test_component_bridge_extension_matches_exhaustive_joint_posterior(seed):
    rng = np.random.default_rng(seed)
    raw = tuple(observation(f'{side}-{time}', side*30.+time*.2, index=side,
        source=time % 2, state_us=1_000_000+time*100_000)
        for time in range(3) for side in range(2))
    _, archive = make_archive(component=bool(seed % 2), observations=raw)
    old = ForestFactors(archive.factors.nodes,
        tuple(tuple((p, float(rng.uniform(-2, 2))) for p, _ in row) for row in archive.factors.rows))
    prior = marginalize_boundary(archive, factors=old)
    assert len(prior.components) == 2 and prior.parent_history_count == 36
    # A new observation can link either component. Another delayed observation
    # shares an old source/frame slot and tests ROOT-wide exclusion after quotient.
    newer = observation('new', 15., state_us=1_450_000)
    late = observation('late', 15., index=2, state_us=1_000_000, arrival_us=1_500_000)
    rows = old.rows+tuple(tuple((p, float(rng.uniform(-2, 2))) for p in range(-1, i)) for i in (6, 7))
    factors = ForestFactors(old.nodes+(newer.node, late.node), rows)
    result = extend_boundary_prior(prior, factors, decision_us=1_600_000)
    assert_exact_extension(result, factors)
    assert result.old_joint_identity_atoms == 25 < prior.parent_history_count
    assert result.enumerated_quotient_leaves < result.parent_history_count
    # The source archive still has the old factors/weights, unmodified.
    archive.validate()
    assert archive.factors != prior.factors


def test_future_evidence_must_not_be_normalized_per_boundary_identity_atom():
    a = observation('a')
    b = observation('b', source=1, state_us=1_100_000)
    _, archive = make_archive(observations=(a, b))
    old = ForestFactors(archive.factors.nodes, (((-1, 0.),), ((-1, 0.), (0, 0.))))
    prior = marginalize_boundary(archive, factors=old)
    late = observation('late-same-source-frame', index=1, arrival_us=1_500_000)
    factors = ForestFactors(old.nodes+(late.node,), old.rows+(((-1, 0.), (0, 0.), (1, 0.)),))
    result = extend_boundary_prior(prior, factors, decision_us=1_600_000)
    mass_split = math.fsum(math.exp(a.log_weight-result.log_partition)
                          for a in result.atoms if a.roots[:2] == (0, 1))
    # Old split and merged priors are both 1/2. Their legal extension masses are
    # 2 and 1. Normalizing each conditional extension independently would erase
    # this evidence, incorrectly leaving the boundary posterior at 1/2.
    assert mass_split == pytest.approx(2/3)
    assert mass_split != pytest.approx(.5)
    assert_exact_extension(result, factors)


def test_rescore_rebuilds_from_raw_support_even_for_previously_unexpanded_history():
    _, archive = make_archive(active_limit=1, expansion_budget=1)
    old = marginalize_boundary(archive)
    changed = ForestFactors(archive.factors.nodes, tuple(tuple((p, 20. if p == -1 else -20.)
                                      for p, _ in row) for row in archive.factors.rows))
    with pytest.raises(ValueError, match='rebuild'):
        extend_boundary_prior(old, changed, decision_us=1_600_000)
    rebuilt = marginalize_boundary(archive, factors=changed, decision_us=1_500_000)
    new = extend_boundary_prior(rebuilt, changed, decision_us=1_600_000)
    best = max(new.atoms, key=lambda a: a.log_weight)
    assert best.roots == (0, 1, 2)
    assert best.representative_parents not in archive.groups[0].snapshot.active
    assert_exact_extension(new, changed)
    assert rebuilt.archive_sha256 == old.archive_sha256 == archive.commit
    assert rebuilt.commit != old.commit


@pytest.mark.parametrize('kind', ['prefix', 'leaf', 'atom', 'negative'])
def test_exact_marginal_never_returns_a_topk_renormalized_answer_on_capacity(kind):
    _, archive = make_archive()
    options = {'max_prefixes': 1} if kind == 'prefix' else {'max_leaves': 1} if kind == 'leaf' else {
        'max_atoms': 1} if kind == 'atom' else {'max_leaves': 0}
    before = archive.payload()
    with pytest.raises(ValueError, match='cap'):
        marginalize_boundary(archive, **options)
    assert archive.payload() == before


@pytest.mark.parametrize('kind', ['joint', 'prefix', 'leaf', 'atom'])
def test_extension_capacity_preserves_original_exact_prior(kind):
    _, archive = make_archive()
    prior = marginalize_boundary(archive)
    before = asdict(prior)
    options = {'max_joint_atoms': 1} if kind == 'joint' else {'max_prefixes': 1} if kind == 'prefix' else {
        'max_leaves': 1} if kind == 'leaf' else {'max_atoms': 1}
    with pytest.raises(ValueError, match='cap'):
        extend_boundary_prior(prior, prior.factors, decision_us=archive.decision_us, **options)
    assert asdict(prior) == before


def test_atom_cap_is_shared_across_independent_components():
    _, archive = make_archive(observations=(observation('a'), observation('b', 30., source=1)))
    with pytest.raises(ValueError, match='identity-atom cap'):
        marginalize_boundary(archive, max_atoms=1)
    assert len(marginalize_boundary(archive, max_atoms=2).components) == 2


@pytest.mark.parametrize('offset', [-900., 900.])
def test_common_log_factor_gauge_does_not_change_root_probabilities(offset):
    _, archive = make_archive()
    base = marginalize_boundary(archive)
    shifted_factors = ForestFactors(archive.factors.nodes,
        tuple(tuple((p, w+offset) for p, w in row) for row in archive.factors.rows))
    shifted = marginalize_boundary(archive, factors=shifted_factors)
    result = extend_boundary_prior(base, base.factors, decision_us=archive.decision_us)
    other = extend_boundary_prior(shifted, shifted.factors, decision_us=archive.decision_us)
    assert other.log_partition-result.log_partition == pytest.approx(3*offset, abs=2e-12)
    for original, moved in zip(result.atoms, other.atoms):
        assert original.roots == moved.roots and original.parent_history_count == moved.parent_history_count
        assert math.exp(original.log_weight-result.log_partition) == pytest.approx(
            math.exp(moved.log_weight-other.log_partition), abs=2e-12)


@pytest.mark.parametrize('kind', ['future', 'withheld', 'changed_old_weights', 'earlier_decision', 'tampered'])
def test_invalid_extension_rejected(kind):
    _, archive = make_archive()
    prior = marginalize_boundary(archive)
    new = observation('new', state_us=1_450_000)
    if kind == 'future':
        new = replace(new, node=replace(new.node, arrival_us=1_800_000))
    elif kind == 'withheld':
        new = observation('withheld', state_us=1_250_000)
    factors = ForestFactors(prior.factors.nodes+(new.node,), prior.factors.rows+(((-1, 0.), (0, .2)),))
    if kind == 'changed_old_weights':
        factors = ForestFactors(factors.nodes, (((-1, 1.),), *factors.rows[1:]))
    if kind == 'tampered':
        prior = replace(prior, log_partition=999.)
    with pytest.raises(ValueError):
        extend_boundary_prior(prior, factors,
            decision_us=1_200_000 if kind == 'earlier_decision' else 1_600_000)


@pytest.mark.parametrize('component', [False, True])
def test_empty_boundary_has_unit_mass_and_starts_new_roots(component):
    _, archive = make_archive(component=component, observations=())
    prior = marginalize_boundary(archive)
    assert prior.components == () and prior.log_partition == 0. and prior.parent_history_count == 1
    empty = extend_boundary_prior(prior, prior.factors, decision_us=archive.decision_us)
    assert empty.log_partition == 0. and empty.parent_history_count == 1
    assert empty.atoms[0].roots == ()
    new = observation('new', state_us=1_450_000)
    factors = ForestFactors((new.node,), (((-1, .2),),))
    result = extend_boundary_prior(prior, factors, decision_us=1_600_000)
    assert_exact_extension(result, factors)
