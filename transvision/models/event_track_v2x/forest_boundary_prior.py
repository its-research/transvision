"""Exhaustive boundary-identity marginalization reference (small scenes only).

An identity partition may have multiple parent-forest histories. Their weights
must be SUMMED, not maximized or normalized separately. Equal full root vectors
have the same raw memberships, exclusion slots and conditional CI replay state;
they can be grouped without averaging distinct identity states.

This oracle proves/checks that quotient for fixed finite ForestFactors. It does
not implement a bounded-memory, recoverable separator-prior search. It explicitly
enumerates every old history and the product of component identity atoms before
extending them, with fail-closed work/storage caps. It is not a real-data result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math

from .forest_boundary import ForestBoundaryArchive
from .forest_components import ForestComponent, split_forest
from .hypothesis_bank import logsumexp
from .identity_forest import ForestFactors, digest


@dataclass(frozen=True)
class RootMassAtom:
    roots: tuple[int, ...]
    representative_parents: tuple[int, ...]
    log_weight: float
    parent_history_count: int


@dataclass(frozen=True)
class ComponentRootMarginal:
    component: ForestComponent
    atoms: tuple[RootMassAtom, ...]
    log_partition: float
    parent_history_count: int


@dataclass(frozen=True)
class ExactBoundaryPrior:
    archive_sha256: str
    archive_decision_us: int
    potential_decision_us: int
    factors: ForestFactors
    components: tuple[ComponentRootMarginal, ...]
    log_partition: float
    parent_history_count: int
    enumerated_prefixes: int
    enumerated_component_leaves: int
    commit: str


@dataclass(frozen=True)
class ExactBoundaryExtension:
    prior_sha256: str
    factors_sha256: str
    decision_us: int
    atoms: tuple[RootMassAtom, ...]
    log_partition: float
    parent_history_count: int
    old_joint_identity_atoms: int
    enumerated_prefixes: int
    enumerated_quotient_leaves: int
    commit: str


def _commit(cls, values):
    prototype = cls(**values, commit='')
    payload = asdict(prototype)
    payload.pop('commit')
    return cls(**values, commit=digest(payload))


class _Budget:
    def __init__(self, max_prefixes, max_leaves, max_atoms):
        if any(type(x) is not int or x < 1 for x in (max_prefixes, max_leaves, max_atoms)):
            raise ValueError('positive integer exhaustive-oracle caps required')
        self.max_prefixes, self.max_leaves, self.max_atoms = max_prefixes, max_leaves, max_atoms
        self.prefixes = self.leaves = self.atoms = 0

    def complete(self, factors, prefix=()):
        # Iterators bound stack memory by depth plus one row's children, not by
        # the entire unresolved search tree. No recursion-limit dependence.
        stack = [iter((prefix,))]
        while stack:
            current = next(stack[-1], None)
            if current is None:
                stack.pop()
                continue
            self.prefixes += 1
            if self.prefixes > self.max_prefixes:
                raise ValueError('exhaustive boundary prefix cap exceeded; no prior/posterior returned')
            if len(current) == len(factors.nodes):
                self.leaves += 1
                if self.leaves > self.max_leaves:
                    raise ValueError('exhaustive boundary leaf cap exceeded; no prior/posterior returned')
                yield current
            else:
                stack.append(iter(factors.children(current)))


def _add(groups, factors, parents, log_weight, count, budget):
    roots = factors.roots(parents)
    if roots not in groups:
        if budget.atoms >= budget.max_atoms:
            raise ValueError('exhaustive boundary identity-atom cap exceeded; no result returned')
        budget.atoms += 1
        groups[roots] = [parents, [], 0]
    entry = groups[roots]
    entry[0] = min(entry[0], parents)
    entry[1].append(log_weight)
    entry[2] += count


def _atoms(groups):
    return tuple(RootMassAtom(roots, parents, logsumexp(weights), count)
                 for roots, (parents, weights, count) in sorted(groups.items()))


def marginalize_boundary(archive, *, factors=None, decision_us=None,
                         max_prefixes=100_000, max_leaves=20_000, max_atoms=10_000):
    """Exact model prior from ALL original histories, never just bank.active.

Optional rescoring rebuilds the entire prior from the archive's original
support. The declared potential decision must not precede its source archive.
The caller must bind these scores to legally arrived evidence; this numerical
oracle cannot certify a learned likelihood or its calibration.
"""
    if type(archive) is not ForestBoundaryArchive:
        raise TypeError('ForestBoundaryArchive required')
    archive.validate()
    factors = archive.factors if factors is None else factors
    decision = archive.decision_us if decision_us is None else decision_us
    if (type(factors) is not ForestFactors or type(decision) is not int or decision < archive.decision_us
            or factors.nodes != archive.factors.nodes
            or any(tuple(p for p, _ in a) != tuple(p for p, _ in b)
                   for a, b in zip(factors.rows, archive.factors.rows))):
        raise ValueError('boundary prior cannot rewrite source support/nodes or use an earlier decision')
    budget = _Budget(max_prefixes, max_leaves, max_atoms)
    result = []
    for component in split_forest(factors):
        groups = {}
        for parents in budget.complete(component.factors):
            _add(groups, component.factors, parents, component.factors.log_weight(parents), 1, budget)
        atoms = _atoms(groups)
        result.append(ComponentRootMarginal(component, atoms, logsumexp(a.log_weight for a in atoms),
                                            sum(a.parent_history_count for a in atoms)))
    return _commit(ExactBoundaryPrior, dict(archive_sha256=archive.commit,
        archive_decision_us=archive.decision_us, potential_decision_us=decision, factors=factors,
        components=tuple(result), log_partition=math.fsum(c.log_partition for c in result),
        parent_history_count=math.prod(c.parent_history_count for c in result),
        enumerated_prefixes=budget.prefixes, enumerated_component_leaves=budget.leaves))


def _validate_prior(prior):
    if type(prior) is not ExactBoundaryPrior:
        raise TypeError('ExactBoundaryPrior required; Top-K conditional priors are not accepted')
    payload = asdict(prior)
    if payload.pop('commit') != digest(payload):
        raise ValueError('boundary prior payload changed')
    if tuple(c.component for c in prior.components) != split_forest(prior.factors):
        raise ValueError('boundary prior component coverage changed')
    for marginal in prior.components:
        if (not marginal.atoms or len({a.roots for a in marginal.atoms}) != len(marginal.atoms)
                or marginal.log_partition != logsumexp(a.log_weight for a in marginal.atoms)
                or marginal.parent_history_count != sum(a.parent_history_count for a in marginal.atoms)):
            raise ValueError('invalid boundary root marginal')
        for atom in marginal.atoms:
            if (type(atom.parent_history_count) is not int or atom.parent_history_count < 1
                    or not math.isfinite(atom.log_weight)
                    or len(atom.representative_parents) != len(marginal.component.indices)
                    or marginal.component.factors.roots(atom.representative_parents) != atom.roots):
                raise ValueError('invalid representative root atom')
    if (prior.log_partition != math.fsum(c.log_partition for c in prior.components)
            or prior.parent_history_count != math.prod(c.parent_history_count for c in prior.components)):
        raise ValueError('invalid global boundary mass/count')
    # Structural integrity is not authorship authentication or an independent
    # proof of exhaustive enumeration. There is deliberately no untrusted load.


def extend_boundary_prior(prior, factors, *, decision_us, max_joint_atoms=10_000,
                          max_prefixes=100_000, max_leaves=20_000, max_atoms=10_000):
    """Exact quotient extension, including bridges between old components.

New potentials must be common raw-node row factors. If a future model depends
on the detailed old parent graph rather than roots/raw membership, this quotient
is not sufficient. In particular, never normalize evidence per boundary atom.
Old rows must equal the prior's rows; rescoring requires marginalize_boundary
again from the original archive (including histories omitted by old Top-K).
"""
    _validate_prior(prior)
    n = len(prior.factors.nodes)
    if (type(factors) is not ForestFactors or type(decision_us) is not int
            or decision_us < prior.potential_decision_us
            or factors.nodes[:n] != prior.factors.nodes or factors.rows[:n] != prior.factors.rows
            or len(factors.nodes) < n or any(o.arrival_us > decision_us for o in factors.nodes)
            or any(o.arrival_us <= prior.archive_decision_us for o in factors.nodes[n:])):
        raise ValueError('future/withheld evidence or changed old factors; rebuild the bound prior before extension')
    if type(max_joint_atoms) is not int or max_joint_atoms < 1:
        raise ValueError('positive integer joint-atom cap required')
    joint_count = math.prod(len(c.atoms) for c in prior.components)
    if joint_count > max_joint_atoms:
        raise ValueError('exhaustive boundary joint-atom cap exceeded before Cartesian enumeration')
    budget = _Budget(max_prefixes, max_leaves, max_atoms)
    groups = {}
    for chosen in itertools.product(*(c.atoms for c in prior.components)):
        parents = [-1]*n
        for marginal, atom in zip(prior.components, chosen):
            for i, p in zip(marginal.component.indices,
                            marginal.component.lift(atom.representative_parents)):
                parents[i] = p
        old_mass = math.fsum(a.log_weight for a in chosen)
        multiplicity = math.prod(a.parent_history_count for a in chosen)
        for full in budget.complete(factors, tuple(parents)):
            # Use summed old mass, NOT the representative old forest's weight.
            log_weight = math.fsum([old_mass, *(dict(factors.rows[i])[full[i]]
                                               for i in range(n, len(full)))])
            _add(groups, factors, full, log_weight, multiplicity, budget)
    atoms = _atoms(groups)
    return _commit(ExactBoundaryExtension, dict(prior_sha256=prior.commit, factors_sha256=factors.digest(),
        decision_us=decision_us, atoms=atoms, log_partition=logsumexp(a.log_weight for a in atoms),
        parent_history_count=sum(a.parent_history_count for a in atoms), old_joint_identity_atoms=joint_count,
        enumerated_prefixes=budget.prefixes, enumerated_quotient_leaves=budget.leaves))
