"""Deterministic global split budgeting for independent local hypothesis banks.

The scheduler prioritizes L times a model omitted-mass upper bound, never a
learned confidence. This quantity bounds truncation regret only under the
common-action Bayes-decision premises; it is not a guarantee on HOTA/IDF1 or
on an uncalibrated real-world posterior. No learned VoI claim is made.

Banks are copied on adoption and on each transaction. The scheduler owns its
copies; caller-owned banks are not mutated. Staging doubles live bank storage
temporarily, and split counts are not equal-wall-time or equal-byte budgets.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Mapping

from .hypothesis_bank import BankSnapshot, EvidenceUpdate, HypothesisBank, regret_upper_bound


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AllocationStep:
    component_id: str
    risk_upper_before: float
    eta_upper_after: float
    consumed_splits: int
    resource_limited: bool


@dataclass(frozen=True, slots=True)
class ComponentAllocation:
    component_id: str
    loss_range: float
    truncation_risk_upper: float
    consumed_splits: int
    status: str
    bank: BankSnapshot


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    request_id: str
    decision_us: int
    requested_splits: int
    consumed_splits: int
    components: tuple[ComponentAllocation, ...]
    trace: tuple[AllocationStep, ...]
    risk_semantics: str
    max_requests: int
    previous_commit: str
    commit: str


class RiskBudgetAllocator:
    """Prioritize the greatest local truncation-risk bound with stable ID ties.

    An allocation accepts at most one factor update per component. Unknown IDs,
    causal violations, conflicting evidence IDs, or history-cap failures abort
    the entire transaction before any owned bank is replaced. Identical request
    retries return their original immutable receipt without spending more work.
    """

    def __init__(self, banks: Mapping[str, HypothesisBank], *,
                 loss_ranges: Mapping[str, float] | None = None, max_requests: int = 128):
        if not isinstance(banks, Mapping) or any(type(key) is not str or not key for key in banks):
            raise ValueError("banks must map nonempty component IDs to HypothesisBank instances")
        if any(type(bank) is not HypothesisBank for bank in banks.values()):
            raise TypeError("every component must contain an exact HypothesisBank")
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        supplied = {} if loss_ranges is None else dict(loss_ranges)
        if set(supplied) - set(banks):
            raise ValueError("loss_ranges contains an unknown component ID")
        ranges = {key: supplied.get(key, 1.) for key in banks}
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value < 0 for value in ranges.values()):
            raise ValueError("loss ranges must be finite nonnegative numbers")
        self._banks = deepcopy(dict(banks))
        self._ranges = {key: float(value) for key, value in ranges.items()}
        self._max_requests = max_requests
        self._requests: dict[str, tuple[str, AllocationSnapshot]] = {}
        self._commits: tuple[AllocationSnapshot, ...] = ()

    @property
    def commits(self) -> tuple[AllocationSnapshot, ...]:
        return self._commits

    def component_snapshot(self, component_id: str) -> BankSnapshot | None:
        bank = self._banks[component_id]
        return bank.commits[-1] if bank.commits else None

    def component_ancestry(self, component_id: str, after_commit: str,
                           through_commit: str | None = None) -> tuple[BankSnapshot, ...]:
        """Return verified owned commits, excluding after and including through.

        ``through_commit=None`` selects the current tip. The all-zero hash is
        the explicit pre-first-commit sentinel. Unknown hashes, reversed ranges,
        forked parents, changed origins, and corrupted commit payloads fail
        closed. The returned tuple contains immutable snapshots, not bank state.
        """
        if type(component_id) is not str or not component_id or component_id not in self._banks:
            raise ValueError("unknown or empty component ID")
        for name, value in (("after_commit", after_commit), ("through_commit", through_commit)):
            if name == "through_commit" and value is None:
                continue
            if (type(value) is not str or len(value) != 64 or
                    any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(name + " must be a nonempty canonical SHA256 hash")
        bank = self._banks[component_id]
        commits = bank.commits
        origin = bank.original_factors.digest()
        previous = "0" * 64
        positions = {previous: -1}
        for index, snapshot in enumerate(commits):
            if type(snapshot) is not BankSnapshot:
                raise ValueError("owned ancestry contains an invalid snapshot type")
            payload = asdict(snapshot)
            digest = payload.pop("commit")
            if (snapshot.original_factors_sha256 != origin or snapshot.previous_commit != previous
                    or digest != _digest(payload) or digest in positions):
                raise ValueError("owned ancestry has a changed origin, forked parent or corrupted commit")
            positions[digest] = index
            previous = digest
        tip = previous if through_commit is None else through_commit
        if after_commit not in positions or tip not in positions:
            raise ValueError("requested ancestry hash is not in the owned component chain")
        start, end = positions[after_commit], positions[tip]
        if end < start:
            raise ValueError("through_commit precedes after_commit")
        return commits[start + 1:end + 1]

    @staticmethod
    def _can_split(bank):
        rows = bank.original_factors.shape[0]
        return any(len(prefix) < rows for prefix in bank.frontier_constraints)

    def allocate(self, request_id: str, *, decision_us: int, split_budget: int,
                 evidence: Mapping[str, EvidenceUpdate] | None = None) -> AllocationSnapshot:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if type(decision_us) is not int or decision_us < 0:
            raise ValueError("decision_us must be a nonnegative integer")
        if type(split_budget) is not int or split_budget < 0:
            raise ValueError("split_budget must be a nonnegative integer")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("evidence must map component IDs to EvidenceUpdate objects")
        updates = dict(evidence or {})
        if set(updates) - set(self._banks):
            raise ValueError("evidence contains an unknown component ID")
        if any(type(update) is not EvidenceUpdate for update in updates.values()):
            raise TypeError("every update must be EvidenceUpdate")
        fingerprint = _digest({"decision_us": decision_us, "split_budget": split_budget,
                               "updates": {key: update.digest() for key, update in sorted(updates.items())}})
        if request_id in self._requests:
            prior, snapshot = self._requests[request_id]
            if prior != fingerprint:
                raise ValueError("conflicting duplicate request ID")
            return snapshot
        if len(self._requests) >= self._max_requests:
            raise ValueError("max_requests reached; close the allocation window explicitly")
        if self._commits and decision_us < self._commits[-1].decision_us:
            raise ValueError("allocation decisions must be nondecreasing")

        # Complete all validation and updates against owned staging copies. A
        # late failing component cannot leave the earlier components committed.
        staged = deepcopy(self._banks)
        states = {}
        for key in sorted(staged):
            states[key] = staged[key].advance(decision_us=decision_us, expansion_budget=0,
                                             evidence=updates.get(key))
        consumed = {key: 0 for key in staged}
        blocked = set()
        trace = []
        remaining = split_budget
        while remaining:
            candidates = [key for key in staged if key not in blocked and self._can_split(staged[key])
                          and self._ranges[key] > 0 and states[key].eta_upper > 0]
            if not candidates:
                break
            key = min(candidates, key=lambda item: (-regret_upper_bound(states[item].eta_upper,
                                                                       loss_range=self._ranges[item]), item))
            risk = regret_upper_bound(states[key].eta_upper, loss_range=self._ranges[key])
            state = staged[key].advance(decision_us=decision_us, expansion_budget=1)
            states[key] = state
            consumed[key] += state.expansions
            remaining -= state.expansions
            trace.append(AllocationStep(key, risk, state.eta_upper, state.expansions, state.resource_limited))
            if state.expansions == 0:
                blocked.add(key)  # At most one zero-progress probe per component.

        components = []
        for key in sorted(staged):
            state = states[key]
            if key in blocked:
                status = "resource_limited"
            elif not staged[key].frontier_constraints:
                status = "complete_support_retained"
            elif not self._can_split(staged[key]):
                status = "active_capacity_limited"
            elif self._ranges[key] == 0:
                status = "zero_loss_range"
            elif state.eta_upper == 0:
                status = "zero_numerical_risk_bound"
            else:
                status = "split_budget_exhausted"
            components.append(ComponentAllocation(key, self._ranges[key],
                regret_upper_bound(state.eta_upper, loss_range=self._ranges[key]), consumed[key], status, state))
        payload = dict(request_id=request_id, decision_us=decision_us, requested_splits=split_budget,
                       consumed_splits=split_budget - remaining, components=tuple(components), trace=tuple(trace),
                       risk_semantics="L_times_model_omitted_mass_upper_requires_common_action_Bayes_premise",
                       max_requests=self._max_requests,
                       previous_commit=self._commits[-1].commit if self._commits else "0" * 64)
        serializable = {**payload, "components": [asdict(item) for item in components],
                        "trace": [asdict(item) for item in trace]}
        snapshot = AllocationSnapshot(**payload, commit=_digest(serializable))
        self._banks = staged
        self._requests[request_id] = (fingerprint, snapshot)
        self._commits += (snapshot,)
        return snapshot
