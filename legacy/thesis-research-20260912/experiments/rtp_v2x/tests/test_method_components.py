#!/usr/bin/env python3

from __future__ import annotations

import math
import random
import sys
import unittest
from itertools import combinations
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from model import (  # noqa: E402
    AssociationCostConfig,
    CausalMemoryError,
    CausalMessageMemory,
    CommunicationCandidate,
    MessageRecord,
    compose_association_cost,
    select_budgeted_messages,
    solve_with_unmatched,
)


def message(
    message_id: str,
    *,
    event_time: int,
    arrival_time: int,
    agent_id: str = "ego",
    payload_kind: str = "query",
    digest_character: str = "a",
) -> MessageRecord:
    return MessageRecord(
        message_id=message_id,
        agent_id=agent_id,
        payload_kind=payload_kind,
        event_time=event_time,
        arrival_time=arrival_time,
        encoded_bytes=128,
        payload_sha256=digest_character * 64,
        reliability=0.8,
    )


def candidate(
    message_id: str,
    *,
    encoded_bytes: int,
    utility: float,
    arrival: int = 5,
    deadline: int = 5,
) -> CommunicationCandidate:
    return CommunicationCandidate(
        message_id=message_id,
        granularity="query",
        encoded_bytes=encoded_bytes,
        net_utility=utility,
        predicted_arrival_time=arrival,
        deadline=deadline,
        wire_sha256="b" * 64,
    )


class CausalMessageMemoryTest(unittest.TestCase):
    def test_late_ingest_cannot_change_an_earlier_snapshot(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=4)
        first = message("m1", event_time=1, arrival_time=1)
        late = message("m2", event_time=2, arrival_time=5)
        memory.ingest(first)
        before = memory.snapshot(decision_time=2)
        memory.commit_output(decision_time=2)
        memory.ingest(late)
        after = memory.snapshot(decision_time=2)
        self.assertEqual(before, (first,))
        self.assertEqual(after, before)

    def test_future_event_time_is_not_visible_even_if_arrival_is_early(self) -> None:
        memory = CausalMessageMemory(max_age=20, max_messages_per_agent=4)
        memory.ingest(message("clock-skew", event_time=8, arrival_time=3))
        self.assertEqual(
            memory.snapshot(decision_time=5),
            (),
        )

    def test_duplicate_identity_is_idempotent_but_conflict_is_rejected(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=4)
        original = message("stable", event_time=1, arrival_time=2)
        self.assertTrue(memory.ingest(original))
        self.assertFalse(memory.ingest(original))
        with self.assertRaisesRegex(CausalMemoryError, "reused"):
            memory.ingest(
                message(
                    "stable",
                    event_time=1,
                    arrival_time=2,
                    digest_character="c",
                )
            )

    def test_capacity_is_applied_independently_to_each_agent(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=1)
        rows = [
            message("ego-1", event_time=1, arrival_time=1),
            message("ego-2", event_time=2, arrival_time=2),
            message("rsu-1", event_time=1, arrival_time=1, agent_id="rsu-1"),
        ]
        memory.ingest_many(rows)
        snapshot = memory.snapshot(decision_time=3)
        self.assertEqual({row.message_id for row in snapshot}, {"ego-2", "rsu-1"})

    def test_capacity_is_shared_across_payload_kinds_for_one_agent(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=1)
        memory.ingest(
            message(
                "older-bev",
                event_time=1,
                arrival_time=1,
                payload_kind="bev",
            )
        )
        newest = message(
            "newer-query",
            event_time=2,
            arrival_time=2,
            payload_kind="query",
        )
        memory.ingest(newest)
        self.assertEqual(memory.snapshot(decision_time=3), (newest,))

    def test_post_commit_backfill_is_rejected_even_with_old_declared_arrival(
        self,
    ) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=4)
        memory.ingest(message("known", event_time=1, arrival_time=1))
        memory.commit_output(decision_time=2)
        with self.assertRaisesRegex(CausalMemoryError, "backfill"):
            memory.ingest(message("forged-old", event_time=1, arrival_time=1))

    def test_later_commits_do_not_rewrite_historical_snapshots(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=4)
        original = message("historical", event_time=1, arrival_time=1)
        memory.ingest(original)
        before = memory.snapshot(decision_time=1)
        memory.commit_output(decision_time=1)
        memory.commit_output(decision_time=20)
        self.assertEqual(memory.snapshot(decision_time=1), before)

    def test_identity_cannot_be_reused_after_later_commits(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=4)
        memory.ingest(message("global-id", event_time=1, arrival_time=1))
        memory.commit_output(decision_time=1)
        memory.commit_output(decision_time=20)
        with self.assertRaisesRegex(CausalMemoryError, "reused"):
            memory.ingest(
                message(
                    "global-id",
                    event_time=21,
                    arrival_time=21,
                    digest_character="c",
                )
            )

    def test_equal_timestamp_capacity_uses_stable_message_identity(self) -> None:
        memory = CausalMessageMemory(max_age=10, max_messages_per_agent=1)
        memory.ingest(
            message(
                "z-bev",
                event_time=1,
                arrival_time=1,
                payload_kind="bev",
            )
        )
        memory.ingest(
            message(
                "a-trajectory",
                event_time=1,
                arrival_time=1,
                payload_kind="trajectory",
            )
        )
        self.assertEqual(
            tuple(row.message_id for row in memory.snapshot(decision_time=1)),
            ("z-bev",),
        )

    def test_zero_max_age_is_rejected(self) -> None:
        with self.assertRaisesRegex(CausalMemoryError, "max_age must be positive"):
            CausalMessageMemory(max_age=0, max_messages_per_agent=1)

    def test_huge_integer_reliability_uses_domain_error(self) -> None:
        with self.assertRaisesRegex(CausalMemoryError, "numeric bound"):
            MessageRecord(
                message_id="huge-reliability",
                agent_id="ego",
                payload_kind="query",
                event_time=1,
                arrival_time=1,
                encoded_bytes=1,
                payload_sha256="a" * 64,
                reliability=10**1000,
            )


class ReliabilityAssociationTest(unittest.TestCase):
    def test_cost_terms_match_the_frozen_equation(self) -> None:
        config = AssociationCostConfig(1.0, 1.0, 1.0, 1.0, 1.0)
        result = compose_association_cost(
            mahalanobis_distance=[[2.0, 0.0]],
            bev_iou=[[0.5, 1.0]],
            embedding_cosine=[[0.25, 1.0]],
            normalized_age=[[0.2, 0.0]],
            reliability=[[0.5, 1.0]],
            gate=[[True, False]],
            config=config,
        )
        self.assertAlmostEqual(result[0][0], 3.45 - math.log(0.5))
        self.assertEqual(result[0][1], math.inf)

    def test_low_cost_pairs_are_matched(self) -> None:
        result = solve_with_unmatched(
            [[1.0, 100.0], [100.0, 1.0]],
            query_unmatched_costs=[5.0, 5.0],
            candidate_new_costs=[5.0, 5.0],
        )
        self.assertEqual(result.matches, ((0, 0), (1, 1)))
        self.assertEqual(result.unmatched_queries, ())
        self.assertEqual(result.new_candidates, ())
        self.assertEqual(result.total_cost, 2.0)

    def test_explicit_unmatched_terms_can_beat_a_bad_pair(self) -> None:
        result = solve_with_unmatched(
            [[20.0]],
            query_unmatched_costs=[3.0],
            candidate_new_costs=[4.0],
        )
        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_queries, (0,))
        self.assertEqual(result.new_candidates, (0,))
        self.assertEqual(result.total_cost, 7.0)

    def test_zero_queries_marks_every_initial_detection_as_new(self) -> None:
        result = solve_with_unmatched(
            [],
            query_unmatched_costs=[],
            candidate_new_costs=[3.0, 4.0],
        )
        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_queries, ())
        self.assertEqual(result.new_candidates, (0, 1))
        self.assertEqual(result.total_cost, 7.0)

    def test_rectangular_problem_accounts_for_every_query_and_candidate(self) -> None:
        result = solve_with_unmatched(
            [[1.0], [2.0]],
            query_unmatched_costs=[4.0, 4.0],
            candidate_new_costs=[3.0],
        )
        self.assertEqual(result.matches, ((0, 0),))
        self.assertEqual(result.unmatched_queries, (1,))
        self.assertEqual(result.new_candidates, ())
        self.assertEqual(result.total_cost, 5.0)

    def test_solver_matches_exhaustive_small_problem_optima(self) -> None:
        rng = random.Random(3407)
        for query_count, candidate_count in ((1, 2), (2, 1), (2, 2), (3, 2)):
            for _ in range(20):
                costs = [
                    [float(rng.randrange(0, 12)) for _ in range(candidate_count)]
                    for _ in range(query_count)
                ]
                query_costs = [float(rng.randrange(0, 8)) for _ in range(query_count)]
                candidate_costs = [
                    float(rng.randrange(0, 8)) for _ in range(candidate_count)
                ]
                observed = solve_with_unmatched(
                    costs,
                    query_unmatched_costs=query_costs,
                    candidate_new_costs=candidate_costs,
                )

                best = math.inf

                def visit(
                    query_index: int,
                    used_candidates: frozenset[int],
                    running_cost: float,
                ) -> None:
                    nonlocal best
                    if query_index == query_count:
                        total = running_cost + sum(
                            candidate_costs[index]
                            for index in range(candidate_count)
                            if index not in used_candidates
                        )
                        best = min(best, total)
                        return
                    visit(
                        query_index + 1,
                        used_candidates,
                        running_cost + query_costs[query_index],
                    )
                    for candidate_index in range(candidate_count):
                        if candidate_index in used_candidates:
                            continue
                        visit(
                            query_index + 1,
                            used_candidates | {candidate_index},
                            running_cost + costs[query_index][candidate_index],
                        )

                visit(0, frozenset(), 0.0)
                self.assertAlmostEqual(observed.total_cost, best)

    def test_extreme_costs_fail_before_aggregate_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric bound"):
            solve_with_unmatched(
                [[math.inf]],
                query_unmatched_costs=[1e308],
                candidate_new_costs=[1e308],
            )

    def test_huge_integer_inputs_use_domain_errors(self) -> None:
        huge = 10**1000
        with self.assertRaisesRegex(ValueError, "numeric bound"):
            AssociationCostConfig(huge, 1.0, 1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "numeric bound"):
            solve_with_unmatched(
                [[huge]],
                query_unmatched_costs=[1.0],
                candidate_new_costs=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "numeric bound"):
            solve_with_unmatched(
                [[1.0]],
                query_unmatched_costs=[huge],
                candidate_new_costs=[1.0],
            )


class CommunicationSelectionTest(unittest.TestCase):
    def test_exact_budget_can_prefer_two_messages_over_one(self) -> None:
        result = select_budgeted_messages(
            [
                candidate("a", encoded_bytes=6, utility=10.0),
                candidate("b", encoded_bytes=5, utility=9.0),
                candidate("c", encoded_bytes=5, utility=9.0),
            ],
            budget_bytes=10,
        )
        self.assertEqual([row.message_id for row in result.selected], ["b", "c"])
        self.assertEqual(result.bytes_used, 10)
        self.assertEqual(result.total_utility, 18.0)

    def test_deadline_and_non_positive_utility_are_rejected_before_selection(
        self,
    ) -> None:
        result = select_budgeted_messages(
            [
                candidate(
                    "late", encoded_bytes=1, utility=100.0, arrival=6, deadline=5
                ),
                candidate("zero", encoded_bytes=1, utility=0.0),
                candidate("valid", encoded_bytes=2, utility=1.0),
            ],
            budget_bytes=2,
        )
        self.assertEqual([row.message_id for row in result.selected], ["valid"])
        self.assertEqual(
            dict(result.rejected),
            {
                "late": "predicted_deadline_miss",
                "zero": "non_positive_utility",
            },
        )

    def test_ties_are_resolved_by_stable_message_identity(self) -> None:
        result = select_budgeted_messages(
            [
                candidate("b", encoded_bytes=1, utility=1.0),
                candidate("a", encoded_bytes=1, utility=1.0),
            ],
            budget_bytes=1,
        )
        self.assertEqual([row.message_id for row in result.selected], ["a"])
        self.assertEqual(
            dict(result.rejected),
            {"b": "not_selected_by_exact_budget_optimization"},
        )

    def test_strictly_larger_utility_wins_even_below_old_tolerance(self) -> None:
        result = select_budgeted_messages(
            [
                candidate("a", encoded_bytes=1, utility=1.0),
                candidate("z", encoded_bytes=1, utility=1.0000000000005),
            ],
            budget_bytes=1,
        )
        self.assertEqual([row.message_id for row in result.selected], ["z"])

    def test_equal_utility_prefers_fewer_bytes_before_identity(self) -> None:
        result = select_budgeted_messages(
            [
                candidate("a", encoded_bytes=2, utility=1.0),
                candidate("z", encoded_bytes=1, utility=1.0),
            ],
            budget_bytes=2,
        )
        self.assertEqual([row.message_id for row in result.selected], ["z"])
        self.assertEqual(result.solver_id, "exact-sparse-pareto-dp-v1")
        self.assertGreaterEqual(result.frontier_peak, 1)

    def test_selector_matches_exhaustive_small_problem_optima(self) -> None:
        rng = random.Random(4909)
        for _ in range(30):
            rows = [
                candidate(
                    f"m{index}",
                    encoded_bytes=rng.randrange(1, 8),
                    utility=float(rng.randrange(1, 15)),
                )
                for index in range(7)
            ]
            budget = rng.randrange(1, 18)
            observed = select_budgeted_messages(rows, budget_bytes=budget)
            best: tuple[float, int, tuple[str, ...]] = (0.0, 0, ())
            for subset_size in range(len(rows) + 1):
                for subset in combinations(rows, subset_size):
                    used = sum(row.encoded_bytes for row in subset)
                    if used > budget:
                        continue
                    state = (
                        sum(row.net_utility for row in subset),
                        used,
                        tuple(sorted(row.message_id for row in subset)),
                    )
                    if state[0] > best[0] or (
                        state[0] == best[0]
                        and (
                            state[1] < best[1]
                            or (state[1] == best[1] and state[2] < best[2])
                        )
                    ):
                        best = state
            self.assertEqual(observed.total_utility, best[0])
            self.assertEqual(observed.bytes_used, best[1])
            self.assertEqual(
                tuple(row.message_id for row in observed.selected), best[2]
            )

    def test_huge_integer_utility_uses_domain_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric bound"):
            candidate("huge", encoded_bytes=1, utility=10**1000)


if __name__ == "__main__":
    unittest.main()
