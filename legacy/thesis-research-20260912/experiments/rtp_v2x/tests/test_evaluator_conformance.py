#!/usr/bin/env python3

from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_ROOT))

from evaluator_conformance import (  # noqa: E402
    CONTRACT_ID,
    EvaluationInputError,
    ForecastSample,
    TimedObservation,
    TrackPoint,
    evaluate_forecasts,
    evaluate_tracking,
    normalize_mode_probabilities,
    run_protocol_conformance,
    select_causal_observations,
)


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "evaluator-conformance-v1.json"
)


def point(frame_index: int, track_id: str, x: float) -> TrackPoint:
    return TrackPoint(frame_index, track_id, (x, 0.0, 0.0))


def forecast(
    sample_id: str,
    *,
    truth: tuple[tuple[float, float], ...],
    modes: tuple[tuple[tuple[float, float], ...], ...],
    weights: tuple[float, ...],
) -> ForecastSample:
    return ForecastSample(sample_id, truth, modes, weights)


class EmbeddedProtocolTest(unittest.TestCase):
    def test_all_embedded_vectors_pass(self) -> None:
        summary = run_protocol_conformance(PROTOCOL_PATH)
        self.assertEqual(summary["contract_id"], CONTRACT_ID)
        self.assertFalse(summary["official_evaluator_equivalence"])
        self.assertEqual(summary["passed_case_count"], 8)
        self.assertRegex(summary["protocol_sha256"], r"^[0-9a-f]{64}$")

    def test_protocol_cannot_escalate_scientific_claim(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        document["scientific_claim_allowed"] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "protocol.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationInputError, "cannot allow"):
                run_protocol_conformance(path)

    def test_empty_or_weakened_protocol_vectors_are_rejected(self) -> None:
        for mutation in ("empty_groups", "empty_expected"):
            with self.subTest(mutation=mutation):
                document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
                if mutation == "empty_groups":
                    document["test_vectors"] = {}
                else:
                    document["test_vectors"]["tracking"][0]["expected"] = {}
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "protocol.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(EvaluationInputError):
                        run_protocol_conformance(path)


class TrackingReferenceTest(unittest.TestCase):
    def test_input_order_does_not_change_global_matching(self) -> None:
        ground_truth = [point(0, "g1", 10.0), point(0, "g0", 0.0)]
        predictions = [point(0, "p1", 10.0), point(0, "p0", 0.0)]
        forward = evaluate_tracking(
            ground_truth, predictions, distance_threshold_m=0.0
        )
        reverse = evaluate_tracking(
            list(reversed(ground_truth)),
            list(reversed(predictions)),
            distance_threshold_m=0.0,
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.true_positives, 2)
        self.assertEqual(forward.idf1, 1.0)

    def test_strict_minimum_distance_precedes_lexicographic_tie_break(self) -> None:
        metrics = evaluate_tracking(
            [point(0, "g0", 0.0), point(1, "g0", 0.0)],
            [
                point(0, "p0", 5e-13),
                point(0, "p1", 0.0),
                point(1, "p1", 0.0),
            ],
            distance_threshold_m=1.0,
        )
        self.assertEqual(metrics.identity_switches, 0)
        self.assertAlmostEqual(metrics.idf1, 0.8)

    def test_distance_equal_to_gate_is_a_match(self) -> None:
        metrics = evaluate_tracking(
            [point(0, "g0", 0.0)],
            [point(0, "p0", 2.0)],
            distance_threshold_m=2.0,
        )
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 0)
        self.assertEqual(metrics.false_negatives, 0)

    def test_identity_change_after_miss_counts_switch_and_fragment(self) -> None:
        metrics = evaluate_tracking(
            [point(0, "g0", 0.0), point(1, "g0", 1.0), point(2, "g0", 2.0)],
            [point(0, "p0", 0.0), point(2, "p1", 2.0)],
            distance_threshold_m=0.0,
        )
        self.assertEqual(metrics.identity_switches, 1)
        self.assertEqual(metrics.fragmentations, 1)

    def test_terminal_miss_without_recovery_is_not_fragmentation(self) -> None:
        metrics = evaluate_tracking(
            [point(0, "g0", 0.0), point(1, "g0", 1.0)],
            [point(0, "p0", 0.0)],
            distance_threshold_m=0.0,
        )
        self.assertEqual(metrics.fragmentations, 0)

    def test_non_finite_tracking_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "finite number"):
            evaluate_tracking(
                [TrackPoint(0, "g0", (math.nan, 0.0, 0.0))],
                [],
                distance_threshold_m=1.0,
            )
        with self.assertRaisesRegex(EvaluationInputError, "finite number"):
            evaluate_tracking(
                [point(0, "g0", 0.0)],
                [],
                distance_threshold_m=math.inf,
            )

    def test_duplicate_identity_in_one_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "duplicate"):
            evaluate_tracking(
                [point(0, "g0", 0.0), point(0, "g0", 1.0)],
                [],
                distance_threshold_m=1.0,
            )

    def test_empty_ground_truth_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "ground_truth"):
            evaluate_tracking([], [], distance_threshold_m=1.0)


class CausalOrderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            TimedObservation("future", "rsu", 2, 5),
            TimedObservation("c", "ego", 2, 3),
            TimedObservation("a", "ego", 0, 2),
            TimedObservation("b", "rsu", 1, 3),
        ]

    def test_future_message_is_excluded_and_order_is_canonical(self) -> None:
        selected = select_causal_observations(self.rows, decision_time=3)
        self.assertEqual([row.message_id for row in selected], ["a", "b", "c"])
        replay = select_causal_observations(
            list(reversed(self.rows)), decision_time=3
        )
        self.assertEqual(selected, replay)

    def test_event_after_arrival_is_rejected_even_when_message_is_future(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "event_time after arrival_time"):
            select_causal_observations(
                [TimedObservation("bad", "rsu", 8, 7)], decision_time=1
            )

    def test_duplicate_message_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "duplicate message_id"):
            select_causal_observations(
                [
                    TimedObservation("same", "ego", 0, 0),
                    TimedObservation("same", "rsu", 0, 1),
                ],
                decision_time=1,
            )


class ForecastReferenceTest(unittest.TestCase):
    def test_probability_weights_are_l1_normalized(self) -> None:
        probabilities = normalize_mode_probabilities((2.0, 1.0, 0.0))
        self.assertAlmostEqual(probabilities[0], 2.0 / 3.0)
        self.assertAlmostEqual(probabilities[1], 1.0 / 3.0)
        self.assertEqual(probabilities[2], 0.0)
        self.assertAlmostEqual(math.fsum(probabilities), 1.0)

    def test_invalid_probability_vectors_are_rejected(self) -> None:
        invalid = [
            (),
            (0.0, 0.0),
            (-1.0, 2.0),
            (math.nan, 1.0),
            (math.inf, 1.0),
        ]
        for weights in invalid:
            with self.subTest(weights=weights):
                with self.assertRaises(EvaluationInputError):
                    normalize_mode_probabilities(weights)

    def test_minade_and_minfde_choose_modes_independently(self) -> None:
        sample = forecast(
            "independent",
            truth=((0.0, 0.0), (0.0, 0.0)),
            modes=(
                ((0.0, 0.0), (2.0, 0.0)),
                ((1.1, 0.0), (1.1, 0.0)),
            ),
            weights=(0.5, 0.5),
        )
        metrics = evaluate_forecasts(
            [sample], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=10
        )
        self.assertAlmostEqual(metrics.min_ade_m, 1.0)
        self.assertAlmostEqual(metrics.min_fde_m, 1.1)
        self.assertEqual(metrics.miss_rate, 0.0)

    def test_miss_threshold_equality_is_a_hit(self) -> None:
        sample = forecast(
            "boundary",
            truth=((0.0, 0.0),),
            modes=(((2.0, 0.0),),),
            weights=(1.0,),
        )
        metrics = evaluate_forecasts(
            [sample], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
        )
        self.assertEqual(metrics.miss_rate, 0.0)
        self.assertEqual(metrics.top_mode_brier, 0.0)
        self.assertEqual(metrics.top_mode_ece, 0.0)
        self.assertEqual(metrics.top_mode_aurc, 0.0)

    def test_fixed_gaussian_nll_includes_all_time_and_dimensions(self) -> None:
        exact = forecast(
            "exact",
            truth=((0.0, 0.0), (1.0, 0.0)),
            modes=(((0.0, 0.0), (1.0, 0.0)),),
            weights=(1.0,),
        )
        metrics = evaluate_forecasts(
            [exact], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=10
        )
        self.assertAlmostEqual(metrics.mixture_nll, 2.0 * math.log(2.0 * math.pi))

    def test_top_mode_tie_uses_lower_index_not_oracle_mode(self) -> None:
        sample = forecast(
            "tie",
            truth=((0.0, 0.0),),
            modes=(((3.0, 0.0),), ((0.0, 0.0),)),
            weights=(1.0, 1.0),
        )
        metrics = evaluate_forecasts(
            [sample], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
        )
        self.assertEqual(metrics.min_fde_m, 0.0)
        self.assertEqual(metrics.miss_rate, 0.0)
        self.assertAlmostEqual(metrics.top_mode_brier, 0.25)
        self.assertAlmostEqual(metrics.top_mode_ece, 0.5)
        self.assertEqual(metrics.top_mode_aurc, 1.0)

    def test_ece_and_aurc_use_top_mode_confidence_and_prefix_risk(self) -> None:
        high_confidence_miss = forecast(
            "a_miss",
            truth=((0.0, 0.0),),
            modes=(((3.0, 0.0),), ((4.0, 0.0),)),
            weights=(0.9, 0.1),
        )
        lower_confidence_hit = forecast(
            "b_hit",
            truth=((0.0, 0.0),),
            modes=(((0.0, 0.0),), ((3.0, 0.0),)),
            weights=(0.6, 0.4),
        )
        metrics = evaluate_forecasts(
            [lower_confidence_hit, high_confidence_miss],
            miss_threshold_m=2.0,
            nll_sigma_m=1.0,
            ece_bins=10,
        )
        self.assertAlmostEqual(metrics.miss_rate, 0.5)
        self.assertAlmostEqual(metrics.top_mode_brier, (0.9**2 + 0.4**2) / 2.0)
        self.assertAlmostEqual(metrics.top_mode_ece, 0.65)
        self.assertAlmostEqual(metrics.top_mode_aurc, 0.75)

    def test_equal_confidence_aurc_tie_is_sample_id_deterministic(self) -> None:
        hit = forecast(
            "a_hit",
            truth=((0.0, 0.0),),
            modes=(((0.0, 0.0),), ((3.0, 0.0),)),
            weights=(0.5, 0.5),
        )
        miss = forecast(
            "b_miss",
            truth=((0.0, 0.0),),
            modes=(((3.0, 0.0),), ((4.0, 0.0),)),
            weights=(0.5, 0.5),
        )
        forward = evaluate_forecasts(
            [miss, hit], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
        )
        replay = evaluate_forecasts(
            [hit, miss], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
        )
        self.assertEqual(forward, replay)
        self.assertAlmostEqual(forward.top_mode_aurc, 0.25)

    def test_non_finite_forecast_input_and_parameters_are_rejected(self) -> None:
        invalid_sample = forecast(
            "bad",
            truth=((math.inf, 0.0),),
            modes=(((0.0, 0.0),),),
            weights=(1.0,),
        )
        with self.assertRaisesRegex(EvaluationInputError, "finite number"):
            evaluate_forecasts(
                [invalid_sample], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
            )
        valid_sample = forecast(
            "valid",
            truth=((0.0, 0.0),),
            modes=(((0.0, 0.0),),),
            weights=(1.0,),
        )
        for kwargs in (
            {"miss_threshold_m": math.nan, "nll_sigma_m": 1.0, "ece_bins": 2},
            {"miss_threshold_m": 2.0, "nll_sigma_m": 0.0, "ece_bins": 2},
            {"miss_threshold_m": 2.0, "nll_sigma_m": 1.0, "ece_bins": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(EvaluationInputError):
                    evaluate_forecasts([valid_sample], **kwargs)

    def test_finite_but_overflowing_forecast_is_rejected(self) -> None:
        overflowing = forecast(
            "overflow",
            truth=((0.0, 0.0),),
            modes=(((1e308, 0.0),),),
            weights=(1.0,),
        )
        with self.assertRaisesRegex(EvaluationInputError, "overflow|non-finite"):
            evaluate_forecasts(
                [overflowing],
                miss_threshold_m=2.0,
                nll_sigma_m=1.0,
                ece_bins=2,
            )

    def test_duplicate_sample_id_is_rejected(self) -> None:
        sample = forecast(
            "duplicate",
            truth=((0.0, 0.0),),
            modes=(((0.0, 0.0),),),
            weights=(1.0,),
        )
        with self.assertRaisesRegex(EvaluationInputError, "duplicate sample_id"):
            evaluate_forecasts(
                [sample, sample], miss_threshold_m=2.0, nll_sigma_m=1.0, ece_bins=2
            )


if __name__ == "__main__":
    unittest.main()
