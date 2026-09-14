#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_ROOT))

import evaluator_conformance as evaluator_v1  # noqa: E402
import evaluator_conformance_v2 as evaluator_v2  # noqa: E402


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "evaluator-conformance-v2.json"
)
IDENTITY_COVARIANCE = ((1.0, 0.0), (0.0, 1.0))


def mode(
    means: tuple[tuple[float, float], ...],
    covariances: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    | None = None,
) -> evaluator_v2.HeteroscedasticForecastMode:
    return evaluator_v2.HeteroscedasticForecastMode(
        means=means,
        covariances=covariances
        if covariances is not None
        else tuple(IDENTITY_COVARIANCE for _ in means),
    )


def sample(
    sample_id: str,
    *,
    truth: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
    modes: tuple[evaluator_v2.HeteroscedasticForecastMode, ...] | None = None,
    weights: tuple[float, ...] = (1.0,),
) -> evaluator_v2.HeteroscedasticForecastSample:
    return evaluator_v2.HeteroscedasticForecastSample(
        sample_id=sample_id,
        truth=truth,
        modes=modes if modes is not None else (mode(truth),),
        mode_weights=weights,
    )


class EmbeddedProtocolTest(unittest.TestCase):
    def test_all_hand_computable_vectors_pass(self) -> None:
        summary = evaluator_v2.run_protocol_conformance(PROTOCOL_PATH)
        self.assertEqual(summary["contract_id"], evaluator_v2.CONTRACT_ID)
        self.assertFalse(summary["scientific_claim_allowed"])
        self.assertFalse(summary["official_evaluator_equivalence"])
        self.assertEqual(summary["passed_case_count"], 3)
        self.assertRegex(summary["protocol_sha256"], r"^[0-9a-f]{64}$")

    def test_v1_contract_remains_a_separate_fixed_isotropic_version(self) -> None:
        self.assertEqual(evaluator_v1.CONTRACT_ID, "RTPV2X-EVALUATOR-CONFORMANCE-v1")
        self.assertNotEqual(evaluator_v1.CONTRACT_ID, evaluator_v2.CONTRACT_ID)
        v1_protocol = json.loads(
            (PROTOCOL_PATH.parent / "evaluator-conformance-v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(
            "fixed-isotropic-Gaussian", v1_protocol["forecasting"]["metrics"]["NLL"]
        )

    def test_protocol_cannot_claim_science_or_official_equivalence(self) -> None:
        for field in ("scientific_claim_allowed", "official_evaluator_equivalence"):
            with self.subTest(field=field):
                document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
                document[field] = True
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "protocol.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(evaluator_v2.EvaluationInputError):
                        evaluator_v2.run_protocol_conformance(path)

    def test_protocol_limits_and_vectors_cannot_be_silently_weakened(self) -> None:
        for mutation in (
            "resource_limit",
            "numeric_limit",
            "missing_case",
            "empty_expected",
        ):
            with self.subTest(mutation=mutation):
                document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
                if mutation == "resource_limit":
                    document["resource_limits"]["maximum_modes_per_sample"] += 1
                elif mutation == "numeric_limit":
                    document["numeric_policy"]["minimum_covariance_eigenvalue_m2"] = 0.0
                elif mutation == "missing_case":
                    document["test_vectors"]["forecasting"].pop()
                else:
                    document["test_vectors"]["forecasting"][0]["expected"] = {}
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "protocol.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(evaluator_v2.EvaluationInputError):
                        evaluator_v2.run_protocol_conformance(path)

    def test_scope_forecasting_and_parity_gate_are_frozen_exactly(self) -> None:
        mutations = (
            ("scope", "purpose", "changed"),
            ("forecasting", "space", "changed"),
            ("parity_gate", "required_before_benchmark_use", False),
        )
        for section, field, replacement in mutations:
            with self.subTest(section=section, field=field):
                document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
                document[section][field] = replacement
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "protocol.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(evaluator_v2.EvaluationInputError):
                        evaluator_v2.run_protocol_conformance(path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = PROTOCOL_PATH.read_text(encoding="utf-8")
        duplicated = raw.replace(
            '"schema_version": 2,',
            '"schema_version": 2,\n  "schema_version": 2,',
            1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "protocol.json"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaisesRegex(
                evaluator_v2.EvaluationInputError, "duplicate key"
            ):
                evaluator_v2.run_protocol_conformance(path)

    def test_protocol_byte_limit_is_enforced_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "oversized.json"
            with path.open("wb") as stream:
                stream.seek(evaluator_v2.MAX_PROTOCOL_BYTES)
                stream.write(b"x")
            with self.assertRaisesRegex(
                evaluator_v2.EvaluationInputError, "resource limit"
            ):
                evaluator_v2.run_protocol_conformance(path)


class HandComputableMetricTest(unittest.TestCase):
    def test_isotropic_covariance_matches_v1_fixed_sigma(self) -> None:
        truth = ((0.0, 0.0), (1.0, 0.0))
        means = (
            ((0.0, 0.0), (1.0, 0.0)),
            ((2.0, 0.0), (3.0, 0.0)),
        )
        sigma = 2.0
        covariance = ((sigma**2, 0.0), (0.0, sigma**2))
        v2_sample = sample(
            "parity",
            truth=truth,
            modes=tuple(
                mode(mode_means, tuple(covariance for _ in truth))
                for mode_means in means
            ),
            weights=(3.0, 1.0),
        )
        v1_sample = evaluator_v1.ForecastSample(
            sample_id="parity",
            truth=truth,
            modes=means,
            mode_weights=(3.0, 1.0),
        )
        actual = evaluator_v2.evaluate_forecasts(
            [v2_sample], miss_threshold_m=2.0, ece_bins=10
        )
        reference = evaluator_v1.evaluate_forecasts(
            [v1_sample],
            miss_threshold_m=2.0,
            nll_sigma_m=sigma,
            ece_bins=10,
        )
        for field, reference_value in asdict(reference).items():
            self.assertAlmostEqual(asdict(actual)[field], reference_value)

    def test_anisotropic_nll_matches_closed_form(self) -> None:
        forecast = sample(
            "anisotropic",
            truth=((1.0, 0.0),),
            modes=(
                mode(
                    ((0.0, 0.0),),
                    (((2.0, 1.0), (1.0, 2.0)),),
                ),
            ),
        )
        metrics = evaluator_v2.evaluate_forecasts(
            [forecast], miss_threshold_m=3.0, ece_bins=5
        )
        expected = math.log(2.0 * math.pi) + 0.5 * math.log(3.0) + 0.5 * (2.0 / 3.0)
        self.assertAlmostEqual(metrics.mixture_nll, expected)
        self.assertAlmostEqual(metrics.min_ade_m, 1.0)

    def test_mixture_nll_sums_steps_before_logsumexp_across_modes(self) -> None:
        forecast = sample(
            "mixture",
            truth=((0.0, 0.0), (0.0, 0.0)),
            modes=(
                mode(((0.0, 0.0), (0.0, 0.0))),
                mode(((1.0, 0.0), (1.0, 0.0))),
            ),
            weights=(3.0, 1.0),
        )
        metrics = evaluator_v2.evaluate_forecasts(
            [forecast], miss_threshold_m=2.0, ece_bins=4
        )
        expected = 2.0 * math.log(2.0 * math.pi) - math.log(
            0.75 + 0.25 * math.exp(-1.0)
        )
        self.assertAlmostEqual(metrics.mixture_nll, expected)
        self.assertAlmostEqual(metrics.top_mode_brier, 0.25**2)
        self.assertAlmostEqual(metrics.top_mode_ece, 0.25)

    def test_minade_and_minfde_still_choose_modes_independently(self) -> None:
        forecast = sample(
            "independent",
            truth=((0.0, 0.0), (0.0, 0.0)),
            modes=(
                mode(((0.0, 0.0), (2.0, 0.0))),
                mode(((1.1, 0.0), (1.1, 0.0))),
            ),
            weights=(0.5, 0.5),
        )
        metrics = evaluator_v2.evaluate_forecasts(
            [forecast], miss_threshold_m=2.0, ece_bins=10
        )
        self.assertAlmostEqual(metrics.min_ade_m, 1.0)
        self.assertAlmostEqual(metrics.min_fde_m, 1.1)
        self.assertEqual(metrics.miss_rate, 0.0)

    def test_top_mode_calibration_semantics_match_v1_and_ignore_covariance(
        self,
    ) -> None:
        v2_samples = (
            sample(
                "a_miss",
                modes=(
                    mode(
                        ((3.0, 0.0),),
                        (((100.0, 0.0), (0.0, 0.1)),),
                    ),
                    mode(((4.0, 0.0),)),
                ),
                weights=(0.9, 0.1),
            ),
            sample(
                "b_hit",
                modes=(mode(((0.0, 0.0),)), mode(((3.0, 0.0),))),
                weights=(0.6, 0.4),
            ),
        )
        v1_samples = (
            evaluator_v1.ForecastSample(
                "a_miss",
                ((0.0, 0.0),),
                (((3.0, 0.0),), ((4.0, 0.0),)),
                (0.9, 0.1),
            ),
            evaluator_v1.ForecastSample(
                "b_hit",
                ((0.0, 0.0),),
                (((0.0, 0.0),), ((3.0, 0.0),)),
                (0.6, 0.4),
            ),
        )
        actual = evaluator_v2.evaluate_forecasts(
            v2_samples, miss_threshold_m=2.0, ece_bins=10
        )
        reference = evaluator_v1.evaluate_forecasts(
            v1_samples,
            miss_threshold_m=2.0,
            nll_sigma_m=1.0,
            ece_bins=10,
        )
        for field in (
            "min_ade_m",
            "min_fde_m",
            "miss_rate",
            "top_mode_brier",
            "top_mode_ece",
            "top_mode_aurc",
        ):
            self.assertAlmostEqual(getattr(actual, field), getattr(reference, field))
        self.assertAlmostEqual(actual.top_mode_aurc, 0.75)


class StrictValidationTest(unittest.TestCase):
    def assert_covariance_rejected(self, covariance: object, pattern: str) -> None:
        invalid_mode = evaluator_v2.HeteroscedasticForecastMode(
            means=((0.0, 0.0),),
            covariances=(covariance,),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(evaluator_v2.EvaluationInputError, pattern):
            evaluator_v2.evaluate_forecasts(
                [sample("bad-covariance", modes=(invalid_mode,))],
                miss_threshold_m=2.0,
                ece_bins=10,
            )

    def test_non_spd_covariances_are_rejected(self) -> None:
        for covariance in (
            ((1.0, 0.0), (0.0, -1.0)),
            ((1.0, 1.0), (1.0, 1.0)),
            ((1e-14, 0.0), (0.0, 1.0)),
        ):
            with self.subTest(covariance=covariance):
                self.assert_covariance_rejected(
                    covariance, "positive definite|minimum eigenvalue"
                )

    def test_numerically_unstable_condition_number_is_rejected(self) -> None:
        self.assert_covariance_rejected(
            ((1.0, 0.0), (0.0, evaluator_v2.MAX_COVARIANCE_CONDITION_NUMBER * 10.0)),
            "condition number",
        )

    def test_asymmetric_covariance_is_rejected_without_tolerance(self) -> None:
        self.assert_covariance_rejected(
            ((1.0, 0.1), (0.10000000000000002, 1.0)), "exactly symmetric"
        )

    def test_covariance_and_trajectory_shape_errors_are_rejected(self) -> None:
        invalid_covariances = (
            (1.0, 1.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((1.0, 0.0),),
        )
        for covariance in invalid_covariances:
            with self.subTest(covariance=covariance):
                self.assert_covariance_rejected(covariance, "2x2")

        wrong_mean = evaluator_v2.HeteroscedasticForecastMode(
            means=((0.0, 0.0, 0.0),),  # type: ignore[arg-type]
            covariances=(IDENTITY_COVARIANCE,),
        )
        mismatched_horizon = evaluator_v2.HeteroscedasticForecastMode(
            means=((0.0, 0.0),),
            covariances=(IDENTITY_COVARIANCE, IDENTITY_COVARIANCE),
        )
        for invalid_mode in (wrong_mean, mismatched_horizon):
            with self.subTest(invalid_mode=invalid_mode):
                with self.assertRaises(evaluator_v2.EvaluationInputError):
                    evaluator_v2.evaluate_forecasts(
                        [sample("bad-shape", modes=(invalid_mode,))],
                        miss_threshold_m=2.0,
                        ece_bins=10,
                    )

    def test_non_finite_and_finite_overflow_scale_inputs_are_rejected(self) -> None:
        invalid_samples = (
            sample("truth-nan", truth=((math.nan, 0.0),)),
            sample("mean-inf", modes=(mode(((math.inf, 0.0),)),)),
            sample(
                "covariance-nan",
                modes=(mode(((0.0, 0.0),), (((math.nan, 0.0), (0.0, 1.0)),)),),
            ),
            sample("weight-inf", weights=(math.inf,)),
            sample("coordinate-overflow", truth=((1e308, 0.0),)),
            sample(
                "covariance-overflow",
                modes=(mode(((0.0, 0.0),), (((1e308, 0.0), (0.0, 1.0)),)),),
            ),
        )
        for invalid in invalid_samples:
            with self.subTest(sample_id=invalid.sample_id):
                with self.assertRaisesRegex(
                    evaluator_v2.EvaluationInputError,
                    "finite|bound|exceeds|covariance",
                ):
                    evaluator_v2.evaluate_forecasts(
                        [invalid], miss_threshold_m=2.0, ece_bins=10
                    )

    def test_invalid_parameters_and_weights_are_rejected(self) -> None:
        valid = sample("valid")
        for kwargs in (
            {"miss_threshold_m": math.nan, "ece_bins": 10},
            {"miss_threshold_m": -1.0, "ece_bins": 10},
            {"miss_threshold_m": 2.0, "ece_bins": 0},
            {"miss_threshold_m": 2.0, "ece_bins": evaluator_v2.MAX_ECE_BINS + 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(evaluator_v2.EvaluationInputError):
                    evaluator_v2.evaluate_forecasts([valid], **kwargs)
        for weights in ((), (0.0,), (-1.0,), (evaluator_v2.MAX_MODE_WEIGHT * 10.0,)):
            with self.subTest(weights=weights):
                with self.assertRaises(evaluator_v2.EvaluationInputError):
                    evaluator_v2.evaluate_forecasts(
                        [sample("weights", weights=weights)],
                        miss_threshold_m=2.0,
                        ece_bins=10,
                    )
        with self.assertRaisesRegex(evaluator_v2.EvaluationInputError, "underflowed"):
            evaluator_v2.evaluate_forecasts(
                [
                    sample(
                        "underflowing-positive-weight",
                        modes=(mode(((0.0, 0.0),)), mode(((100.0, 0.0),))),
                        weights=(5e-324, evaluator_v2.MAX_MODE_WEIGHT),
                    )
                ],
                miss_threshold_m=2.0,
                ece_bins=10,
            )

    def test_duplicate_sample_ids_are_rejected(self) -> None:
        duplicate = sample("duplicate")
        with self.assertRaisesRegex(
            evaluator_v2.EvaluationInputError, "duplicate sample_id"
        ):
            evaluator_v2.evaluate_forecasts(
                [duplicate, duplicate], miss_threshold_m=2.0, ece_bins=10
            )

    def test_invalid_zero_weight_mode_is_still_validated(self) -> None:
        invalid_hidden_mode = evaluator_v2.HeteroscedasticForecastMode(
            means=((0.0, 0.0),),
            covariances=(((1.0, 1.0), (1.0, 1.0)),),
        )
        with self.assertRaisesRegex(
            evaluator_v2.EvaluationInputError, "positive definite"
        ):
            evaluator_v2.evaluate_forecasts(
                [
                    sample(
                        "zero-weight-invalid",
                        modes=(mode(((0.0, 0.0),)), invalid_hidden_mode),
                        weights=(1.0, 0.0),
                    )
                ],
                miss_threshold_m=2.0,
                ece_bins=10,
            )

    def test_resource_bounds_are_enforced_before_scoring(self) -> None:
        valid = sample("valid")
        with self.assertRaisesRegex(evaluator_v2.EvaluationInputError, "sample count"):
            evaluator_v2.evaluate_forecasts(
                [valid] * (evaluator_v2.MAX_SAMPLES + 1),
                miss_threshold_m=2.0,
                ece_bins=10,
            )
        excessive_modes = tuple(
            mode(((float(index), 0.0),)) for index in range(evaluator_v2.MAX_MODES + 1)
        )
        with self.assertRaisesRegex(evaluator_v2.EvaluationInputError, "mode count"):
            evaluator_v2.evaluate_forecasts(
                [
                    sample(
                        "too-many-modes",
                        modes=excessive_modes,
                        weights=tuple(1.0 for _ in excessive_modes),
                    )
                ],
                miss_threshold_m=2.0,
                ece_bins=10,
            )
        long_truth = tuple(
            (0.0, 0.0) for _ in range(evaluator_v2.MAX_HORIZON_STEPS + 1)
        )
        with self.assertRaisesRegex(evaluator_v2.EvaluationInputError, "horizon"):
            evaluator_v2.evaluate_forecasts(
                [sample("too-long", truth=long_truth)],
                miss_threshold_m=2.0,
                ece_bins=10,
            )


if __name__ == "__main__":
    unittest.main()
