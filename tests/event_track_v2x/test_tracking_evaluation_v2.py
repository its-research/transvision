"""Fail-closed archive checks plus real-engine golden cases when installed."""
import copy
import hashlib
import importlib.util
import unittest

import numpy as np

from transvision.models.event_track_v2x.tracking_evaluation_v2 import (
    canonical, golden_cases, oriented_bev_iou, validate_predictions,
)


def fixture():
    gt, predictions = [], []
    previous = "0" * 64
    for n, stamp in enumerate((1000000, 1100100, 1750000)):
        gt.append({"sequence_id": "0003", "frame_id": str(n), "box_reference_timestamp_us": stamp})
        p = {"sequence_id": "0003", "frame_id": str(n), "box_reference_timestamp_us": stamp,
             "decision_timestamp_us": stamp + 100000, "coordinate_frame": "world",
             "state_layout": "gravity_xyz_length_width_height_yaw_vxy",
             "previous_commit_sha256": previous,
             "predictions": [{"track_id": "0003:1", "class_label": "car", "score": .9,
                              "mean": [0., 0., 0., 4., 2., 1.5, 0., 0., 0.],
                              "covariance": np.eye(9).tolist()}]}
        p["commit_sha256"] = hashlib.sha256(canonical(p)).hexdigest()
        previous = p["commit_sha256"]
        predictions.append(p)
    return gt, predictions


def reseal(predictions):
    previous = "0" * 64
    for p in predictions:
        p.pop("commit_sha256", None)
        p["previous_commit_sha256"] = previous
        p["commit_sha256"] = hashlib.sha256(canonical(p)).hexdigest()
        previous = p["commit_sha256"]


class PredictionValidation(unittest.TestCase):
    def test_complete_native_schedule_and_commit_chain(self):
        gt, p = fixture()
        self.assertEqual(validate_predictions(p, gt)["frames"], 3)

    def test_missing_or_reordered_frame_fails(self):
        gt, p = fixture()
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_predictions(p[:-1], gt)
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_predictions(p[::-1], gt)

    def test_empty_predictions_remain_explicit_frames(self):
        gt, p = fixture()
        p[1]["predictions"] = []
        reseal(p)
        self.assertEqual(validate_predictions(p, gt)["empty_frames"], 1)

    def test_tampered_seal_and_chain_fail(self):
        gt, p = fixture()
        p[0]["predictions"][0]["score"] = .7
        with self.assertRaisesRegex(ValueError, "seal"):
            validate_predictions(p, gt)
        gt, p = fixture()
        p[1]["previous_commit_sha256"] = "1" * 64
        p[1]["commit_sha256"] = hashlib.sha256(canonical({k: v for k, v in p[1].items() if k != "commit_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError, "chain"):
            validate_predictions(p, gt)

    def test_duplicates_and_non_psd_covariance_fail(self):
        gt, p = fixture()
        p[1]["predictions"].append(copy.deepcopy(p[1]["predictions"][0]))
        reseal(p)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_predictions(p, gt)
        gt, p = fixture()
        p[1]["predictions"][0]["covariance"][0][0] = -1
        reseal(p)
        with self.assertRaisesRegex(ValueError, "covariance"):
            validate_predictions(p, gt)

    def test_wrong_state_time_and_convention_fail(self):
        gt, p = fixture()
        p[0]["box_reference_timestamp_us"] += 1
        reseal(p)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_predictions(p, gt)
        gt, p = fixture()
        p[0]["coordinate_frame"] = "lidar"
        reseal(p)
        with self.assertRaisesRegex(ValueError, "convention"):
            validate_predictions(p, gt)

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "Shapely is not installed")
    def test_rotated_bev_iou(self):
        box = {"mean": [0., 0., 0., 4., 2., 1.5, 0., 0., 0.]}
        rotated = {"mean": [0., 0., 0., 4., 2., 1.5, np.pi / 2, 0., 0.]}
        distant = {"mean": [100., 0., 0., 4., 2., 1.5, 0., 0., 0.]}
        self.assertAlmostEqual(oriented_bev_iou([box], [box])[0, 0], 1.)
        self.assertAlmostEqual(oriented_bev_iou([box], [rotated])[0, 0], 1. / 3)
        self.assertEqual(oriented_bev_iou([box], [distant])[0, 0], 0.)
        self.assertEqual(oriented_bev_iou([], [box]).shape, (0, 1))

    @unittest.skipUnless(importlib.util.find_spec("nuscenes") and importlib.util.find_spec("trackeval"),
                         "official evaluator runtime is not installed")
    def test_official_golden_cases(self):
        self.assertTrue(golden_cases()["passed"])


if __name__ == "__main__":
    unittest.main()
