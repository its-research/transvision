#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from causal_tracking_canary import generate_observations, run_method  # noqa: E402


class CausalTrackingCanaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = generate_observations(seed=3407, steps=48, packet_loss=0.08, max_latency=5)

    def test_replay_is_deterministic(self) -> None:
        replay = generate_observations(seed=3407, steps=48, packet_loss=0.08, max_latency=5)
        self.assertEqual(self.events, replay)

    def test_no_future_message_is_consumed(self) -> None:
        result = run_method(self.events, steps=48, method="reliability")
        self.assertEqual(result["future_messages_consumed"], 0)

    def test_reliability_update_is_not_regressive(self) -> None:
        naive = run_method(self.events, steps=48, method="naive")
        reliability = run_method(self.events, steps=48, method="reliability")
        self.assertLessEqual(reliability["position_rmse"], naive["position_rmse"] * 1.05)


if __name__ == "__main__":
    unittest.main()
