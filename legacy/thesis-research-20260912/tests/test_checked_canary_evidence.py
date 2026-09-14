from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "clearml" / "download_canary_evidence.py"
SPEC = importlib.util.spec_from_file_location(
    "checked_canary_download_validator", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TASK_ID = "b824e7114467492aa71f87c0dad0d555"
SCRIPT_SHA256 = "cf29d5f5fcfea582d9caeb93fb3855ba9066c79bad967f252648fdd135805521"
CONFIG_SHA256 = "c56a9dd437ffd86895251261e40cbe190ceb6447c41adc051a70a77ebbc2b5dc"
RECEIPT_SHA256 = "8d487a783879bef7767054e99e7d78bd09e5fb35dafa512ce95634973d5f156b"
EVIDENCE = ROOT / "evidence" / "clearml" / "synth-causal-canary" / TASK_ID


class CheckedCanaryEvidenceTest(unittest.TestCase):
    def test_committed_bundle_matches_published_receipt(self) -> None:
        staged = {
            "metrics": EVIDENCE / "metrics.json",
            "events": EVIDENCE / "events.jsonl",
            "run_manifest": EVIDENCE / "run_manifest.json",
        }
        expected_receipt = MODULE.validate_staged_bundle(
            staged,
            expected_task_id=TASK_ID,
            expected_protocol=MODULE.PROTOCOL_ID,
            expected_script_sha256=SCRIPT_SHA256,
            expected_diff_sha256=SCRIPT_SHA256,
            expected_config_sha256=CONFIG_SHA256,
            require_a100=True,
        )
        expected_receipt["task_status"] = "published"

        receipt_path = EVIDENCE / "receipt.json"
        actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(actual_receipt, expected_receipt)
        self.assertEqual(MODULE.file_sha256(receipt_path), RECEIPT_SHA256)


if __name__ == "__main__":
    unittest.main()
