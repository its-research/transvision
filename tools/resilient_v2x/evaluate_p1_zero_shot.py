#!/usr/bin/env python3
"""Run the P1 inference config with a separately owned P0 checkpoint.

This is deliberately a thin, extension-only adapter.  The sealed 26-subject
evaluator remains byte-for-byte unchanged; this process installs one additional
configuration subject and then delegates all 12-condition planning, execution,
and evidence generation to that evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.resilient_v2x import evaluate_controlled_baselines as evaluator  # noqa: E402


CONFIG_SUBJECT = "reliability_gated_residual"
CONFIG_PATH = (
    ROOT
    / "configs"
    / "resilient_v2x"
    / "improvements"
    / "reliability_gated_residual.py"
)


def install_extension_subject() -> None:
    """Install exactly one inference-only config without editing the producer."""

    if not CONFIG_PATH.is_file() or CONFIG_PATH.is_symlink():
        raise RuntimeError(f"P1 config is unavailable: {CONFIG_PATH}")
    existing = evaluator.EVALUATION_CONFIGS.get(CONFIG_SUBJECT)
    if existing is not None and Path(existing).resolve() != CONFIG_PATH.resolve():
        raise RuntimeError("P1 evaluator subject is already bound to another config")
    if CONFIG_SUBJECT not in evaluator.IMPROVEMENTS:
        evaluator.IMPROVEMENTS = (*evaluator.IMPROVEMENTS, CONFIG_SUBJECT)
    if CONFIG_SUBJECT not in evaluator.EVALUATION_SUBJECTS:
        evaluator.EVALUATION_SUBJECTS = (
            *evaluator.EVALUATION_SUBJECTS,
            CONFIG_SUBJECT,
        )
    evaluator.IMPROVEMENT_CONFIGS[CONFIG_SUBJECT] = CONFIG_PATH
    evaluator.EVALUATION_CONFIGS[CONFIG_SUBJECT] = CONFIG_PATH


def main() -> int:
    install_extension_subject()
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
