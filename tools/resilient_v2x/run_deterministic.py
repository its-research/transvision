#!/usr/bin/env python3
"""Run a Python entry point with the controlled CUDA determinism flags set."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


_TORCH_TRUE_VALUES = frozenset({"1", "y", "yes", "true"})


def _enable_trusted_mmengine_checkpoint_loading() -> None:
    """Restore legacy loading for provenance-checked MMEngine checkpoints."""

    force_weights_only = os.environ.get(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        "",
    ).strip().casefold()
    if force_weights_only in _TORCH_TRUE_VALUES:
        raise RuntimeError(
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD conflicts with the trusted "
            "MMEngine checkpoint compatibility policy"
        )
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def _disable_tf32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("a Python entry point is required")

    entry_point = Path(arguments[0])
    _enable_trusted_mmengine_checkpoint_loading()
    _disable_tf32()
    sys.argv = [str(entry_point), *arguments[1:]]
    runpy.run_path(str(entry_point), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
