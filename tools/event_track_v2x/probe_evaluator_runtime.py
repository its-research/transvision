#!/usr/bin/env python3
"""Report whether the pinned TrackEval and nuScenes runtimes are available."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.evaluator_runtime import (  # noqa: E402
    probe_evaluator_runtime_v1,
)


def main() -> int:
    status = probe_evaluator_runtime_v1()
    print(
        json.dumps(
            {
                "nuscenes_devkit_version": status.nuscenes_devkit_version,
                "observed_version": status.observed_version,
                "ready": status.ready,
                "reason": status.reason,
                "trackeval_version": status.trackeval_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
