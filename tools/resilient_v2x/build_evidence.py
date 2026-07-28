#!/usr/bin/env python3
"""Bundle existing evaluation outputs and a complexity profile without inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.evaluation.resilient_v2x_evidence import (  # noqa: E402
    PROFILE_DOCUMENT_TYPE,
    EvidenceError,
    build_evidence_document,
    digest_artifact,
    read_document,
    write_document,
)


def _parse_role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("artifact must have the form ROLE=PATH")
    return role, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hashed experiment-evidence bundle from metrics and "
            "predictions already written by the evaluation run. No model is loaded."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--metrics-key", help="optional dotted path to the metrics object")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_parse_role_path,
        metavar="ROLE=PATH",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON object: {path}") from error
    if not isinstance(value, Mapping):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return dict(value)


def _select_mapping(
    value: Mapping[str, object],
    dotted_key: str | None,
) -> dict[str, object]:
    selected: object = value
    if dotted_key:
        for component in dotted_key.split("."):
            if not component or not isinstance(selected, Mapping) or component not in selected:
                raise EvidenceError(f"metrics key does not exist: {dotted_key}")
            selected = selected[component]
    if not isinstance(selected, Mapping):
        raise EvidenceError("selected metrics value must be an object")
    return dict(selected)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_root = _read_json_object(args.metrics)
    metrics = _select_mapping(metrics_root, args.metrics_key)
    profile = read_document(args.profile, expected_type=PROFILE_DOCUMENT_TYPE)
    artifacts = [
        digest_artifact(args.metrics, "evaluation_metrics"),
        digest_artifact(args.profile, "complexity_profile"),
    ]
    prediction_role = None
    if args.predictions is not None:
        prediction_role = "evaluation_predictions"
        artifacts.append(digest_artifact(args.predictions, prediction_role))
    conditions: dict[str, object] = {}
    if args.conditions is not None:
        conditions = _read_json_object(args.conditions)
        artifacts.append(digest_artifact(args.conditions, "condition_spec"))
    artifacts.extend(digest_artifact(path, role) for role, path in args.artifact)
    document = build_evidence_document(
        run_id=args.run_id,
        metrics=metrics,
        complexity_profile=profile,
        artifacts=artifacts,
        conditions=conditions,
        measured_at_utc=_utc_now(),
        prediction_artifact_role=prediction_role,
    )
    destination = write_document(args.out, document)
    print(
        json.dumps(
            {
                "output": str(destination),
                "content_sha256": document["content_sha256"],
                "inference_performed": False,
                "metric_recomputation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
