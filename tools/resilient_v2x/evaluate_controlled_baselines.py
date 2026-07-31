#!/usr/bin/env python3
"""Evaluate one controlled fusion baseline on the fixed 12-condition matrix."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import keyword
import os
import pprint
import re
import runpy
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.resilient_v2x_manifest import (  # noqa: E402
    canonical_json_bytes,
    content_sha256,
)
from transvision.dataset.resilient_v2x_schedule import (  # noqa: E402
    ScheduleError,
    read_overlay,
)


BASELINES = ("v2x_vit", "cobevt", "coformernet", "bevfusion", "ffnet")
DELAYS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
DEFAULT_OVERLAY_INDEX = Path(
    "artifacts/resilient_v2x/dair_v2/evaluation_overlays.json"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EVALUATION_INDEX_TYPE = "resilient_v2x_evaluation_overlays"

BASELINE_CONFIGS = {
    name: ROOT / "configs" / "resilient_v2x" / "baselines" / f"{name}.py"
    for name in BASELINES
}
CONDITION_CONFIGS = {
    **{
        (delay, "Full"): (
            ROOT
            / "configs"
            / "resilient_v2x"
            / "conditions"
            / f"global_delay_{delay:03d}_full.py"
        )
        for delay in DELAYS
    },
    **{
        (delay, condition): (
            ROOT
            / "configs"
            / "resilient_v2x"
            / "conditions"
            / (
                f"causal_delay_{delay:03d}_"
                f"{'l_fail' if condition == 'L-Fail' else 'c_fail'}.py"
            )
        )
        for delay in DELAYS
        for condition in ("L-Fail", "C-Fail")
    },
}


class ControlledBaselineEvaluationError(ValueError):
    """Raised when a controlled evaluation cannot be resolved safely."""


def _sha256(value: object, context: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ControlledBaselineEvaluationError(
            f"{context} must be lowercase hexadecimal SHA-256"
        )
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ControlledBaselineEvaluationError(f"{context} must be a mapping")
    return dict(value)


def _sequence(value: object, context: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ControlledBaselineEvaluationError(f"{context} must be a sequence")
    return list(value)


def _load_json_mapping(path: Path, context: str) -> tuple[Path, dict[str, object]]:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ControlledBaselineEvaluationError(f"{context} must be a file")
        value = json.loads(source.read_text(encoding="utf-8"))
    except ControlledBaselineEvaluationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledBaselineEvaluationError(
            f"unable to read {context}: {error}"
        ) from error
    return source, _mapping(value, context)


def _verify_content_hash(document: Mapping[str, object], context: str) -> str:
    expected = _sha256(document.get("content_sha256"), f"{context}.content_sha256")
    try:
        actual = content_sha256(document)
    except (TypeError, ValueError) as error:
        raise ControlledBaselineEvaluationError(
            f"unable to hash {context}: {error}"
        ) from error
    if actual != expected:
        raise ControlledBaselineEvaluationError(f"{context} content hash mismatch")
    return expected


def _sha256_file(path: Path, context: str = "checkpoint") -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ControlledBaselineEvaluationError(
            f"unable to hash {context}: {error}"
        ) from error
    return digest.hexdigest()


def _ordered_subset(
    values: Sequence[object],
    order: tuple[object, ...],
    context: str,
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ControlledBaselineEvaluationError(f"{context} must be a sequence")
    selected = tuple(values)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(value not in order for value in selected)
    ):
        raise ControlledBaselineEvaluationError(
            f"{context} must be a unique non-empty protocol subset"
        )
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _merge(base: object, update: object) -> object:
    if isinstance(update, dict) and update.get("_delete_") is True:
        return {
            key: copy.deepcopy(value)
            for key, value in update.items()
            if key != "_delete_"
        }
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key == "_delete_":
            continue
        merged[key] = _merge(merged.get(key), value)
    return merged


def _load_python_config(
    path: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, object]:
    source = Path(path).resolve()
    if source in stack:
        raise ControlledBaselineEvaluationError(
            f"cyclic config inheritance at {source}"
        )
    if not source.is_file():
        raise ControlledBaselineEvaluationError(f"missing config: {source}")
    namespace = runpy.run_path(str(source))
    bases = namespace.get("_base_", ())
    if isinstance(bases, str):
        bases = (bases,)
    if not isinstance(bases, (tuple, list)):
        raise ControlledBaselineEvaluationError(
            f"invalid _base_ declaration in {source}"
        )
    merged: dict[str, object] = {}
    for base in bases:
        if type(base) is not str:
            raise ControlledBaselineEvaluationError(
                f"invalid base config path in {source}"
            )
        parent = _load_python_config(source.parent / base, (*stack, source))
        merged = _merge(merged, parent)  # type: ignore[assignment]
    own = {key: value for key, value in namespace.items() if not key.startswith("_")}
    return _merge(merged, own)  # type: ignore[return-value]


def _resolve_overlay_path(index_dir: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value:
        raise ControlledBaselineEvaluationError(f"{context}.path must be a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ControlledBaselineEvaluationError(
            f"{context}.path must be relative to the overlay index"
        )
    return (index_dir / relative).resolve()


def _overlay_descriptor(
    entry: Mapping[str, object],
    index_dir: Path,
    context: str,
) -> tuple[Path, str]:
    overlay = _mapping(entry.get("overlay"), f"{context}.overlay")
    overlay_path = _resolve_overlay_path(index_dir, overlay.get("path"), context)
    if not overlay_path.is_file():
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay.path does not name a file: {overlay_path}"
        )
    compressed_digest = _sha256(
        overlay.get("compressed_sha256"),
        f"{context}.overlay.compressed_sha256",
    )
    compressed_size = overlay.get("compressed_size")
    if type(compressed_size) is not int or compressed_size < 0:
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay.compressed_size must be a non-negative integer"
        )
    try:
        actual_size = overlay_path.stat().st_size
    except OSError as error:
        raise ControlledBaselineEvaluationError(
            f"unable to inspect {context}.overlay.path: {error}"
        ) from error
    if actual_size != compressed_size:
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay compressed size mismatch"
        )
    if _sha256_file(overlay_path, f"{context}.overlay") != compressed_digest:
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay compressed SHA-256 mismatch"
        )
    digest = _sha256(
        overlay.get("uncompressed_sha256"),
        f"{context}.overlay.uncompressed_sha256",
    )
    return overlay_path, digest


def _evaluation_sample_ids(index: Mapping[str, object]) -> tuple[str, ...]:
    raw = _sequence(index.get("sample_ids"), "overlay index.sample_ids")
    if (
        not raw
        or any(type(value) is not str or not value for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise ControlledBaselineEvaluationError(
            "overlay index.sample_ids must be unique non-empty strings"
        )
    sample_ids = tuple(str(value) for value in raw)
    expected = _sha256(
        index.get("sample_ids_sha256"),
        "overlay index.sample_ids_sha256",
    )
    actual = hashlib.sha256(canonical_json_bytes(sample_ids)).hexdigest()
    if actual != expected:
        raise ControlledBaselineEvaluationError(
            "overlay index sample_ids SHA-256 mismatch"
        )
    return sample_ids


def _verify_overlay_cohort(
    entry: Mapping[str, object],
    path: Path,
    uncompressed_sha256: str,
    expected_sample_ids: tuple[str, ...],
    context: str,
) -> None:
    overlay = _mapping(entry.get("overlay"), f"{context}.overlay")
    record_count = overlay.get("record_count")
    if type(record_count) is not int or record_count <= 0:
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay.record_count must be a positive integer"
        )
    try:
        records = read_overlay(path, uncompressed_sha256)
    except (OSError, ScheduleError) as error:
        raise ControlledBaselineEvaluationError(
            f"unable to verify {context}.overlay stream: {error}"
        ) from error
    if len(records) != record_count:
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay record_count mismatch"
        )
    actual_sample_ids = {record.get("sample_id") for record in records}
    if actual_sample_ids != set(expected_sample_ids):
        raise ControlledBaselineEvaluationError(
            f"{context}.overlay sample IDs differ from the fixed cohort"
        )


def _index_overlays(
    index: Mapping[str, object],
    index_dir: Path,
    manifest_sha256: str,
) -> tuple[
    dict[int, tuple[Path, str]],
    dict[tuple[int, str, str, int], tuple[Path, str, str]],
]:
    expected_sample_ids = _evaluation_sample_ids(index)
    transports: dict[int, tuple[Path, str]] = {}
    for offset, raw in enumerate(
        _sequence(index.get("transport_overlays"), "transport_overlays")
    ):
        entry = _mapping(raw, f"transport_overlays[{offset}]")
        delay = entry.get("delay_ms")
        if type(delay) is not int or delay not in DELAYS:
            raise ControlledBaselineEvaluationError(
                f"transport_overlays[{offset}].delay_ms is invalid"
            )
        if delay in transports:
            raise ControlledBaselineEvaluationError(
                f"duplicate transport overlay for delay {delay}"
            )
        descriptor = _overlay_descriptor(
            entry,
            index_dir,
            f"transport_overlays[{offset}]",
        )
        _verify_overlay_cohort(
            entry,
            *descriptor,
            expected_sample_ids,
            f"transport_overlays[{offset}]",
        )
        transports[delay] = descriptor

    faults: dict[tuple[int, str, str, int], tuple[Path, str, str]] = {}
    for offset, raw in enumerate(
        _sequence(index.get("fault_overlays"), "fault_overlays")
    ):
        entry = _mapping(raw, f"fault_overlays[{offset}]")
        delay = entry.get("delay_ms")
        condition = entry.get("condition")
        scope = entry.get("agent_scope")
        duration = entry.get("duration")
        if type(delay) is not int or delay not in DELAYS:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}].delay_ms is invalid"
            )
        if condition not in CONDITIONS or type(condition) is not str:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}].condition is invalid"
            )
        if scope not in ("E+R", "E-only", "R-only") or type(scope) is not str:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}].agent_scope is invalid"
            )
        agents = tuple(
            _sequence(entry.get("agents"), f"fault_overlays[{offset}].agents")
        )
        expected_agents = {
            "E+R": ("ego", "rsu"),
            "E-only": ("ego",),
            "R-only": ("rsu",),
        }[scope]
        if agents != expected_agents:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}].agents does not match agent_scope"
            )
        if type(duration) is not int or duration <= 0:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}].duration is invalid"
            )
        temporal_digest = _sha256(
            entry.get("temporal_manifest_sha256"),
            f"fault_overlays[{offset}].temporal_manifest_sha256",
        )
        if temporal_digest != manifest_sha256:
            raise ControlledBaselineEvaluationError(
                f"fault_overlays[{offset}] temporal manifest hash mismatch"
            )
        transport_digest = _sha256(
            entry.get("transport_overlay_sha256"),
            f"fault_overlays[{offset}].transport_overlay_sha256",
        )
        key = (delay, condition, scope, duration)
        if key in faults:
            raise ControlledBaselineEvaluationError(
                f"duplicate fault overlay for {key}"
            )
        overlay_path, overlay_digest = _overlay_descriptor(
            entry,
            index_dir,
            f"fault_overlays[{offset}]",
        )
        _verify_overlay_cohort(
            entry,
            overlay_path,
            overlay_digest,
            expected_sample_ids,
            f"fault_overlays[{offset}]",
        )
        faults[key] = (overlay_path, overlay_digest, transport_digest)
    return transports, faults


def _atomic_write(path: Path, raw: bytes) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _atomic_json(path: Path, value: object) -> Path:
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ControlledBaselineEvaluationError(
            f"value cannot be serialized as JSON: {error}"
        ) from error
    return _atomic_write(path, raw)


def _render_config(config: Mapping[str, object]) -> bytes:
    lines = [
        "# Generated by tools/resilient_v2x/evaluate_controlled_baselines.py.",
        "# Do not edit: regenerate from the baseline, condition, and overlay index.",
        "",
    ]
    for key in sorted(config):
        if not key.isidentifier() or keyword.iskeyword(key):
            raise ControlledBaselineEvaluationError(
                f"config key cannot be rendered as Python: {key!r}"
            )
        rendered = pprint.pformat(config[key], width=88, sort_dicts=False)
        lines.extend((f"{key} = {rendered}", ""))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _condition_id(delay: int, condition: str) -> str:
    return f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"


def _verify_condition_contract(
    config: Mapping[str, object],
    delay: int,
    condition: str,
) -> None:
    experiment = _mapping(config.get("experiment"), "condition experiment")
    contract = _mapping(experiment.get("condition"), "experiment.condition")
    if contract.get("delay_ms") != delay or contract.get("fault") != condition:
        raise ControlledBaselineEvaluationError(
            "condition config does not match its delay/fault matrix position"
        )
    if condition != "Full" and (
        contract.get("scope") != "E+R" or contract.get("duration_ticks") != 1
    ):
        raise ControlledBaselineEvaluationError(
            "fault condition config must be E+R with duration one tick"
        )


def _resolved_condition_config(
    condition_config: Mapping[str, object],
    baseline_model: Mapping[str, object],
    *,
    data_root: Path,
    expected_split_hash: str,
    manifest_path: Path,
    transport_path: Path,
    transport_sha256: str,
    fault_path: Path | None,
    fault_sha256: str | None,
    checkpoint: Path,
    checkpoint_sha256: str,
    overlay_index: Path,
    overlay_index_sha256: str,
    output_dir: Path,
    predictions: Path,
    baseline: str,
    delay: int,
    condition: str,
) -> dict[str, object]:
    resolved = copy.deepcopy(dict(condition_config))
    resolved["model"] = copy.deepcopy(dict(baseline_model))
    resolved["load_from"] = str(checkpoint)
    resolved["resume"] = False
    resolved["launcher"] = "none"
    resolved["work_dir"] = str(output_dir)

    dataloader = _mapping(resolved.get("test_dataloader"), "test_dataloader")
    dataset = _mapping(dataloader.get("dataset"), "test_dataloader.dataset")
    dataset.update(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        expected_split_hash=expected_split_hash,
        transport_overlay_path=str(transport_path),
        transport_overlay_sha256=transport_sha256,
        fault_overlay_path=None if fault_path is None else str(fault_path),
        fault_overlay_sha256=fault_sha256,
    )
    dataloader["dataset"] = dataset
    resolved["test_dataloader"] = dataloader

    evaluator = _mapping(resolved.get("test_evaluator"), "test_evaluator")
    evaluator["prediction_output"] = str(predictions)
    resolved["test_evaluator"] = evaluator
    resolved["controlled_baseline_evaluation"] = {
        "baseline": baseline,
        "delay_ms": delay,
        "condition": condition,
        "agent_scope": "E+R",
        "duration_ticks": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "overlay_index": str(overlay_index),
        "overlay_index_content_sha256": overlay_index_sha256,
    }
    return resolved


def build_evaluation_plan(
    *,
    baseline: str,
    checkpoint: Path,
    overlay_index: Path = DEFAULT_OVERLAY_INDEX,
    work_dir: Path,
    delays: Sequence[int] = DELAYS,
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, object]:
    """Resolve configs and provenance for a controlled evaluation matrix."""

    if baseline not in BASELINES:
        raise ControlledBaselineEvaluationError(f"unsupported baseline: {baseline}")
    selected_delays = _ordered_subset(delays, DELAYS, "delays")
    selected_conditions = _ordered_subset(conditions, CONDITIONS, "conditions")

    try:
        checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    except OSError as error:
        raise ControlledBaselineEvaluationError(
            f"unable to resolve checkpoint: {error}"
        ) from error
    if not checkpoint_path.is_file():
        raise ControlledBaselineEvaluationError("checkpoint must be a file")
    checkpoint_digest = _sha256_file(checkpoint_path)

    manifest_value = os.environ.get("RESILIENT_V2X_MANIFEST")
    if not manifest_value:
        raise ControlledBaselineEvaluationError(
            "RESILIENT_V2X_MANIFEST must name the temporal manifest JSON"
        )
    manifest_path, manifest = _load_json_mapping(
        Path(manifest_value),
        "RESILIENT_V2X_MANIFEST",
    )
    manifest_digest = _verify_content_hash(manifest, "temporal manifest")
    manifest_file_digest = _sha256_file(manifest_path, "temporal manifest")
    manifest_split_digest = _sha256(
        manifest.get("split_sha256"),
        "temporal manifest.split_sha256",
    )

    data_root_value = os.environ.get("RESILIENT_V2X_DATA_ROOT")
    if not data_root_value:
        raise ControlledBaselineEvaluationError(
            "RESILIENT_V2X_DATA_ROOT must name the prepared DAIR-V2X directory"
        )
    try:
        data_root = Path(data_root_value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ControlledBaselineEvaluationError(
            f"unable to resolve RESILIENT_V2X_DATA_ROOT: {error}"
        ) from error
    if not data_root.is_dir():
        raise ControlledBaselineEvaluationError(
            "RESILIENT_V2X_DATA_ROOT must name a directory"
        )
    split_value = os.environ.get("RESILIENT_V2X_SPLIT_SHA256")
    split_digest = _sha256(
        split_value,
        "RESILIENT_V2X_SPLIT_SHA256",
    )
    if split_digest != manifest_split_digest:
        raise ControlledBaselineEvaluationError(
            "RESILIENT_V2X_SPLIT_SHA256 does not match temporal manifest"
        )

    index_path, index = _load_json_mapping(Path(overlay_index), "overlay index")
    if index.get("artifact_type") != EVALUATION_INDEX_TYPE:
        raise ControlledBaselineEvaluationError(
            f"overlay index artifact_type must be {EVALUATION_INDEX_TYPE!r}"
        )
    index_digest = _verify_content_hash(index, "overlay index")
    index_file_digest = _sha256_file(index_path, "overlay index")
    indexed_manifest_digest = _sha256(
        index.get("temporal_manifest_sha256"),
        "overlay index.temporal_manifest_sha256",
    )
    if indexed_manifest_digest != manifest_digest:
        raise ControlledBaselineEvaluationError(
            "overlay index temporal manifest content hash does not match "
            "RESILIENT_V2X_MANIFEST"
        )

    transports, faults = _index_overlays(
        index,
        index_path.parent,
        manifest_digest,
    )
    baseline_config_path = BASELINE_CONFIGS[baseline]
    baseline_config = _load_python_config(baseline_config_path)
    baseline_config_digest = _sha256_file(
        baseline_config_path,
        "baseline config",
    )
    baseline_model = _mapping(baseline_config.get("model"), "baseline model")
    if baseline_model.get("type") != "ControlledCooperativeBaselineNet":
        raise ControlledBaselineEvaluationError(
            "baseline config does not use ControlledCooperativeBaselineNet"
        )
    if baseline_model.get("baseline_name") != baseline:
        raise ControlledBaselineEvaluationError("baseline config name mismatch")

    output_root = Path(work_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    if metrics_path.exists():
        raise ControlledBaselineEvaluationError(
            f"refusing to overwrite stale metrics evidence: {metrics_path}"
        )
    runs: list[dict[str, object]] = []
    for raw_delay in selected_delays:
        delay = int(raw_delay)
        if delay not in transports:
            raise ControlledBaselineEvaluationError(
                f"overlay index is missing transport delay {delay}"
            )
        transport_path, transport_digest = transports[delay]
        for raw_condition in selected_conditions:
            condition = str(raw_condition)
            fault_path: Path | None = None
            fault_digest: str | None = None
            if condition != "Full":
                fault_key = (delay, condition, "E+R", 1)
                if fault_key not in faults:
                    raise ControlledBaselineEvaluationError(
                        f"overlay index is missing E+R duration-1 fault {fault_key}"
                    )
                fault_path, fault_digest, bound_transport = faults[fault_key]
                if bound_transport != transport_digest:
                    raise ControlledBaselineEvaluationError(
                        f"fault {fault_key} references a different transport overlay"
                    )

            condition_config_path = CONDITION_CONFIGS[(delay, condition)]
            condition_config = _load_python_config(condition_config_path)
            _verify_condition_contract(condition_config, delay, condition)
            condition_name = _condition_id(delay, condition)
            condition_dir = output_root / condition_name
            resolved_config_path = condition_dir / "resolved_config.py"
            predictions_path = condition_dir / "predictions.json"
            checkpoint_hash_path = condition_dir / "checkpoint.sha256"
            if predictions_path.exists():
                raise ControlledBaselineEvaluationError(
                    "refusing to reuse stale predictions evidence: "
                    f"{predictions_path}"
                )
            resolved = _resolved_condition_config(
                condition_config,
                baseline_model,
                data_root=data_root,
                expected_split_hash=split_digest,
                manifest_path=manifest_path,
                transport_path=transport_path,
                transport_sha256=transport_digest,
                fault_path=fault_path,
                fault_sha256=fault_digest,
                checkpoint=checkpoint_path,
                checkpoint_sha256=checkpoint_digest,
                overlay_index=index_path,
                overlay_index_sha256=index_digest,
                output_dir=condition_dir,
                predictions=predictions_path,
                baseline=baseline,
                delay=delay,
                condition=condition,
            )
            resolved_raw = _render_config(resolved)
            resolved_digest = hashlib.sha256(resolved_raw).hexdigest()
            _atomic_write(resolved_config_path, resolved_raw)
            _atomic_write(
                checkpoint_hash_path,
                f"{checkpoint_digest}\n".encode("ascii"),
            )
            runs.append(
                {
                    "condition_id": condition_name,
                    "delay_ms": delay,
                    "condition": condition,
                    "agent_scope": "E+R",
                    "duration_ticks": 1,
                    "condition_config": str(condition_config_path.resolve()),
                    "resolved_config": str(resolved_config_path.resolve()),
                    "resolved_config_sha256": resolved_digest,
                    "predictions": str(predictions_path.resolve()),
                    "checkpoint_sha256_file": str(checkpoint_hash_path.resolve()),
                    "transport_overlay": str(transport_path),
                    "transport_overlay_sha256": transport_digest,
                    "fault_overlay": (
                        None if fault_path is None else str(fault_path)
                    ),
                    "fault_overlay_sha256": fault_digest,
                }
            )

    plan: dict[str, object] = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "baseline": baseline,
        "baseline_config": str(baseline_config_path.resolve()),
        "baseline_config_sha256": baseline_config_digest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "data_root": str(data_root),
        "split_sha256": split_digest,
        "manifest": str(manifest_path),
        "manifest_content_sha256": manifest_digest,
        "manifest_file_sha256": manifest_file_digest,
        "overlay_index": str(index_path),
        "overlay_index_content_sha256": index_digest,
        "overlay_index_file_sha256": index_file_digest,
        "work_dir": str(output_root),
        "delays_ms": list(selected_delays),
        "conditions": list(selected_conditions),
        "runs": runs,
        "metrics_output": str(metrics_path.resolve()),
    }
    plan_path = (output_root / "evaluation_plan.json").resolve()
    plan["plan_path"] = str(plan_path)
    plan["content_sha256"] = content_sha256(plan)
    _atomic_json(plan_path, plan)
    return plan


def _jsonable(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise ControlledBaselineEvaluationError(
        f"metric value is not JSON serializable: {type(value).__name__}"
    )


def execute_evaluation_plan(plan: Mapping[str, object]) -> dict[str, object]:
    """Run every resolved config sequentially and atomically publish metrics."""

    try:
        from mmengine import Config
        from mmengine.runner import Runner
        import torch

        from transvision import register_all_modules
    except ImportError as error:
        raise ControlledBaselineEvaluationError(
            "non-dry-run evaluation requires the locked MMEngine/MMDet3D runtime"
        ) from error

    _verify_content_hash(plan, "evaluation plan")
    for path_key, hash_key, context in (
        ("baseline_config", "baseline_config_sha256", "baseline config"),
        ("manifest", "manifest_file_sha256", "temporal manifest"),
        ("overlay_index", "overlay_index_file_sha256", "overlay index"),
    ):
        try:
            sealed_path = Path(str(plan[path_key])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise ControlledBaselineEvaluationError(
                f"unable to resolve planned {context}: {error}"
            ) from error
        expected_digest = _sha256(
            plan.get(hash_key),
            f"plan.{hash_key}",
        )
        if _sha256_file(sealed_path, context) != expected_digest:
            raise ControlledBaselineEvaluationError(
                f"{context} changed after the evaluation plan was generated"
            )

    manifest_path, manifest = _load_json_mapping(
        Path(str(plan["manifest"])),
        "planned temporal manifest",
    )
    manifest_digest = _verify_content_hash(manifest, "planned temporal manifest")
    if manifest_digest != plan.get("manifest_content_sha256"):
        raise ControlledBaselineEvaluationError(
            "planned temporal manifest content identity changed"
        )
    split_digest = _sha256(plan.get("split_sha256"), "plan.split_sha256")
    if _sha256(
        manifest.get("split_sha256"),
        "planned temporal manifest.split_sha256",
    ) != split_digest:
        raise ControlledBaselineEvaluationError(
            "planned split identity does not match temporal manifest"
        )
    try:
        data_root = Path(str(plan["data_root"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ControlledBaselineEvaluationError(
            f"unable to resolve planned data root: {error}"
        ) from error
    if not data_root.is_dir():
        raise ControlledBaselineEvaluationError(
            "planned data root must remain a directory"
        )
    index_path, index = _load_json_mapping(
        Path(str(plan["overlay_index"])),
        "planned overlay index",
    )
    index_digest = _verify_content_hash(index, "planned overlay index")
    if index_digest != plan.get("overlay_index_content_sha256"):
        raise ControlledBaselineEvaluationError(
            "planned overlay index content identity changed"
        )
    if _sha256(
        index.get("temporal_manifest_sha256"),
        "planned overlay index.temporal_manifest_sha256",
    ) != manifest_digest:
        raise ControlledBaselineEvaluationError(
            "planned overlay index no longer matches temporal manifest"
        )
    transports, faults = _index_overlays(
        index,
        index_path.parent,
        manifest_digest,
    )

    try:
        checkpoint = Path(str(plan["checkpoint"])).resolve(strict=True)
    except OSError as error:
        raise ControlledBaselineEvaluationError(
            f"unable to resolve planned checkpoint: {error}"
        ) from error
    planned_checkpoint_digest = _sha256(
        plan.get("checkpoint_sha256"),
        "plan.checkpoint_sha256",
    )
    if _sha256_file(checkpoint) != planned_checkpoint_digest:
        raise ControlledBaselineEvaluationError(
            "checkpoint content changed after the evaluation plan was generated"
        )

    register_all_modules()
    raw_runs = _sequence(plan.get("runs"), "plan.runs")
    completed: list[dict[str, object]] = []
    output = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": False,
        "planned_run_count": len(raw_runs),
        "baseline": plan["baseline"],
        "checkpoint": plan["checkpoint"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "manifest_content_sha256": plan["manifest_content_sha256"],
        "overlay_index_content_sha256": plan["overlay_index_content_sha256"],
        "runs": completed,
    }
    metrics_path = Path(str(plan["metrics_output"]))
    if metrics_path.exists():
        raise ControlledBaselineEvaluationError(
            f"refusing to overwrite stale metrics evidence: {metrics_path}"
        )
    for offset, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"plan.runs[{offset}]")
        delay = run.get("delay_ms")
        condition = run.get("condition")
        if type(delay) is not int or delay not in transports:
            raise ControlledBaselineEvaluationError(
                f"plan.runs[{offset}].delay_ms is invalid"
            )
        if type(condition) is not str or condition not in CONDITIONS:
            raise ControlledBaselineEvaluationError(
                f"plan.runs[{offset}].condition is invalid"
            )
        transport_path, transport_digest = transports[delay]
        if (
            Path(str(run.get("transport_overlay"))).resolve() != transport_path
            or run.get("transport_overlay_sha256") != transport_digest
        ):
            raise ControlledBaselineEvaluationError(
                f"plan.runs[{offset}] transport overlay identity mismatch"
            )
        if condition == "Full":
            if run.get("fault_overlay") is not None or run.get(
                "fault_overlay_sha256"
            ) is not None:
                raise ControlledBaselineEvaluationError(
                    f"plan.runs[{offset}] Full condition must not use a fault overlay"
                )
        else:
            fault_key = (delay, condition, "E+R", 1)
            if fault_key not in faults:
                raise ControlledBaselineEvaluationError(
                    f"planned overlay index is missing fault {fault_key}"
                )
            fault_path, fault_digest, bound_transport = faults[fault_key]
            if (
                Path(str(run.get("fault_overlay"))).resolve() != fault_path
                or run.get("fault_overlay_sha256") != fault_digest
                or bound_transport != transport_digest
            ):
                raise ControlledBaselineEvaluationError(
                    f"plan.runs[{offset}] fault overlay identity mismatch"
                )
        resolved_config = Path(str(run["resolved_config"])).resolve(strict=True)
        if _sha256_file(resolved_config, "resolved config") != _sha256(
            run.get("resolved_config_sha256"),
            f"plan.runs[{offset}].resolved_config_sha256",
        ):
            raise ControlledBaselineEvaluationError(
                f"plan.runs[{offset}] resolved config changed after planning"
            )
        predictions = Path(str(run["predictions"]))
        if predictions.exists():
            raise ControlledBaselineEvaluationError(
                f"refusing to reuse stale predictions evidence: {predictions}"
            )
        config = Config.fromfile(str(resolved_config))
        runner = Runner.from_cfg(config)
        metrics = _jsonable(runner.test())
        if not predictions.is_file():
            raise ControlledBaselineEvaluationError(
                f"evaluation did not produce predictions: {predictions}"
            )
        completed.append(
            {
                "condition_id": run["condition_id"],
                "delay_ms": run["delay_ms"],
                "condition": run["condition"],
                "metrics": metrics,
                "predictions": str(predictions),
            }
        )
        _atomic_json(metrics_path, output)
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output["complete"] = True
    _atomic_json(metrics_path, output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an adapted comparison baseline on the controlled "
            "DAIR-V2X delay/fault matrix."
        )
    )
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--overlay-index",
        type=Path,
        default=DEFAULT_OVERLAY_INDEX,
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--delays",
        nargs="+",
        type=int,
        choices=DELAYS,
        default=DELAYS,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_evaluation_plan(
            baseline=args.baseline,
            checkpoint=args.checkpoint,
            overlay_index=args.overlay_index,
            work_dir=args.work_dir,
            delays=args.delays,
            conditions=args.conditions,
        )
        if args.dry_run:
            print(json.dumps({"dry_run": True, **plan}, sort_keys=True))
            return 0
        metrics = execute_evaluation_plan(plan)
        print(
            json.dumps(
                {
                    "dry_run": False,
                    "metrics_output": plan["metrics_output"],
                    "run_count": len(metrics["runs"]),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ControlledBaselineEvaluationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BASELINES",
    "CONDITIONS",
    "DEFAULT_OVERLAY_INDEX",
    "DELAYS",
    "ControlledBaselineEvaluationError",
    "build_evaluation_plan",
    "execute_evaluation_plan",
    "main",
    "parse_args",
)
