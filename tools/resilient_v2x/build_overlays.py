#!/usr/bin/env python3
"""Build immutable ResilientV2X protocol cohorts and overlay matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.resilient_v2x_manifest import (  # noqa: E402
    ManifestError,
    TemporalManifest,
    TemporalSampleRecord,
    canonical_json_bytes,
    content_sha256,
    load_temporal_manifest,
)
from transvision.dataset.resilient_v2x_schedule import (  # noqa: E402
    CausalFaultPlan,
    FaultPlan,
    OverlayDigest,
    ScheduleError,
    TransportPlan,
    read_overlay,
    write_causal_fault_overlay,
    write_fault_overlay,
    write_transport_overlay,
)


SCHEMA_VERSION = 1
COHORT_TYPE = "resilient_v2x_evaluation_cohort"
TRAIN_INDEX_TYPE = "resilient_v2x_training_overlays"
EVALUATION_INDEX_TYPE = "resilient_v2x_evaluation_overlays"
DELAYS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
AGENT_SCOPES: dict[str, tuple[str, ...]] = {
    "E+R": ("ego", "rsu"),
    "E-only": ("ego",),
    "R-only": ("rsu",),
}
BRANCHES = (
    ("ego", "lidar"),
    ("ego", "camera"),
    ("rsu", "lidar"),
    ("rsu", "camera"),
)


class ProtocolBuildError(ValueError):
    """Raised when a protocol artifact cannot be built reproducibly."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(artifact_type: str, payload: Mapping[str, object]) -> dict[str, object]:
    reserved = {"schema_version", "artifact_type", "content_sha256"}
    if reserved.intersection(payload):
        raise ProtocolBuildError("payload contains a reserved artifact field")
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        **dict(payload),
    }
    document["content_sha256"] = content_sha256(document)
    canonical_json_bytes(document)
    return document


def _regular_bytes(path: Path, context: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ProtocolBuildError(f"unable to inspect {context}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ProtocolBuildError(f"{context} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ProtocolBuildError(f"unable to read {context}") from error


def _publish_canonical_json(path: Path, document: Mapping[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(document)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError:
            if _regular_bytes(output, "existing artifact") != raw:
                raise ProtocolBuildError("artifact destination conflict")
        directory = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output
    except ProtocolBuildError:
        raise
    except OSError as error:
        raise ProtocolBuildError("artifact publication failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _available_branch_sources(
    sample: TemporalSampleRecord,
    agent: str,
    modality: str,
    max_delay_ms: int,
    delta_t_ms: int,
    history_limit: int,
) -> tuple[tuple[int, ...], dict[str, int], int | None]:
    available: list[int] = []
    causal_endpoint_candidates: list[int] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for source in sample.source_slices:
        if (source.agent, source.modality) != (agent, modality):
            continue
        if source.n_s > sample.n_t:
            reject("future_source")
            continue
        if source.n_s < sample.n_t - history_limit:
            reject("outside_supported_history")
            continue
        arrived = agent == "ego" or source.tau_s_ms + max_delay_ms <= sample.tau_t_ms
        if arrived:
            causal_endpoint_candidates.append(source.n_s)
        if source.tau_s_ms != source.n_s * delta_t_ms:
            reject("invalid_timestamp")
            continue
        metadata_invalid = False
        for valid, reason in (
            (source.payload_valid, "invalid_payload"),
            (source.pose_valid, "invalid_pose"),
            (source.calibration_valid, "invalid_calibration"),
        ):
            if not valid:
                reject(reason)
                metadata_invalid = True
        if metadata_invalid:
            continue
        if not arrived:
            reject("not_arrived_at_max_delay")
            continue
        available.append(source.n_s)
    endpoint = max(causal_endpoint_candidates, default=None)
    return (
        tuple(sorted(available, reverse=True)),
        dict(sorted(rejected.items())),
        endpoint,
    )


def build_cohort_document(
    manifest: TemporalManifest,
    *,
    split: str,
    max_delay_ms: int = 300,
    max_continuous_fault_duration: int = 1,
) -> dict[str, object]:
    """Select one cohort that is valid for every requested worst-case branch."""

    if not isinstance(manifest, TemporalManifest):
        raise ProtocolBuildError("manifest must be a TemporalManifest")
    if split not in ("val", "test"):
        raise ProtocolBuildError("evaluation cohort split must be val or test")
    if type(max_delay_ms) is not int or max_delay_ms not in DELAYS:
        raise ProtocolBuildError("max_delay_ms must be 0, 100, 200, or 300")
    if (
        type(max_continuous_fault_duration) is not int
        or max_continuous_fault_duration <= 0
        or max_continuous_fault_duration > manifest.history_limit + 1
    ):
        raise ProtocolBuildError(
            "max continuous fault duration exceeds represented history"
        )
    if (
        max_delay_ms + (max_continuous_fault_duration - 1) * manifest.delta_t_ms
        > manifest.history_limit * manifest.delta_t_ms
    ):
        raise ProtocolBuildError(
            "joint delay/fault history requirement exceeds represented history: "
            f"{max_delay_ms} / {manifest.delta_t_ms} + "
            f"{max_continuous_fault_duration} - 1 > {manifest.history_limit}"
        )

    candidates = tuple(sample for sample in manifest.samples if sample.split == split)
    included: list[str] = []
    excluded: list[dict[str, object]] = []
    for sample in candidates:
        reasons: list[dict[str, object]] = []
        for agent, modality in BRANCHES:
            available, rejected, endpoint = _available_branch_sources(
                sample,
                agent,
                modality,
                max_delay_ms,
                manifest.delta_t_ms,
                manifest.history_limit,
            )
            endpoint_valid = bool(available) and available[0] == endpoint
            if not endpoint_valid or len(available) < max_continuous_fault_duration:
                reasons.append(
                    {
                        "code": (
                            "invalid_or_missing_causal_endpoint"
                            if not endpoint_valid
                            else "insufficient_causal_history_at_max_delay"
                        ),
                        "agent": agent,
                        "modality": modality,
                        "required_source_count": max_continuous_fault_duration,
                        "available_source_count": len(available),
                        "latest_available_n_s": available[0] if available else None,
                        "expected_endpoint_n_s": endpoint,
                        "rejected_source_counts": rejected,
                    }
                )
        if reasons:
            excluded.append(
                {
                    "sample_id": sample.sample_id,
                    "sequence_id": sample.sequence_id,
                    "n_t": sample.n_t,
                    "tau_t_ms": sample.tau_t_ms,
                    "reasons": reasons,
                }
            )
        else:
            included.append(sample.sample_id)

    return _seal(
        COHORT_TYPE,
        {
            "temporal_manifest_sha256": manifest.content_sha256,
            "split": split,
            "max_delay_ms": max_delay_ms,
            "max_continuous_fault_duration": max_continuous_fault_duration,
            "eligibility_branch_order": [list(branch) for branch in BRANCHES],
            "candidate_sample_count": len(candidates),
            "included_sample_count": len(included),
            "excluded_sample_count": len(excluded),
            "sample_ids": included,
            "sample_ids_sha256": _sha256_bytes(canonical_json_bytes(included)),
            "excluded_samples": excluded,
        },
    )


def write_cohort(
    manifest: TemporalManifest,
    output: Path,
    *,
    split: str,
    max_delay_ms: int = 300,
    max_continuous_fault_duration: int = 1,
) -> dict[str, object]:
    document = build_cohort_document(
        manifest,
        split=split,
        max_delay_ms=max_delay_ms,
        max_continuous_fault_duration=max_continuous_fault_duration,
    )
    _publish_canonical_json(output, document)
    return document


def _decode_json_object(path: Path, context: str) -> dict[str, object]:
    raw = _regular_bytes(path, context)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolBuildError(f"duplicate {context} key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ProtocolBuildError(f"non-finite {context} value: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ProtocolBuildError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolBuildError(f"invalid {context} JSON") from error
    if not isinstance(value, Mapping):
        raise ProtocolBuildError(f"{context} must be a JSON object")
    plain = dict(value)
    if canonical_json_bytes(plain) != raw:
        raise ProtocolBuildError(f"{context} JSON is not canonical")
    return plain


def load_cohort(
    path: Path,
    manifest: TemporalManifest,
) -> dict[str, object]:
    """Load a cohort and recompute its membership and exclusion reasons."""

    value = _decode_json_object(Path(path), "cohort")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolBuildError("unsupported cohort schema_version")
    if value.get("artifact_type") != COHORT_TYPE:
        raise ProtocolBuildError("unexpected cohort artifact_type")
    if value.get("temporal_manifest_sha256") != manifest.content_sha256:
        raise ProtocolBuildError("cohort temporal manifest digest mismatch")
    split = value.get("split")
    max_delay = value.get("max_delay_ms")
    max_duration = value.get("max_continuous_fault_duration")
    if split not in ("val", "test"):
        raise ProtocolBuildError("cohort split is invalid")
    if type(max_delay) is not int or type(max_duration) is not int:
        raise ProtocolBuildError("cohort eligibility parameters are invalid")
    expected = build_cohort_document(
        manifest,
        split=split,
        max_delay_ms=max_delay,
        max_continuous_fault_duration=max_duration,
    )
    if value != expected:
        raise ProtocolBuildError("cohort content or self-hash mismatch")
    return value


def _samples_from_ids(
    manifest: TemporalManifest,
    split: str,
    sample_ids: Sequence[object],
) -> tuple[TemporalSampleRecord, ...]:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise ProtocolBuildError("cohort sample_ids must be a sequence")
    if not sample_ids:
        raise ProtocolBuildError("cohort contains no eligible samples")
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    samples: list[TemporalSampleRecord] = []
    for sample_id in sample_ids:
        if type(sample_id) is not str or sample_id not in by_id:
            raise ProtocolBuildError("cohort references an unknown sample")
        sample = by_id[sample_id]
        if sample.split != split:
            raise ProtocolBuildError("cohort sample split mismatch")
        samples.append(sample)
    if len(samples) != len({sample.sample_id for sample in samples}):
        raise ProtocolBuildError("cohort sample IDs must be unique")
    return tuple(samples)


def _overlay_entry(digest: OverlayDigest, output_dir: Path) -> dict[str, object]:
    try:
        relative = digest.path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError as error:
        raise ProtocolBuildError("overlay path escapes output directory") from error
    return {
        "path": relative,
        "record_count": digest.record_count,
        "uncompressed_size": digest.uncompressed_size,
        "uncompressed_sha256": digest.uncompressed_sha256,
        "compressed_size": digest.compressed_size,
        "compressed_sha256": digest.compressed_sha256,
    }


def _assert_overlay_sample_ids(
    records: Sequence[Mapping[str, object]],
    expected_ids: tuple[str, ...],
    context: str,
) -> None:
    actual = {record.get("sample_id") for record in records}
    if actual != set(expected_ids):
        raise ProtocolBuildError(f"{context} sample IDs differ from the cohort")


def _transport_filename(split: str, delay_ms: int) -> str:
    delay = f"{delay_ms:03d}" if delay_ms == 0 else str(delay_ms)
    return f"{split}_transport_delay_{delay}.jsonl.zst"


def _fault_filename(
    split: str,
    delay_ms: int,
    condition: str,
    scope: str,
    duration: int,
) -> str:
    delay = f"{delay_ms:03d}"
    if condition in ("L-Fail", "C-Fail") and scope == "E+R" and duration == 1:
        condition_slug = condition.lower().replace("-", "_")
        return f"{split}_causal_delay_{delay}_{condition_slug}.jsonl.zst"
    if condition in ("L-Fail", "C-Fail"):
        modality = "lidar" if condition == "L-Fail" else "camera"
        return f"{split}_causal_{delay}_{scope}_{modality}_d{duration}.jsonl.zst"
    scope_slug = scope.lower().replace("+", "_").replace("-", "_")
    return f"{split}_causal_{delay}_{scope_slug}_full_d1.jsonl.zst"


def _diagnostic_fault_filename(
    split: str,
    delay_ms: int,
    condition: str,
    scope: str,
    duration: int,
) -> str:
    modality = "lidar" if condition == "L-Fail" else "camera"
    return f"{split}_causal_{delay_ms:03d}_{scope}_{modality}_d{duration}.jsonl.zst"


def _publish_file_alias(source: Path, destination: Path) -> Path:
    raw = _regular_bytes(source, "overlay alias source")
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        if _regular_bytes(destination, "existing overlay alias") != raw:
            raise ProtocolBuildError("overlay alias destination conflict")
    except OSError as error:
        raise ProtocolBuildError("overlay alias publication failed") from error
    directory = os.open(
        destination.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination


def build_training_overlays(
    manifest: TemporalManifest,
    output_dir: Path,
    *,
    protocol_seed: int,
    epochs: Sequence[int],
    p_lidar: float,
    p_camera: float,
) -> dict[str, object]:
    """Build deterministic epoch-indexed random transport and fault overlays."""

    if type(protocol_seed) is not int or protocol_seed < 0:
        raise ProtocolBuildError("protocol_seed must be a non-negative integer")
    epoch_values = tuple(sorted(epochs))
    if (
        not epoch_values
        or any(type(epoch) is not int or epoch < 0 for epoch in epoch_values)
        or len(epoch_values) != len(set(epoch_values))
    ):
        raise ProtocolBuildError("epochs must be unique non-negative integers")
    for probability, name in ((p_lidar, "p_lidar"), (p_camera, "p_camera")):
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ProtocolBuildError(f"{name} must be in [0,1]")
    samples = tuple(
        sample
        for sample in manifest.samples
        if sample.split == "train" and sample.n_t >= manifest.history_limit
    )
    if not samples:
        raise ProtocolBuildError("manifest has no history-eligible training samples")
    if len(samples) != manifest.history_eligible_train_count:
        raise ProtocolBuildError("training history eligibility count mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transport = write_transport_overlay(
        TransportPlan(
            temporal_manifest_sha256=manifest.content_sha256,
            split="train",
            samples=samples,
            mode="train_random",
            protocol_seed=protocol_seed,
            epochs=epoch_values,
            delay_values_ms=DELAYS,
            fixed_delay_ms=None,
        ),
        output_dir / "train_transport.jsonl.zst",
    )
    fault = write_fault_overlay(
        FaultPlan(
            temporal_manifest_sha256=manifest.content_sha256,
            split="train",
            samples=samples,
            mode="train_random",
            protocol_seed=protocol_seed,
            epochs=epoch_values,
            condition=None,
            p_lidar=float(p_lidar),
            p_camera=float(p_camera),
            agents=("ego", "rsu"),
            modality=None,
            duration=None,
        ),
        output_dir / "train_fault.jsonl.zst",
    )
    sample_ids = tuple(sample.sample_id for sample in samples)
    for digest, context in (
        (transport, "training transport"),
        (fault, "training fault"),
    ):
        records = read_overlay(digest.path, digest.uncompressed_sha256)
        _assert_overlay_sample_ids(records, sample_ids, context)
    document = _seal(
        TRAIN_INDEX_TYPE,
        {
            "temporal_manifest_sha256": manifest.content_sha256,
            "split": "train",
            "protocol_seed": protocol_seed,
            "epochs": list(epoch_values),
            "delay_values_ms": list(DELAYS),
            "p_lidar": float(p_lidar),
            "p_camera": float(p_camera),
            "sample_ids": list(sample_ids),
            "sample_ids_sha256": _sha256_bytes(canonical_json_bytes(sample_ids)),
            "overlays": {
                "transport": _overlay_entry(transport, output_dir),
                "fault": _overlay_entry(fault, output_dir),
            },
        },
    )
    _publish_canonical_json(output_dir / "training_overlays.json", document)
    return document


def _ordered_subset(values: Sequence[object], order: tuple[object, ...], name: str):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProtocolBuildError(f"{name} must be a sequence")
    if (
        not values
        or len(values) != len(set(values))
        or any(value not in order for value in values)
    ):
        raise ProtocolBuildError(f"{name} must be a unique non-empty protocol subset")
    selected = set(values)
    return tuple(value for value in order if value in selected)


def build_evaluation_overlays(
    manifest: TemporalManifest,
    cohort_path: Path,
    output_dir: Path,
    *,
    delays: Sequence[int] = DELAYS,
    conditions: Sequence[str] = CONDITIONS,
    agent_scopes: Sequence[str] = tuple(AGENT_SCOPES),
    duration: int = 1,
) -> dict[str, object]:
    """Build a fixed causal matrix over one immutable shared cohort."""

    cohort = load_cohort(cohort_path, manifest)
    delay_values = _ordered_subset(delays, DELAYS, "delays")
    condition_values = _ordered_subset(conditions, CONDITIONS, "conditions")
    scope_values = _ordered_subset(
        agent_scopes,
        tuple(AGENT_SCOPES),
        "agent_scopes",
    )
    if type(duration) is not int or duration <= 0:
        raise ProtocolBuildError("duration must be a positive integer")
    if duration > cohort["max_continuous_fault_duration"]:
        raise ProtocolBuildError("duration exceeds the cohort eligibility contract")
    if max(delay_values) > cohort["max_delay_ms"]:
        raise ProtocolBuildError(
            "requested delay exceeds the cohort eligibility contract"
        )
    split = cohort["split"]
    assert isinstance(split, str)
    samples = _samples_from_ids(manifest, split, cohort["sample_ids"])
    sample_ids = tuple(sample.sample_id for sample in samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transports: list[dict[str, object]] = []
    faults: list[dict[str, object]] = []

    for delay in delay_values:
        transport = write_transport_overlay(
            TransportPlan(
                temporal_manifest_sha256=manifest.content_sha256,
                split=split,  # type: ignore[arg-type]
                samples=samples,
                mode="fixed_evaluation",
                protocol_seed=None,
                epochs=(),
                delay_values_ms=(),
                fixed_delay_ms=delay,
            ),
            output_dir / _transport_filename(split, delay),
        )
        transport_records = read_overlay(
            transport.path,
            transport.uncompressed_sha256,
        )
        _assert_overlay_sample_ids(
            transport_records,
            sample_ids,
            f"transport delay {delay}",
        )
        transports.append(
            {
                "delay_ms": delay,
                "overlay": _overlay_entry(transport, output_dir),
            }
        )
        for condition in condition_values:
            actual_duration = 1 if condition == "Full" else duration
            for scope in scope_values:
                fault = write_causal_fault_overlay(
                    CausalFaultPlan(
                        temporal_manifest_sha256=manifest.content_sha256,
                        transport_overlay_sha256=transport.uncompressed_sha256,
                        split=split,  # type: ignore[arg-type]
                        samples=samples,
                        condition=condition,  # type: ignore[arg-type]
                        agents=AGENT_SCOPES[scope],  # type: ignore[arg-type]
                        duration=actual_duration,
                        fixed_delay_ms=delay,  # type: ignore[arg-type]
                    ),
                    manifest,
                    transport_records,
                    output_dir
                    / _fault_filename(
                        split,
                        delay,
                        condition,
                        scope,
                        actual_duration,
                    ),
                )
                fault_records = read_overlay(
                    fault.path,
                    fault.uncompressed_sha256,
                )
                _assert_overlay_sample_ids(
                    fault_records,
                    sample_ids,
                    f"fault delay {delay} {condition} {scope}",
                )
                aliases: list[str] = []
                if (
                    condition in ("L-Fail", "C-Fail")
                    and scope == "E+R"
                    and actual_duration == 1
                ):
                    alias = output_dir / _diagnostic_fault_filename(
                        split,
                        delay,
                        condition,
                        scope,
                        actual_duration,
                    )
                    _publish_file_alias(fault.path, alias)
                    aliases.append(alias.relative_to(output_dir).as_posix())
                faults.append(
                    {
                        "delay_ms": delay,
                        "condition": condition,
                        "agent_scope": scope,
                        "agents": list(AGENT_SCOPES[scope]),
                        "duration": actual_duration,
                        "temporal_manifest_sha256": manifest.content_sha256,
                        "transport_overlay_sha256": transport.uncompressed_sha256,
                        "overlay": _overlay_entry(fault, output_dir),
                        "alias_paths": aliases,
                    }
                )

    cohort_raw = _regular_bytes(Path(cohort_path), "cohort")
    document = _seal(
        EVALUATION_INDEX_TYPE,
        {
            "temporal_manifest_sha256": manifest.content_sha256,
            "cohort_content_sha256": cohort["content_sha256"],
            "cohort_file_sha256": _sha256_bytes(cohort_raw),
            "split": split,
            "sample_ids": list(sample_ids),
            "sample_ids_sha256": cohort["sample_ids_sha256"],
            "requested_duration": duration,
            "delays_ms": list(delay_values),
            "conditions": list(condition_values),
            "agent_scopes": list(scope_values),
            "transport_overlays": transports,
            "fault_overlays": faults,
        },
    )
    _publish_canonical_json(output_dir / "evaluation_overlays.json", document)
    return document


def _common_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--allow-fixture", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable causal transport/fault protocol overlays."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cohort = subparsers.add_parser("cohort", help="freeze a shared evaluation cohort")
    _common_manifest_arguments(cohort)
    cohort.add_argument("--split", required=True, choices=("val", "test"))
    cohort.add_argument("--max-delay-ms", type=int, choices=DELAYS, default=300)
    cohort.add_argument("--max-duration", type=int, default=1)
    cohort.add_argument("--out", required=True, type=Path)

    train = subparsers.add_parser("train", help="build random training overlays")
    _common_manifest_arguments(train)
    train.add_argument("--protocol-seed", required=True, type=int)
    train.add_argument("--epochs", required=True, type=int, nargs="+")
    train.add_argument("--p-lidar", required=True, type=float)
    train.add_argument("--p-camera", required=True, type=float)
    train.add_argument("--out-dir", required=True, type=Path)

    evaluation = subparsers.add_parser(
        "evaluation",
        help="build fixed causal evaluation overlays from one cohort",
    )
    _common_manifest_arguments(evaluation)
    evaluation.add_argument("--cohort", required=True, type=Path)
    evaluation.add_argument(
        "--delays", type=int, nargs="+", choices=DELAYS, default=DELAYS
    )
    evaluation.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )
    evaluation.add_argument(
        "--agents",
        nargs="+",
        choices=tuple(AGENT_SCOPES),
        default=tuple(AGENT_SCOPES),
    )
    evaluation.add_argument("--duration", type=int, default=1)
    evaluation.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _load_from_args(args: argparse.Namespace) -> TemporalManifest:
    return load_temporal_manifest(
        args.manifest,
        expected_split_hash=args.expected_split_sha256,
        allow_fixture=args.allow_fixture,
    )


def _summary(document: Mapping[str, object], output: Path) -> None:
    print(
        json.dumps(
            {
                "artifact_type": document["artifact_type"],
                "content_sha256": document["content_sha256"],
                "output": str(output.resolve()),
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _load_from_args(args)
        if args.command == "cohort":
            document = write_cohort(
                manifest,
                args.out,
                split=args.split,
                max_delay_ms=args.max_delay_ms,
                max_continuous_fault_duration=args.max_duration,
            )
            output = args.out
        elif args.command == "train":
            document = build_training_overlays(
                manifest,
                args.out_dir,
                protocol_seed=args.protocol_seed,
                epochs=args.epochs,
                p_lidar=args.p_lidar,
                p_camera=args.p_camera,
            )
            output = args.out_dir / "training_overlays.json"
        else:
            document = build_evaluation_overlays(
                manifest,
                args.cohort,
                args.out_dir,
                delays=args.delays,
                conditions=args.conditions,
                agent_scopes=args.agents,
                duration=args.duration,
            )
            output = args.out_dir / "evaluation_overlays.json"
        _summary(document, output)
        return 0
    except (ManifestError, ScheduleError, ProtocolBuildError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AGENT_SCOPES",
    "COHORT_TYPE",
    "CONDITIONS",
    "DELAYS",
    "EVALUATION_INDEX_TYPE",
    "ProtocolBuildError",
    "TRAIN_INDEX_TYPE",
    "build_cohort_document",
    "build_evaluation_overlays",
    "build_training_overlays",
    "load_cohort",
    "main",
    "parse_args",
    "write_cohort",
)
