#!/usr/bin/env python3
"""Deterministic causal tracking canary for ClearML routing and evidence export.

This synthetic canary validates message-arrival semantics, reliability-aware state
updates, deterministic replay, artifact hashing, and GPU worker visibility. Its
metrics are diagnostic only and MUST NOT be reported as thesis performance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


PROTOCOL_ID = "SYNTH-CAUSAL-CANARY-v1"
CLEARML_PROJECT = "Thesis/RTP-V2X"
CLEARML_TASK_PREFIX = "rtpv2x__synth-causal-canary-v1__"
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class Observation:
    source: str
    truth_id: int
    event_time: int
    arrival_time: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    reliability: float


@dataclass
class Track:
    position: np.ndarray
    velocity: np.ndarray
    reliability: float = 1.0


def truth_state(object_id: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    definitions = (
        (np.array([-12.0, -0.7]), np.array([0.24, 0.018])),
        (np.array([12.0, 0.7]), np.array([-0.24, -0.018])),
        (np.array([-4.0, 7.0]), np.array([0.09, -0.13])),
    )
    origin, velocity = definitions[object_id]
    curve = np.array([0.0, 0.0015 * (step - 30.0) ** 2 * (object_id - 1)])
    return origin + velocity * step + curve, velocity + np.array(
        [0.0, 0.003 * (step - 30.0) * (object_id - 1)]
    )


def generate_observations(
    seed: int, steps: int, packet_loss: float, max_latency: int
) -> list[Observation]:
    rng = np.random.default_rng(seed)
    events: list[Observation] = []
    rsu_burst_remaining = 0
    for step in range(steps):
        if rsu_burst_remaining == 0 and rng.random() < 0.035:
            rsu_burst_remaining = int(rng.integers(2, 7))
        for source in ("ego", "rsu"):
            if source == "rsu" and rsu_burst_remaining > 0:
                continue
            if rng.random() < packet_loss * (1.35 if source == "rsu" else 0.45):
                continue
            latency = 0 if source == "ego" else int(rng.integers(0, max_latency + 1))
            noise_sigma = 0.52 if source == "ego" else 0.24
            source_reliability = 0.72 if source == "ego" else 0.93
            for object_id in range(3):
                position, velocity = truth_state(object_id, step)
                local_sigma = noise_sigma * (1.0 + 0.7 * (object_id == 2 and step > 35))
                noisy_position = position + rng.normal(0.0, local_sigma, size=2)
                noisy_velocity = velocity + rng.normal(0.0, local_sigma * 0.10, size=2)
                reliability = source_reliability / (1.0 + local_sigma**2)
                events.append(
                    Observation(
                        source=source,
                        truth_id=object_id,
                        event_time=step,
                        arrival_time=step + latency,
                        position=tuple(float(x) for x in noisy_position),
                        velocity=tuple(float(x) for x in noisy_velocity),
                        reliability=float(reliability),
                    )
                )
        if rsu_burst_remaining > 0:
            rsu_burst_remaining -= 1
    return sorted(events, key=lambda row: (row.arrival_time, row.source, row.truth_id))


def best_assignment(cost: np.ndarray) -> tuple[int, ...]:
    rows, columns = cost.shape
    if rows != columns or rows > 7:
        raise ValueError("canary assignment expects a small square cost matrix")
    return min(
        itertools.permutations(range(columns)),
        key=lambda permutation: sum(cost[row, column] for row, column in enumerate(permutation)),
    )


def initialize_tracks() -> list[Track]:
    tracks: list[Track] = []
    for object_id in range(3):
        position, velocity = truth_state(object_id, 0)
        tracks.append(Track(position=position.copy(), velocity=velocity.copy()))
    return tracks


def update_from_source(
    tracks: list[Track], observations: list[Observation], decision_time: int, method: str
) -> None:
    if len(observations) != len(tracks):
        return
    predicted = np.stack([track.position + track.velocity for track in tracks])
    candidates = []
    weights = []
    for observation in observations:
        position = np.asarray(observation.position, dtype=np.float64)
        age = decision_time - observation.event_time
        if method == "reliability":
            position = position + np.asarray(observation.velocity) * age
            weight = observation.reliability * math.exp(-age / 3.0)
        else:
            weight = 0.50
        candidates.append(position)
        weights.append(float(np.clip(weight, 0.05, 0.95)))
    candidate_array = np.stack(candidates)
    assignment = best_assignment(
        np.linalg.norm(predicted[:, None, :] - candidate_array[None, :, :], axis=-1)
    )
    for track_index, candidate_index in enumerate(assignment):
        track = tracks[track_index]
        previous = track.position.copy()
        gain = weights[candidate_index]
        track.position = gain * candidate_array[candidate_index] + (1.0 - gain) * predicted[track_index]
        measured_velocity = track.position - previous
        track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
        track.reliability = gain


def evaluate_mapping(tracks: list[Track], step: int) -> tuple[tuple[int, ...], float]:
    truth = np.stack([truth_state(object_id, step)[0] for object_id in range(3)])
    estimates = np.stack([track.position for track in tracks])
    assignment = best_assignment(
        np.linalg.norm(estimates[:, None, :] - truth[None, :, :], axis=-1)
    )
    rmse = math.sqrt(
        float(np.mean([np.sum((estimates[index] - truth[truth_id]) ** 2) for index, truth_id in enumerate(assignment)]))
    )
    return assignment, rmse


def prediction_error(tracks: list[Track], step: int, horizon: int = 6) -> tuple[float, float]:
    ade_values: list[float] = []
    fde_values: list[float] = []
    for track_index, track in enumerate(tracks):
        errors = []
        for offset in range(1, horizon + 1):
            predicted = track.position + track.velocity * offset
            target, _ = truth_state(track_index, step + offset)
            errors.append(float(np.linalg.norm(predicted - target)))
        ade_values.append(float(np.mean(errors)))
        fde_values.append(errors[-1])
    return float(np.mean(ade_values)), float(np.mean(fde_values))


def run_method(events: list[Observation], steps: int, method: str) -> dict[str, float | int]:
    tracks = initialize_tracks()
    previous_mapping = tuple(range(3))
    identity_switches = 0
    rmse_values: list[float] = []
    ade_values: list[float] = []
    fde_values: list[float] = []
    future_messages_consumed = 0
    late_messages_consumed = 0

    by_arrival: dict[int, list[Observation]] = {}
    for event in events:
        by_arrival.setdefault(event.arrival_time, []).append(event)

    for step in range(steps):
        for track in tracks:
            track.position = track.position + track.velocity
        arrived = by_arrival.get(step, [])
        if any(event.event_time > step or event.arrival_time > step for event in arrived):
            future_messages_consumed += 1
        late_messages_consumed += sum(event.event_time < step for event in arrived)
        for source in ("ego", "rsu"):
            source_rows = [event for event in arrived if event.source == source]
            update_from_source(tracks, source_rows, step, method)
        mapping, rmse = evaluate_mapping(tracks, step)
        if step > 0:
            identity_switches += sum(a != b for a, b in zip(mapping, previous_mapping))
        previous_mapping = mapping
        rmse_values.append(rmse)
        ade, fde = prediction_error(tracks, step)
        ade_values.append(ade)
        fde_values.append(fde)

    return {
        "position_rmse": float(np.mean(rmse_values)),
        "prediction_ade": float(np.mean(ade_values)),
        "prediction_fde": float(np.mean(fde_values)),
        "identity_switches": identity_switches,
        "future_messages_consumed": future_messages_consumed,
        "late_messages_consumed": late_messages_consumed,
        "messages_available": len(events),
    }


def nvidia_summary() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return {"available": bool(rows), "devices": rows}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def initialize_clearml(mode: str) -> object | None:
    if mode == "off":
        return None
    try:
        from clearml import Task  # type: ignore
    except Exception as exc:
        raise RuntimeError("required ClearML SDK is unavailable") from exc
    try:
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name="RTP-V2X synthetic causal canary",
        )
    except Exception as exc:
        raise RuntimeError("required ClearML task context could not be initialized") from exc
    task_id = getattr(task, "id", None)
    task_name = getattr(task, "name", None)
    project_name = task.get_project_name() if task is not None else None
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError("required ClearML task context has no task ID")
    if not isinstance(task_name, str) or not task_name.startswith(CLEARML_TASK_PREFIX):
        raise RuntimeError("ClearML task name does not match the synthetic canary contract")
    if project_name != CLEARML_PROJECT:
        raise RuntimeError("ClearML project does not match the synthetic canary contract")
    return task


def task_diff_sha256(task: object) -> str:
    data = getattr(task, "data", None)
    script = getattr(data, "script", None)
    diff = getattr(script, "diff", None)
    if not isinstance(diff, str) or not diff:
        raise RuntimeError("ClearML task has no standalone source diff")
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    def parse_bool(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--packet-loss", type=float, default=0.08)
    parser.add_argument("--max-latency", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/causal-tracking-canary")
    parser.add_argument("--require-gpu", type=parse_bool, default=False)
    parser.add_argument("--require-a100", type=parse_bool, default=False)
    parser.add_argument("--clearml-mode", choices=("required", "off"), default="required")
    parser.add_argument("--expected-script-sha256")
    parser.add_argument("--expected-diff-sha256")
    return parser.parse_args()


def preparse_clearml_mode() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clearml-mode", choices=("required", "off"), default="required")
    args, _ = parser.parse_known_args()
    return args.clearml_mode


def main() -> None:
    preparse_mode = preparse_clearml_mode()
    task = initialize_clearml(preparse_mode)
    args = parse_args()
    if args.clearml_mode != preparse_mode:
        raise RuntimeError("ClearML mode changed after task initialization")
    if args.steps < 24 or not 0.0 <= args.packet_loss < 0.8 or args.max_latency < 0:
        raise SystemExit("invalid canary configuration")

    script_path = Path(__file__).resolve()
    observed_script_sha256 = file_sha256(script_path)
    observed_diff_sha256 = None
    if task is not None:
        expected_script_sha256 = require_sha256(
            args.expected_script_sha256, "expected script hash"
        )
        if observed_script_sha256 != expected_script_sha256:
            raise RuntimeError("executed script hash does not match the trusted expectation")
        expected_diff_sha256 = require_sha256(
            args.expected_diff_sha256, "expected standalone diff hash"
        )
        observed_diff_sha256 = task_diff_sha256(task)
        if observed_diff_sha256 != expected_diff_sha256:
            raise RuntimeError("ClearML standalone diff hash does not match the trusted expectation")

    config = {
        "protocol_id": PROTOCOL_ID,
        "seed": args.seed,
        "steps": args.steps,
        "packet_loss": args.packet_loss,
        "max_latency": args.max_latency,
        "require_gpu": args.require_gpu,
        "require_a100": args.require_a100,
        "clearml_mode": args.clearml_mode,
    }
    events = generate_observations(args.seed, args.steps, args.packet_loss, args.max_latency)
    replay = generate_observations(args.seed, args.steps, args.packet_loss, args.max_latency)
    if events != replay:
        raise RuntimeError("deterministic replay check failed")

    naive = run_method(events, args.steps, "naive")
    reliability = run_method(events, args.steps, "reliability")
    gpu = nvidia_summary()
    if args.require_gpu and not gpu.get("available"):
        raise RuntimeError("GPU canary requested but nvidia-smi reported no device")
    if args.require_a100 and not any("A100" in row for row in gpu.get("devices", [])):
        raise RuntimeError("A100 canary requested but no A100 device was reported")
    if reliability["future_messages_consumed"] != 0:
        raise RuntimeError("causal leakage detected")

    metrics = {
        "protocol_id": PROTOCOL_ID,
        "scientific_claim_allowed": False,
        "diagnostic_only": True,
        "naive": naive,
        "reliability": reliability,
        "delta": {
            "position_rmse": float(reliability["position_rmse"] - naive["position_rmse"]),
            "prediction_ade": float(reliability["prediction_ade"] - naive["prediction_ade"]),
            "identity_switches": int(reliability["identity_switches"] - naive["identity_switches"]),
        },
        "deterministic_replay": True,
        "gpu": gpu,
    }
    if reliability["position_rmse"] > naive["position_rmse"] * 1.05:
        raise RuntimeError("reliability-aware synthetic sanity check regressed position RMSE")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    events_path = output_dir / "events.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    write_json(metrics_path, metrics)
    with events_path.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")

    task_id = getattr(task, "id", None)
    task_name = getattr(task, "name", None)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_claim_allowed": False,
        "diagnostic_only": True,
        "clearml_task_id": task_id,
        "clearml_task_name": task_name,
        "clearml_project": CLEARML_PROJECT if task is not None else None,
        "clearml_required": args.clearml_mode == "required",
        "config": config,
        "config_sha256": canonical_sha256(config),
        "script_sha256": observed_script_sha256,
        "standalone_diff_sha256": observed_diff_sha256,
        "metrics_sha256": file_sha256(metrics_path),
        "events_sha256": file_sha256(events_path),
        "artifact_commit_order": ["metrics", "events", "run_manifest"],
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    write_json(manifest_path, manifest)

    if task is not None:
        logger = task.get_logger()
        for method in ("naive", "reliability"):
            for key in ("position_rmse", "prediction_ade", "prediction_fde"):
                logger.report_scalar("canary", f"{method}/{key}", float(metrics[method][key]), 0)
        for artifact_name, artifact_path in (
            ("metrics", metrics_path),
            ("events", events_path),
            ("run_manifest", manifest_path),
        ):
            uploaded = task.upload_artifact(
                artifact_name,
                artifact_object=str(artifact_path),
                wait_on_upload=True,
                retries=2,
            )
            if not uploaded:
                raise RuntimeError(f"ClearML artifact upload failed: {artifact_name}")
        task.flush(wait_for_uploads=True)

    safe_status = {
        "artifacts": ["metrics", "events", "run_manifest"],
        "clearml_task_id": task_id,
        "deterministic_replay": True,
        "diagnostic_only": True,
        "future_messages_consumed": reliability["future_messages_consumed"],
        "protocol_id": PROTOCOL_ID,
    }
    print("RTPV2X_CANARY_STATUS=" + json.dumps(safe_status, sort_keys=True))


if __name__ == "__main__":
    main()
