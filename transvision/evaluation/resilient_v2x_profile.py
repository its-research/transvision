"""Complexity and synchronized runtime profiling for ResilientV2X.

The generic measurement functions are framework-light and dependency
injected, which makes the synchronization boundary explicit and testable.
Torch is imported lazily only by the optional FLOP profiler.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from transvision.evaluation.resilient_v2x_evidence import (
    ArtifactDigest,
    EvidenceError,
    PROFILE_DOCUMENT_TYPE,
    seal_document,
)


class ProfilingError(ValueError):
    """Raised when a profile request or measurement is invalid."""


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProfilingError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ProfilingError(f"{name} must be positive")
    return result


def _module_path_matches(name: str, excluded_path: str) -> bool:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name == excluded_path or name.startswith(excluded_path + ".")


def _is_excluded(name: str, excluded_paths: tuple[str, ...]) -> bool:
    return any(_module_path_matches(name, path) for path in excluded_paths)


@dataclass(frozen=True)
class ParameterProfile:
    """Student-only parameter/storage counts and explicitly excluded state."""

    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    buffer_count: int
    buffer_bytes: int
    excluded_parameter_count: int
    excluded_parameter_bytes: int
    excluded_buffer_count: int
    excluded_buffer_bytes: int
    excluded_module_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "parameter_count",
            "trainable_parameter_count",
            "parameter_bytes",
            "buffer_count",
            "buffer_bytes",
            "excluded_parameter_count",
            "excluded_parameter_bytes",
            "excluded_buffer_count",
            "excluded_buffer_bytes",
        ):
            _nonnegative_integer(getattr(self, name), name)
        if self.trainable_parameter_count > self.parameter_count:
            raise ProfilingError("trainable parameters exceed included parameters")
        if any(
            type(path) is not str or not path or path != path.strip(".")
            for path in self.excluded_module_paths
        ):
            raise ProfilingError("excluded module paths must be canonical")
        if tuple(sorted(set(self.excluded_module_paths))) != self.excluded_module_paths:
            raise ProfilingError("excluded module paths must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_bytes": self.parameter_bytes,
            "buffer_count": self.buffer_count,
            "buffer_bytes": self.buffer_bytes,
            "excluded_parameter_count": self.excluded_parameter_count,
            "excluded_parameter_bytes": self.excluded_parameter_bytes,
            "excluded_buffer_count": self.excluded_buffer_count,
            "excluded_buffer_bytes": self.excluded_buffer_bytes,
            "excluded_module_paths": list(self.excluded_module_paths),
        }


def count_parameters(
    model: object,
    *,
    excluded_module_paths: Sequence[str] = ("teacher",),
) -> ParameterProfile:
    """Count unique named model state, excluding the frozen teacher by default."""

    if not hasattr(model, "named_parameters") or not hasattr(model, "named_buffers"):
        raise ProfilingError("model must provide named_parameters and named_buffers")
    if isinstance(excluded_module_paths, (str, bytes)):
        raise ProfilingError("excluded module paths must be a sequence of paths")
    paths = tuple(sorted(set(excluded_module_paths)))
    if any(
        type(path) is not str or not path or path != path.strip(".") for path in paths
    ):
        raise ProfilingError("excluded module paths must be canonical")

    included_parameters = trainable_parameters = included_parameter_bytes = 0
    excluded_parameters = excluded_parameter_bytes = 0
    seen_parameters: set[int] = set()
    for name, parameter in model.named_parameters():  # type: ignore[attr-defined]
        identity = id(parameter)
        if identity in seen_parameters:
            continue
        seen_parameters.add(identity)
        count = _nonnegative_integer(parameter.numel(), f"parameter {name} size")
        size = count * _positive_integer(
            parameter.element_size(),
            f"parameter {name} element_size",
        )
        if _is_excluded(name, paths):
            excluded_parameters += count
            excluded_parameter_bytes += size
        else:
            included_parameters += count
            included_parameter_bytes += size
            if bool(parameter.requires_grad):
                trainable_parameters += count

    included_buffers = included_buffer_bytes = 0
    excluded_buffers = excluded_buffer_bytes = 0
    seen_buffers: set[int] = set()
    for name, buffer in model.named_buffers():  # type: ignore[attr-defined]
        identity = id(buffer)
        if identity in seen_buffers:
            continue
        seen_buffers.add(identity)
        count = _nonnegative_integer(buffer.numel(), f"buffer {name} size")
        size = count * _positive_integer(
            buffer.element_size(),
            f"buffer {name} element_size",
        )
        if _is_excluded(name, paths):
            excluded_buffers += count
            excluded_buffer_bytes += size
        else:
            included_buffers += count
            included_buffer_bytes += size

    return ParameterProfile(
        parameter_count=included_parameters,
        trainable_parameter_count=trainable_parameters,
        parameter_bytes=included_parameter_bytes,
        buffer_count=included_buffers,
        buffer_bytes=included_buffer_bytes,
        excluded_parameter_count=excluded_parameters,
        excluded_parameter_bytes=excluded_parameter_bytes,
        excluded_buffer_count=excluded_buffers,
        excluded_buffer_bytes=excluded_buffer_bytes,
        excluded_module_paths=paths,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ProfilingError("cannot compute a quantile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


@dataclass(frozen=True)
class LatencyProfile:
    warmup_iterations: int
    measured_iterations: int
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float
    synchronized: bool

    def __post_init__(self) -> None:
        _nonnegative_integer(self.warmup_iterations, "warmup_iterations")
        _positive_integer(self.measured_iterations, "measured_iterations")
        if len(self.samples_ms) != self.measured_iterations:
            raise ProfilingError("latency sample count does not match iterations")
        for value in (
            *self.samples_ms,
            self.mean_ms,
            self.median_ms,
            self.p90_ms,
            self.minimum_ms,
            self.maximum_ms,
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ProfilingError(
                    "latency values must be finite non-negative floats"
                )
        if type(self.synchronized) is not bool:
            raise ProfilingError("synchronized must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "samples_ms": list(self.samples_ms),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "synchronized": self.synchronized,
        }


@dataclass(frozen=True)
class GpuMemoryProfile:
    status: str
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    method: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("measured", "unavailable"):
            raise ProfilingError("GPU memory status must be measured or unavailable")
        if self.status == "measured":
            if self.peak_allocated_bytes is None or self.peak_reserved_bytes is None:
                raise ProfilingError("measured GPU memory requires peak values")
            _nonnegative_integer(self.peak_allocated_bytes, "peak_allocated_bytes")
            _nonnegative_integer(self.peak_reserved_bytes, "peak_reserved_bytes")
            if not self.method or self.reason is not None:
                raise ProfilingError("measured GPU memory has inconsistent metadata")
        elif (
            self.peak_allocated_bytes is not None
            or self.peak_reserved_bytes is not None
            or self.method is not None
            or not self.reason
        ):
            raise ProfilingError("unavailable GPU memory has inconsistent metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "method": self.method,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RuntimeProfile:
    latency: LatencyProfile
    gpu_memory: GpuMemoryProfile

    def to_dict(self) -> dict[str, object]:
        return {
            "latency": self.latency.to_dict(),
            "gpu_memory": self.gpu_memory.to_dict(),
        }


def profile_runtime(
    run_once: Callable[[], Any],
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 100,
    synchronize: Callable[[], None] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    reset_peak_memory: Callable[[], None] | None = None,
    peak_allocated_bytes: Callable[[], int] | None = None,
    peak_reserved_bytes: Callable[[], int] | None = None,
    memory_method: str = "torch.cuda.max_memory_allocated/max_memory_reserved",
) -> RuntimeProfile:
    """Measure warm-cache callable latency with explicit device synchronization.

    The callable is executed exactly ``warmup_iterations + measured_iterations``
    times.  Returned predictions are discarded and are never evaluated here.
    """

    if not callable(run_once):
        raise ProfilingError("run_once must be callable")
    warmup_iterations = _nonnegative_integer(
        warmup_iterations,
        "warmup_iterations",
    )
    measured_iterations = _positive_integer(
        measured_iterations,
        "measured_iterations",
    )
    sync = synchronize or (lambda: None)
    memory_hooks = (
        reset_peak_memory,
        peak_allocated_bytes,
        peak_reserved_bytes,
    )
    if any(hook is not None for hook in memory_hooks) and not all(
        callable(hook) for hook in memory_hooks
    ):
        raise ProfilingError("GPU memory hooks must be supplied together")

    for _ in range(warmup_iterations):
        run_once()
        sync()
    sync()
    if reset_peak_memory is not None:
        reset_peak_memory()

    samples_ms: list[float] = []
    for _ in range(measured_iterations):
        sync()
        started = clock_ns()
        run_once()
        sync()
        stopped = clock_ns()
        if type(started) is not int or type(stopped) is not int:
            raise ProfilingError("clock_ns must return monotonic integer nanoseconds")
        elapsed = stopped - started
        if elapsed < 0:
            raise ProfilingError("clock_ns must return monotonic integer nanoseconds")
        samples_ms.append(float(elapsed) / 1_000_000.0)

    mean_ms = float(sum(samples_ms) / len(samples_ms))
    latency = LatencyProfile(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        samples_ms=tuple(samples_ms),
        mean_ms=mean_ms,
        median_ms=_quantile(samples_ms, 0.5),
        p90_ms=_quantile(samples_ms, 0.9),
        minimum_ms=float(min(samples_ms)),
        maximum_ms=float(max(samples_ms)),
        synchronized=synchronize is not None,
    )
    if reset_peak_memory is None:
        memory = GpuMemoryProfile(
            status="unavailable",
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            method=None,
            reason="GPU memory hooks were not provided",
        )
    else:
        allocated = peak_allocated_bytes()  # type: ignore[misc]
        reserved = peak_reserved_bytes()  # type: ignore[misc]
        memory = GpuMemoryProfile(
            status="measured",
            peak_allocated_bytes=_nonnegative_integer(
                allocated,
                "peak_allocated_bytes",
            ),
            peak_reserved_bytes=_nonnegative_integer(
                reserved,
                "peak_reserved_bytes",
            ),
            method=memory_method,
        )
    return RuntimeProfile(latency=latency, gpu_memory=memory)


@dataclass(frozen=True)
class FlopProfile:
    status: str
    flop_count: int | None
    method: str
    operator_events_with_flops: int
    coverage_note: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("measured", "unavailable"):
            raise ProfilingError("FLOP status must be measured or unavailable")
        _nonnegative_integer(
            self.operator_events_with_flops,
            "operator_events_with_flops",
        )
        if not self.method or not self.coverage_note:
            raise ProfilingError("FLOP method and coverage note are required")
        if self.status == "measured":
            if self.flop_count is None:
                raise ProfilingError("measured FLOPs require flop_count")
            _positive_integer(self.flop_count, "flop_count")
            if self.reason is not None:
                raise ProfilingError("measured FLOPs must not have a reason")
        elif self.flop_count is not None or not self.reason:
            raise ProfilingError("unavailable FLOPs have inconsistent metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "flop_count": self.flop_count,
            "method": self.method,
            "operator_events_with_flops": self.operator_events_with_flops,
            "coverage_note": self.coverage_note,
            "reason": self.reason,
        }


def unavailable_flops(reason: str) -> FlopProfile:
    if type(reason) is not str or not reason:
        raise ProfilingError("unavailable FLOPs require a reason")
    return FlopProfile(
        status="unavailable",
        flop_count=None,
        method="torch.profiler(with_flops=True)",
        operator_events_with_flops=0,
        coverage_note=(
            "Operator-level profiler coverage is partial; unsupported custom "
            "operators are not estimated."
        ),
        reason=reason,
    )


def profile_torch_flops(
    run_once: Callable[[], Any],
    *,
    include_cuda_activity: bool,
    synchronize: Callable[[], None] | None = None,
) -> FlopProfile:
    """Measure supported operator FLOPs for one forward, or report unavailable."""

    try:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if include_cuda_activity:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sync = synchronize or (lambda: None)
        sync()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            run_once()
            sync()
        events = profiler.key_averages()
        positive = [int(event.flops) for event in events if (event.flops or 0) > 0]
        total = sum(positive)
        if total <= 0:
            return unavailable_flops(
                "the profiler reported no supported operator FLOPs"
            )
        return FlopProfile(
            status="measured",
            flop_count=total,
            method="torch.profiler(with_flops=True)",
            operator_events_with_flops=len(positive),
            coverage_note=(
                "Count includes only operators for which torch.profiler exposes "
                "a FLOP formula; custom CUDA operators may be omitted."
            ),
        )
    except Exception as error:  # profiling is an optional evidence channel
        return unavailable_flops(
            f"{type(error).__name__}: {str(error).strip() or 'profiler failed'}"
        )


def build_profile_document(
    *,
    parameters: ParameterProfile,
    runtime: RuntimeProfile,
    flops: FlopProfile,
    boundary: str,
    batch_size: int,
    device: Mapping[str, object],
    artifacts: Sequence[ArtifactDigest] = (),
    source_state: Mapping[str, object] | None = None,
    measured_at_utc: str | None = None,
) -> dict[str, object]:
    """Build a content-addressed complexity profile document."""

    if not isinstance(parameters, ParameterProfile):
        raise ProfilingError("parameters must be a ParameterProfile")
    if not isinstance(runtime, RuntimeProfile):
        raise ProfilingError("runtime must be a RuntimeProfile")
    if not isinstance(flops, FlopProfile):
        raise ProfilingError("flops must be a FlopProfile")
    if type(boundary) is not str or not boundary or boundary != boundary.strip():
        raise ProfilingError("boundary must be trimmed and non-empty")
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(device, Mapping) or not device:
        raise ProfilingError("device must be a non-empty mapping")
    roles = [artifact.role for artifact in artifacts]
    if len(roles) != len(set(roles)):
        raise ProfilingError("artifact roles must be unique")
    payload: dict[str, object] = {
        "measurement_boundary": boundary,
        "batch_size": batch_size,
        "parameters": parameters.to_dict(),
        "flops": flops.to_dict(),
        "runtime": runtime.to_dict(),
        "device": dict(device),
        "artifacts": [
            artifact.to_dict()
            for artifact in sorted(artifacts, key=lambda item: item.role)
        ],
        "source_state": dict(source_state or {}),
    }
    if measured_at_utc is not None:
        if (
            type(measured_at_utc) is not str
            or not measured_at_utc
            or measured_at_utc != measured_at_utc.strip()
        ):
            raise ProfilingError("measured_at_utc must be trimmed and non-empty")
        payload["measured_at_utc"] = measured_at_utc
    try:
        return seal_document(PROFILE_DOCUMENT_TYPE, payload)
    except EvidenceError as error:
        raise ProfilingError("profile contains non-canonical evidence") from error


__all__ = (
    "FlopProfile",
    "GpuMemoryProfile",
    "LatencyProfile",
    "ParameterProfile",
    "ProfilingError",
    "RuntimeProfile",
    "build_profile_document",
    "count_parameters",
    "profile_runtime",
    "profile_torch_flops",
    "unavailable_flops",
)
