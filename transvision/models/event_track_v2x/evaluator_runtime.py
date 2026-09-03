"""Fail-closed runtime probe for the two pinned tracking evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util

from .contracts import EvaluatorContractV1


EVALUATOR_NAME_V1 = "nuscenes-tracking+TrackEval"


class EvaluatorRuntimeUnavailable(RuntimeError):
    """Raised when a requested official evaluator cannot be proved available."""


def _module_available(name: str) -> bool:
    try:
        return util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class EvaluatorRuntimeStatusV1:
    ready: bool
    trackeval_version: str | None
    nuscenes_devkit_version: str | None
    observed_version: str | None
    reason: str


def probe_evaluator_runtime_v1() -> EvaluatorRuntimeStatusV1:
    trackeval_version = _package_version("trackeval")
    nuscenes_version = _package_version("nuscenes-devkit")
    trackeval_module = _module_available("trackeval")
    nuscenes_module = _module_available("nuscenes")
    ready = (
        trackeval_version is not None
        and nuscenes_version is not None
        and trackeval_module
        and nuscenes_module
    )
    observed = (
        f"trackeval={trackeval_version};nuscenes-devkit={nuscenes_version}"
        if ready
        else None
    )
    missing: list[str] = []
    if trackeval_version is None or not trackeval_module:
        missing.append("trackeval")
    if nuscenes_version is None or not nuscenes_module:
        missing.append("nuscenes-devkit")
    reason = "ready" if ready else "missing official evaluator runtime: " + ", ".join(missing)
    return EvaluatorRuntimeStatusV1(
        ready=ready,
        trackeval_version=trackeval_version,
        nuscenes_devkit_version=nuscenes_version,
        observed_version=observed,
        reason=reason,
    )


def require_evaluator_runtime_v1(
    contract: EvaluatorContractV1,
) -> EvaluatorRuntimeStatusV1:
    if not isinstance(contract, EvaluatorContractV1):
        raise TypeError("contract must be EvaluatorContractV1")
    if contract.evaluator_name != EVALUATOR_NAME_V1:
        raise EvaluatorRuntimeUnavailable(
            f"unsupported evaluator contract: {contract.evaluator_name!r}"
        )
    status = probe_evaluator_runtime_v1()
    if not status.ready:
        raise EvaluatorRuntimeUnavailable(status.reason)
    if contract.evaluator_version != status.observed_version:
        raise EvaluatorRuntimeUnavailable(
            "evaluator version mismatch: "
            f"contract={contract.evaluator_version!r}, observed={status.observed_version!r}"
        )
    return status


__all__ = [
    "EVALUATOR_NAME_V1",
    "EvaluatorRuntimeStatusV1",
    "EvaluatorRuntimeUnavailable",
    "probe_evaluator_runtime_v1",
    "require_evaluator_runtime_v1",
]
