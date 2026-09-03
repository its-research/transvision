"""Fail-closed detector-locked baseline adapters.

The adapters in this module separate two claims that must not be conflated:

* :class:`ReferenceMultiTargetTracker` is a locally testable lifecycle backend;
* an upstream tracker is a reproduced baseline only after a dedicated bridge and
  all of its declared dependencies are available.

Every adapter accepts only :class:`DetectionCacheV1` records bound to one
immutable detector lock.  The wrapper preserves the public tracker lifecycle,
rejects cache provenance drift before mutating the backend, and records the
digest of every applied *input* cache even when a baseline-specific view of the
cache is passed to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
from importlib.util import find_spec
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import DetectionCacheV1, TrackingPredictionV1
from .tracker import (
    ReferenceMultiTargetTracker,
    TrackerAdapter,
    TrackerIngestResult,
    TrackerIngestStatus,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DetectionInputMismatchError(ValueError):
    """Raised before state mutation when a cache violates the detector lock."""


class UnavailableBaselineError(RuntimeError):
    """Raised when an upstream baseline cannot be reproduced in this checkout."""


class AtomicBatchUnsupportedError(RuntimeError):
    """Raised before mutation when a backend lacks atomic batch ingestion."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    value = _nonempty(value, name)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class DetectionInputLock:
    """Provenance shared by all methods in one detector-locked comparison.

    ``cache_manifest_sha256`` binds the immutable cache as a whole and
    ``frame_sha256s`` is the exact allow-list returned by cache verification.
    The other fields are checked against every ingested frame so a tracker
    cannot silently switch detector, checkpoint, dataset, coordinate frame, or
    synthesize an unlisted frame mid-sequence.
    """

    cache_manifest_sha256: str
    dataset_sha256: str
    detector_config_sha256: str
    checkpoint_sha256: str
    coordinate_frame: str
    frame_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cache_manifest_sha256",
            "dataset_sha256",
            "detector_config_sha256",
            "checkpoint_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "coordinate_frame",
            _nonempty(self.coordinate_frame, "coordinate_frame"),
        )
        if not isinstance(self.frame_sha256s, tuple) or not self.frame_sha256s:
            raise ValueError("frame_sha256s must be a non-empty tuple")
        validated = tuple(
            _sha256(value, f"frame_sha256s[{index}]")
            for index, value in enumerate(self.frame_sha256s)
        )
        if len(set(validated)) != len(validated):
            raise ValueError("frame_sha256s must be unique")
        object.__setattr__(self, "frame_sha256s", tuple(sorted(validated)))

    @classmethod
    def from_cache(
        cls, cache: DetectionCacheV1, *, cache_manifest_sha256: str
    ) -> "DetectionInputLock":
        if not isinstance(cache, DetectionCacheV1):
            raise TypeError("cache must be DetectionCacheV1")
        return cls(
            cache_manifest_sha256=cache_manifest_sha256,
            dataset_sha256=cache.dataset_sha256,
            detector_config_sha256=cache.detector_config_sha256,
            checkpoint_sha256=cache.checkpoint_sha256,
            coordinate_frame=cache.coordinate_frame,
            frame_sha256s=(cache.digest(),),
        )

    @classmethod
    def from_caches(
        cls,
        caches: Iterable[DetectionCacheV1],
        *,
        cache_manifest_sha256: str,
    ) -> "DetectionInputLock":
        """Build an exact membership lock from already verified cache frames.

        Callers handling a cache directory should normally use
        :meth:`from_verified_directory`.  This constructor exists for in-memory
        verification pipelines and tests; it never infers membership from
        provenance fields alone.
        """

        frames = tuple(caches)
        if not frames:
            raise ValueError("caches must contain at least one frame")
        if not all(isinstance(frame, DetectionCacheV1) for frame in frames):
            raise TypeError("every cache must be DetectionCacheV1")
        first = frames[0]
        expected = (
            first.dataset_sha256,
            first.detector_config_sha256,
            first.checkpoint_sha256,
            first.coordinate_frame,
        )
        if any(
            (
                frame.dataset_sha256,
                frame.detector_config_sha256,
                frame.checkpoint_sha256,
                frame.coordinate_frame,
            )
            != expected
            for frame in frames[1:]
        ):
            raise DetectionInputMismatchError(
                "verified cache frames do not share one detector lock"
            )
        return cls(
            cache_manifest_sha256=cache_manifest_sha256,
            dataset_sha256=first.dataset_sha256,
            detector_config_sha256=first.detector_config_sha256,
            checkpoint_sha256=first.checkpoint_sha256,
            coordinate_frame=first.coordinate_frame,
            frame_sha256s=tuple(frame.digest() for frame in frames),
        )

    @classmethod
    def from_verified_directory(
        cls,
        directory: Path,
        *,
        expected_cohort_sha256: str,
        expected_frames: Iterable[tuple[str, str, str, float]],
        expected_frame_contract_sha256: str,
        expected_class_names: tuple[str, ...],
    ) -> "DetectionInputLock":
        """Verify an immutable cache directory and bind its exact frame set."""

        from .detection_cache import verify_detection_cache

        manifest_sha256, frames = verify_detection_cache(
            Path(directory),
            expected_cohort_sha256=expected_cohort_sha256,
            expected_frames=expected_frames,
            expected_frame_contract_sha256=expected_frame_contract_sha256,
            expected_class_names=expected_class_names,
        )
        return cls.from_caches(
            frames,
            cache_manifest_sha256=manifest_sha256,
        )

    def validate(self, cache: DetectionCacheV1) -> None:
        if not isinstance(cache, DetectionCacheV1):
            raise TypeError("cache must be DetectionCacheV1")
        expected = {
            "dataset_sha256": self.dataset_sha256,
            "detector_config_sha256": self.detector_config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "coordinate_frame": self.coordinate_frame,
        }
        mismatches = tuple(
            name
            for name, value in expected.items()
            if getattr(cache, name) != value
        )
        if mismatches:
            raise DetectionInputMismatchError(
                "detection cache violates detector lock: " + ", ".join(mismatches)
            )
        if cache.digest() not in self.frame_sha256s:
            raise DetectionInputMismatchError(
                "detection cache frame is not a member of the verified manifest"
            )


class DetectorLockedTrackerAdapter:
    """Lifecycle guard around an arbitrary verified :class:`TrackerAdapter`."""

    adapter_role = "detector-locked"

    def __init__(
        self,
        detection_lock: DetectionInputLock,
        *,
        backend: TrackerAdapter | None = None,
    ) -> None:
        if not isinstance(detection_lock, DetectionInputLock):
            raise TypeError("detection_lock must be DetectionInputLock")
        selected_backend = (
            ReferenceMultiTargetTracker() if backend is None else backend
        )
        if not isinstance(selected_backend, TrackerAdapter):
            raise TypeError("backend must implement TrackerAdapter")
        self._detection_lock = detection_lock
        self._backend = selected_backend
        self._initialized = False
        self._finalized = False
        self._sequence_id = ""
        self._processed_input_digests: set[str] = set()
        self._processed_backend_digests: set[str] = set()
        self._input_receipts: list[str] = []

    @property
    def adapter_name(self) -> str:
        return f"{self.adapter_role}:{self._backend.adapter_name}"

    @property
    def current_time(self) -> float:
        return self._backend.current_time

    @property
    def detection_lock(self) -> DetectionInputLock:
        return self._detection_lock

    @property
    def input_cache_sha256s(self) -> tuple[str, ...]:
        """Original cache digests in successful canonical application order."""

        return tuple(self._input_receipts)

    @property
    def supports_atomic_ingest_batch(self) -> bool:
        """Whether the selected backend exposes the required atomic operation.

        This is a runtime interface capability, not evidence that an upstream
        implementation was reproduced or is eligible for a paper ranking.
        """

        return callable(getattr(self._backend, "ingest_batch", None))

    @property
    def ranking_eligible(self) -> bool:
        """The wrapper alone never establishes official-baseline eligibility."""

        return False

    def _ensure_active(self) -> None:
        if not self._initialized:
            raise RuntimeError("adapter must be reset before use")
        if self._finalized:
            raise RuntimeError("adapter has been finalized")

    def reset(self, *, sequence_id: str, initial_time: float) -> None:
        if self._initialized:
            raise RuntimeError(
                "adapter instances are single-sequence; create a new instance "
                "instead of erasing committed history"
            )
        sequence_id = _nonempty(sequence_id, "sequence_id")
        self._backend.reset(sequence_id=sequence_id, initial_time=initial_time)
        self._initialized = True
        self._finalized = False
        self._sequence_id = sequence_id
        self._processed_input_digests = set()
        self._processed_backend_digests = set()
        self._input_receipts = []

    def _validate_cache_role(self, detections: DetectionCacheV1) -> None:
        del detections

    def _backend_cache(self, detections: DetectionCacheV1) -> DetectionCacheV1:
        return detections

    def ingest(
        self, detections: DetectionCacheV1, *, arrival_time: float
    ) -> TrackerIngestResult:
        self._ensure_active()
        if not isinstance(detections, DetectionCacheV1):
            raise TypeError("detections must be DetectionCacheV1")
        input_digest = detections.digest()
        if input_digest in self._processed_input_digests:
            return TrackerIngestResult(
                TrackerIngestStatus.DUPLICATE_CACHE, input_digest
            )
        self._detection_lock.validate(detections)
        if detections.sequence_id != self._sequence_id:
            return TrackerIngestResult(
                TrackerIngestStatus.SEQUENCE_MISMATCH, input_digest
            )
        self._validate_cache_role(detections)
        backend_cache = self._backend_cache(detections)
        if not isinstance(backend_cache, DetectionCacheV1):
            raise TypeError("baseline cache view must be DetectionCacheV1")
        backend_digest = backend_cache.digest()
        if backend_digest in self._processed_backend_digests:
            raise DetectionInputMismatchError(
                "distinct detector-cache inputs collapse to one baseline view"
            )
        backend_result = self._backend.ingest(
            backend_cache, arrival_time=arrival_time
        )
        if backend_result.cache_sha256 != backend_digest:
            raise RuntimeError("backend returned a cache digest it did not ingest")
        if backend_result.status is TrackerIngestStatus.DUPLICATE_CACHE:
            raise DetectionInputMismatchError(
                "distinct detector-cache inputs collapse to one baseline view"
            )
        if backend_result.applied:
            self._processed_input_digests.add(input_digest)
            self._processed_backend_digests.add(backend_digest)
            self._input_receipts.append(input_digest)
        return TrackerIngestResult(
            status=backend_result.status,
            cache_sha256=input_digest,
            matched_track_ids=backend_result.matched_track_ids,
            born_track_ids=backend_result.born_track_ids,
            pruned_track_ids=backend_result.pruned_track_ids,
        )

    def ingest_batch(
        self,
        batch: Iterable[tuple[DetectionCacheV1, float]],
    ) -> tuple[TrackerIngestResult, ...]:
        """Preflight and atomically ingest simultaneously available caches.

        The wrapper never emulates a batch by sequentially calling ``ingest``:
        that would make multi-agent association depend on input order and could
        leave partial state after a later rejection.  Instead, every detector
        lock, role, sequence, duplicate, transformed-view, and arrival check is
        completed first.  A canonicalized batch is then passed exactly once to
        a backend-provided atomic ``ingest_batch`` implementation.
        """

        self._ensure_active()
        backend_batch = getattr(self._backend, "ingest_batch", None)
        if not callable(backend_batch):
            raise AtomicBatchUnsupportedError(
                "backend lacks atomic ingest_batch; sequential fallback is forbidden"
            )
        try:
            items = tuple(batch)
        except TypeError as error:
            raise TypeError(
                "batch must be an iterable of (DetectionCacheV1, arrival_time)"
            ) from error
        if not items:
            return ()

        prepared: list[
            tuple[
                tuple[float, str, str, float, str],
                str,
                str,
                DetectionCacheV1,
                float,
            ]
        ] = []
        seen_input_digests: set[str] = set()
        seen_backend_digests: set[str] = set()
        for item_index, item in enumerate(items):
            try:
                detections, raw_arrival_time = item
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "batch items must be (DetectionCacheV1, arrival_time) pairs"
                ) from error
            if not isinstance(detections, DetectionCacheV1):
                raise TypeError(
                    f"batch item {item_index} detections must be DetectionCacheV1"
                )
            try:
                arrival_time = float(raw_arrival_time)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"batch[{item_index}].arrival_time must be finite"
                ) from error
            if not np.isfinite(arrival_time):
                raise ValueError(
                    f"batch[{item_index}].arrival_time must be finite"
                )

            input_digest = detections.digest()
            if (
                input_digest in self._processed_input_digests
                or input_digest in seen_input_digests
            ):
                raise DetectionInputMismatchError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.DUPLICATE_CACHE.value}"
                )
            self._detection_lock.validate(detections)
            if detections.sequence_id != self._sequence_id:
                raise DetectionInputMismatchError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.SEQUENCE_MISMATCH.value}"
                )
            self._validate_cache_role(detections)
            backend_cache = self._backend_cache(detections)
            if not isinstance(backend_cache, DetectionCacheV1):
                raise TypeError("baseline cache view must be DetectionCacheV1")
            backend_digest = backend_cache.digest()
            if (
                backend_digest in self._processed_backend_digests
                or backend_digest in seen_backend_digests
            ):
                raise DetectionInputMismatchError(
                    "distinct detector-cache inputs collapse to one baseline view"
                )

            seen_input_digests.add(input_digest)
            seen_backend_digests.add(backend_digest)
            sort_key = (
                arrival_time,
                detections.agent_id,
                detections.frame_id,
                detections.event_time,
                input_digest,
            )
            prepared.append(
                (
                    sort_key,
                    input_digest,
                    backend_digest,
                    backend_cache,
                    arrival_time,
                )
            )

        prepared.sort(key=lambda item: item[0])
        try:
            backend_results = tuple(
                backend_batch(
                    (backend_cache, arrival_time)
                    for _, _, _, backend_cache, arrival_time in prepared
                )
            )
        except TypeError as error:
            raise RuntimeError(
                "backend atomic ingest_batch returned a non-iterable result"
            ) from error
        if len(backend_results) != len(prepared) or not all(
            isinstance(result, TrackerIngestResult) for result in backend_results
        ):
            raise RuntimeError(
                "backend atomic ingest_batch returned incompatible results"
            )
        by_backend_digest: dict[str, TrackerIngestResult] = {}
        for result in backend_results:
            if result.cache_sha256 in by_backend_digest:
                raise RuntimeError(
                    "backend atomic ingest_batch returned duplicate cache receipts"
                )
            if not result.applied:
                raise DetectionInputMismatchError(
                    "backend rejected a detector-lock batch after wrapper preflight: "
                    f"{result.status.value}"
                )
            by_backend_digest[result.cache_sha256] = result
        expected_backend_digests = {
            backend_digest for _, _, backend_digest, _, _ in prepared
        }
        if set(by_backend_digest) != expected_backend_digests:
            raise RuntimeError(
                "backend atomic ingest_batch receipts do not match submitted caches"
            )

        translated: list[TrackerIngestResult] = []
        for _, input_digest, backend_digest, _, _ in prepared:
            backend_result = by_backend_digest[backend_digest]
            translated.append(
                TrackerIngestResult(
                    status=backend_result.status,
                    cache_sha256=input_digest,
                    matched_track_ids=backend_result.matched_track_ids,
                    born_track_ids=backend_result.born_track_ids,
                    pruned_track_ids=backend_result.pruned_track_ids,
                )
            )
        self._processed_input_digests.update(seen_input_digests)
        self._processed_backend_digests.update(seen_backend_digests)
        self._input_receipts.extend(result.cache_sha256 for result in translated)
        return tuple(translated)

    def advance(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        return self._backend.advance(decision_time)

    def commit(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        return self._backend.commit(decision_time)

    def finalize(self) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        result = self._backend.finalize()
        self._finalized = True
        return result


class VehicleOnlyTrackerAdapter(DetectorLockedTrackerAdapter):
    """Detector-locked vehicle-only baseline with explicit agent isolation."""

    adapter_role = "vehicle-only"

    def __init__(
        self,
        detection_lock: DetectionInputLock,
        *,
        vehicle_agent_id: str = "vehicle",
        backend: TrackerAdapter | None = None,
    ) -> None:
        super().__init__(detection_lock, backend=backend)
        self._vehicle_agent_id = _nonempty(vehicle_agent_id, "vehicle_agent_id")

    @property
    def vehicle_agent_id(self) -> str:
        return self._vehicle_agent_id

    def _validate_cache_role(self, detections: DetectionCacheV1) -> None:
        if detections.agent_id != self._vehicle_agent_id:
            raise DetectionInputMismatchError(
                "vehicle-only adapter rejects non-vehicle cache: "
                f"expected {self._vehicle_agent_id!r}, got {detections.agent_id!r}"
            )


class AsynchronousFusionMode(str, Enum):
    """Predefined interpretations of a delayed remote detection."""

    DIRECT_STALE = "direct-stale"
    CONSTANT_VELOCITY = "constant-velocity"


class AsynchronousFusionTrackerAdapter(DetectorLockedTrackerAdapter):
    """Shared-cache asynchronous late-fusion baseline.

    ``DIRECT_STALE`` zeros velocity only for remote caches before the reference
    backend projects the measurement to decision time.  Consequently the box
    center remains at its stale source-time location.  ``CONSTANT_VELOCITY``
    preserves the cache velocity and lets the backend compensate the elapsed
    event time.  Local detections are unchanged in both modes.

    This is an auditable detector-locked fusion adapter, not a claim that an
    upstream tracking paper has been reproduced.
    """

    adapter_role = "asynchronous-fusion"

    def __init__(
        self,
        detection_lock: DetectionInputLock,
        *,
        mode: AsynchronousFusionMode,
        local_agent_id: str = "vehicle",
        backend: TrackerAdapter | None = None,
    ) -> None:
        super().__init__(detection_lock, backend=backend)
        if not isinstance(mode, AsynchronousFusionMode):
            raise TypeError("mode must be AsynchronousFusionMode")
        self._mode = mode
        self._local_agent_id = _nonempty(local_agent_id, "local_agent_id")

    @property
    def mode(self) -> AsynchronousFusionMode:
        return self._mode

    @property
    def adapter_name(self) -> str:
        return (
            f"{self.adapter_role}:{self._mode.value}:"
            f"{self._backend.adapter_name}"
        )

    def _backend_cache(self, detections: DetectionCacheV1) -> DetectionCacheV1:
        if (
            self._mode is AsynchronousFusionMode.CONSTANT_VELOCITY
            or detections.agent_id == self._local_agent_id
        ):
            return detections
        return DetectionCacheV1(
            sequence_id=detections.sequence_id,
            frame_id=detections.frame_id,
            event_time=detections.event_time,
            agent_id=detections.agent_id,
            coordinate_frame=detections.coordinate_frame,
            boxes_3d=detections.boxes_3d,
            scores=detections.scores,
            class_labels=detections.class_labels,
            velocities=np.zeros_like(detections.velocities),
            covariances=detections.covariances,
            dataset_sha256=detections.dataset_sha256,
            detector_config_sha256=detections.detector_config_sha256,
            checkpoint_sha256=detections.checkpoint_sha256,
        )


# The bridge is intentionally separate from the legacy offline scripts.  A
# baseline is ready only when both upstream dependencies and this contract-aware
# bridge exist.  Merely finding a source directory is not reproduction evidence.
@dataclass(frozen=True, slots=True)
class _OfficialBackendSpec:
    name: str
    required_modules: tuple[str, ...]
    bridge_module: str


@dataclass(frozen=True, slots=True)
class OfficialBackendStatus:
    """Import discovery only; never grants scientific ranking eligibility."""

    name: str
    ready: bool
    missing_modules: tuple[str, ...]
    bridge_module: str
    detail: str

    @property
    def ranking_eligible(self) -> bool:
        """Qualification is external to dependency/bridge discovery."""

        return False


_OFFICIAL_BACKENDS: Mapping[str, _OfficialBackendSpec] = {
    "ab3dmot": _OfficialBackendSpec(
        name="ab3dmot",
        required_modules=(
            "AB3DMOT",
            "filterpy",
            "xinshuo_io",
            "xinshuo_miscellaneous",
        ),
        bridge_module="transvision.models.event_track_v2x.backends.ab3dmot",
    ),
    "simpletrack": _OfficialBackendSpec(
        name="simpletrack",
        required_modules=("simpletrack",),
        bridge_module="transvision.models.event_track_v2x.backends.simpletrack",
    ),
    "immortaltracker": _OfficialBackendSpec(
        name="immortaltracker",
        required_modules=("immortaltracker",),
        bridge_module=(
            "transvision.models.event_track_v2x.backends.immortaltracker"
        ),
    ),
}


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_official_backend(name: str) -> OfficialBackendStatus:
    """Inspect import availability without importing untrusted tracker code."""

    normalized = _nonempty(name, "baseline name").lower()
    try:
        spec = _OFFICIAL_BACKENDS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown official baseline: {name}") from exc
    candidates = (*spec.required_modules, spec.bridge_module)
    missing = tuple(module for module in candidates if not _module_exists(module))
    if missing:
        detail = "missing required modules or verified bridge: " + ", ".join(
            missing
        )
    else:
        detail = (
            "dependencies and DetectionCacheV1 bridge are discoverable; atomic batch "
            "semantics and scientific qualification still require runtime validation"
        )
    return OfficialBackendStatus(
        name=normalized,
        ready=not missing,
        missing_modules=missing,
        bridge_module=spec.bridge_module,
        detail=detail,
    )


def build_official_backend(name: str, **kwargs: Any) -> TrackerAdapter:
    """Build a verified upstream bridge or fail closed.

    Each bridge must expose ``build_tracker(**kwargs)`` and return an object
    satisfying :class:`TrackerAdapter`.  No fallback to the reference tracker is
    allowed because that would mislabel an experiment as an official baseline.
    """

    status = probe_official_backend(name)
    if not status.ready:
        raise UnavailableBaselineError(
            f"official baseline {status.name!r} is unavailable: {status.detail}"
        )
    try:
        module = importlib.import_module(status.bridge_module)
    except Exception as exc:
        raise UnavailableBaselineError(
            f"verified bridge {status.bridge_module!r} failed to import"
        ) from exc
    factory = getattr(module, "build_tracker", None)
    if not callable(factory):
        raise UnavailableBaselineError(
            f"verified bridge {status.bridge_module!r} has no build_tracker factory"
        )
    try:
        backend = factory(**kwargs)
    except Exception as exc:
        raise UnavailableBaselineError(
            f"verified bridge {status.bridge_module!r} failed to build a tracker"
        ) from exc
    if not isinstance(backend, TrackerAdapter):
        raise UnavailableBaselineError(
            f"verified bridge {status.bridge_module!r} returned an incompatible backend"
        )
    if not callable(getattr(backend, "ingest_batch", None)):
        raise UnavailableBaselineError(
            f"verified bridge {status.bridge_module!r} lacks atomic ingest_batch"
        )
    return backend


def official_backend_statuses() -> tuple[OfficialBackendStatus, ...]:
    return tuple(probe_official_backend(name) for name in sorted(_OFFICIAL_BACKENDS))


__all__ = [
    "AsynchronousFusionMode",
    "AsynchronousFusionTrackerAdapter",
    "AtomicBatchUnsupportedError",
    "DetectionInputLock",
    "DetectionInputMismatchError",
    "DetectorLockedTrackerAdapter",
    "OfficialBackendStatus",
    "UnavailableBaselineError",
    "VehicleOnlyTrackerAdapter",
    "build_official_backend",
    "official_backend_statuses",
    "probe_official_backend",
]
