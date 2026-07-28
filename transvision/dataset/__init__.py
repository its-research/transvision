from __future__ import annotations

from importlib import import_module


_MANIFEST_EXPORTS = (
    "ManifestError",
    "MANIFEST_SCHEMA_VERSION",
    "OFFICIAL_COOPERATIVE_SPLIT_SHA256",
    "RawSliceRecord",
    "GroundTruthBoxRecord",
    "TemporalSampleRecord",
    "ReleaseInventoryEntry",
    "PreparedArtifactRecord",
    "TemporalManifest",
    "canonical_json_bytes",
    "content_sha256",
    "build_release_inventory",
    "release_inventory_sha256",
    "load_temporal_manifest",
)
_SCHEDULE_EXPORTS = (
    "ScheduleError",
    "TransportPlan",
    "FaultPlan",
    "ArrivalRelativeFaultPlan",
    "CausalFaultPlan",
    "OverlayDigest",
    "TransportOverlayRecord",
    "FaultOverlayRecord",
    "stable_uint64",
    "bernoulli_from_hash",
    "delay_from_hash",
    "write_transport_overlay",
    "write_fault_overlay",
    "write_arrival_relative_fault_overlay",
    "write_causal_fault_overlay",
    "read_overlay",
    "augmentation_seed",
)
_RUNTIME_EXPORTS = (
    "RUNTIME_BRANCH_ORDER",
    "RuntimeProtocolError",
    "RuntimeOverlayIndex",
    "ResolvedHistorySlot",
    "ResolvedBranch",
    "ResolvedTemporalSample",
    "resolve_temporal_sample",
    "ResilientTemporalDataset",
    "collate_resilient_samples",
    "EpochIndexSampler",
)
__all__ = (
    *_MANIFEST_EXPORTS,
    "SUPPROTED_DATASETS",
    "BEVLoadMultiViewImageFromFiles",
)


def __getattr__(name: str) -> object:
    if name in _MANIFEST_EXPORTS:
        module = import_module(".resilient_v2x_manifest", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SCHEDULE_EXPORTS:
        module = import_module(".resilient_v2x_schedule", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _RUNTIME_EXPORTS:
        module = import_module(".resilient_v2x_runtime", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "SUPPROTED_DATASETS":
        module = import_module(".dair_v2x_for_detection", __name__)
        value = {
            "dair-v2x-v": module.DAIRV2XV,
            "dair-v2x-i": module.DAIRV2XI,
            "vic-sync": module.VICSyncDataset,
            "vic-async": module.VICAsyncDataset,
            "dair-v2x-v-spd": module.DAIRV2XVSPD,
            "dair-v2x-i-spd": module.DAIRV2XISPD,
            "vic-sync-spd": module.VICSyncDatasetSPD,
            "vic-async-spd": module.VICAsyncDatasetSPD,
        }
        globals()[name] = value
        return value
    if name == "BEVLoadMultiViewImageFromFiles":
        module = import_module(".transforms.loading", __name__)
        value = module.BEVLoadMultiViewImageFromFiles
        globals()[name] = value
        return value
    if name in ("transforms", "v2x_dataset"):
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
