"""Honest V2X-Seq glue for a CenterPoint-inspired tracking baseline.

This package is not part of the official CenterPoint or V2X-Seq projects.  It
contains only local conversion and orchestration contracts; no third-party
source code is vendored here.
"""

from .adapter import (
    AdapterContractError,
    BASELINE_KIND,
    CENTERPOINT_COMMIT,
    CenterPointStyleTracker,
    CenterPointV2XSeqAdapter,
    IntegrationBlockedError,
    VehicleOnlyFrameConverter,
    build_adapter,
    create_adapter,
)
from .calibration import (
    DAIR_V2X_COMMIT,
    INFRASTRUCTURE_LIDAR_FRAME,
    POINT_EQUATION,
    VEHICLE_LIDAR_FRAME,
    VEHICLE_NOVATEL_FRAME,
    V2XSEQ_COMMIT,
    WORLD_FRAME,
    CoordinateTransform,
    SpdCalibrationBridge,
    SpdCalibrationRole,
    build_spd_calibration_bridge,
    loads_spd_calibration_json,
    parse_spd_calibration,
)

__all__ = [
    "AdapterContractError",
    "BASELINE_KIND",
    "CENTERPOINT_COMMIT",
    "CenterPointStyleTracker",
    "CenterPointV2XSeqAdapter",
    "IntegrationBlockedError",
    "DAIR_V2X_COMMIT",
    "INFRASTRUCTURE_LIDAR_FRAME",
    "POINT_EQUATION",
    "CoordinateTransform",
    "SpdCalibrationBridge",
    "SpdCalibrationRole",
    "VEHICLE_LIDAR_FRAME",
    "VEHICLE_NOVATEL_FRAME",
    "V2XSEQ_COMMIT",
    "VehicleOnlyFrameConverter",
    "WORLD_FRAME",
    "build_adapter",
    "build_spd_calibration_bridge",
    "create_adapter",
    "loads_spd_calibration_json",
    "parse_spd_calibration",
]
