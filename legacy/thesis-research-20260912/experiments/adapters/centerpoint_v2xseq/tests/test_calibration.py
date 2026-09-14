from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest
from typing import Any

from experiments.adapters.centerpoint_v2xseq import calibration


INFRASTRUCTURE_FRAME = "000001"
VEHICLE_FRAME = "000101"
INFRASTRUCTURE_TO_WORLD_PATH = (
    "infrastructure-side/calib/virtuallidar_to_world/000001.json"
)
VEHICLE_TO_NOVATEL_PATH = "vehicle-side/calib/lidar_to_novatel/000101.json"
NOVATEL_TO_WORLD_PATH = "vehicle-side/calib/novatel_to_world/000101.json"

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ROTATE_Z_90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
UPSTREAM_INVERTIBLE_MATRIX = [
    [-0.0638033225610772, -0.9910914864003576, -0.04429948490729328],
    [-0.2102873406178483, 0.043997692433495696, -0.7987692871343754],
    [0.97575114561348, -0.06031492538699515, -0.17158543199893228],
]


def direct(
    *,
    rotation: Any = None,
    translation: Any = None,
) -> dict[str, Any]:
    return {
        "rotation": deepcopy(IDENTITY if rotation is None else rotation),
        "translation": deepcopy(
            [[0.0], [0.0], [0.0]] if translation is None else translation
        ),
    }


def wrapped(
    *,
    rotation: Any = None,
    translation: Any = None,
) -> dict[str, Any]:
    return {"transform": direct(rotation=rotation, translation=translation)}


def build_bridge(
    *,
    infrastructure_to_world: Any = None,
    vehicle_to_novatel: Any = None,
    novatel_to_world: Any = None,
    offset: Any = None,
) -> calibration.SpdCalibrationBridge:
    return calibration.build_spd_calibration_bridge(
        infrastructure_frame_id=INFRASTRUCTURE_FRAME,
        vehicle_frame_id=VEHICLE_FRAME,
        infrastructure_lidar_to_world_document=(
            direct(rotation=ROTATE_Z_90, translation=[[10.0], [0.0], [0.0]])
            if infrastructure_to_world is None
            else infrastructure_to_world
        ),
        infrastructure_lidar_to_world_path=INFRASTRUCTURE_TO_WORLD_PATH,
        vehicle_lidar_to_novatel_document=(
            wrapped(translation=[[1.0], [0.0], [0.0]])
            if vehicle_to_novatel is None
            else vehicle_to_novatel
        ),
        vehicle_lidar_to_novatel_path=VEHICLE_TO_NOVATEL_PATH,
        vehicle_novatel_to_world_document=(
            direct(translation=[[5.0], [0.0], [0.0]])
            if novatel_to_world is None
            else novatel_to_world
        ),
        vehicle_novatel_to_world_path=NOVATEL_TO_WORLD_PATH,
        system_error_offset=(
            {"delta_x": 0.5, "delta_y": -0.25} if offset is None else offset
        ),
    )


def observation(**overrides: Any) -> dict[str, Any]:
    value = {
        "infrastructure_frame_id": INFRASTRUCTURE_FRAME,
        "infrastructure_lidar_path": (
            "/sealed/infrastructure-side/velodyne/000001.pcd"
        ),
        "infrastructure_sequence_id": "0000",
        "infrastructure_timestamp": 100,
        "vehicle_frame_id": VEHICLE_FRAME,
        "vehicle_lidar_path": "/sealed/vehicle-side/velodyne/000101.pcd",
        "vehicle_sequence_id": "0010",
        "vehicle_timestamp": 110,
    }
    value.update(overrides)
    return value


class SpdCalibrationTest(unittest.TestCase):
    def test_three_official_roles_have_explicit_directions(self) -> None:
        infrastructure_to_world = calibration.parse_spd_calibration(
            direct(),
            role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
            relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
            expected_frame_id=INFRASTRUCTURE_FRAME,
        )
        vehicle_to_novatel = calibration.parse_spd_calibration(
            wrapped(),
            role=calibration.SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL,
            relative_path=VEHICLE_TO_NOVATEL_PATH,
            expected_frame_id=VEHICLE_FRAME,
        )
        novatel_to_world = calibration.parse_spd_calibration(
            direct(),
            role=calibration.SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD,
            relative_path=NOVATEL_TO_WORLD_PATH,
            expected_frame_id=VEHICLE_FRAME,
        )
        self.assertEqual(
            (
                infrastructure_to_world.source_frame,
                infrastructure_to_world.target_frame,
            ),
            (calibration.INFRASTRUCTURE_LIDAR_FRAME, calibration.WORLD_FRAME),
        )
        self.assertEqual(
            (vehicle_to_novatel.source_frame, vehicle_to_novatel.target_frame),
            (calibration.VEHICLE_LIDAR_FRAME, calibration.VEHICLE_NOVATEL_FRAME),
        )
        self.assertEqual(
            (novatel_to_world.source_frame, novatel_to_world.target_frame),
            (calibration.VEHICLE_NOVATEL_FRAME, calibration.WORLD_FRAME),
        )

    def test_spd_chain_and_offset_are_applied_in_target_axes(self) -> None:
        bridge = build_bridge()
        self.assertEqual(
            bridge.raw_transform.transform_point([1, 0, 0]), (4.0, 1.0, 0.0)
        )
        self.assertEqual(bridge.transform_point([1, 0, 0]), (4.5, 0.75, 0.0))
        restored = bridge.inverse_transform_point(bridge.transform_point([2, -3, 4]))
        for observed, expected in zip(restored, (2.0, -3.0, 4.0), strict=True):
            self.assertAlmostEqual(observed, expected)
        payload = bridge.to_backend_payload()
        self.assertEqual(
            payload["system_error_offset"]["application_frame"],
            calibration.VEHICLE_LIDAR_FRAME,
        )
        self.assertEqual(
            payload["system_error_offset"]["application_order"],
            "after_coordinate_chain",
        )
        self.assertEqual(
            payload["transform"]["point_equation"], calibration.POINT_EQUATION
        )

    def test_direction_mismatch_and_wrong_role_path_fail_closed(self) -> None:
        infrastructure_to_world = calibration.parse_spd_calibration(
            direct(),
            role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
            relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
            expected_frame_id=INFRASTRUCTURE_FRAME,
        )
        vehicle_to_novatel = calibration.parse_spd_calibration(
            wrapped(),
            role=calibration.SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL,
            relative_path=VEHICLE_TO_NOVATEL_PATH,
            expected_frame_id=VEHICLE_FRAME,
        )
        with self.assertRaisesRegex(
            calibration.AdapterContractError, "coordinate direction mismatch"
        ):
            infrastructure_to_world.then(vehicle_to_novatel)
        with self.assertRaisesRegex(
            calibration.AdapterContractError, "does not prove role"
        ):
            calibration.parse_spd_calibration(
                direct(),
                role=calibration.SpdCalibrationRole.VEHICLE_NOVATEL_TO_WORLD,
                relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                expected_frame_id=VEHICLE_FRAME,
            )
        with self.assertRaisesRegex(calibration.AdapterContractError, "role"):
            calibration.parse_spd_calibration(
                direct(),
                role="vehicle_novatel_to_world",  # type: ignore[arg-type]
                relative_path=NOVATEL_TO_WORLD_PATH,
                expected_frame_id=VEHICLE_FRAME,
            )

    def test_unknown_ambiguous_and_wrongly_wrapped_schemas_are_rejected(self) -> None:
        unknown = direct()
        unknown["labels"] = []
        ambiguous = direct()
        ambiguous["transform"] = direct()
        for document in (unknown, ambiguous, wrapped()):
            with self.subTest(document=document):
                with self.assertRaisesRegex(
                    calibration.AdapterContractError, "official schema exactly"
                ):
                    calibration.parse_spd_calibration(
                        document,
                        role=(
                            calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD
                        ),
                        relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                        expected_frame_id=INFRASTRUCTURE_FRAME,
                    )
        with self.assertRaisesRegex(
            calibration.AdapterContractError, "official schema exactly"
        ):
            calibration.parse_spd_calibration(
                direct(),
                role=calibration.SpdCalibrationRole.VEHICLE_LIDAR_TO_NOVATEL,
                relative_path=VEHICLE_TO_NOVATEL_PATH,
                expected_frame_id=VEHICLE_FRAME,
            )

    def test_nonfinite_singular_and_reflective_matrices_are_rejected(self) -> None:
        invalid = (
            ([[math.nan, 0, 0], [0, 1, 0], [0, 0, 1]], "finite"),
            ([[1e308, 0, 0], [0, 1e308, 0], [0, 0, 1e308]], "finite"),
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], "singular"),
            ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "handedness"),
        )
        for rotation, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(calibration.AdapterContractError, message):
                    calibration.parse_spd_calibration(
                        direct(rotation=rotation),
                        role=(
                            calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD
                        ),
                        relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                        expected_frame_id=INFRASTRUCTURE_FRAME,
                    )
        with self.assertRaisesRegex(calibration.AdapterContractError, "finite"):
            calibration.parse_spd_calibration(
                direct(translation=[[math.inf], [0], [0]]),
                role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
                relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                expected_frame_id=INFRASTRUCTURE_FRAME,
            )

    def test_upstream_invertible_nonorthonormal_matrix_round_trips(self) -> None:
        transform = calibration.parse_spd_calibration(
            direct(
                rotation=UPSTREAM_INVERTIBLE_MATRIX,
                translation=[-5.779144404715124, 6.037615758600886, 1.0636424034755758],
            ),
            role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
            relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
            expected_frame_id=INFRASTRUCTURE_FRAME,
        )
        point = (2.5, -3.0, 7.25)
        restored = transform.inverse().transform_point(transform.transform_point(point))
        for observed, expected in zip(restored, point, strict=True):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_system_error_offset_schema_and_values_are_strict(self) -> None:
        invalid = (
            "",
            {"delta_x": 0.0},
            {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0},
            {"delta_x": math.nan, "delta_y": 0.0},
        )
        for offset in invalid:
            with self.subTest(offset=offset):
                with self.assertRaises(calibration.AdapterContractError):
                    build_bridge(offset=offset)

    def test_json_entry_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        duplicate = (
            '{"rotation": [[1,0,0],[0,1,0],[0,0,1]], '
            '"translation": [0,0,0], "translation": [1,0,0]}'
        )
        nonfinite = '{"rotation": [[1,0,0],[0,1,0],[0,0,1]], "translation": [NaN,0,0]}'
        with self.assertRaisesRegex(calibration.AdapterContractError, "duplicate"):
            calibration.loads_spd_calibration_json(
                duplicate,
                role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
                relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                expected_frame_id=INFRASTRUCTURE_FRAME,
            )
        with self.assertRaisesRegex(calibration.AdapterContractError, "non-finite"):
            calibration.loads_spd_calibration_json(
                nonfinite,
                role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
                relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
                expected_frame_id=INFRASTRUCTURE_FRAME,
            )

    def test_paths_are_canonical_and_bound_to_expected_frame(self) -> None:
        invalid_paths = (
            "/" + INFRASTRUCTURE_TO_WORLD_PATH,
            "infrastructure-side/calib/../virtuallidar_to_world/000001.json",
            "infrastructure-side\\calib\\virtuallidar_to_world\\000001.json",
            "infrastructure-side/calib/virtuallidar_to_world/000002.json",
        )
        for path in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(calibration.AdapterContractError):
                    calibration.parse_spd_calibration(
                        direct(),
                        role=(
                            calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD
                        ),
                        relative_path=path,
                        expected_frame_id=INFRASTRUCTURE_FRAME,
                    )

    def test_inference_binding_rejects_any_label_bearing_or_unknown_field(self) -> None:
        bridge = build_bridge()
        clean = bridge.bind_inference_observation(observation())
        serialized = json.dumps(clean, sort_keys=True)
        for forbidden in ("labels", "annotations", "targets", "ground_truth"):
            self.assertNotIn(forbidden, serialized)
        for forbidden in ("labels", "annotations", "targets", "ground_truth"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(
                    calibration.AdapterContractError, "official schema exactly"
                ):
                    bridge.bind_inference_observation(
                        observation(**{forbidden: [{"future": True}]})
                    )

    def test_inference_binding_checks_frame_ids_paths_and_timestamps(self) -> None:
        bridge = build_bridge()
        invalid = (
            observation(infrastructure_frame_id="000002"),
            observation(vehicle_frame_id="000102"),
            observation(vehicle_lidar_path="relative/000101.pcd"),
            observation(
                vehicle_lidar_path="/sealed/vehicle-side/label/lidar/000101.json"
            ),
            observation(
                infrastructure_lidar_path=(
                    "/sealed/infrastructure-side/velodyne/../label/000001.pcd"
                )
            ),
            observation(vehicle_sequence_id="not-a-sequence"),
            observation(infrastructure_timestamp=True),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(calibration.AdapterContractError):
                    bridge.bind_inference_observation(value)

    def test_flat_and_column_translations_are_both_unambiguous(self) -> None:
        flat = calibration.parse_spd_calibration(
            direct(translation=[1, 2, 3]),
            role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
            relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
            expected_frame_id=INFRASTRUCTURE_FRAME,
        )
        column = calibration.parse_spd_calibration(
            direct(translation=[[1], [2], [3]]),
            role=calibration.SpdCalibrationRole.INFRASTRUCTURE_LIDAR_TO_WORLD,
            relative_path=INFRASTRUCTURE_TO_WORLD_PATH,
            expected_frame_id=INFRASTRUCTURE_FRAME,
        )
        self.assertEqual(flat.translation, column.translation)


if __name__ == "__main__":
    unittest.main()
