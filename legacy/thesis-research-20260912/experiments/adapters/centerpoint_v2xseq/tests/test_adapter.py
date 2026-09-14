from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ADAPTER_PATH = Path(__file__).resolve().parents[1] / "adapter.py"
SPEC = importlib.util.spec_from_file_location(
    "centerpoint_v2xseq_adapter_test", ADAPTER_PATH
)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

RUNNER_ROOT = Path(__file__).resolve().parents[3] / "rtp_v2x"
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))
import v2xseq_real_canary  # noqa: E402


def label(**overrides: Any) -> dict[str, Any]:
    value = {
        "type": "Truck",
        "track_id": 17,
        "3d_dimensions": {"l": 4.5, "w": 2.0, "h": 1.8},
        "3d_location": {"x": 10.0, "y": -2.0, "z": 0.3},
        "rotation": 0.25,
    }
    value.update(overrides)
    return value


def frame(*, labels: Any = None, timestamp: int = 100) -> dict[str, Any]:
    return {
        "sequence_id": "0001",
        "frame_id": "000001",
        "timestamp": timestamp,
        "assets": [
            {
                "agent": "vehicle",
                "modality": "lidar",
                "absolute_path": "/sealed/vehicle/000001.pcd",
            },
            {
                "agent": "infrastructure",
                "modality": "lidar",
                "absolute_path": "/sealed/infrastructure/000001.pcd",
            },
        ],
        "labels": [label()] if labels is None else labels,
    }


def frozen_context() -> dict[str, Any]:
    return {
        "protocol": {
            "status": "frozen",
            "coordinate_frame": adapter.COORDINATE_FRAME,
            "classes": ["vehicle"],
            "tracking_metrics": ["MOTA"],
        }
    }


class PoisonLabels(dict[str, Any]):
    def get(self, key: str, default: Any = None) -> Any:
        if key == "labels":
            raise AssertionError("inference touched future ground truth")
        return super().get(key, default)


class FakeBackend:
    def __init__(self, runtime_environment: Any = None) -> None:
        self.inference_samples: list[dict[str, Any]] = []
        self.environment = (
            {
                "framework": "PyTorch",
                "framework_version": "2.7.1",
                "device_name": "NVIDIA A100-SXM4-80GB",
            }
            if runtime_environment is None
            else runtime_environment
        )

    def training_forward(self, sample: Any) -> Any:
        return {"opaque_loss": sample}

    def backward(self, forward_output: Any) -> dict[str, Any]:
        return {"loss": 1.0, "did_backward": True}

    def optimizer_step(self) -> None:
        return None

    def infer(self, sample: Any) -> list[dict[str, Any]]:
        self.inference_samples.append(dict(sample))
        return [{"class_name": "vehicle", "translation": [0, 0, 0], "score": 0.9}]

    def save_checkpoint(self, path: str) -> None:
        Path(path).write_bytes(b"test-only-checkpoint")

    def runtime_environment(self) -> Any:
        return self.environment


class FakeEvaluator:
    def evaluate(self, predictions: Any, payloads: Any) -> dict[str, float]:
        return {"MOTA": 0.5}


def build_injected_adapter_for_runner(context: dict[str, Any]) -> Any:
    """Runner-loaded factory whose result is the production adapter class."""

    return adapter.create_adapter(
        context,
        backend=FakeBackend(),
        evaluator=FakeEvaluator(),
        max_distance_m=2.0,
        timestamp_to_seconds=0.1,
        max_age_frames=0,
        use_velocity=False,
    )


class CenterPointV2XSeqAdapterTest(unittest.TestCase):
    def test_spd_label_mapping_swaps_length_width_without_velocity(self) -> None:
        observed = adapter.convert_spd_label(label())
        self.assertEqual(
            observed,
            {
                "track_id": "17",
                "class_name": "vehicle",
                "source_class": "Truck",
                "box_3d": [10.0, -2.0, 0.3, 2.0, 4.5, 1.8, 0.25],
                "box_3d_order": "x,y,z,w,l,h,raw_spd_rotation",
                "velocity_available": False,
            },
        )

    def test_official_spd_car_maps_to_vehicle(self) -> None:
        self.assertEqual(
            adapter.convert_spd_label(label(type="Car"))["class_name"], "vehicle"
        )

    def test_official_spd_van_maps_to_vehicle(self) -> None:
        self.assertEqual(
            adapter.convert_spd_label(label(type="Van"))["class_name"], "vehicle"
        )

    def test_official_spd_truck_maps_to_vehicle(self) -> None:
        self.assertEqual(
            adapter.convert_spd_label(label(type="Truck"))["class_name"],
            "vehicle",
        )

    def test_official_spd_bus_maps_to_vehicle(self) -> None:
        self.assertEqual(
            adapter.convert_spd_label(label(type="Bus"))["class_name"], "vehicle"
        )

    def test_unknown_class_and_nonfinite_geometry_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            adapter.AdapterContractError, "unsupported SPD class"
        ):
            adapter.convert_spd_label(label(type="Pedestrian"))
        with self.assertRaisesRegex(adapter.AdapterContractError, "finite"):
            adapter.convert_spd_label(
                label(**{"3d_location": {"x": math.nan, "y": 0, "z": 0}})
            )

    def test_inference_conversion_never_reads_or_emits_labels(self) -> None:
        observed = adapter.VehicleOnlyFrameConverter().convert(
            PoisonLabels(frame()), training=False
        )
        self.assertNotIn("annotations", observed)
        self.assertNotIn("labels", observed)
        self.assertTrue(observed["vehicle_lidar_path"].endswith("000001.pcd"))
        self.assertEqual(observed["fusion_scope"], "vehicle_only")

    def test_converter_requires_exactly_one_vehicle_lidar(self) -> None:
        missing = frame()
        missing["assets"] = missing["assets"][1:]
        with self.assertRaisesRegex(adapter.AdapterContractError, "exactly one"):
            adapter.VehicleOnlyFrameConverter().convert(missing, training=False)
        duplicate = frame()
        duplicate["assets"].append(dict(duplicate["assets"][0]))
        with self.assertRaisesRegex(adapter.AdapterContractError, "exactly one"):
            adapter.VehicleOnlyFrameConverter().convert(duplicate, training=False)

    def test_velocity_target_is_not_silently_fabricated(self) -> None:
        converter = adapter.VehicleOnlyFrameConverter(require_velocity_targets=True)
        with self.assertRaisesRegex(adapter.AdapterContractError, "no velocity target"):
            converter.convert(frame(), training=True)

    def test_center_tracker_keeps_and_rekeys_id_at_distance_gate(self) -> None:
        tracker = adapter.CenterPointStyleTracker(
            max_distance_m=1.0,
            timestamp_to_seconds=0.1,
            max_age_frames=0,
            use_velocity=False,
        )
        first = tracker.step(
            [{"class_name": "vehicle", "translation": [0, 0, 0], "score": 0.9}],
            timestamp=10,
        )
        second = tracker.step(
            [
                {
                    "class_name": "vehicle",
                    "translation": [0.5, 0, 0],
                    "score": 0.8,
                }
            ],
            timestamp=11,
        )
        third = tracker.step(
            [
                {
                    "class_name": "vehicle",
                    "translation": [3.0, 0, 0],
                    "score": 0.7,
                }
            ],
            timestamp=12,
        )
        self.assertEqual(first[0]["track_id"], "1")
        self.assertEqual(second[0]["track_id"], "1")
        self.assertEqual(third[0]["track_id"], "2")

    def test_velocity_backprojection_and_strict_time_order(self) -> None:
        tracker = adapter.CenterPointStyleTracker(
            max_distance_m=0.2,
            timestamp_to_seconds=1.0,
            max_age_frames=0,
            use_velocity=True,
        )
        tracker.step(
            [
                {
                    "class_name": "vehicle",
                    "translation": [0, 0, 0],
                    "velocity": [1, 0],
                    "score": 1,
                }
            ],
            timestamp=1,
        )
        observed = tracker.step(
            [
                {
                    "class_name": "vehicle",
                    "translation": [1, 0, 0],
                    "velocity": [1, 0],
                    "score": 1,
                }
            ],
            timestamp=2,
        )
        self.assertEqual(observed[0]["track_id"], "1")
        with self.assertRaisesRegex(adapter.AdapterContractError, "increase strictly"):
            tracker.step([], timestamp=2)

    def test_tracker_rejects_unmapped_backend_class(self) -> None:
        tracker = adapter.CenterPointStyleTracker(
            max_distance_m=1.0,
            timestamp_to_seconds=1.0,
            max_age_frames=0,
            use_velocity=False,
        )
        with self.assertRaisesRegex(adapter.AdapterContractError, "class_name"):
            tracker.step(
                [
                    {
                        "class_name": "pedestrian",
                        "translation": [0, 0, 0],
                        "score": 1,
                    }
                ],
                timestamp=1,
            )

    def test_injected_adapter_matches_canary_without_label_leakage(self) -> None:
        backend = FakeBackend()
        instance = adapter.create_adapter(
            frozen_context(),
            backend=backend,
            evaluator=FakeEvaluator(),
            max_distance_m=2.0,
            timestamp_to_seconds=0.1,
            max_age_frames=0,
            use_velocity=False,
        )
        training_output = instance.forward(frame(), training=True)
        self.assertIn("opaque_loss", training_output)
        self.assertIs(instance.backward(training_output)["did_backward"], True)
        instance.optimizer_step()
        prediction = instance.forward(PoisonLabels(frame()), training=False)
        self.assertEqual(prediction["baseline_kind"], adapter.BASELINE_KIND)
        self.assertNotIn("annotations", backend.inference_samples[0])
        metrics = instance.evaluate(
            [{"frame_id": "000001", "prediction": prediction}], [frame()]
        )
        self.assertEqual(metrics, {"MOTA": 0.5})
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.bin"
            instance.save_checkpoint(str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), b"test-only-checkpoint")

    def test_runtime_environment_is_exact_and_runner_loadable(self) -> None:
        instance = build_injected_adapter_for_runner(frozen_context())
        self.assertEqual(
            instance.runtime_environment(),
            {
                "framework": "PyTorch",
                "framework_version": "2.7.1",
                "device_name": "NVIDIA A100-SXM4-80GB",
            },
        )

        loaded = v2xseq_real_canary.load_adapter(
            Path(__file__).resolve(),
            "build_injected_adapter_for_runner",
            frozen_context(),
        )
        self.assertEqual(loaded.runtime_environment(), instance.runtime_environment())

    def test_runtime_environment_rejects_nonexact_or_empty_metadata(self) -> None:
        invalid_environments = (
            {
                "framework": "PyTorch",
                "framework_version": "2.7.1",
                "device_name": "NVIDIA A100-SXM4-80GB",
                "cuda_version": "12.6",
            },
            {
                "framework": "PyTorch",
                "framework_version": "2.7.1",
            },
            {
                "framework": "PyTorch",
                "framework_version": " ",
                "device_name": "NVIDIA A100-SXM4-80GB",
            },
            ["PyTorch", "2.7.1", "NVIDIA A100-SXM4-80GB"],
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                instance = adapter.create_adapter(
                    frozen_context(),
                    backend=FakeBackend(environment),
                    evaluator=FakeEvaluator(),
                    max_distance_m=2.0,
                    timestamp_to_seconds=0.1,
                    max_age_frames=0,
                    use_velocity=False,
                )
                with self.assertRaises(adapter.AdapterContractError):
                    instance.runtime_environment()

    def test_metric_mismatch_and_production_factory_fail_closed(self) -> None:
        class WrongEvaluator:
            def evaluate(self, predictions: Any, payloads: Any) -> dict[str, float]:
                return {"invented": 1.0}

        instance = adapter.create_adapter(
            frozen_context(),
            backend=FakeBackend(),
            evaluator=WrongEvaluator(),
            max_distance_m=2.0,
            timestamp_to_seconds=1.0,
            max_age_frames=0,
            use_velocity=False,
        )
        with self.assertRaisesRegex(
            adapter.AdapterContractError, "metrics do not match"
        ):
            instance.evaluate([], [])
        with self.assertRaisesRegex(adapter.IntegrationBlockedError, "not official"):
            adapter.build_adapter(frozen_context())


if __name__ == "__main__":
    unittest.main()
