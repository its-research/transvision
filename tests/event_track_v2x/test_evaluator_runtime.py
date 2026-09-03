import pytest

from transvision.models.event_track_v2x.contracts import EvaluatorContractV1
from transvision.models.event_track_v2x.evaluator_runtime import (
    EVALUATOR_NAME_V1,
    EvaluatorRuntimeUnavailable,
    probe_evaluator_runtime_v1,
    require_evaluator_runtime_v1,
)


def _contract(version: str) -> EvaluatorContractV1:
    return EvaluatorContractV1(
        dataset_name="V2X-Seq-SPD",
        frequency_hz=10.0,
        split_name="val",
        sample_count=3316,
        roi=(-100.0, -40.0, -5.0, 100.0, 40.0, 5.0),
        class_mapping=(("Car", "car"),),
        matching_thresholds=(("center_distance_m", 2.0),),
        evaluator_name=EVALUATOR_NAME_V1,
        evaluator_version=version,
        cohort_sha256="a" * 64,
    )


def test_evaluator_runtime_requires_both_official_packages(monkeypatch) -> None:
    monkeypatch.setattr(
        "transvision.models.event_track_v2x.evaluator_runtime._package_version",
        lambda name: None if name == "trackeval" else "1.2.3",
    )
    monkeypatch.setattr(
        "transvision.models.event_track_v2x.evaluator_runtime._module_available",
        lambda name: name == "nuscenes",
    )
    status = probe_evaluator_runtime_v1()
    assert not status.ready
    with pytest.raises(EvaluatorRuntimeUnavailable, match="trackeval"):
        require_evaluator_runtime_v1(_contract("anything"))


def test_evaluator_runtime_requires_exact_pinned_version(monkeypatch) -> None:
    versions = {"trackeval": "1.0.0", "nuscenes-devkit": "1.2.0"}
    monkeypatch.setattr(
        "transvision.models.event_track_v2x.evaluator_runtime._package_version",
        versions.get,
    )
    monkeypatch.setattr(
        "transvision.models.event_track_v2x.evaluator_runtime._module_available",
        lambda name: True,
    )
    expected = "trackeval=1.0.0;nuscenes-devkit=1.2.0"
    assert require_evaluator_runtime_v1(_contract(expected)).ready
    with pytest.raises(EvaluatorRuntimeUnavailable, match="version mismatch"):
        require_evaluator_runtime_v1(_contract("trackeval=0;nuscenes-devkit=0"))
