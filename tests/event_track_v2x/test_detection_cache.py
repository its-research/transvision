import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import DetectionCacheV1
from transvision.models.event_track_v2x.detection_cache import (
    DetectionCacheError,
    build_detection_cache,
    detection_frame_contract_sha256,
    verify_detection_cache,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


def _frame(
    frame_id: str = "000001",
    *,
    event_time: float = 1.0,
    class_label: str = "car",
) -> DetectionCacheV1:
    return DetectionCacheV1(
        sequence_id="0001",
        frame_id=frame_id,
        event_time=event_time,
        agent_id="vehicle",
        coordinate_frame="world",
        boxes_3d=np.asarray([[1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1]]),
        scores=np.asarray([0.9]),
        class_labels=(class_label,),
        velocities=np.asarray([[2.0, 0.0]]),
        covariances=np.asarray([np.eye(9)]),
        dataset_sha256="a" * 64,
        detector_config_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
    )


def _contract(
    *frames: DetectionCacheV1,
) -> tuple[tuple[str, str, str, float], ...]:
    return tuple(
        (frame.sequence_id, frame.frame_id, frame.agent_id, frame.event_time)
        for frame in frames
    )


def _verify(destination: Path, *frames: DetectionCacheV1):
    return verify_detection_cache(
        destination,
        expected_cohort_sha256="d" * 64,
        expected_frames=_contract(*frames),
        expected_frame_contract_sha256=detection_frame_contract_sha256(
            _contract(*frames)
        ),
        expected_class_names=("car",),
    )


def test_detection_cache_round_trip_is_sealed_and_gt_free(tmp_path: Path) -> None:
    destination = tmp_path / "cache"
    first = _frame()
    second = _frame("000002", event_time=1.1)
    digest = build_detection_cache(
        [first, second],
        destination,
        cache_id="cooptrack-r50-camera-v1",
        cohort_sha256="d" * 64,
        allowed_class_names=("car",),
    )

    observed, frames = _verify(destination, first, second)
    manifest = json.loads(
        (destination / "detection-cache-manifest.json").read_text()
    )
    assert observed == digest == manifest["content_sha256"]
    assert manifest["ground_truth_included"] is False
    assert [frame.frame_id for frame in frames] == ["000001", "000002"]
    assert all("track" not in path.name for path in destination.rglob("*.json"))


def test_detection_cache_rejects_duplicate_frames_and_existing_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(DetectionCacheError, match="identities"):
        build_detection_cache(
            [_frame(), _frame()],
            tmp_path / "duplicate",
            cache_id="cache-v1",
            cohort_sha256="d" * 64,
            allowed_class_names=("car",),
        )

    with pytest.raises(DetectionCacheError, match="class vocabulary"):
        build_detection_cache(
            [_frame(class_label="truck")],
            tmp_path / "wrong-class",
            cache_id="cache-v1",
            cohort_sha256="d" * 64,
            allowed_class_names=("car",),
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(DetectionCacheError, match="must not already exist"):
        build_detection_cache(
            [_frame()],
            existing,
            cache_id="cache-v1",
            cohort_sha256="d" * 64,
            allowed_class_names=("car",),
        )


@pytest.mark.parametrize("attack", ["tamper", "extra", "manifest-space"])
def test_detection_cache_rejects_mutation_and_extra_files(
    tmp_path: Path, attack: str
) -> None:
    destination = tmp_path / "cache"
    build_detection_cache(
        [_frame()],
        destination,
        cache_id="cache-v1",
        cohort_sha256="d" * 64,
        allowed_class_names=("car",),
    )
    if attack == "tamper":
        frame_path = next((destination / "frames").rglob("*.json"))
        frame_path.write_bytes(frame_path.read_bytes() + b" ")
    elif attack == "extra":
        (destination / "ground-truth.json").write_text("{}")
    else:
        manifest = destination / "detection-cache-manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(DetectionCacheError):
        _verify(destination, _frame())


def test_detection_cache_rejects_symbolic_links(tmp_path: Path) -> None:
    destination = tmp_path / "cache"
    build_detection_cache(
        [_frame()],
        destination,
        cache_id="cache-v1",
        cohort_sha256="d" * 64,
        allowed_class_names=("car",),
    )
    target = tmp_path / "outside"
    target.write_text("secret")
    try:
        (destination / "leak").symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this filesystem")

    with pytest.raises(DetectionCacheError, match="symbolic link"):
        _verify(destination, _frame())


def test_detection_cache_rejects_resealed_deleted_cohort_frame(tmp_path: Path) -> None:
    destination = tmp_path / "cache"
    first = _frame()
    second = _frame("000002", event_time=1.1)
    build_detection_cache(
        [first, second],
        destination,
        cache_id="cache-v1",
        cohort_sha256="d" * 64,
        allowed_class_names=("car",),
    )
    manifest_path = destination / "detection-cache-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    removed = manifest["entries"].pop()
    manifest["frame_count"] = 1
    payload = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (destination / removed["relative_path"]).unlink()

    with pytest.raises(DetectionCacheError, match="cohort mismatch"):
        _verify(destination, first, second)
