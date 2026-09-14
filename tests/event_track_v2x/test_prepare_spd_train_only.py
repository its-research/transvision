from __future__ import annotations

import json
from pathlib import Path
import stat
import zipfile

import pytest

from tools.event_track_v2x.prepare_spd_train_only import (
    INFRASTRUCTURE_IMAGE_ARCHIVES,
    METADATA_ARCHIVE,
    VEHICLE_IMAGE_ARCHIVE,
    TrainOnlyPreparationError,
    prepare_spd_train_only,
)


def _side_row(side: str, frame: str, sequence: str) -> dict[str, object]:
    if side == "vehicle":
        return {
            "image_path": f"image/{frame}.jpg",
            "pointcloud_path": f"velodyne/{frame}.pcd",
            "calib_camera_intrinsic_path": (f"calib/camera_intrinsic/{frame}.json"),
            "calib_lidar_to_camera_path": (f"calib/lidar_to_camera/{frame}.json"),
            "calib_lidar_to_novatel_path": (f"calib/lidar_to_novatel/{frame}.json"),
            "calib_novatel_to_world_path": (f"calib/novatel_to_world/{frame}.json"),
            "label_camera_std_path": f"label/camera/{frame}.json",
            "label_lidar_std_path": f"label/lidar/{frame}.json",
            "frame_id": frame,
            "sequence_id": sequence,
        }
    return {
        "image_path": f"image/{frame}.jpg",
        "pointcloud_path": f"velodyne/{frame}.pcd",
        "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{frame}.json",
        "calib_virtuallidar_to_camera_path": (
            f"calib/virtuallidar_to_camera/{frame}.json"
        ),
        "calib_virtuallidar_to_world_path": (
            f"calib/virtuallidar_to_world/{frame}.json"
        ),
        "label_camera_std_path": f"label/camera/{frame}.json",
        "label_lidar_std_path": f"label/virtuallidar/{frame}.json",
        "frame_id": frame,
        "sequence_id": sequence,
    }


def _split_document() -> dict[str, object]:
    return {
        "batch_split": {
            "train": ["0001"],
            "val": ["0002"],
            "test": ["0003"],
            "test_A": ["0003"],
        },
        "vehicle_split": {
            "train": ["000001", "000002"],
            "val": ["000003"],
            "test": ["000004"],
            "test_A": ["000004"],
        },
        "infrastructure_split": {
            "train": ["100001", "100002"],
            "val": ["100003"],
            "test": ["100004"],
            "test_A": ["100004"],
        },
        "cooperative_split": {
            "train": ["000001", "000002"],
            "val": ["000003"],
            "test": ["000004"],
            "test_A": ["000004"],
        },
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    archives = root / "archives"
    archives.mkdir(parents=True)
    split_path = root / "cooperative-split-data-spd.json"
    split_path.write_text(json.dumps(_split_document()), encoding="utf-8")

    vehicle_rows = [
        _side_row("vehicle", "000001", "0001"),
        _side_row("vehicle", "000002", "0001"),
        _side_row("vehicle", "000003", "0002"),
    ]
    infrastructure_rows = [
        _side_row("infrastructure", "100001", "0001"),
        _side_row("infrastructure", "100002", "0001"),
        _side_row("infrastructure", "100003", "0002"),
    ]
    cooperative_rows = [
        {
            "vehicle_frame": vehicle,
            "infrastructure_frame": infrastructure,
            "vehicle_sequence": sequence,
            "infrastructure_sequence": sequence,
            "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0},
        }
        for vehicle, infrastructure, sequence in (
            ("000001", "100001", "0001"),
            ("000002", "100002", "0001"),
            ("000003", "100003", "0002"),
        )
    ]

    with zipfile.ZipFile(archives / METADATA_ARCHIVE, "w") as archive:
        archive.writestr(
            "V2X-Seq-SPD/vehicle-side/data_info.json",
            json.dumps(vehicle_rows),
        )
        archive.writestr(
            "V2X-Seq-SPD/infrastructure-side/data_info.json",
            json.dumps(infrastructure_rows),
        )
        archive.writestr(
            "V2X-Seq-SPD/cooperative/data_info.json",
            json.dumps(cooperative_rows),
        )
        for side, rows in (
            ("vehicle", vehicle_rows),
            ("infrastructure", infrastructure_rows),
        ):
            for row in rows:
                for key, value in row.items():
                    if key.startswith("calib_") or key.startswith("label_"):
                        archive.writestr(
                            f"V2X-Seq-SPD/{side}-side/{value}",
                            b"{}" if key.startswith("calib_") else b"[]",
                        )
        for row in cooperative_rows:
            archive.writestr(
                f"V2X-Seq-SPD/cooperative/label/{row['vehicle_frame']}.json",
                b"[]",
            )
        archive.writestr("V2X-Seq-SPD/maps/yizhuang06.json", b"{}")
        archive.writestr("V2X-Seq-SPD/README.md", b"fixture")

    infrastructure_roots = tuple(
        name.split("_11616163645886464.zip", maxsplit=1)[0]
        for name in INFRASTRUCTURE_IMAGE_ARCHIVES
    )
    with zipfile.ZipFile(archives / INFRASTRUCTURE_IMAGE_ARCHIVES[0], "w") as archive:
        archive.writestr(f"{infrastructure_roots[0]}/100001.jpg", b"i-train-1")
    with zipfile.ZipFile(archives / INFRASTRUCTURE_IMAGE_ARCHIVES[1], "w") as archive:
        archive.writestr(f"{infrastructure_roots[1]}/100002.jpg", b"i-train-2")
        archive.writestr(f"{infrastructure_roots[1]}/100003.jpg", b"i-val")
    with zipfile.ZipFile(archives / VEHICLE_IMAGE_ARCHIVE, "w") as archive:
        for frame, payload in (
            ("000001", b"v-train-1"),
            ("000002", b"v-train-2"),
            ("000003", b"v-val"),
        ):
            archive.writestr(f"V2X-Seq-SPD-vehicle-side-image/{frame}.jpg", payload)
    return archives, split_path


def _append_member(
    archive_path: Path,
    name: str,
    payload: bytes = b"malicious",
    *,
    mode: int | None = None,
) -> None:
    with zipfile.ZipFile(archive_path, "a") as archive:
        if mode is None:
            archive.writestr(name, payload)
            return
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = mode << 16
        archive.writestr(info, payload)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_materializes_exact_train_camera_projection_and_manifest(
    tmp_path: Path,
) -> None:
    archives, split_path = _write_fixture(tmp_path)
    output = tmp_path / "train-only"

    manifest = prepare_spd_train_only(
        archives,
        split_path,
        output,
        expected_split_sha256=None,
    )

    assert manifest["status"] == "materialized"
    assert manifest["policy"]["point_cloud_payloads_included"] is False
    assert manifest["counts"]["infrastructure_images"] == 2
    assert manifest["counts"]["vehicle_images"] == 2
    assert manifest["counts"]["frame_json"] == 24
    assert manifest["counts"]["output_payload_files"] == 31
    assert len(manifest["plan_sha256"]) == 64
    assert len(manifest["output_tree_sha256"]) == 64
    assert len(manifest["content_sha256"]) == 64
    assert len(manifest["source_archives"]) == 4
    assert all(len(item["sha256"]) == 64 for item in manifest["source_archives"])

    dataset = output / "V2X-Seq-SPD"
    assert sorted(path.name for path in (dataset / "vehicle-side/image").iterdir()) == [
        "000001.jpg",
        "000002.jpg",
    ]
    assert sorted(
        path.name for path in (dataset / "infrastructure-side/image").iterdir()
    ) == ["100001.jpg", "100002.jpg"]
    assert {
        row["frame_id"] for row in _read_json(dataset / "vehicle-side/data_info.json")
    } == {"000001", "000002"}
    assert {
        row["frame_id"]
        for row in _read_json(dataset / "infrastructure-side/data_info.json")
    } == {"100001", "100002"}
    assert {
        row["vehicle_frame"]
        for row in _read_json(dataset / "cooperative/data_info.json")
    } == {"000001", "000002"}
    relative_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert not any(
        path.endswith(".pcd") or "/velodyne/" in path for path in relative_files
    )
    assert not any(
        "000003.jpg" in path or "100003.jpg" in path for path in relative_files
    )
    assert _read_json(output / "train-only-manifest.json") == manifest


def test_dry_run_only_audits_and_preserves_plan_identity(tmp_path: Path) -> None:
    archives, split_path = _write_fixture(tmp_path)
    dry_output = tmp_path / "dry-output"

    audit = prepare_spd_train_only(
        archives,
        split_path,
        dry_output,
        dry_run=True,
        expected_split_sha256=None,
    )

    assert audit["status"] == "audit-only"
    assert audit["output_tree_sha256"] is None
    assert not dry_output.exists()
    materialized = prepare_spd_train_only(
        archives,
        split_path,
        tmp_path / "materialized",
        expected_split_sha256=None,
    )
    assert audit["plan_sha256"] == materialized["plan_sha256"]


def test_refuses_overwrite_before_touching_existing_output(tmp_path: Path) -> None:
    archives, split_path = _write_fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not change", encoding="utf-8")

    with pytest.raises(TrainOnlyPreparationError, match="overwrite"):
        prepare_spd_train_only(
            archives,
            split_path,
            output,
            dry_run=True,
            expected_split_sha256=None,
        )

    assert marker.read_text(encoding="utf-8") == "do not change"


@pytest.mark.parametrize(
    ("name", "mode", "message"),
    [
        ("../escape.jpg", None, "canonical POSIX"),
        (
            "V2X-Seq-SPD-vehicle-side-image/symlink.jpg",
            stat.S_IFLNK | 0o777,
            "special archive member",
        ),
    ],
)
def test_rejects_path_traversal_and_special_members_before_output(
    tmp_path: Path,
    name: str,
    mode: int | None,
    message: str,
) -> None:
    archives, split_path = _write_fixture(tmp_path)
    _append_member(archives / VEHICLE_IMAGE_ARCHIVE, name, mode=mode)
    output = tmp_path / "output"

    with pytest.raises(TrainOnlyPreparationError, match=message):
        prepare_spd_train_only(
            archives,
            split_path,
            output,
            dry_run=True,
            expected_split_sha256=None,
        )

    assert not output.exists()
    assert not (tmp_path / "escape.jpg").exists()


def test_rejects_duplicate_frame_across_image_archives(tmp_path: Path) -> None:
    archives, split_path = _write_fixture(tmp_path)
    second_root = INFRASTRUCTURE_IMAGE_ARCHIVES[1].split(
        "_11616163645886464.zip", maxsplit=1
    )[0]
    _append_member(
        archives / INFRASTRUCTURE_IMAGE_ARCHIVES[1],
        f"{second_root}/100001.jpg",
    )

    with pytest.raises(TrainOnlyPreparationError, match="across archives"):
        prepare_spd_train_only(
            archives,
            split_path,
            tmp_path / "output",
            dry_run=True,
            expected_split_sha256=None,
        )


def test_rejects_duplicate_zip_member_and_pointcloud_payload(tmp_path: Path) -> None:
    archives, split_path = _write_fixture(tmp_path)
    member = "V2X-Seq-SPD-vehicle-side-image/000001.jpg"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _append_member(archives / VEHICLE_IMAGE_ARCHIVE, member)
    with pytest.raises(TrainOnlyPreparationError, match="duplicate"):
        prepare_spd_train_only(
            archives,
            split_path,
            tmp_path / "duplicate-output",
            dry_run=True,
            expected_split_sha256=None,
        )

    archives, split_path = _write_fixture(tmp_path / "pointcloud-fixture")
    _append_member(
        archives / METADATA_ARCHIVE,
        "V2X-Seq-SPD/vehicle-side/velodyne/000001.pcd",
    )
    with pytest.raises(TrainOnlyPreparationError, match="unexpected metadata"):
        prepare_spd_train_only(
            archives,
            split_path,
            tmp_path / "pointcloud-output",
            dry_run=True,
            expected_split_sha256=None,
        )


def test_rejects_split_leakage_and_unpinned_split_by_default(tmp_path: Path) -> None:
    archives, split_path = _write_fixture(tmp_path)
    with pytest.raises(TrainOnlyPreparationError, match="SHA-256 mismatch"):
        prepare_spd_train_only(
            archives,
            split_path,
            tmp_path / "unpinned-output",
            dry_run=True,
        )

    split = _split_document()
    split["vehicle_split"]["val"] = ["000001"]
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(TrainOnlyPreparationError, match="leakage"):
        prepare_spd_train_only(
            archives,
            split_path,
            tmp_path / "leak-output",
            dry_run=True,
            expected_split_sha256=None,
        )
