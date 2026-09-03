import json
from pathlib import Path
import zipfile

import pytest

from transvision.dataset.event_track_v2x_release_audit import (
    SPDReleaseAuditError,
    audit_spd_release,
)


def _write_split(path: Path) -> None:
    document = {
        "batch_split": {},
        "cooperative_split": {
            "test": ["000003"],
            "train": ["000001"],
            "val": ["000002"],
        },
        "infrastructure_split": {},
        "vehicle_split": {},
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_archive(path: Path, *, include_test: bool) -> None:
    rows = [
        {
            "vehicle_frame": "000001",
            "infrastructure_frame": "100001",
            "vehicle_sequence": "0001",
            "infrastructure_sequence": "0001",
        },
        {
            "vehicle_frame": "000002",
            "infrastructure_frame": "100002",
            "vehicle_sequence": "0002",
            "infrastructure_sequence": "0002",
        },
    ]
    if include_test:
        rows.append(
            {
                "vehicle_frame": "000003",
                "infrastructure_frame": "100003",
                "vehicle_sequence": "0003",
                "infrastructure_sequence": "0003",
            }
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "V2X-Seq-SPD/cooperative/data_info.json", json.dumps(rows)
        )
        for row in rows:
            archive.writestr(
                f"V2X-Seq-SPD/cooperative/label/{row['vehicle_frame']}.json",
                "[]",
            )


def test_spd_release_audit_marks_synthetic_protocol_cohorts_not_ready(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    archive = tmp_path / "spd.zip"
    _write_split(split)
    _write_archive(archive, include_test=True)
    report = audit_spd_release(archive, split)
    assert report["split_reports"]["train"]["protocol_required"] is True
    assert report["split_reports"]["train"]["protocol_role"] == "development"
    assert report["split_reports"]["train"]["protocol_ready"] is False
    assert report["split_reports"]["val"]["protocol_required"] is True
    assert report["split_reports"]["val"]["protocol_role"] == "confirmatory"
    assert report["split_reports"]["val"]["protocol_ready"] is False
    assert report["split_reports"]["test"]["protocol_required"] is False
    assert report["split_reports"]["test"]["protocol_role"] == "excluded"
    assert report["split_reports"]["test"]["protocol_ready"] is False
    assert report["official_test_policy"] == "excluded"
    assert len(report["content_sha256"]) == 64


def test_spd_release_audit_does_not_promote_absent_test_to_a_required_cohort(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    archive = tmp_path / "spd.zip"
    _write_split(split)
    _write_archive(archive, include_test=False)
    report = audit_spd_release(archive, split)
    test = report["split_reports"]["test"]
    assert test["protocol_ready"] is False
    assert test["protocol_required"] is False
    assert test["protocol_role"] == "excluded"
    assert test["missing_metadata_count"] == 1
    assert test["missing_label_count"] == 1


def test_spd_release_audit_rejects_train_val_sequence_overlap(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    archive = tmp_path / "spd.zip"
    _write_split(split)
    _write_archive(archive, include_test=True)
    with zipfile.ZipFile(archive, "a") as archive_file:
        rows = [
            {
                "vehicle_frame": "000001",
                "infrastructure_frame": "100001",
                "vehicle_sequence": "shared-vehicle",
                "infrastructure_sequence": "shared-infrastructure",
            },
            {
                "vehicle_frame": "000002",
                "infrastructure_frame": "100002",
                "vehicle_sequence": "shared-vehicle",
                "infrastructure_sequence": "shared-infrastructure",
            },
            {
                "vehicle_frame": "000003",
                "infrastructure_frame": "100003",
                "vehicle_sequence": "excluded-test",
                "infrastructure_sequence": "excluded-test",
            },
        ]
        archive_file.writestr(
            "V2X-Seq-SPD/cooperative/data_info-overlap.json", json.dumps(rows)
        )
    report = audit_spd_release(
        archive,
        split,
        metadata_member="V2X-Seq-SPD/cooperative/data_info-overlap.json",
    )
    assert report["protocol_partition_disjoint"] is False
    assert report["split_reports"]["train"]["protocol_ready"] is False
    assert report["split_reports"]["val"]["protocol_ready"] is False


def test_spd_release_audit_rejects_duplicate_frames(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    _write_split(split)
    document = json.loads(split.read_text())
    document["cooperative_split"]["train"] = ["000001", "000001"]
    split.write_text(json.dumps(document))
    archive = tmp_path / "spd.zip"
    _write_archive(archive, include_test=True)
    with pytest.raises(SPDReleaseAuditError, match="duplicate"):
        audit_spd_release(archive, split)
