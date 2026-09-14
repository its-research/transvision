from __future__ import annotations

import hashlib
import json

import pytest

from tools.event_track_v2x.prepare_cooptrack_fold_inputs import prepare
from transvision.models.event_track_v2x.development_split import (
    build_development_split_manifest_v1,
)


@pytest.fixture
def inputs(tmp_path):
    sequences = [f"{i:04d}" for i in range(46)]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"batch_split": {"train": sequences}}))
    manifest = build_development_split_manifest_v1(
        sequences, split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    manifest_path = tmp_path / "development.json"
    manifest_path.write_bytes(manifest.canonical_bytes)
    root = tmp_path / "source"
    for side in ("vehicle-side", "infrastructure-side"):
        side_root = root / side
        side_root.mkdir(parents=True)
        rows = []
        for sequence in sequences:
            frame = f"{int(sequence):06d}"
            rows.append({"sequence_id": sequence, "frame_id": frame,
                         "image_path": f"image/{frame}.jpg",
                         "label_lidar_std_path": f"label/{frame}.json"})
            # Held-out payloads deliberately do not exist: they must not be read.
            if sequence in manifest.folds[0].fit_sequence_ids:
                for relative, raw in ((f"image/{frame}.jpg", b"image"),
                                      (f"label/{frame}.json", b"[{\"raw\":true}]")):
                    target = side_root / relative
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
        (side_root / "data_info.json").write_text(json.dumps(rows))
    return root, split, manifest_path, tmp_path / "output", manifest


def test_fit_only_copy_preserves_labels_and_excludes_held_out(inputs):
    root, split, manifest_path, output, manifest = inputs
    report = prepare(root, split, manifest_path, 0, output)
    assert report["frame_counts"] == {"vehicle-side": 36, "infrastructure-side": 36}
    assert report["formal_training_authorized_by_this_artifact"] is False
    assert not report["raw_labels_modified"]
    for side in ("vehicle-side", "infrastructure-side"):
        rows = json.loads((output / side / "data_info.json").read_bytes())
        assert {row["sequence_id"] for row in rows} == set(manifest.folds[0].fit_sequence_ids)
        for row in rows:
            rel = f"{side}/{row['label_lidar_std_path']}"
            assert (output / rel).read_bytes() == (root / rel).read_bytes()
    assert json.loads((output / "fold-split.json").read_bytes())["batch_split"]["val"] == []
    with pytest.raises(FileExistsError, match="create-once"):
        prepare(root, split, manifest_path, 0, output)


def test_rejects_non_train_metadata_before_writing(inputs):
    root, split, manifest_path, output, _ = inputs
    path = root / "vehicle-side/data_info.json"
    rows = json.loads(path.read_bytes())
    rows.append({"sequence_id": "9999", "frame_id": "999999"})
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="complete official train"):
        prepare(root, split, manifest_path, 0, output)
    assert not output.exists()


def test_rejects_wrong_official_split(inputs):
    root, split, manifest_path, output, _ = inputs
    split.write_bytes(split.read_bytes() + b" ")
    with pytest.raises(ValueError, match="split hash"):
        prepare(root, split, manifest_path, 0, output)


def test_rejects_traversal_and_symlink(inputs):
    root, split, manifest_path, output, manifest = inputs
    path = root / "vehicle-side/data_info.json"
    rows = json.loads(path.read_bytes())
    row = next(row for row in rows if row["sequence_id"] in manifest.folds[0].fit_sequence_ids)
    row["image_path"] = "../escape.jpg"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="relative payload"):
        prepare(root, split, manifest_path, 0, output)


def test_rejects_output_inside_source(inputs):
    root, split, manifest_path, _, _ = inputs
    with pytest.raises(ValueError, match="outside the source"):
        prepare(root, split, manifest_path, 0, root / "output")


def test_full_train_copies_all_46_sequences_but_no_val_or_test(inputs):
    root, split, manifest_path, output, manifest = inputs
    for side in ("vehicle-side", "infrastructure-side"):
        for sequence in manifest.folds[0].held_out_sequence_ids:
            frame = f"{int(sequence):06d}"
            (root / side / "image" / (frame + ".jpg")).write_bytes(b"image")
            (root / side / "label" / (frame + ".json")).write_bytes(b"[]")
    report = prepare(root, split, manifest_path, None, output, cohort="full-train")
    assert report["cohort"] == "full-train" and report["fold_id"] is None
    assert report["frame_counts"] == {"vehicle-side": 46, "infrastructure-side": 46}
    assert report["excluded_held_out_sequence_ids"] == []
    assert set(report["fit_sequence_ids"]) == set(manifest.sequence_ids)
    selected = json.loads((output / "fold-split.json").read_bytes())["batch_split"]
    assert len(selected["train"]) == 46
    assert selected["val"] == selected["test"] == selected["test_A"] == []
