from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from transvision.models.event_track_v2x.v2v4real_inputs import (
    V2V4RealInputError, load_prepared_frames, load_raw_yaml, pose_projection,
    pose_to_world, read_annotations, source_to_target,
)
from tools.event_track_v2x.prepare_v2v4real_inputs import (
    inventory_native_root, main, prepare_native_inputs,
)
from tools.event_track_v2x import prepare_v2v4real_inputs as preparer


def _metadata(raw_class="Car"):
    return {"lidar_pose": [1., 2., 3., 10., 20., 30.], "vehicles": {
        17: {"location": [10., 20., 0.], "center": [.5, .25, 1.],
             "angle": [0., 90., 0.], "extent": [2., 1., .75],
             "ass_id": "native-id-9", "obj_type": raw_class}},
        "speed": 99., "arbitrary_gt_only_field": "do-not-export"}


def _fixture(tmp_path):
    source = tmp_path / "validation_dir_actually_test"
    sequence = "testoutput_CAV_data_session_0"
    for cav in ("0", "1"):
        folder = source / sequence / cav
        folder.mkdir(parents=True)
        for key in ("00000", "00001"):
            (folder / (key + ".yaml")).write_text(yaml.safe_dump(_metadata()))
            (folder / (key + ".pcd")).write_text(
                "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                "WIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n1 2 3\n")
    evidence = tmp_path / "source-provenance.txt"
    evidence.write_text("Synthetic fixture; not an official source or split certificate.\n")
    return source, {sequence: "1"}, evidence


def _prepare(tmp_path):
    source, ego, evidence = _fixture(tmp_path)
    output = tmp_path / "projection"
    receipt = prepare_native_inputs(source, output, dataset_split="test", ego_agents=ego, source_evidence=evidence)
    return source, output, receipt


def _mutate_record(output, mutation):
    path = output / "inputs" / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(records)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path = output / "inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_native_geometry_matches_fixed_official_equations():
    # Independent closed-form oracle from the pinned native convention.
    rng = np.random.default_rng(1337)
    for _ in range(128):
        pose = rng.uniform(-170, 170, 6)
        r, y, p = map(math.radians, pose[3:])
        cr, sr, cy, sy, cp, sp = math.cos(r), math.sin(r), math.cos(y), math.sin(y), math.cos(p), math.sin(p)
        expected = np.array([
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ])
        actual = pose_to_world(pose)
        np.testing.assert_allclose(actual[:3, :3], expected, atol=1e-14)
        np.testing.assert_allclose(actual[:3, 3], pose[:3])
        np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-14)
        assert np.linalg.det(actual[:3, :3]) == pytest.approx(1.)


@pytest.mark.parametrize("angle_index,expected", [(3, [0, 0, -1]), (4, [-1, 0, 0]), (5, [0, 1, 0])])
def test_angles_are_degrees_in_roll_yaw_pitch_order(angle_index, expected):
    pose = [0.] * 6
    pose[angle_index] = 90.
    np.testing.assert_allclose(pose_to_world(pose)[:3, :3] @ [0, 1, 0], expected, atol=1e-15)


def test_coordinate_round_trip_and_translation():
    a, b = [3, 4, 1, 15, 76, -9], [-3, 1, 9, 6, -41, 18]
    np.testing.assert_allclose(source_to_target(a, b) @ source_to_target(b, a), np.eye(4), atol=1e-14)
    np.testing.assert_allclose(source_to_target(a, b), np.linalg.inv(pose_to_world(b)) @ pose_to_world(a), atol=1e-14)
    np.testing.assert_allclose(source_to_target([1, 2, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0])[:3, 3], [1, 2, 3])


@pytest.mark.parametrize("pose", [[0]*5, [0]*7, [True, 0, 0, 0, 0, 0], [float("nan")]*6, [float("inf")]*6, ["1"]*6, None, np.array(0), np.zeros((6, 1))])
def test_invalid_pose_rejected(pose):
    with pytest.raises(V2V4RealInputError):
        pose_to_world(pose)


def test_safe_yaml_and_gt_free_projection():
    metadata = load_raw_yaml(yaml.safe_dump(_metadata()).encode())
    projection = pose_projection(metadata)
    assert set(projection) == {"lidar_pose", "source_to_world"}
    altered = copy.deepcopy(metadata)
    altered["vehicles"][17]["location"] = [-100, -100, 900]
    altered["vehicles"][17]["obj_type"] = "Bus"
    assert projection == pose_projection(altered)
    assert "native-id-9" not in json.dumps(projection)
    assert "timestamp" not in json.dumps(projection)


def test_scientific_yaml_does_not_change_global_safe_loader():
    before = yaml.safe_load("x: 1e-3")
    assert load_raw_yaml(b"lidar_pose: [1e-3, 2E3, -1e+2, 0, 0, 0]")["lidar_pose"][:3] == [.001, 2000., -100.]
    assert yaml.safe_load("x: 1e-3") == before


@pytest.mark.parametrize("raw", [
    b"!!python/object/apply:builtins.str [unsafe]", b"yaml_parser: arbitrary_code",
    b"x: 1\nx: 2", b"x: {y: 1, y: 2}", b"x: &a [1]\ny: *a",
    b"x: {<<: {a: 1}}", b"x: .nan", b"x: .inf", b"true: 1", b"x: !!set {a: null}",
    b"x: 2026-09-13", b"---\nx: 1\n---\ny: 2", b"[not, a, mapping]",
    b"x: " + b"[" * 34 + b"0" + b"]" * 34,
])
def test_unsafe_ambiguous_yaml_is_rejected(raw):
    with pytest.raises(V2V4RealInputError):
        load_raw_yaml(raw)


@pytest.mark.parametrize("raw_class", ["Car", "Van", "Pickup Truck", "Semi-truck", "Bus", "Pedestrian", "Unmapped Label"])
def test_raw_category_and_native_ids_are_preserved_without_mapping(raw_class):
    annotation, = read_annotations(_metadata(raw_class))
    assert annotation.raw_class == raw_class
    assert annotation.object_id == "17"
    assert annotation.associated_id == "native-id-9"
    assert annotation.center_world == (10.5, 20.25, 1.)
    corners = annotation.corners_in([0]*6)
    np.testing.assert_allclose(corners.mean(axis=0), annotation.center_world)
    np.testing.assert_allclose(corners.max(axis=0)-corners.min(axis=0), [2., 4., 1.5])


@pytest.mark.parametrize("field,value", [("obj_type", None), ("obj_type", ""), ("extent", [0, 1, 1]), ("extent", [-1, 1, 1]), ("angle", [0, 0]), ("ass_id", True)])
def test_annotation_contract_does_not_guess(field, value):
    metadata = _metadata()
    metadata["vehicles"][17][field] = value
    with pytest.raises(V2V4RealInputError):
        read_annotations(metadata)


def test_missing_class_and_normalized_identity_collision_rejected():
    metadata = _metadata()
    del metadata["vehicles"][17]["obj_type"]
    with pytest.raises(V2V4RealInputError, match="obj_type"):
        read_annotations(metadata)
    metadata = _metadata()
    metadata["vehicles"]["17"] = metadata["vehicles"][17]
    with pytest.raises(V2V4RealInputError, match="collide"):
        read_annotations(metadata)


def test_inventory_reads_no_payloads(tmp_path, monkeypatch):
    source, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("payload read in inventory"))
    inventory = inventory_native_root(source)
    assert inventory["paired_frame_count"] == 2
    assert inventory["source_frame_count"] == 4


@pytest.mark.parametrize("change", ["missing_yaml", "missing_pcd", "unaligned", "extra_cav", "symlink", "mixed_width"])
def test_no_silent_frame_intersection_or_unsafe_entries(tmp_path, change):
    source, ego, evidence = _fixture(tmp_path)
    folder = source / next(iter(ego)) / "0"
    if change == "missing_yaml":
        (folder / "00001.yaml").unlink()
    elif change == "missing_pcd":
        (folder / "00001.pcd").unlink()
    elif change == "unaligned":
        for suffix in ("yaml", "pcd"):
            (folder / f"00001.{suffix}").rename(folder / f"00002.{suffix}")
    elif change == "extra_cav":
        (folder.parent / "2").mkdir()
    elif change == "symlink":
        (folder / "private.txt").symlink_to(evidence)
    else:
        for suffix in ("yaml", "pcd"):
            (folder / f"00001.{suffix}").rename(folder / f"1.{suffix}")
    with pytest.raises(V2V4RealInputError):
        inventory_native_root(source)


@pytest.mark.parametrize("split", ["train", "test"])
def test_full_projection_round_trip_and_explicit_split(tmp_path, split):
    source, ego, evidence = _fixture(tmp_path)
    output = tmp_path / "projection"
    receipt = prepare_native_inputs(source, output, dataset_split=split, ego_agents=ego, source_evidence=evidence)
    manifest, records = load_prepared_frames(output / "inputs")
    assert manifest["dataset_split"] == split
    assert manifest["source_frame_count"] == len(records) == 4
    assert manifest["paired_frame_count"] == 2
    assert manifest["time_basis"] == "ordinal-only-no-clock"
    assert manifest["official_split_membership_verified"] is False
    assert receipt["paper_eligible"] is False
    assert receipt["raw_yaml_parsed_by_preparer"] is True
    assert all(r["is_ego"] == (r["cav_id"] == "1") for r in records)
    assert not any(p.suffix == ".yaml" for p in (output / "inputs").rglob("*"))
    assert not any(term in (output / "inputs" / "frames.jsonl").read_text() for term in ("vehicles", "obj_type", "ass_id", "native-id-9"))
    original = next(source.rglob("*.pcd"))
    original.write_text("changed original")
    load_prepared_frames(output / "inputs")  # Copies, not hard links/references.


@pytest.mark.parametrize("split", ["val", "validate", "test_A", "", None])
def test_split_aliases_cannot_bypass_dataset_scope(tmp_path, split):
    source, ego, evidence = _fixture(tmp_path)
    with pytest.raises(V2V4RealInputError, match="split"):
        prepare_native_inputs(source, tmp_path / "out", dataset_split=split, ego_agents=ego, source_evidence=evidence)


@pytest.mark.parametrize("ego", [{}, {"unknown": "0"}, {"testoutput_CAV_data_session_0": 1}, {"testoutput_CAV_data_session_0": "7"}])
def test_ego_cannot_be_guessed_from_sort_order(tmp_path, ego):
    source, _, evidence = _fixture(tmp_path)
    with pytest.raises(V2V4RealInputError, match="ego"):
        prepare_native_inputs(source, tmp_path / "out", dataset_split="test", ego_agents=ego, source_evidence=evidence)


def test_no_overwrite_or_output_inside_source(tmp_path):
    source, ego, evidence = _fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user-owned")
    for path in (output, source / "projection", tmp_path):
        with pytest.raises(V2V4RealInputError):
            prepare_native_inputs(source, path, dataset_split="test", ego_agents=ego, source_evidence=evidence)
    assert sentinel.read_text() == "user-owned"


def test_bad_yaml_fails_without_publishing_or_leaving_staging(tmp_path):
    source, ego, evidence = _fixture(tmp_path)
    next(source.rglob("*.yaml")).write_text("lidar_pose: [0, 0, 0]")
    with pytest.raises(V2V4RealInputError):
        prepare_native_inputs(source, tmp_path / "out", dataset_split="test", ego_agents=ego, source_evidence=evidence)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.preparing-*"))


@pytest.mark.parametrize("change", ["pcd", "extra_gt", "index", "symlink"])
def test_projection_tampering_is_detected(tmp_path, change):
    _, output, _ = _prepare(tmp_path)
    inputs = output / "inputs"
    if change == "pcd":
        path = next(inputs.rglob("*.pcd"))
        path.write_bytes(path.read_bytes().replace(b"1 2 3", b"9 9 9"))
    elif change == "extra_gt":
        (inputs / "labels.json").write_text("[]")
    elif change == "index":
        (inputs / "frames.jsonl").write_text("{}\n")
    else:
        (inputs / "leak").symlink_to(output / "audit.json")
    with pytest.raises(V2V4RealInputError):
        load_prepared_frames(inputs)


def test_external_digest_rejects_split_relabeling(tmp_path):
    _, output, receipt = _prepare(tmp_path)
    expected = receipt["input_manifest_sha256"]
    load_prepared_frames(output / "inputs", expected_manifest_sha256=expected)
    path = output / "inputs" / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["dataset_split"] = "train"
    path.write_text(json.dumps(manifest))
    with pytest.raises(V2V4RealInputError, match="pinned"):
        load_prepared_frames(output / "inputs", expected_manifest_sha256=expected)


def test_mid_preparation_source_change_does_not_publish(tmp_path, monkeypatch):
    source, ego, evidence = _fixture(tmp_path)
    original_copy = preparer._copy_pcd

    def change_source_after_read(inp, out):
        result = original_copy(inp, out)
        inp.with_suffix(".yaml").write_text("lidar_pose: [0, 0, 0, 0, 0, 0]")
        return result

    monkeypatch.setattr(preparer, "_copy_pcd", change_source_after_read)
    with pytest.raises(V2V4RealInputError, match="changed"):
        prepare_native_inputs(source, tmp_path / "out", dataset_split="test", ego_agents=ego, source_evidence=evidence)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.preparing-*"))


@pytest.mark.parametrize("mutation", [
    lambda rs: rs[0].update(obj_type="Car"),
    lambda rs: rs[0].update(source_to_world=np.eye(4).tolist()),
    lambda rs: rs[0].update(pcd_path="../../secret"),
    lambda rs: rs[0].update(is_ego=True),
    lambda rs: rs[0].update(frame_ordinal=True),
    lambda rs: rs[0].update(sequence_id=".."),
    lambda rs: rs.pop(),
    lambda rs: rs.append(rs[0]),
    lambda rs: rs.reverse(),
])
def test_rehashed_but_invalid_frame_records_rejected(tmp_path, mutation):
    _, output, _ = _prepare(tmp_path)
    _mutate_record(output, mutation)
    with pytest.raises(V2V4RealInputError):
        load_prepared_frames(output / "inputs")


def test_cli_inventory_and_preparation(tmp_path, capsys):
    source, ego, evidence = _fixture(tmp_path)
    assert main(["--source-root", str(source), "--inventory-only"]) == 0
    assert json.loads(capsys.readouterr().out)["source_frame_count"] == 4
    ego_path = tmp_path / "ego.json"
    ego_path.write_text(json.dumps(ego))
    assert main(["--source-root", str(source), "--split", "test", "--ego-agents", str(ego_path),
                 "--source-evidence", str(evidence), "--output", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["paper_eligible"] is False
    load_prepared_frames(tmp_path / "out" / "inputs")
