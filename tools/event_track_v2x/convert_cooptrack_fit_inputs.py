#!/usr/bin/env python3
"""Run pinned upstream CoopTrack conversion without deleting/interpolating GT.

Run inside the isolated Python 3.8 runtime. Inputs are the fit-only view produced
by prepare_cooptrack_fold_inputs.py, never the complete SPD archive/dataset.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


UPSTREAM_COMMIT = "29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def preserve_original_annotations(source_name, samples, sequence_frames,
                                  instance_tokens, annotations, transforms):
    """Do not interpolate occluded boxes into the official native cohort."""
    return annotations


def configure_converter(converter):
    converter.to_remove_list_veh = []
    converter.to_remove_list_coop = []
    converter._generate_unvisible_annotations = preserve_original_annotations


def write_location_tables(samples, output, version="v1.0-trainval"):
    """Preserve unreported locations instead of inventing a named junction.

    Upstream templates enumerate only six named junctions; native SPD also has
    blank intersection_loc metadata. Maps are not used by this image detector.
    """
    groups = {}
    for sample in samples.values():
        groups.setdefault(sample["scene_token"], []).append(sample)
    scene_locations = {scene: tuple(sorted({frame["location"] for frame in frames}))
                       for scene, frames in groups.items()}
    locations = sorted(set(scene_locations.values()))
    tokens = {value: hashlib.sha256(b"spd-log:" + canonical(value)).hexdigest()
              for value in locations}
    logs = [{"token": tokens[value], "logfile": "", "vehicle": "",
             "date_captured": "", "location": value[0] if len(value) == 1 else "",
             "source_locations": list(value)} for value in locations]
    maps = [{"token": hashlib.sha256(b"spd-map:" + canonical(value)).hexdigest(),
             "category": "semantic_prior", "filename": "", "log_tokens": [tokens[value]]}
            for value in locations]
    scenes = []
    for scene, frames in groups.items():
        scenes.append({"token": scene, "name": scene, "description": "",
                       "log_token": tokens[scene_locations[scene]],
                       "nbr_samples": len(frames), "first_sample_token": frames[0]["token"],
                       "last_sample_token": frames[-1]["token"]})
    for name, records in (("log", logs), ("map", maps), ("scene", scenes)):
        with (output / version / (name + ".json")).open("xb") as stream:
            stream.write(canonical(records))
    return sum(sample["location"] == "" for sample in samples.values())


def convert(upstream_root, input_root, output):
    if output.exists():
        raise FileExistsError("converted output is create-once")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise ValueError("unexpected CoopTrack source commit")
    if subprocess.check_output(
            ["git", "-C", str(upstream_root), "status", "--porcelain"], text=True):
        raise ValueError("upstream checkout must be clean")
    manifest_raw = (input_root / "input-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    expected_hash = manifest.pop("content_sha256")
    if hashlib.sha256(canonical(manifest)).hexdigest() != expected_hash:
        raise ValueError("fit input manifest hash mismatch")
    if (manifest["kind"] != "cooptrack_fit_input_preparation_v1"
            or manifest["raw_labels_modified"]
            or manifest["official_val_read"] or manifest["test_or_test_A_read"]):
        raise ValueError("input preparation policy mismatch")
    split = json.loads((input_root / "fold-split.json").read_bytes())["batch_split"]
    if (split["train"] != manifest["fit_sequence_ids"]
            or any(split[name] for name in ("val", "test", "test_A"))):
        raise ValueError("fit split contains an excluded cohort")
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_root / "tools/spd_data_converter"))
    converter = importlib.import_module("spd_to_uniad")
    configure_converter(converter)
    nusc = importlib.import_module("spd_to_nuscenes")
    output.mkdir(parents=True, exist_ok=False)
    source_inventory = {item["path"]: item for item in manifest["payload_inventory"]}
    converted_counts = {}
    for side in ("vehicle-side", "infrastructure-side"):
        (output / side).mkdir()
        rows = json.loads((input_root / side / "data_info.json").read_bytes())
        if {row["sequence_id"] for row in rows} != set(split["train"]):
            raise ValueError("non-fit sequence present")
        expected_annotations = {}
        for row in rows:
            relative = side + "/" + row["label_lidar_std_path"]
            raw = (input_root / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source_inventory[relative]["sha256"]:
                raise ValueError("raw training label changed")
            labels = json.loads(raw)
            tokens = {label["token"] for label in labels}
            if len(tokens) != len(labels):
                raise ValueError("duplicate annotation token")
            expected_annotations[row["frame_id"]] = tokens
        annotations, samples, infos = converter.create_spd_infos(
            str(input_root), str(output), side, str(input_root / "fold-split.json"),
            "", "spd", forecasting=False)
        if (len(infos) != len(rows)
                or {info["token"] for info in infos} != set(expected_annotations)):
            raise ValueError("converter removed or added frames")
        for frame, labels in annotations.items():
            if set(labels) != expected_annotations[frame]:
                raise ValueError("converter removed or interpolated labels")
        kwargs = dict(version="v1.0-trainval",
                      local_root=str(upstream_root / "tools/spd_data_converter/nuscenes_jsons"))
        target = str(output / side)
        for name in ("category", "attribute", "visibility", "sensor"):
            getattr(nusc, "generate_" + name + "_json")(target, **kwargs)
        nusc.generate_instance_json(annotations, samples, target, **kwargs)
        nusc.generate_calibrated_sensor_json(infos, target, **kwargs)
        nusc.generate_ego_pose_json(infos, target, **kwargs)
        unreported_locations = write_location_tables(samples, output / side)
        nusc.generate_sample_json(samples, target, **kwargs)
        nusc.generate_sample_data_json(infos, target, **kwargs)
        # The pinned upstream helper reads these two module globals.
        nusc.sample_info_mappings = samples
        nusc.spd_infos = infos
        nusc.generate_sample_annotation_json(annotations, target, **kwargs)
        converted_counts[side] = {"frames": len(infos),
                                 "annotations": sum(map(len, annotations.values())),
                                 "frames_with_unreported_location": unreported_locations}
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            inventory.append({"path": path.relative_to(output).as_posix(),
                              "size_bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
    report = {
        "kind": "cooptrack_fit_conversion_v1", "upstream_commit": actual_commit,
        "input_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "fold_id": manifest["fold_id"], "fit_sequence_ids": split["train"],
        "cohort": manifest.get("cohort", "fit-fold"),
        "counts": converted_counts, "removed_frames": 0,
        "interpolated_annotations": 0, "raw_labels_modified": False,
        "forecasting_targets_generated": False, "inventory": inventory,
        "location_metadata_policy": "preserve-frame-values-scene-mixed-locations-unreported",
    }
    report["content_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    with (output / "conversion-manifest.json").open("xb") as stream:
        stream.write(canonical(report))
    print(json.dumps({key: report[key] for key in ("kind", "counts", "content_sha256")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.upstream_root, args.input_root, args.output)
