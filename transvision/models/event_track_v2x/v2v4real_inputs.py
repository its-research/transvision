"""Native V2V4Real geometry and preparation-only YAML readers.

This is not DetectionCacheV2 and does not relax its SPD split/feature contract.
Native filename stems are frame keys, NOT source-clock timestamps. Annotation
objects never enter the pose projection. Raw categories are preserved verbatim;
missing ``obj_type`` is an error rather than an implicit Car label.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np


OFFICIAL_SOURCE_COMMIT = "5a821e13753bafc611f95c47bc1a306acdcb0f7c"
POSE_CONVENTION = "native-source-to-world-matrix-or-xyz-roll-yaw-pitch-degrees-v2"
MAX_YAML_BYTES = 16 * 1024 * 1024
MATRIX_RIGID_ATOL = 1e-5  # Native calibration serialization precision; never orthogonalize input.
PROJECTION_KIND = "v2v4real_pose_lidar_projection_v2"


class V2V4RealInputError(ValueError):
    """An input cannot be interpreted without an unsupported assumption."""


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1) or len(value) != size):
        raise V2V4RealInputError(f"{name} must contain {size} finite numbers")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in value):
        raise V2V4RealInputError(f"{name} must contain numbers, not strings or bools")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise V2V4RealInputError(f"{name} must contain {size} finite numbers")
    return result


def pose_to_world(pose: Any) -> np.ndarray:
    """Column-vector source-to-world transform for [x,y,z,roll,yaw,pitch].

    The independently factored rotation reproduces the native OpenCOOD
    convention: Rz(yaw) Ry(-pitch) Rx(-roll), with angles in degrees.
    """
    # Real release frames carry T_source_to_world directly as a 4x4 ndarray.
    # Do not squeeze it into an assumed Euler pose or discard roll/pitch.
    if (isinstance(pose, (list, tuple, np.ndarray))
            and not (isinstance(pose, np.ndarray) and pose.ndim == 0) and len(pose) == 4):
        rows = [_vector(row, 4, "lidar_pose matrix row") for row in pose]
        matrix = np.asarray(rows)
        if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=MATRIX_RIGID_ATOL, rtol=0.)
                or abs(np.linalg.det(matrix[:3, :3]) - 1.) > MATRIX_RIGID_ATOL):
            raise V2V4RealInputError("lidar_pose must be a rigid source-to-world transform")
        return matrix.copy()
    x, y, z, roll, yaw, pitch = _vector(pose, 6, "lidar_pose")
    r, y_angle, p = map(math.radians, (-roll, yaw, -pitch))
    cr, sr, cy, sy, cp, sp = (
        math.cos(r), math.sin(r), math.cos(y_angle), math.sin(y_angle),
        math.cos(p), math.sin(p),
    )
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = (x, y, z)
    return result


def source_to_target(source_pose: Any, target_pose: Any) -> np.ndarray:
    """Transform source LiDAR column vectors into target LiDAR coordinates."""
    return np.linalg.solve(pose_to_world(target_pose), pose_to_world(source_pose))


def load_raw_yaml(raw: bytes) -> dict[Any, Any]:
    """Read preparation metadata without arbitrary YAML object construction.

    Decode a closed native numerical-tag grammar without executing Python
    constructors. Only dtype/ndarray aliases are allowed. Reject merge keys,
    duplicate keys, executable parser directives, excessive structure and NaNs.
    """
    import yaml  # Optional preparation dependency, not a package import dependency.
    from .v2v4real_numpy_yaml import ALIAS_TAGS, install_numeric_constructors

    if not isinstance(raw, bytes) or len(raw) > MAX_YAML_BYTES:
        raise V2V4RealInputError("raw YAML exceeds the bounded metadata input size")

    class NativeSafeLoader(yaml.SafeLoader):
        node_count = 0
        depth = 0

        def compose_node(self, parent, index):
            if self.check_event(yaml.AliasEvent):
                anchor = self.peek_event().anchor
                target = self.anchors.get(anchor)
                if target is None or target.tag not in ALIAS_TAGS:
                    raise V2V4RealInputError("only native numerical YAML aliases are accepted")
            self.node_count += 1
            self.depth += 1
            if self.node_count > 200_000 or self.depth > 32:
                raise V2V4RealInputError("YAML structure exceeds preparation limits")
            try:
                return super().compose_node(parent, index)
            finally:
                self.depth -= 1

        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    raise V2V4RealInputError("YAML merge keys are not accepted")
                key = self.construct_object(key_node, deep=deep)
                if type(key) not in (str, int):
                    raise V2V4RealInputError("YAML mapping keys must be strings or integers")
                if key in result:
                    raise V2V4RealInputError(f"duplicate YAML key: {key!r}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    # Native YAML uses scientific notation without a decimal point (e.g. 1e-3).
    # This resolver is local to the safe subclass and does not mutate SafeLoader.
    NativeSafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
        list("-+0123456789."),
    )
    install_numeric_constructors(NativeSafeLoader, V2V4RealInputError)
    try:
        result = yaml.load(raw, Loader=NativeSafeLoader)
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise V2V4RealInputError("invalid or unsupported raw YAML") from exc
    if not isinstance(result, dict) or "yaml_parser" in result:
        raise V2V4RealInputError("expected raw frame metadata, not a parser configuration")

    def validate(value):
        if isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate(item)
        elif isinstance(value, np.ndarray):
            if value.dtype != np.float64 or value.shape not in ((4, 4), (6,)) or not np.isfinite(value).all():
                raise V2V4RealInputError("unsupported or non-finite native numerical array")
        elif value is not None and type(value) not in (str, int, float, bool):
            raise V2V4RealInputError("unsupported YAML scalar type")
        elif isinstance(value, float) and not math.isfinite(value):
            raise V2V4RealInputError("non-finite YAML scalar")

    validate(result)
    return result


def pose_projection(metadata: dict[Any, Any]) -> dict[str, Any]:
    """Whitelist ONLY source pose, without labels, object IDs or clock guesses."""
    if "lidar_pose" not in metadata:
        raise V2V4RealInputError("raw frame is missing lidar_pose")
    matrix = pose_to_world(metadata["lidar_pose"])
    return {"lidar_pose": np.asarray(metadata["lidar_pose"]).tolist(), "source_to_world": matrix.tolist()}


@dataclass(frozen=True)
class NativeAnnotation:
    object_id: str
    associated_id: str
    raw_class: str
    center_world: tuple[float, float, float]
    angle_degrees: tuple[float, float, float]
    half_extents: tuple[float, float, float]

    def corners_in(self, lidar_pose: Any) -> np.ndarray:
        object_pose = (*self.center_world, *self.angle_degrees)
        transform = source_to_target(object_pose, lidar_pose)
        corners = np.asarray(list(product((-1., 1.), repeat=3))) * self.half_extents
        return corners @ transform[:3, :3].T + transform[:3, 3]


def _identity(value: Any, name: str) -> str:
    if type(value) not in (int, str) or str(value) == "":
        raise V2V4RealInputError(f"{name} must be a nonempty string or integer")
    return str(value)


def read_annotations(metadata: dict[Any, Any]) -> tuple[NativeAnnotation, ...]:
    """Evaluator/preparation API only; preserve all raw labels without mapping.

    No ROI, class merge, deduplication, or default category is applied. Such
    choices belong to the separately frozen native evaluation protocol.
    """
    objects = metadata.get("vehicles")
    if not isinstance(objects, dict):
        raise V2V4RealInputError("raw frame is missing the vehicles annotation mapping")
    result = []
    seen = set()
    for object_id, item in objects.items():
        identity = _identity(object_id, "object_id")
        if identity in seen:
            raise V2V4RealInputError("object IDs collide after string normalization")
        seen.add(identity)
        if not isinstance(item, dict) or not isinstance(item.get("obj_type"), str) or not item["obj_type"]:
            raise V2V4RealInputError(f"object {identity} has no explicit obj_type")
        location = _vector(item.get("location"), 3, "location")
        center = _vector(item.get("center"), 3, "center")
        angle = _vector(item.get("angle"), 3, "angle")
        extent = _vector(item.get("extent"), 3, "extent")
        if (extent <= 0).any():
            raise V2V4RealInputError("annotation half extents must be positive")
        with np.errstate(over="ignore", invalid="ignore"):
            world_center = location + center
        if not np.isfinite(world_center).all():
            raise V2V4RealInputError("annotation world center is non-finite")
        result.append(NativeAnnotation(
            identity, _identity(item.get("ass_id", object_id), "ass_id"),
            item["obj_type"], tuple(world_center), tuple(angle), tuple(extent),
        ))
    return tuple(result)


def load_prepared_frames(
    inputs_root: Path, *, expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Verify pose/PCD projection integrity before consumption, including all PCDs.

    No access to the parent audit, provenance document or raw YAML is required.
    This is an integrity check, not proof of official provenance or PCD validity.
    Returned ordinals are deliberately NOT mapped to event/arrival times.
    """
    root = Path(inputs_root)
    if root.is_symlink() or not root.is_dir():
        raise V2V4RealInputError("input projection must be a non-symlink directory")
    actual_files = set()
    # rglob does not follow directory symlinks; reject them explicitly as well.
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise V2V4RealInputError("projection contains an unsupported filesystem entry")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())

    def load_json(raw):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise V2V4RealInputError("duplicate projection JSON key")
                result[key] = value
            return result
        try:
            return json.loads(raw, object_pairs_hook=pairs,
                              parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        except (ValueError, UnicodeError) as exc:
            raise V2V4RealInputError("invalid projection JSON") from exc

    manifest_raw = (root / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None:
        if (not isinstance(expected_manifest_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
                or hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256):
            raise V2V4RealInputError("projection differs from the pinned manifest digest")
    manifest = load_json(manifest_raw)
    fields = {
        "kind", "dataset", "dataset_split", "time_basis", "pose_convention",
        "source_frame_count", "paired_frame_count", "sequence_count", "ego_agents",
        "frames_path", "frames_sha256", "gt_in_projection", "detection_cache_created",
        "official_split_membership_verified", "paper_eligible",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise V2V4RealInputError("unexpected projection manifest schema")
    versions = {PROJECTION_KIND: POSE_CONVENTION,
                "v2v4real_pose_lidar_projection_v1": "xyz-roll-yaw-pitch-degrees-RzRyNegRxNeg-v1"}
    if (manifest["kind"] not in versions or manifest["dataset"] != "V2V4Real"
            or manifest["dataset_split"] not in ("train", "test")
            or manifest["time_basis"] != "ordinal-only-no-clock"
            or manifest["pose_convention"] != versions[manifest["kind"]]
            or manifest["frames_path"] != "frames.jsonl"
            or any(manifest[k] is not False for k in (
                "gt_in_projection", "detection_cache_created", "official_split_membership_verified", "paper_eligible"))):
        raise V2V4RealInputError("invalid projection scope or unsupported claims")
    if not isinstance(manifest["ego_agents"], dict) or not manifest["ego_agents"]:
        raise V2V4RealInputError("projection is missing explicit ego assignments")
    for name in ("source_frame_count", "paired_frame_count", "sequence_count"):
        if type(manifest[name]) is not int or manifest[name] < 1:
            raise V2V4RealInputError("invalid projection counts")
    frame_bytes = (root / "frames.jsonl").read_bytes()
    if hashlib.sha256(frame_bytes).hexdigest() != manifest["frames_sha256"]:
        raise V2V4RealInputError("frame index digest mismatch")
    records = tuple(load_json(line) for line in frame_bytes.splitlines())
    record_fields = {"sequence_id", "frame_key", "frame_ordinal", "cav_id", "is_ego",
                     "pcd_path", "pcd_sha256", "pcd_size_bytes", "lidar_pose", "source_to_world"}
    groups = {}
    expected_files = {"manifest.json", "frames.jsonl"}
    for record in records:
        if not isinstance(record, dict) or set(record) != record_fields:
            raise V2V4RealInputError("unexpected frame record (labels/extra fields forbidden)")
        sequence, key, cav = record["sequence_id"], record["frame_key"], record["cav_id"]
        if (not isinstance(sequence, str) or sequence in ("", ".", "..") or "/" in sequence or "\\" in sequence
                or not isinstance(key, str) or not re.fullmatch(r"[0-9]+", key)
                or not isinstance(cav, str) or not re.fullmatch(r"[0-9]+", cav)):
            raise V2V4RealInputError("invalid native source identifiers")
        if sequence not in manifest["ego_agents"] or record["is_ego"] is not (cav == manifest["ego_agents"][sequence]):
            raise V2V4RealInputError("ego assignment mismatch")
        ordinal = record["frame_ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            raise V2V4RealInputError("frame ordinal must be nonnegative integer")
        sequence_frames = groups.setdefault(sequence, {})
        frame = sequence_frames.setdefault(ordinal, {"key": key, "cavs": set()})
        if frame["key"] != key or cav in frame["cavs"]:
            raise V2V4RealInputError("duplicate or mismatched CAV frame")
        frame["cavs"].add(cav)
        pcd = (Path("pcd") / sequence / cav / (key + ".pcd")).as_posix()
        if record["pcd_path"] != pcd:
            raise V2V4RealInputError("PCD path is not the canonical relative frame path")
        expected_files.add(pcd)
        if manifest['kind'] == 'v2v4real_pose_lidar_projection_v1':
            _vector(record['lidar_pose'], 6, 'legacy lidar_pose')
        expected_transform = pose_to_world(record["lidar_pose"])
        matrix = record["source_to_world"]
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise V2V4RealInputError("pose transform must contain four rows")
        actual_transform = np.asarray([_vector(row, 4, "pose transform row") for row in matrix])
        if actual_transform.shape != (4, 4) or not np.allclose(actual_transform, expected_transform, rtol=0, atol=1e-12):
            raise V2V4RealInputError("pose transform does not match native convention")
        path = root / pcd
        if type(record["pcd_size_bytes"]) is not int or record["pcd_size_bytes"] < 1 or path.stat().st_size != record["pcd_size_bytes"]:
            raise V2V4RealInputError("PCD size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != record["pcd_sha256"]:
            raise V2V4RealInputError("PCD digest mismatch")
    if set(groups) != set(manifest["ego_agents"]) or actual_files != expected_files:
        raise V2V4RealInputError("unexpected or missing input payloads/sequences")
    paired_count = 0
    for sequence, frames in groups.items():
        keys = [frames[o]["key"] for o in sorted(frames)]
        cav_sets = [frames[o]["cavs"] for o in sorted(frames)]
        if (set(frames) != set(range(len(frames))) or len(set(keys)) != len(keys)
                or keys != sorted(keys) or len({len(key) for key in keys}) != 1
                or any(len(c) != 2 or c != cav_sets[0] for c in cav_sets)
                or manifest["ego_agents"][sequence] not in cav_sets[0]):
            raise V2V4RealInputError("incomplete two-CAV frame coverage")
        paired_count += len(frames)
    order = [(r["sequence_id"], r["frame_ordinal"], r["cav_id"]) for r in records]
    if order != sorted(order):
        raise V2V4RealInputError("frame records are not in canonical sequence/ordinal/CAV order")
    if (len(records) != manifest["source_frame_count"] or len(records) != 2 * paired_count
            or paired_count != manifest["paired_frame_count"] or len(groups) != manifest["sequence_count"]):
        raise V2V4RealInputError("projection frame/sequence counts disagree")
    return manifest, records
