from __future__ import annotations

import errno
import hashlib
import io
import os
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pypcd4 import Encoding, PointCloud


__all__ = ("PreparedPointCloud", "convert_pcd_to_bin")

_FIELDS = ("x", "y", "z", "intensity")
_READ_CHUNK_SIZE = 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024


@dataclass(frozen=True)
class PreparedPointCloud:
    destination: Path
    point_count: int
    size: int
    sha256: str
    dtype: Literal["<f4"]
    fields: tuple[
        Literal["x"],
        Literal["y"],
        Literal["z"],
        Literal["intensity"],
    ]


def _read_stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(f"source must not be a symlink: {path}") from error
        raise ValueError(f"cannot open regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            raise ValueError(f"source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _header(raw: bytes) -> tuple[dict[str, tuple[str, ...]], int]:
    stream = io.BytesIO(raw)
    values: dict[str, tuple[str, ...]] = {}
    while stream.tell() <= _MAX_HEADER_BYTES:
        line_bytes = stream.readline()
        if not line_bytes:
            break
        try:
            line = line_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PCD header must be UTF-8") from error
        if not line or line.startswith("#"):
            continue
        key, separator, remainder = line.partition(" ")
        if not separator:
            raise ValueError("malformed PCD header line")
        key = key.upper()
        if key in values:
            raise ValueError(f"duplicate PCD header field: {key}")
        values[key] = tuple(remainder.split())
        if key == "DATA":
            return values, stream.tell()
    raise ValueError("PCD header is missing DATA")


def _header_integer(
    header: dict[str, tuple[str, ...]],
    key: str,
) -> int:
    value = header.get(key)
    if value is None or len(value) != 1:
        raise ValueError(f"PCD header requires one {key} value")
    try:
        parsed = int(value[0], 10)
    except ValueError as error:
        raise ValueError(f"PCD {key} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"PCD {key} must be nonnegative")
    return parsed


def _validate_header(
    raw: bytes,
) -> tuple[int, int, str]:
    header, data_offset = _header(raw)
    field_names = header.get("FIELDS")
    counts = header.get("COUNT")
    sizes = header.get("SIZE")
    types = header.get("TYPE")
    if field_names is None:
        raise ValueError("PCD header is missing required FIELDS")
    if counts is None:
        counts = tuple("1" for _ in field_names)
    if sizes is None or types is None:
        raise ValueError("PCD header is missing required SIZE or TYPE")
    if not (
        len(field_names) == len(counts) == len(sizes) == len(types)
    ):
        raise ValueError("PCD field metadata lengths are inconsistent")
    parsed_counts: list[int] = []
    for value in counts:
        try:
            parsed_counts.append(int(value, 10))
        except ValueError as error:
            raise ValueError("PCD COUNT values must be integers") from error
    for required in _FIELDS:
        matches = [
            index for index, field in enumerate(field_names) if field == required
        ]
        if not matches:
            raise ValueError(f"PCD is missing required field {required}")
        if len(matches) != 1:
            raise ValueError(
                f"PCD required field {required} must occur exactly once"
            )
        if parsed_counts[matches[0]] != 1:
            raise ValueError(
                f"PCD required field {required} must have scalar count one"
            )
    width = _header_integer(header, "WIDTH")
    height = _header_integer(header, "HEIGHT")
    points = _header_integer(header, "POINTS")
    if width * height != points:
        raise ValueError("PCD point count disagrees with WIDTH and HEIGHT")
    data_values = header.get("DATA")
    if data_values is None or len(data_values) != 1:
        raise ValueError("PCD header requires one DATA value")
    encoding = data_values[0]
    if encoding not in {
        Encoding.ASCII.value,
        Encoding.BINARY.value,
        Encoding.BINARY_COMPRESSED.value,
    }:
        raise ValueError(f"unsupported PCD encoding: {encoding}")

    try:
        item_size = sum(
            int(size, 10) * count
            for size, count in zip(sizes, parsed_counts)
        )
    except ValueError as error:
        raise ValueError("PCD SIZE values must be integers") from error
    body = raw[data_offset:]
    expected_uncompressed = points * item_size
    if encoding == Encoding.BINARY.value:
        if len(body) != expected_uncompressed:
            raise ValueError("PCD binary point count is inconsistent")
    elif encoding == Encoding.BINARY_COMPRESSED.value:
        if len(body) < 8:
            raise ValueError("PCD compressed payload is truncated")
        compressed_size, uncompressed_size = struct.unpack("<II", body[:8])
        if uncompressed_size != expected_uncompressed:
            raise ValueError("PCD compressed point count is inconsistent")
        if len(body) != 8 + compressed_size:
            raise ValueError("PCD compressed payload size is inconsistent")
    return points, data_offset, encoding


def _pcd_output(raw: bytes) -> tuple[bytes, int]:
    expected_points, _, _ = _validate_header(raw)
    try:
        cloud = PointCloud.from_fileobj(io.BytesIO(raw))
        selected = np.asarray(cloud.numpy(_FIELDS))
    except Exception as error:
        raise ValueError(f"invalid PCD payload: {error}") from error
    if selected.ndim != 2 or selected.shape != (expected_points, 4):
        raise ValueError("PCD point count is inconsistent")
    if not np.isfinite(selected).all():
        raise ValueError("PCD selected fields must be finite")
    little_endian = np.ascontiguousarray(selected, dtype=np.dtype("<f4"))
    output = little_endian.tobytes(order="C")
    if len(output) != expected_points * 16:
        raise ValueError("prepared PCD byte size is inconsistent")
    return output, expected_points


def _directory_fsync(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_bytes(path: Path) -> bytes | None:
    try:
        return _read_stable_regular_bytes(path)
    except ValueError as error:
        if not path.exists() and not path.is_symlink():
            return None
        raise ValueError(f"destination conflict: {path}") from error


def _publish_immutable_bytes(destination: Path, data: bytes) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_bytes(destination)
    if existing is not None:
        if existing == data:
            return
        raise ValueError(f"destination conflict: {destination}")

    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
        temporary_path = Path(raw_temporary_path)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            written = stream.write(data)
            if written != len(data):
                raise OSError(
                    f"short write: expected {len(data)} bytes, wrote {written}"
                )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            raced = _existing_bytes(destination)
            if raced != data:
                raise ValueError(f"destination conflict: {destination}")
        _directory_fsync(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def convert_pcd_to_bin(
    source: Path,
    destination: Path,
) -> PreparedPointCloud:
    source = Path(source)
    destination = Path(destination)
    raw = _read_stable_regular_bytes(source)
    output, point_count = _pcd_output(raw)
    _publish_immutable_bytes(destination, output)
    return PreparedPointCloud(
        destination=destination,
        point_count=point_count,
        size=len(output),
        sha256=hashlib.sha256(output).hexdigest(),
        dtype="<f4",
        fields=_FIELDS,
    )


def _convert_verified_pcd_to_bin(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> PreparedPointCloud:
    source = Path(source)
    destination = Path(destination)
    raw = _read_stable_regular_bytes(source)
    actual_hash = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_size or actual_hash != expected_sha256:
        raise ValueError(f"raw PCD differs from frozen inventory: {source}")
    output, point_count = _pcd_output(raw)
    revalidated = _read_stable_regular_bytes(source)
    if (
        len(revalidated) != expected_size
        or hashlib.sha256(revalidated).hexdigest() != expected_sha256
        or revalidated != raw
    ):
        raise ValueError(f"raw PCD changed before publication: {source}")
    _publish_immutable_bytes(destination, output)
    return PreparedPointCloud(
        destination=destination,
        point_count=point_count,
        size=len(output),
        sha256=hashlib.sha256(output).hexdigest(),
        dtype="<f4",
        fields=_FIELDS,
    )
