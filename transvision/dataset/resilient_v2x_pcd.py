from __future__ import annotations

import errno
import hashlib
import io
import os
import secrets
import stat
import struct
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


_EntryIdentity = tuple[int, int, int]


def _entry_identity(metadata: os.stat_result) -> _EntryIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


@dataclass
class _AnchoredParent:
    path: Path
    component_names: tuple[str, ...]
    descriptors: list[int]
    identities: list[_EntryIdentity]
    leaf_name: str

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for descriptor, expected in zip(
            self.descriptors,
            self.identities,
        ):
            if _entry_identity(os.fstat(descriptor)) != expected:
                raise ValueError(
                    f"directory chain changed during operation: {self.path}"
                )
        for index, component in enumerate(self.component_names):
            try:
                current = os.stat(
                    component,
                    dir_fd=self.descriptors[index],
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ValueError(
                    f"directory chain changed during operation: {self.path}"
                ) from error
            if (
                stat.S_ISLNK(current.st_mode)
                or _entry_identity(current) != self.identities[index + 1]
            ):
                raise ValueError(
                    f"directory chain changed during operation: {self.path}"
                )

    def close(self) -> None:
        first_error: OSError | None = None
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        self.descriptors.clear()
        if first_error is not None:
            raise first_error


def _path_components(path: Path) -> tuple[str, tuple[str, ...], str]:
    path = Path(path)
    parts = path.parts
    if not parts:
        raise ValueError(f"path must name a file: {path}")
    if path.is_absolute():
        anchor = os.sep
        relative_parts = parts[1:]
    else:
        anchor = "."
        relative_parts = parts
    if (
        not relative_parts
        or relative_parts[-1] in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in relative_parts[:-1])
    ):
        raise ValueError(f"path must not contain ambiguous components: {path}")
    return anchor, tuple(relative_parts[:-1]), relative_parts[-1]


def _open_anchored_parent(
    path: Path,
    *,
    create: bool,
) -> _AnchoredParent:
    path = Path(path)
    anchor, component_names, leaf_name = _path_components(path)
    descriptors: list[int] = []
    identities: list[_EntryIdentity] = []
    successful = False
    try:
        descriptor = os.open(anchor, _directory_open_flags())
        descriptors.append(descriptor)
        identities.append(_entry_identity(os.fstat(descriptor)))
        for component in component_names:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(
                    f"path ancestor must not be a symlink: {path}"
                )
            if not stat.S_ISDIR(before.st_mode):
                error = NotADirectoryError(
                    errno.ENOTDIR,
                    os.strerror(errno.ENOTDIR),
                    str(path),
                )
                raise ValueError(
                    f"path ancestor must be a directory: {path}"
                ) from error
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ValueError(
                    f"path ancestor is not a stable directory: {path}"
                ) from error
            descriptors.append(child)
            opened = os.fstat(child)
            after = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(after.st_mode)
                or _entry_identity(before) != _entry_identity(opened)
                or _entry_identity(opened) != _entry_identity(after)
            ):
                raise ValueError(
                    f"path ancestor changed while opening: {path}"
                )
            identities.append(_entry_identity(opened))
            descriptor = child
        parent = _AnchoredParent(
            path=path,
            component_names=component_names,
            descriptors=descriptors,
            identities=identities,
            leaf_name=leaf_name,
        )
        parent.verify()
        successful = True
        return parent
    finally:
        if not successful:
            first_error: OSError | None = None
            for opened_descriptor in reversed(descriptors):
                try:
                    os.close(opened_descriptor)
                except OSError as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error


def _read_opened_regular_bytes(
    parent: _AnchoredParent,
    *,
    role: str,
) -> bytes:
    parent.verify()
    try:
        before_entry = os.stat(
            parent.leaf_name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before_entry.st_mode):
        raise ValueError(f"{role} must not be a symlink: {parent.path}")
    if not stat.S_ISREG(before_entry.st_mode):
        raise ValueError(f"{role} must be a regular file: {parent.path}")
    try:
        descriptor = os.open(
            parent.leaf_name,
            _regular_open_flags(),
            dir_fd=parent.descriptor,
        )
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(
                f"{role} must not be a symlink: {parent.path}"
            ) from error
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"{role} must be a regular file: {parent.path}"
            )
        after_open_entry = os.stat(
            parent.leaf_name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            _entry_identity(before_entry) != _entry_identity(before)
            or _entry_identity(before) != _entry_identity(after_open_entry)
        ):
            raise ValueError(f"{role} changed while opening: {parent.path}")
        parent.verify()
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
            raise ValueError(f"{role} changed while reading: {parent.path}")
        after_read_entry = os.stat(
            parent.leaf_name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(after_read_entry) != _entry_identity(after):
            raise ValueError(f"{role} changed while reading: {parent.path}")
        parent.verify()
        return raw
    finally:
        os.close(descriptor)


def _filesystem_value_error(
    operation: str,
    path: Path,
    error: OSError,
) -> ValueError:
    detail = error.strerror or str(error) or type(error).__name__
    one_line_detail = " ".join(detail.splitlines())
    return ValueError(
        f"filesystem error {operation} {path}: {one_line_detail}"
    )


def _read_stable_regular_bytes_impl(path: Path) -> bytes:
    parent: _AnchoredParent | None = None
    try:
        parent = _open_anchored_parent(path, create=False)
        return _read_opened_regular_bytes(parent, role="source")
    finally:
        if parent is not None:
            parent.close()


def _read_stable_regular_bytes(path: Path) -> bytes:
    path = Path(path)
    try:
        return _read_stable_regular_bytes_impl(path)
    except OSError as error:
        raise _filesystem_value_error("reading", path, error) from error


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


def _existing_bytes(parent: _AnchoredParent) -> bytes | None:
    try:
        return _read_opened_regular_bytes(parent, role="destination")
    except FileNotFoundError:
        return None
    except ValueError as error:
        raise ValueError(f"destination conflict: {parent.path}") from error


def _open_exclusive_temporary(
    parent_descriptor: int,
    destination_name: str,
) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _ in range(128):
        temporary_name = (
            f".{destination_name}.tmp-{secrets.token_hex(12)}"
        )
        try:
            return (
                os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                ),
                temporary_name,
            )
        except FileExistsError:
            continue
    raise OSError("could not allocate a unique temporary file")


def _publish_immutable_bytes_impl(destination: Path, data: bytes) -> None:
    destination = Path(destination)
    parent: _AnchoredParent | None = None
    descriptor = -1
    temporary_name: str | None = None
    try:
        parent = _open_anchored_parent(destination, create=True)
        existing = _existing_bytes(parent)
        if existing is not None:
            if existing == data:
                return
            raise ValueError(f"destination conflict: {destination}")

        descriptor, temporary_name = _open_exclusive_temporary(
            parent.descriptor,
            parent.leaf_name,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            written = stream.write(data)
            if written != len(data):
                raise OSError(
                    f"short write: expected {len(data)} bytes, wrote {written}"
                )
            stream.flush()
            os.fsync(stream.fileno())
        temporary_parent = _AnchoredParent(
            path=destination.with_name(temporary_name),
            component_names=parent.component_names,
            descriptors=parent.descriptors,
            identities=parent.identities,
            leaf_name=temporary_name,
        )
        if _read_opened_regular_bytes(
            temporary_parent,
            role="temporary file",
        ) != data:
            raise ValueError(
                f"temporary file verification failed: {destination}"
            )
        parent.verify()
        try:
            os.link(
                temporary_name,
                parent.leaf_name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raced = _existing_bytes(parent)
            if raced != data:
                raise ValueError(f"destination conflict: {destination}")
        parent.verify()
        if _existing_bytes(parent) != data:
            raise ValueError(
                f"published file verification failed: {destination}"
            )
        os.fsync(parent.descriptor)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            if parent is not None:
                try:
                    if temporary_name is not None:
                        try:
                            os.unlink(
                                temporary_name,
                                dir_fd=parent.descriptor,
                            )
                        except FileNotFoundError:
                            pass
                finally:
                    parent.close()


def _publish_immutable_bytes(destination: Path, data: bytes) -> None:
    destination = Path(destination)
    try:
        _publish_immutable_bytes_impl(destination, data)
    except OSError as error:
        raise _filesystem_value_error(
            "publishing",
            destination,
            error,
        ) from error


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
