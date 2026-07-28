from __future__ import annotations

import hashlib
import io
import importlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pypcd4 import Encoding, PointCloud


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests/resilient_v2x/fixtures/dair-mini"
XYZI = np.array(
    [
        [1.0, 2.0, 3.0, 0.25],
        [-4.0, 5.5, 6.25, 0.75],
    ],
    dtype=np.float32,
)
COMPRESSIBLE_XYZI = np.repeat(XYZI, 128, axis=0)
IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
CAMERA_INTRINSIC = (
    (1000.0, 0.0, 960.0),
    (0.0, 1000.0, 540.0),
    (0.0, 0.0, 1.0),
)
BRANCH_ORDER = (
    ("ego", "lidar"),
    ("rsu", "lidar"),
    ("ego", "camera"),
    ("rsu", "camera"),
)


def _pcd_bytes(
    points: np.ndarray,
    *,
    encoding: Encoding,
    extra_field: bool = False,
) -> bytes:
    if extra_field:
        values = np.column_stack([points[:, 0], np.arange(len(points)), points[:, 1:]])
        fields = ("x", "ring", "y", "z", "intensity")
        types = (np.float32, np.uint16, np.float32, np.float32, np.float32)
    else:
        values = points
        fields = ("x", "y", "z", "intensity")
        types = (np.float32,) * 4
    cloud = PointCloud.from_points(values, fields, types)
    buffer = io.BytesIO()
    cloud.save(buffer, encoding=encoding)
    return buffer.getvalue()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _digest(prefix: str, value: object) -> str:
    return prefix + hashlib.sha256(_canonical(value)).hexdigest()


def _normalized_record(
    index: int,
    timestamps: tuple[int, int, int, int],
    *,
    vehicle_batch_id: str = "vehicle-batch-a",
    infrastructure_batch_id: str = "infrastructure-batch-a",
    split: str = "train",
) -> dict[str, object]:
    slices: list[dict[str, object]] = []
    for branch_index, (agent, modality) in enumerate(BRANCH_ORDER):
        frame_id = f"{agent}-{modality}-{index:06d}"
        slices.append(
            {
                "agent": agent,
                "modality": modality,
                "capture_timestamp_us": timestamps[branch_index],
                "frame_id": frame_id,
                "relative_path": (
                    f"{agent}-side/{modality}/{frame_id}."
                    f"{'pcd' if modality == 'lidar' else 'jpg'}"
                ),
                "world_from_agent": IDENTITY_4X4,
                "agent_from_sensor": IDENTITY_4X4,
                "calibration_relative_path": (
                    f"{agent}-side/calib/{modality}/{frame_id}.json"
                ),
                "calibration_sha256": f"{branch_index + 1:064x}",
                "camera_intrinsic": (None if modality == "lidar" else CAMERA_INTRINSIC),
            }
        )
    return {
        "sample_id": f"{index:06d}",
        "split": split,
        "vehicle_batch_id": vehicle_batch_id,
        "infrastructure_batch_id": infrastructure_batch_id,
        "slices": slices,
        "annotation_path": f"cooperative/label_world/{index:06d}.json",
        "annotation_sha256": f"{index + 10:064x}",
        "ground_truth": [
            {
                "class_name": "Car",
                "x": 10.0,
                "y": 0.0,
                "z_bottom": 0.0,
                "length": 4.0,
                "width": 2.0,
                "height": 2.0,
                "yaw": 0.0,
                "source_annotation_index": 0,
            }
        ],
    }


def _fixture_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "dair-mini"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination


def _fixture_kwargs(root: Path, output: Path) -> dict[str, object]:
    split = root / "split.json"
    return {
        "data_root": root,
        "split_path": split,
        "output_path": output,
        "expected_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "protocol_scope": "fixture",
        "delta_t_ms": 100,
        "history_limit": 3,
        "interval_min_ms": 50,
        "interval_max_ms": 150,
        "max_capture_skew_ms": 50,
    }


def _rewrite_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _assert_no_publication(root: Path, output: Path) -> None:
    assert not output.exists()
    assert not (root / "prepared").exists()
    assert not [
        path
        for path in output.parent.iterdir()
        if path.name.startswith(f".{output.name}.tmp-")
    ]


def _inject_atomic_failure(
    monkeypatch: pytest.MonkeyPatch,
    pcd_module: object,
    stage: str,
    destination_name: str,
) -> None:
    def failure(*args: object, **kwargs: object) -> object:
        raise OSError(f"injected {stage} failure")

    if stage == "temp":
        monkeypatch.setattr(
            pcd_module,
            "_open_exclusive_temporary",
            failure,
        )
        return
    if stage == "write":
        original_fdopen = pcd_module.os.fdopen

        class FailingWriter:
            def __init__(self, stream: object) -> None:
                self.stream = stream

            def __enter__(self) -> "FailingWriter":
                self.stream.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self.stream.__exit__(*args)

            def write(self, data: bytes) -> int:
                raise OSError("injected write failure")

            def flush(self) -> None:
                self.stream.flush()

            def fileno(self) -> int:
                return self.stream.fileno()

        def failing_fdopen(*args: object, **kwargs: object) -> FailingWriter:
            return FailingWriter(original_fdopen(*args, **kwargs))

        monkeypatch.setattr(pcd_module.os, "fdopen", failing_fdopen)
        return
    if stage in {"file_fsync", "directory_fsync"}:
        original_fsync = pcd_module.os.fsync

        def failing_fsync(descriptor: int) -> None:
            mode = os.fstat(descriptor).st_mode
            if (
                stage == "file_fsync"
                and stat.S_ISREG(mode)
                or stage == "directory_fsync"
                and stat.S_ISDIR(mode)
            ):
                raise OSError(f"injected {stage} failure")
            original_fsync(descriptor)

        monkeypatch.setattr(pcd_module.os, "fsync", failing_fsync)
        return
    if stage == "link":
        monkeypatch.setattr(pcd_module.os, "link", failure)
        return
    if stage == "cleanup":
        original_unlink = pcd_module.os.unlink

        def failing_unlink(
            path: object,
            *args: object,
            **kwargs: object,
        ) -> None:
            if Path(path).name.startswith(f".{destination_name}.tmp-"):
                raise OSError("injected cleanup failure")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(pcd_module.os, "unlink", failing_unlink)
        return
    raise AssertionError(f"unknown stage {stage}")


def test_public_modules_import_and_cli_help_lists_exact_flags() -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")

    assert callable(module.main)
    assert callable(module.prepare_manifest)
    assert callable(module.build_protocol_sequences)
    assert callable(pcd_module.convert_pcd_to_bin)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/resilient_v2x/prepare_data.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    flags = {
        token
        for token in result.stdout.replace(",", " ").split()
        if token.startswith("--")
    }
    assert flags == {
        "--data-root",
        "--split-file",
        "--expected-split-sha256",
        "--protocol-scope",
        "--output",
        "--delta-t-ms",
        "--history-limit",
        "--interval-min-ms",
        "--interval-max-ms",
        "--max-capture-skew-ms",
        "--help",
    }


@pytest.mark.parametrize(
    "encoding",
    (Encoding.ASCII, Encoding.BINARY, Encoding.BINARY_COMPRESSED),
)
def test_pcd_encodings_preserve_order_and_emit_exact_little_endian_bytes(
    tmp_path: Path,
    encoding: Encoding,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    points = COMPRESSIBLE_XYZI if encoding == Encoding.BINARY_COMPRESSED else XYZI
    source = tmp_path / f"cloud-{encoding.value}.pcd"
    source.write_bytes(
        _pcd_bytes(
            points,
            encoding=encoding,
            extra_field=encoding == Encoding.ASCII,
        )
    )
    destination = tmp_path / "prepared" / f"{encoding.value}.bin"

    prepared = pcd_module.convert_pcd_to_bin(source, destination)

    expected = np.asarray(points, dtype="<f4", order="C").tobytes(order="C")
    assert destination.read_bytes() == expected
    assert prepared.destination == destination
    assert prepared.point_count == len(points)
    assert prepared.size == len(points) * 16
    assert prepared.sha256 == hashlib.sha256(expected).hexdigest()
    assert prepared.dtype == "<f4"
    assert prepared.fields == ("x", "y", "z", "intensity")


def test_pcd_accepts_only_canonical_pcl_zero_padding(tmp_path: Path) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    raw = _pcd_bytes(COMPRESSIBLE_XYZI, encoding=Encoding.BINARY_COMPRESSED)
    marker = b"DATA binary_compressed\n"
    data_offset = raw.index(marker) + len(marker)
    padding = b"\0" * (4096 - data_offset)
    source = tmp_path / "pcl-padded.pcd"
    source.write_bytes(raw + padding)
    destination = tmp_path / "pcl-padded.bin"

    prepared = pcd_module.convert_pcd_to_bin(source, destination)

    expected = np.asarray(COMPRESSIBLE_XYZI, dtype="<f4", order="C").tobytes(order="C")
    assert destination.read_bytes() == expected
    assert prepared.point_count == len(COMPRESSIBLE_XYZI)


@pytest.mark.parametrize(
    "trailing",
    (
        b"\0",
        b"unexpected",
    ),
)
def test_pcd_rejects_noncanonical_compressed_trailing_bytes(
    tmp_path: Path,
    trailing: bytes,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    raw = _pcd_bytes(COMPRESSIBLE_XYZI, encoding=Encoding.BINARY_COMPRESSED)
    source = tmp_path / "invalid-padded.pcd"
    source.write_bytes(raw + trailing)
    destination = tmp_path / "invalid-padded.bin"

    with pytest.raises(ValueError, match="payload size"):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert not destination.exists()


def test_pcd_rejects_nonzero_canonical_length_padding(tmp_path: Path) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    raw = _pcd_bytes(COMPRESSIBLE_XYZI, encoding=Encoding.BINARY_COMPRESSED)
    marker = b"DATA binary_compressed\n"
    data_offset = raw.index(marker) + len(marker)
    padding = bytearray(4096 - data_offset)
    padding[-1] = 1
    source = tmp_path / "nonzero-padded.pcd"
    source.write_bytes(raw + padding)
    destination = tmp_path / "nonzero-padded.bin"

    with pytest.raises(ValueError, match="payload size"):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert not destination.exists()


def test_pcd_zero_points_is_valid_and_idempotent(tmp_path: Path) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "empty.pcd"
    source.write_text(
        "\n".join(
            (
                "VERSION 0.7",
                "FIELDS x y z intensity",
                "SIZE 4 4 4 4",
                "TYPE F F F F",
                "COUNT 1 1 1 1",
                "WIDTH 0",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                "POINTS 0",
                "DATA ascii",
                "",
            )
        )
    )
    destination = tmp_path / "empty.bin"

    first = pcd_module.convert_pcd_to_bin(source, destination)
    second = pcd_module.convert_pcd_to_bin(source, destination)

    assert first == second
    assert destination.read_bytes() == b""
    assert first.point_count == 0
    assert first.size == 0


@pytest.mark.parametrize(
    ("header", "message"),
    (
        (
            "\n".join(
                (
                    "VERSION 0.7",
                    "FIELDS x y z",
                    "SIZE 4 4 4",
                    "TYPE F F F",
                    "COUNT 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "1 2 3",
                )
            ),
            "required",
        ),
        (
            "\n".join(
                (
                    "VERSION 0.7",
                    "FIELDS x y z intensity intensity",
                    "SIZE 4 4 4 4 4",
                    "TYPE F F F F F",
                    "COUNT 1 1 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "1 2 3 4 5",
                )
            ),
            "exactly once",
        ),
        (
            "\n".join(
                (
                    "VERSION 0.7",
                    "FIELDS x y z intensity",
                    "SIZE 4 4 4 4",
                    "TYPE F F F F",
                    "COUNT 2 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "1 9 2 3 4",
                )
            ),
            "scalar",
        ),
        (
            "\n".join(
                (
                    "VERSION 0.7",
                    "FIELDS x y z intensity",
                    "SIZE 4 4 4 4",
                    "TYPE F F F F",
                    "COUNT 1 1 1 1",
                    "WIDTH 2",
                    "HEIGHT 1",
                    "POINTS 2",
                    "DATA ascii",
                    "1 2 3 4",
                )
            ),
            "point count",
        ),
        (
            "\n".join(
                (
                    "VERSION 0.7",
                    "FIELDS x y z intensity",
                    "SIZE 4 4 4 4",
                    "TYPE F F F F",
                    "COUNT 1 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "nan 2 3 4",
                )
            ),
            "finite",
        ),
    ),
)
def test_pcd_rejects_invalid_fields_counts_rows_and_values(
    tmp_path: Path,
    header: str,
    message: str,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "invalid.pcd"
    source.write_text(header)
    destination = tmp_path / "invalid.bin"

    with pytest.raises(ValueError, match=message):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert not destination.exists()


def test_pcd_refuses_symlink_source_and_conflicting_destination(
    tmp_path: Path,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    real_source = tmp_path / "real.pcd"
    real_source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    symlink_source = tmp_path / "linked.pcd"
    symlink_source.symlink_to(real_source)

    with pytest.raises(ValueError, match="symlink"):
        pcd_module.convert_pcd_to_bin(
            symlink_source,
            tmp_path / "linked.bin",
        )

    destination = tmp_path / "conflict.bin"
    destination.write_bytes(b"different")
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="conflict"):
        pcd_module.convert_pcd_to_bin(real_source, destination)
    assert destination.read_bytes() == before
    assert not [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(f".{destination.name}.tmp-")
    ]


def test_pcd_rejects_source_with_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    source = real_parent / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    destination = tmp_path / "prepared.bin"

    with pytest.raises(ValueError, match="symlink"):
        pcd_module.convert_pcd_to_bin(
            linked_parent / source.name,
            destination,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    "linked_relative",
    ("prepared", "prepared/resilient_v2x"),
)
def test_prepare_rejects_symlinked_prepared_ancestor(
    tmp_path: Path,
    linked_relative: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = root / linked_relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    assert not output.exists()
    assert not list(outside.rglob("*.bin"))


def test_prepare_rejects_symlinked_output_parent(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(outside, target_is_directory=True)
    output = linked_parent / "manifest.json"

    with pytest.raises(ValueError, match="symlink"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    assert not (outside / output.name).exists()


def test_pcd_parent_replacement_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    parent = tmp_path / "publish-parent"
    parent.mkdir()
    destination = parent / "prepared.bin"
    displaced = tmp_path / "displaced-parent"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_link = pcd_module.os.link
    replaced = False

    def replace_parent_then_link(
        source_name: object,
        destination_name: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(displaced)
            parent.symlink_to(outside, target_is_directory=True)
            (outside / Path(source_name).name).write_bytes(
                np.asarray(XYZI, dtype="<f4", order="C").tobytes()
            )
        original_link(
            source_name,
            destination_name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(pcd_module.os, "link", replace_parent_then_link)

    with pytest.raises(ValueError, match="changed"):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert not destination.exists()
    assert not (outside / destination.name).exists()


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_pcd_final_symlink_or_directory_is_a_conflict(
    tmp_path: Path,
    kind: str,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    destination = tmp_path / "prepared.bin"
    target = tmp_path / "target.bin"
    target.write_bytes(b"sentinel")
    if kind == "symlink":
        destination.symlink_to(target)
    else:
        destination.mkdir()

    with pytest.raises(ValueError, match="conflict"):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert target.read_bytes() == b"sentinel"
    if kind == "directory":
        assert list(destination.iterdir()) == []


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_pcd_final_symlink_or_directory_replacement_is_a_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    destination = tmp_path / "prepared.bin"
    target = tmp_path / "target.bin"
    target.write_bytes(b"sentinel")
    original_link = pcd_module.os.link

    def replace_final_then_link(
        source_name: object,
        destination_name: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if kind == "symlink":
            destination.symlink_to(target)
        else:
            destination.mkdir()
        original_link(
            source_name,
            destination_name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(pcd_module.os, "link", replace_final_then_link)

    with pytest.raises(ValueError, match="conflict"):
        pcd_module.convert_pcd_to_bin(source, destination)

    assert target.read_bytes() == b"sentinel"
    if kind == "directory":
        assert list(destination.iterdir()) == []


def test_protocol_sequences_build_exact_history_and_stable_identifiers() -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    records = [
        _normalized_record(
            index,
            tuple(1_000_000 + index * 100_000 + branch * 10_000 for branch in range(4)),
        )
        for index in range(4)
    ]

    samples, boundaries = module.build_protocol_sequences(
        records,
        delta_t_ms=100,
        history_limit=3,
        interval_min_ms=50,
        interval_max_ms=150,
        max_capture_skew_ms=50,
    )

    source_id = _digest(
        "src-",
        {
            "infrastructure_batch_id": "infrastructure-batch-a",
            "sequence_lane": 0,
            "split": "train",
            "vehicle_batch_id": "vehicle-batch-a",
        },
    )
    sequence_id = _digest(
        "seq-",
        {"segment_index": 0, "source_sequence_id": source_id},
    )
    assert boundaries == []
    assert [sample.n_t for sample in samples] == [0, 1, 2, 3]
    assert [sample.tau_t_ms for sample in samples] == [0, 100, 200, 300]
    assert [len(sample.source_slices) for sample in samples] == [4, 8, 12, 16]
    assert {sample.sequence_id for sample in samples} == {sequence_id}
    assert [(item.agent, item.modality) for item in samples[-1].source_slices] == list(
        BRANCH_ORDER
    ) * 4
    first = samples[0].source_slices[0]
    assert first.packet_id == _digest(
        "pkt-",
        {
            "agent": "ego",
            "frame_id": "ego-lidar-000000",
            "modality": "lidar",
            "protocol_sequence_id": sequence_id,
        },
    )
    assert samples[-1].source_slices[0] == first
    assert samples[-1].ground_truth[0].x == 10.0


@pytest.mark.parametrize(
    ("interval_us", "splits"),
    ((49_999, 1), (50_000, 0), (150_000, 0), (150_001, 1)),
)
def test_protocol_interval_boundaries_are_exact_microseconds(
    interval_us: int,
    splits: int,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    records = [
        _normalized_record(0, (1_000_000, 1_010_000, 1_020_000, 1_030_000)),
        _normalized_record(
            1,
            tuple(
                timestamp + interval_us
                for timestamp in (1_000_000, 1_010_000, 1_020_000, 1_030_000)
            ),
        ),
    ]

    samples, boundaries = module.build_protocol_sequences(records, 100, 3, 50, 150, 50)

    assert len(boundaries) == splits
    assert samples[-1].n_t == (0 if splits else 1)
    assert len(samples[-1].source_slices) == (4 if splits else 8)


@pytest.mark.parametrize("branch_index", range(4))
@pytest.mark.parametrize("delta_us", (0, -1))
def test_protocol_rejects_nonincreasing_each_stream_without_reordering(
    branch_index: int,
    delta_us: int,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    first = [1_000_000, 1_010_000, 1_020_000, 1_030_000]
    second = [value + 100_000 for value in first]
    second[branch_index] = first[branch_index] + delta_us

    with pytest.raises(ValueError, match="strictly increasing"):
        module.build_protocol_sequences(
            [
                _normalized_record(0, tuple(first)),
                _normalized_record(1, tuple(second)),
            ],
            100,
            3,
            50,
            150,
            50,
        )


@pytest.mark.parametrize(("skew_us", "valid"), ((50_000, True), (50_001, False)))
def test_protocol_capture_skew_boundary_is_inclusive(
    skew_us: int,
    valid: bool,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    record = _normalized_record(
        0,
        (1_000_000, 1_010_000, 1_020_000, 1_000_000 + skew_us),
    )
    if valid:
        samples, _ = module.build_protocol_sequences([record], 100, 3, 50, 150, 50)
        assert len(samples) == 1
    else:
        with pytest.raises(ValueError, match="skew"):
            module.build_protocol_sequences([record], 100, 3, 50, 150, 50)


def test_protocol_merges_interval_triggers_in_branch_order() -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    first = (1_000_000, 1_010_000, 1_020_000, 1_030_000)
    second = tuple(value + 150_001 for value in first)

    samples, boundaries = module.build_protocol_sequences(
        [
            _normalized_record(0, first),
            _normalized_record(1, second),
        ],
        100,
        3,
        50,
        150,
        50,
    )

    assert [sample.n_t for sample in samples] == [0, 0]
    assert len(boundaries) == 1
    assert [
        (trigger["agent"], trigger["modality"]) for trigger in boundaries[0]["triggers"]
    ] == list(BRANCH_ORDER)
    assert {trigger["interval_us"] for trigger in boundaries[0]["triggers"]} == {
        150_001
    }


def test_source_batch_transition_resets_history_without_interval_boundary() -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    samples, boundaries = module.build_protocol_sequences(
        [
            _normalized_record(0, (1_000_000, 1_010_000, 1_020_000, 1_030_000)),
            _normalized_record(
                1,
                (2_000_000, 2_010_000, 2_020_000, 2_030_000),
                vehicle_batch_id="vehicle-batch-b",
                infrastructure_batch_id="infrastructure-batch-b",
            ),
        ],
        100,
        3,
        50,
        150,
        50,
    )

    assert boundaries == []
    assert [sample.n_t for sample in samples] == [0, 0]
    assert len(samples[1].source_slices) == 4
    assert samples[0].sequence_id != samples[1].sequence_id


def test_source_batch_pair_cannot_reappear_noncontiguously() -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    records = [
        _normalized_record(0, (1_000_000, 1_010_000, 1_020_000, 1_030_000)),
        _normalized_record(
            1,
            (2_000_000, 2_010_000, 2_020_000, 2_030_000),
            vehicle_batch_id="vehicle-batch-b",
            infrastructure_batch_id="infrastructure-batch-b",
        ),
        _normalized_record(2, (3_000_000, 3_010_000, 3_020_000, 3_030_000)),
    ]

    with pytest.raises(ValueError, match="non-contiguously"):
        module.build_protocol_sequences(records, 100, 3, 50, 150, 50)


def test_prepare_four_tick_fixture_is_canonical_complete_and_repeatable(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    manifest_module = importlib.import_module(
        "transvision.dataset.resilient_v2x_manifest"
    )
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    kwargs = _fixture_kwargs(root, output)

    manifest = module.prepare_manifest(**kwargs)
    first_bytes = output.read_bytes()
    payload = json.loads(first_bytes)
    loaded = manifest_module.load_temporal_manifest(
        output,
        expected_split_hash=kwargs["expected_split_sha256"],
        allow_fixture=True,
    )

    assert manifest == loaded
    assert payload["protocol_scope"] == "fixture"
    assert [sample["n_t"] for sample in payload["samples"]] == [0, 1, 2, 3]
    assert payload["samples"][3]["tau_t_ms"] == 300
    assert len(payload["samples"][3]["source_slices"]) == 16
    assert payload["samples"][3]["ground_truth"] == [
        {
            "class_name": "Car",
            "x": 10.0,
            "y": 0.0,
            "z_bottom": 0.0,
            "length": 4.0,
            "width": 2.0,
            "height": 2.0,
            "yaw": 0.0,
            "source_annotation_index": 0,
        }
    ]
    assert payload["sequence_splits"] == []
    assert payload["excluded_samples"] == []
    assert payload["history_eligible_train_count"] == 1
    assert len(payload["prepared_artifacts"]) == 8
    assert [
        item["source_relative_path"] for item in payload["prepared_artifacts"]
    ] == sorted(item["source_relative_path"] for item in payload["prepared_artifacts"])
    inventory_paths = {item["relative_path"] for item in payload["release_inventory"]}
    assert {
        "cooperative/data_info.json",
        "vehicle-side/data_info.json",
        "infrastructure-side/data_info.json",
        "vehicle-side/velodyne/000000.pcd",
        "infrastructure-side/velodyne/100003.pcd",
        "vehicle-side/image/000002.jpg",
        "infrastructure-side/image/100001.jpg",
        "cooperative/label_world/000003.json",
        "vehicle-side/calib/novatel_to_world/000001.json",
        "infrastructure-side/calib/camera_intrinsic/100002.json",
    }.issubset(inventory_paths)
    assert "split.json" not in inventory_paths
    assert output.name not in inventory_paths
    assert not any(path.endswith(".bin") for path in inventory_paths)
    assert all(
        Path(item["prepared_relative_path"]).is_relative_to(
            Path("prepared/resilient_v2x")
        )
        for item in payload["prepared_artifacts"]
    )

    second = module.prepare_manifest(**kwargs)
    assert second == manifest
    assert output.read_bytes() == first_bytes


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "overlap", "test_a", "malformed", "missing", "unknown"),
)
def test_split_and_cooperative_id_failures_precede_publication(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    split_path = root / "split.json"
    split = json.loads(split_path.read_text())
    cooperative_path = root / "cooperative/data_info.json"
    cooperative = json.loads(cooperative_path.read_text())
    if mutation == "duplicate":
        split["cooperative_split"]["train"].append("000000")
    elif mutation == "overlap":
        split["cooperative_split"]["val"].append("000000")
    elif mutation == "test_a":
        split["cooperative_split"]["test_A"].append("not-in-test")
    elif mutation == "malformed":
        split["cooperative_split"]["train"][0] = "../000000"
    elif mutation == "missing":
        split["cooperative_split"]["train"].append("999999")
    else:
        cooperative[0]["vehicle_image_path"] = "vehicle-side/image/999999.jpg"
        cooperative[0]["vehicle_pointcloud_path"] = "vehicle-side/velodyne/999999.pcd"
        cooperative[0]["cooperative_label_path"] = "cooperative/label_world/999999.json"
        _rewrite_json(cooperative_path, cooperative)
    _rewrite_json(split_path, split)
    kwargs = _fixture_kwargs(root, output)

    with pytest.raises((ValueError, RuntimeError)):
        module.prepare_manifest(**kwargs)

    _assert_no_publication(root, output)


def test_split_duplicate_json_keys_and_controlled_hash_are_fail_closed(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    split_path = root / "split.json"
    split_path.write_bytes(
        b'{"cooperative_split":{"train":[],"train":[],"val":[],"test":[]}}'
    )
    kwargs = _fixture_kwargs(root, output)
    with pytest.raises(ValueError, match="duplicate"):
        module.prepare_manifest(**kwargs)
    _assert_no_publication(root, output)

    root = _fixture_copy(tmp_path / "controlled")
    output = tmp_path / "controlled.json"
    kwargs = _fixture_kwargs(root, output)
    kwargs["protocol_scope"] = "controlled"
    with pytest.raises(ValueError, match="official"):
        module.prepare_manifest(**kwargs)
    _assert_no_publication(root, output)


def test_checked_in_official_split_counts_hash_and_test_a_subset() -> None:
    prepare_module = importlib.import_module("tools.resilient_v2x.prepare_data")
    manifest_module = importlib.import_module(
        "transvision.dataset.resilient_v2x_manifest"
    )
    path = ROOT / "data/split_datas/cooperative-split-data.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    split = payload["cooperative_split"]

    assert hashlib.sha256(raw).hexdigest() == (
        manifest_module.OFFICIAL_COOPERATIVE_SPLIT_SHA256
    )
    assert [len(split[name]) for name in ("train", "val", "test")] == [
        4813,
        1783,
        2688,
    ]
    assert all(
        len(split[name]) == len(set(split[name])) for name in ("train", "val", "test")
    )
    assert not (
        set(split["train"]) & set(split["val"])
        or set(split["train"]) & set(split["test"])
        or set(split["val"]) & set(split["test"])
    )
    assert set(split["test_A"]).issubset(set(split["test"]))

    split_by_id, release_ids = prepare_module._parse_split(
        raw,
        actual_sha256=manifest_module.OFFICIAL_COOPERATIVE_SPLIT_SHA256,
        expected_sha256=manifest_module.OFFICIAL_COOPERATIVE_SPLIT_SHA256,
        protocol_scope="controlled",
        protocol_values=(100, 3, 50, 150, 200),
    )
    assert len(release_ids) == 4813 + 1783
    assert set(split_by_id.values()) == {"train", "val"}
    assert not set(split["test"]) & release_ids


def test_prepare_derives_side_frame_ids_from_official_style_paths(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    for side in ("vehicle-side", "infrastructure-side"):
        path = root / side / "data_info.json"
        records = json.loads(path.read_text())
        for record in records:
            record.pop("frame_id")
        _rewrite_json(path, records)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))

    assert len(manifest.samples) == 4
    assert manifest.samples[0].sample_id == "dairc-v000000-i100000"
    assert [
        (item.agent, item.modality, item.frame_id)
        for item in manifest.samples[0].source_slices
    ] == [
        ("ego", "lidar", "000000"),
        ("rsu", "lidar", "100000"),
        ("ego", "camera", "000000"),
        ("rsu", "camera", "100000"),
    ]


def test_prepare_preserves_duplicate_target_pair_variants_in_stable_lanes(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"

    infrastructure_path = root / "infrastructure-side/data_info.json"
    infrastructure = json.loads(infrastructure_path.read_text())
    alternate = dict(infrastructure[0])
    alternate.update(
        {
            "frame_id": "100004",
            "image_path": "image/100004.jpg",
            "pointcloud_path": "velodyne/100004.pcd",
            "image_timestamp": "1025000",
            "pointcloud_timestamp": "1005000",
            "calib_camera_intrinsic_path": ("calib/camera_intrinsic/100004.json"),
            "calib_virtuallidar_to_camera_path": (
                "calib/virtuallidar_to_camera/100004.json"
            ),
            "calib_virtuallidar_to_world_path": (
                "calib/virtuallidar_to_world/100004.json"
            ),
        }
    )
    infrastructure.append(alternate)
    _rewrite_json(infrastructure_path, infrastructure)
    shutil.copyfile(
        root / "infrastructure-side/image/100000.jpg",
        root / "infrastructure-side/image/100004.jpg",
    )
    shutil.copyfile(
        root / "infrastructure-side/velodyne/100000.pcd",
        root / "infrastructure-side/velodyne/100004.pcd",
    )
    for directory in (
        "camera_intrinsic",
        "virtuallidar_to_camera",
        "virtuallidar_to_world",
    ):
        shutil.copyfile(
            root / f"infrastructure-side/calib/{directory}/100000.json",
            root / f"infrastructure-side/calib/{directory}/100004.json",
        )

    cooperative_path = root / "cooperative/data_info.json"
    cooperative = json.loads(cooperative_path.read_text())
    alternate_pair = dict(cooperative[0])
    alternate_pair.update(
        {
            "infrastructure_image_path": ("infrastructure-side/image/100004.jpg"),
            "infrastructure_pointcloud_path": (
                "infrastructure-side/velodyne/100004.pcd"
            ),
        }
    )
    cooperative.append(alternate_pair)
    _rewrite_json(cooperative_path, cooperative)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    sample_ids = [sample.sample_id for sample in manifest.samples]

    assert sample_ids == [
        "dairc-v000000-i100004",
        "dairc-v000001-i100001",
        "dairc-v000002-i100002",
        "dairc-v000003-i100003",
        "dairc-v000000-i100000",
    ]
    assert [sample.n_t for sample in manifest.samples] == [0, 1, 2, 3, 0]
    assert len({sample.sequence_id for sample in manifest.samples[:4]}) == 1
    assert manifest.samples[-1].sequence_id != manifest.samples[0].sequence_id

    _rewrite_json(cooperative_path, list(reversed(cooperative)))
    reordered = module.prepare_manifest(
        **_fixture_kwargs(root, tmp_path / "reordered-manifest.json")
    )
    assert reordered.samples == manifest.samples


@pytest.mark.parametrize(
    "failure",
    (
        "conflicting_join",
        "camera_timestamp",
        "path_traversal",
        "missing_calibration",
        "symlink_payload",
    ),
)
def test_metadata_path_and_timestamp_failures_leave_no_artifacts(
    tmp_path: Path,
    failure: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    if failure == "conflicting_join":
        vehicle_path = root / "vehicle-side/data_info.json"
        vehicle = json.loads(vehicle_path.read_text())
        vehicle[1]["image_path"] = "image/000000.jpg"
        _rewrite_json(vehicle_path, vehicle)
    elif failure == "camera_timestamp":
        vehicle_path = root / "vehicle-side/data_info.json"
        vehicle = json.loads(vehicle_path.read_text())
        vehicle[1]["image_timestamp"] = vehicle[0]["image_timestamp"]
        _rewrite_json(vehicle_path, vehicle)
    elif failure == "path_traversal":
        cooperative_path = root / "cooperative/data_info.json"
        cooperative = json.loads(cooperative_path.read_text())
        cooperative[0]["vehicle_image_path"] = "vehicle-side/image/../../outside.jpg"
        _rewrite_json(cooperative_path, cooperative)
    elif failure == "missing_calibration":
        (root / "vehicle-side/calib/lidar_to_novatel/000000.json").unlink()
    else:
        source = root / "vehicle-side/velodyne/000000.pcd"
        target = root / "vehicle-side/velodyne/000001.pcd"
        source.unlink()
        source.symlink_to(target)
    kwargs = _fixture_kwargs(root, output)

    with pytest.raises((ValueError, RuntimeError)):
        module.prepare_manifest(**kwargs)

    _assert_no_publication(root, output)


def test_manifest_uses_independent_side_timestamps_without_filename_inference(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    final_tick = manifest.samples[-1].source_slices[-4:]

    assert [item.capture_timestamp_us for item in final_tick] == [
        1_300_000,
        1_310_000,
        1_320_000,
        1_330_000,
    ]


def test_calibration_chain_camera_inverse_and_offset_are_exact(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    novatel_path = root / "vehicle-side/calib/novatel_to_world/000000.json"
    novatel = json.loads(novatel_path.read_text())
    novatel["translation"] = [[5.0], [0.0], [0.0]]
    _rewrite_json(novatel_path, novatel)
    camera_path = root / "vehicle-side/calib/lidar_to_camera/000000.json"
    camera = json.loads(camera_path.read_text())
    camera["translation"] = [[1.0], [2.0], [3.0]]
    _rewrite_json(camera_path, camera)
    cooperative_path = root / "cooperative/data_info.json"
    cooperative = json.loads(cooperative_path.read_text())
    cooperative[0]["system_error_offset"] = {
        "delta_x": "2.5",
        "delta_y": "-3.5",
    }
    _rewrite_json(cooperative_path, cooperative)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    first_slices = manifest.samples[0].source_slices

    assert manifest.samples[0].ground_truth[0].x == 5.0
    assert first_slices[0].world_from_agent[0][3] == 5.0
    assert [first_slices[1].world_from_agent[index][3] for index in range(3)] == [
        2.5,
        -3.5,
        0.0,
    ]
    assert [first_slices[2].agent_from_sensor[index][3] for index in range(3)] == [
        -1.0,
        -2.0,
        -3.0,
    ]
    assert first_slices[2].camera_intrinsic == CAMERA_INTRINSIC


def test_infrastructure_camera_preserves_official_proper_affine_extrinsic(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    camera_path = root / "infrastructure-side/calib/virtuallidar_to_camera/100000.json"
    camera = json.loads(camera_path.read_text())
    camera["rotation"] = [
        [1.0, 0.1, 0.0],
        [0.0, 0.82, 0.0],
        [0.0, 0.0, 1.0],
    ]
    _rewrite_json(camera_path, camera)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    rsu_camera = manifest.samples[0].source_slices[3]
    expected = np.linalg.inv(
        np.asarray(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 0.82, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    assert np.allclose(np.asarray(rsu_camera.agent_from_sensor), expected)


@pytest.mark.parametrize(
    "offset",
    (
        {"delta_x": 1.0},
        {"delta_x": True, "delta_y": 0.0},
        {"delta_x": "", "delta_y": 0.0},
        {"delta_x": "nan", "delta_y": 0.0},
        [],
    ),
)
def test_malformed_cooperative_offset_fails_before_publication(
    tmp_path: Path,
    offset: object,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    cooperative_path = root / "cooperative/data_info.json"
    cooperative = json.loads(cooperative_path.read_text())
    cooperative[0]["system_error_offset"] = offset
    _rewrite_json(cooperative_path, cooperative)

    with pytest.raises((ValueError, RuntimeError), match="offset|delta"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    _assert_no_publication(root, output)


@pytest.mark.parametrize("calibration_failure", ("nonrigid", "nonfinite", "singular"))
def test_invalid_calibration_fails_before_publication(
    tmp_path: Path,
    calibration_failure: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    path = root / "vehicle-side/calib/lidar_to_camera/000000.json"
    calibration = json.loads(path.read_text())
    if calibration_failure == "nonrigid":
        calibration["rotation"][0][0] = 2.0
    elif calibration_failure == "nonfinite":
        calibration["translation"][0][0] = float("nan")
    else:
        calibration["rotation"][2] = [0, 0, 0]
    _rewrite_json(path, calibration)

    with pytest.raises((ValueError, RuntimeError)):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    _assert_no_publication(root, output)


def test_nonrigid_source_calibrations_cannot_cancel_to_a_rigid_composition(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    lidar_path = root / "vehicle-side/calib/lidar_to_novatel/000000.json"
    lidar = json.loads(lidar_path.read_text())
    lidar["transform"]["rotation"][0][0] = 2.0
    _rewrite_json(lidar_path, lidar)
    world_path = root / "vehicle-side/calib/novatel_to_world/000000.json"
    world = json.loads(world_path.read_text())
    world["rotation"][0][0] = 0.5
    _rewrite_json(world_path, world)

    with pytest.raises((ValueError, RuntimeError), match="rigid|orthonormal"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    _assert_no_publication(root, output)


def test_world_box_nonidentity_transform_yaw_wrap_and_ignored_provenance(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    pose_path = root / "vehicle-side/calib/novatel_to_world/000000.json"
    pose = json.loads(pose_path.read_text())
    pose["rotation"] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    pose["translation"] = [[20], [0], [0]]
    _rewrite_json(pose_path, pose)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    box = manifest.samples[0].ground_truth[0]
    annotation = next(
        entry
        for entry in manifest.release_inventory
        if entry.relative_path == "cooperative/label_world/000000.json"
    )

    assert (box.x, box.y, box.z_bottom) == (10.0, 0.0, 0.0)
    assert (box.length, box.width, box.height) == (4.0, 2.0, 2.0)
    assert box.yaw == -math.pi
    assert len(manifest.samples[0].ground_truth) == 1
    assert (
        annotation.sha256
        == hashlib.sha256((root / annotation.relative_path).read_bytes()).hexdigest()
    )


def test_official_lowercase_car_label_is_canonicalized(tmp_path: Path) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    path = root / "cooperative/label_world/000000.json"
    labels = json.loads(path.read_text())
    labels[0]["type"] = "car"
    _rewrite_json(path, labels)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))

    assert len(manifest.samples[0].ground_truth) == 1
    assert manifest.samples[0].ground_truth[0].class_name == "Car"


@pytest.mark.parametrize(
    "label_failure",
    (
        "ambiguous",
        "duplicate_corner",
        "noncuboid",
        "dimension",
        "nonfinite",
    ),
)
def test_invalid_world_box_fails_before_publication(
    tmp_path: Path,
    label_failure: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    path = root / "cooperative/label_world/000000.json"
    labels = json.loads(path.read_text())
    car = labels[0]
    if label_failure == "ambiguous":
        car["3d_location"] = {"x": 10, "y": 0, "z": 0}
        car.pop("world_8_points")
    elif label_failure == "duplicate_corner":
        car["world_8_points"][1] = car["world_8_points"][0]
    elif label_failure == "noncuboid":
        car["world_8_points"][6][0] += 0.25
    elif label_failure == "dimension":
        car["3d_dimensions"]["l"] = 4.5
    else:
        car["world_8_points"][0][0] = float("nan")
    _rewrite_json(path, labels)

    with pytest.raises((ValueError, RuntimeError)):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    _assert_no_publication(root, output)


def test_stable_pcd_revalidation_aborts_manifest_and_leaves_only_complete_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    drifting_source = root / "vehicle-side/velodyne/000000.pcd"
    original_read = pcd_module._read_stable_regular_bytes
    calls = 0

    def drifting_read(path: Path) -> bytes:
        nonlocal calls
        if Path(path) == drifting_source:
            calls += 1
            if calls == 2:
                drifting_source.write_bytes(drifting_source.read_bytes() + b"\n")
        return original_read(Path(path))

    monkeypatch.setattr(
        pcd_module,
        "_read_stable_regular_bytes",
        drifting_read,
    )

    with pytest.raises(ValueError, match="changed"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    assert not output.exists()
    bins = sorted((root / "prepared").rglob("*.bin"))
    assert bins
    assert all(path.stat().st_size == 32 for path in bins)
    assert not list((root / "prepared").rglob("*.tmp-*"))


def test_prepared_and_manifest_conflicts_are_never_overwritten(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    first_prepared = (
        root / "prepared/resilient_v2x/infrastructure-side/velodyne/100000.bin"
    )
    first_prepared.parent.mkdir(parents=True)
    first_prepared.write_bytes(b"conflicting prepared bytes")

    with pytest.raises(ValueError, match="conflict"):
        module.prepare_manifest(**_fixture_kwargs(root, output))
    assert first_prepared.read_bytes() == b"conflicting prepared bytes"
    assert not output.exists()

    shutil.rmtree(root / "prepared")
    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))
    original = output.read_bytes()
    output.write_bytes(b"conflicting manifest bytes")
    with pytest.raises(ValueError, match="conflict"):
        module.prepare_manifest(**_fixture_kwargs(root, output))
    assert output.read_bytes() == b"conflicting manifest bytes"
    assert (
        manifest.content_sha256
        != hashlib.sha256(b"conflicting manifest bytes").hexdigest()
    )
    assert not [
        path
        for path in output.parent.iterdir()
        if path.name.startswith(f".{output.name}.tmp-")
    ]
    output.write_bytes(original)


def test_cli_success_error_and_argparse_exit_contracts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    split = root / "split.json"
    argv = [
        "--data-root",
        str(root),
        "--split-file",
        str(split),
        "--expected-split-sha256",
        hashlib.sha256(split.read_bytes()).hexdigest(),
        "--protocol-scope",
        "fixture",
        "--output",
        str(output),
        "--delta-t-ms",
        "100",
        "--history-limit",
        "3",
        "--interval-min-ms",
        "50",
        "--interval-max-ms",
        "150",
        "--max-capture-skew-ms",
        "50",
    ]

    assert module.main(argv) == 0
    assert output.exists()
    assert capsys.readouterr() == ("", "")

    argv[argv.index("--expected-split-sha256") + 1] = "0" * 64
    assert module.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1

    with pytest.raises(SystemExit) as help_exit:
        module.main(["--help"])
    assert help_exit.value.code == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as usage_exit:
        module.main(["--unknown"])
    assert usage_exit.value.code == 2


def test_cli_filesystem_error_is_one_line_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    split = root / "split.json"
    argv = [
        "--data-root",
        str(root),
        "--split-file",
        str(split),
        "--expected-split-sha256",
        hashlib.sha256(split.read_bytes()).hexdigest(),
        "--protocol-scope",
        "fixture",
        "--output",
        "/dev/null/manifest.json",
        "--delta-t-ms",
        "100",
        "--history-limit",
        "3",
        "--interval-min-ms",
        "50",
        "--interval-max-ms",
        "150",
        "--max-capture-skew-ms",
        "50",
    ]

    assert module.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert "traceback" not in captured.err.lower()


def test_prepare_manifest_filesystem_error_has_oserror_cause(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)

    with pytest.raises(ValueError, match="directory") as raised:
        module.prepare_manifest(
            **_fixture_kwargs(root, Path("/dev/null/manifest.json"))
        )

    assert isinstance(raised.value.__cause__, OSError)


def test_public_library_filesystem_error_is_value_error_with_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    destination = tmp_path / "prepared.bin"

    def failing_fsync(descriptor: int) -> None:
        raise OSError("injected public boundary failure")

    monkeypatch.setattr(pcd_module.os, "fsync", failing_fsync)

    with pytest.raises(ValueError, match="filesystem") as raised:
        pcd_module.convert_pcd_to_bin(source, destination)

    assert isinstance(raised.value.__cause__, OSError)


def test_plain_imports_do_not_load_forbidden_runtime_modules() -> None:
    code = r"""
import sys
from importlib.metadata import version

assert version("pypcd4") == "1.4.3"
baseline_modules = set(sys.modules)
import transvision.dataset.resilient_v2x_pcd
import tools.resilient_v2x.prepare_data

forbidden_prefixes = (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "transvision.models",
)
forbidden_fragments = (
    "bev_pool",
    "voxel_layer",
    "custom_op",
    "cuda",
)
loaded = [
    name
    for name in set(sys.modules) - baseline_modules
    if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden_prefixes
    )
    or any(fragment in name.lower() for fragment in forbidden_fragments)
]
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert result.returncode == 0, result.stderr


def test_fixture_manifest_is_rejected_without_explicit_fixture_permission(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    manifest_module = importlib.import_module(
        "transvision.dataset.resilient_v2x_manifest"
    )
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    kwargs = _fixture_kwargs(root, output)
    module.prepare_manifest(**kwargs)

    with pytest.raises(manifest_module.ManifestError, match="fixture"):
        manifest_module.load_temporal_manifest(
            output,
            expected_split_hash=kwargs["expected_split_sha256"],
        )


@pytest.mark.parametrize(
    "stage",
    (
        "temp",
        "write",
        "file_fsync",
        "link",
        "directory_fsync",
        "cleanup",
    ),
)
def test_prepared_atomic_publication_faults_are_absent_or_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    destination = tmp_path / "prepared.bin"
    sentinel = tmp_path / ".sentinel.tmp-keep"
    sentinel.write_bytes(b"sentinel")
    _inject_atomic_failure(
        monkeypatch,
        pcd_module,
        stage,
        destination.name,
    )

    with pytest.raises(ValueError, match="injected") as raised:
        pcd_module.convert_pcd_to_bin(source, destination)
    assert isinstance(raised.value.__cause__, OSError)

    final_expected = stage in {"directory_fsync", "cleanup"}
    assert destination.exists() is final_expected
    if final_expected:
        assert (
            destination.read_bytes()
            == np.asarray(XYZI, dtype="<f4", order="C").tobytes()
        )
    assert sentinel.read_bytes() == b"sentinel"
    owned_temps = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(f".{destination.name}.tmp-")
    ]
    assert bool(owned_temps) is (stage == "cleanup")
    monkeypatch.undo()
    prepared = pcd_module.convert_pcd_to_bin(source, destination)
    assert prepared.size == 32
    assert destination.stat().st_size == 32
    for path in owned_temps:
        path.unlink()


def test_atomic_short_write_never_publishes_a_truncated_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    source = tmp_path / "source.pcd"
    source.write_bytes(_pcd_bytes(XYZI, encoding=Encoding.ASCII))
    destination = tmp_path / "prepared.bin"
    original_fdopen = pcd_module.os.fdopen

    class ShortWriter:
        def __init__(self, stream: object) -> None:
            self.stream = stream

        def __enter__(self) -> "ShortWriter":
            self.stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.stream.__exit__(*args)

        def write(self, data: bytes) -> int:
            self.stream.write(data[:-1])
            return len(data) - 1

        def flush(self) -> None:
            self.stream.flush()

        def fileno(self) -> int:
            return self.stream.fileno()

    monkeypatch.setattr(
        pcd_module.os,
        "fdopen",
        lambda *args, **kwargs: ShortWriter(original_fdopen(*args, **kwargs)),
    )

    with pytest.raises(ValueError, match="short write") as raised:
        pcd_module.convert_pcd_to_bin(source, destination)
    assert isinstance(raised.value.__cause__, OSError)

    assert not destination.exists()
    assert not [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(f".{destination.name}.tmp-")
    ]


@pytest.mark.parametrize(
    "stage",
    (
        "temp",
        "write",
        "file_fsync",
        "link",
        "directory_fsync",
        "cleanup",
    ),
)
def test_manifest_atomic_publication_faults_are_absent_or_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    pcd_module = importlib.import_module("transvision.dataset.resilient_v2x_pcd")
    manifest_module = importlib.import_module(
        "transvision.dataset.resilient_v2x_manifest"
    )
    root = _fixture_copy(tmp_path)
    seed_output = tmp_path / "seed.json"
    kwargs = _fixture_kwargs(root, seed_output)
    module.prepare_manifest(**kwargs)
    target = tmp_path / "target.json"
    kwargs["output_path"] = target
    sentinel = tmp_path / ".sentinel.tmp-keep"
    sentinel.write_bytes(b"sentinel")
    _inject_atomic_failure(monkeypatch, pcd_module, stage, target.name)

    with pytest.raises(ValueError, match="injected") as raised:
        module.prepare_manifest(**kwargs)
    assert isinstance(raised.value.__cause__, OSError)

    final_expected = stage in {"directory_fsync", "cleanup"}
    assert target.exists() is final_expected
    assert sentinel.read_bytes() == b"sentinel"
    owned_temps = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(f".{target.name}.tmp-")
    ]
    assert bool(owned_temps) is (stage == "cleanup")
    monkeypatch.undo()
    if final_expected:
        manifest_module.load_temporal_manifest(
            target,
            expected_split_hash=kwargs["expected_split_sha256"],
            allow_fixture=True,
        )
    recovered = module.prepare_manifest(**kwargs)
    assert recovered.content_sha256
    assert target.exists()
    for path in owned_temps:
        path.unlink()


def test_test_a_is_validated_but_not_emitted_as_a_fourth_split(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    split_path = root / "split.json"
    split = json.loads(split_path.read_text())
    sample_ids = split["cooperative_split"]["train"]
    split["cooperative_split"]["train"] = []
    split["cooperative_split"]["test"] = sample_ids
    split["cooperative_split"]["test_A"] = sample_ids[:2]
    _rewrite_json(split_path, split)

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))

    assert {sample.split for sample in manifest.samples} == {"test"}
    assert manifest.history_eligible_train_count == 0
    assert "test_A" not in output.read_text()


@pytest.mark.parametrize(
    "kind",
    ("payload", "pose", "label", "metadata", "calibration"),
)
@pytest.mark.parametrize("operation", ("missing", "symlink"))
def test_missing_or_symlinked_release_inputs_fail_before_publication(
    tmp_path: Path,
    kind: str,
    operation: str,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    targets = {
        "payload": (
            root / "vehicle-side/image/000000.jpg",
            root / "vehicle-side/image/000001.jpg",
        ),
        "pose": (
            root / "vehicle-side/calib/novatel_to_world/000000.json",
            root / "vehicle-side/calib/novatel_to_world/000001.json",
        ),
        "label": (
            root / "cooperative/label_world/000000.json",
            root / "cooperative/label_world/000001.json",
        ),
        "metadata": (
            root / "cooperative/data_info.json",
            root / "vehicle-side/data_info.json",
        ),
        "calibration": (
            root / "vehicle-side/calib/lidar_to_camera/000000.json",
            root / "vehicle-side/calib/lidar_to_camera/000001.json",
        ),
    }
    target, replacement = targets[kind]
    target.unlink()
    if operation == "symlink":
        target.symlink_to(replacement)

    with pytest.raises((ValueError, RuntimeError)):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    _assert_no_publication(root, output)


def test_raw_inventory_exactly_covers_selected_release_and_no_runtime_state(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"

    manifest = module.prepare_manifest(**_fixture_kwargs(root, output))

    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != root / "split.json"
        and "prepared" not in path.relative_to(root).parts
    }
    actual = {entry.relative_path for entry in manifest.release_inventory}
    assert actual == expected
    assert len(actual) == 51
    assert len(manifest.prepared_artifacts) == len(
        {item.source_relative_path for item in manifest.prepared_artifacts}
    )
    assert len(manifest.prepared_artifacts) == len(
        {item.prepared_relative_path for item in manifest.prepared_artifacts}
    )
    raw = output.read_bytes()
    assert str(root).encode() not in raw
    assert str(tmp_path).encode() not in raw
    assert b".tmp-" not in raw
    assert b'"pid"' not in raw
    assert b'"hostname"' not in raw
    assert b'"generated_at"' not in raw


def test_raw_inventory_drift_after_conversion_leaves_bins_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    label = root / "cooperative/label_world/000003.json"
    original_build = module.build_release_inventory
    calls = 0

    def drifting_inventory(
        inventory_root: Path,
        relative_paths: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 4:
            label.write_bytes(label.read_bytes() + b"\n")
        return original_build(inventory_root, relative_paths)

    monkeypatch.setattr(
        module,
        "build_release_inventory",
        drifting_inventory,
    )

    with pytest.raises(ValueError, match="changed"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    assert calls == 4
    assert not output.exists()
    bins = list((root / "prepared").rglob("*.bin"))
    assert len(bins) == 8
    assert all(path.stat().st_size == 32 for path in bins)


def test_split_drift_after_conversion_leaves_bins_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tools.resilient_v2x.prepare_data")
    root = _fixture_copy(tmp_path)
    output = tmp_path / "manifest.json"
    split = root / "split.json"
    original_read = module._read_stable_regular_bytes
    split_reads = 0

    def drifting_read(path: Path) -> bytes:
        nonlocal split_reads
        if Path(path) == split:
            split_reads += 1
            if split_reads == 2:
                split.write_bytes(split.read_bytes() + b"\n")
        return original_read(Path(path))

    monkeypatch.setattr(module, "_read_stable_regular_bytes", drifting_read)

    with pytest.raises(ValueError, match="split file changed"):
        module.prepare_manifest(**_fixture_kwargs(root, output))

    assert split_reads == 2
    assert not output.exists()
    assert len(list((root / "prepared").rglob("*.bin"))) == 8
