"""Bounded native ASCII XYZRGB -> XYZI, using the official red-channel recipe.

This is the observed V2V4Real release schema, not a generic PCL file reader.
No point removal, order change, ROI, ego mask, intensity scaling choice or pose
transform is performed. Unknown encodings fail instead of being guessed.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

import numpy as np

from .v2v4real_inputs import V2V4RealInputError

RECIPE = 'v2v4real-ascii-xyzrgb-f32-red-u8-div255-xyzi-f32-v1'
MAX_PCD_BYTES = 128 * 1024**2
MAX_POINTS = 2_000_000
HEADER_FIELDS = ('VERSION', 'FIELDS', 'SIZE', 'TYPE', 'COUNT', 'WIDTH', 'HEIGHT', 'VIEWPOINT', 'POINTS', 'DATA')


@dataclass(frozen=True)
class NativePointCloud:
    xyzi: np.ndarray
    rgb_bits: np.ndarray
    header: dict
    source_sha256: str


def decode_pcd(raw: bytes) -> NativePointCloud:
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PCD_BYTES:
        raise V2V4RealInputError('PCD input exceeds bounded file size')
    stream = io.BytesIO(raw)
    header = {}
    for _ in range(64):
        line = stream.readline(1025)
        if len(line) > 1024 or not line:
            raise V2V4RealInputError('missing or oversized PCD header line')
        try:
            tokens = line.decode('ascii').strip().split()
        except UnicodeError as exc:
            raise V2V4RealInputError('PCD header must be ASCII') from exc
        if not tokens or tokens[0].startswith('#'):
            continue
        key, values = tokens[0], tokens[1:]
        if key not in HEADER_FIELDS or key in header or not values:
            raise V2V4RealInputError('unknown, duplicate or empty PCD header field')
        header[key] = values
        if key == 'DATA':
            break
    if (tuple(header) != HEADER_FIELDS or header['VERSION'] not in (['.7'], ['0.7'])
            or header['FIELDS'] != ['x', 'y', 'z', 'rgb'] or header['SIZE'] != ['4'] * 4
            or header['TYPE'] != ['F'] * 4 or header['COUNT'] != ['1'] * 4 or header['DATA'] != ['ascii']):
        raise V2V4RealInputError('expected native PCD 0.7 ASCII XYZRGB float32 schema')
    dimensions = []
    for key in ('WIDTH', 'HEIGHT', 'POINTS'):
        value = header[key]
        if len(value) != 1 or re.fullmatch(r'[0-9]+', value[0]) is None:
            raise V2V4RealInputError('invalid PCD dimensions')
        dimensions.append(int(value[0]))
    width, height, points = dimensions
    if width < 1 or height < 1 or width * height != points or not 0 < points <= MAX_POINTS:
        raise V2V4RealInputError('inconsistent or oversized PCD point count')
    try:
        viewpoint = np.asarray(header['VIEWPOINT'], dtype=np.float64)
        if viewpoint.shape != (7,) or not np.isfinite(viewpoint).all():
            raise ValueError('invalid viewpoint')
        if abs(np.linalg.norm(viewpoint[3:]) - 1.) > 1e-6:
            raise ValueError('invalid viewpoint quaternion')
        # loadtxt checks every token and row, unlike prefix-tolerant fromstring.
        values = np.loadtxt(stream, dtype=np.float64, comments=None, ndmin=2, encoding='ascii')
        if values.shape != (points, 4) or not np.isfinite(values).all():
            raise ValueError('point shape or finiteness differs')
        with np.errstate(over='raise', invalid='raise'):
            rgb = np.ascontiguousarray(values[:, 3], dtype='<f4').view('<u4')
            xyzi = np.empty((points, 4), dtype=np.float32)
            xyzi[:, :3] = values[:, :3]
            xyzi[:, 3] = ((rgb >> 16) & 255).astype(np.float64) / 255.
        if not np.isfinite(xyzi).all():
            raise ValueError('nonfinite float32 output')
    except (ValueError, UnicodeError, FloatingPointError) as exc:
        raise V2V4RealInputError('invalid native PCD point payload or viewpoint') from exc
    xyzi.flags.writeable = rgb.flags.writeable = False
    return NativePointCloud(xyzi, rgb, header, hashlib.sha256(raw).hexdigest())


def read_native_pcd(path: Path, *, expected_sha256: str) -> NativePointCloud:
    """Consume only a hash-pinned point file; never search raw YAML/GT siblings."""
    from tools.event_track_v2x.prepare_v2v4real_inputs import _read
    if re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or '') is None:
        raise V2V4RealInputError('explicit point-cloud SHA-256 required')
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise V2V4RealInputError('point-cloud symlink is not permitted')
    raw = _read(path, MAX_PCD_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise V2V4RealInputError('point-cloud SHA-256 differs')
    return decode_pcd(raw)
