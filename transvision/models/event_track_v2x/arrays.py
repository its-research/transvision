"""Immutable NumPy storage for hashable EventTrack-V2X values."""

from __future__ import annotations

import numpy as np


def immutable_float64(value: object) -> np.ndarray:
    """Return a C-contiguous float64 array with irreversible read-only backing.

    NumPy arrays that own mutable memory can undo ``write=False``.  A view over
    Python ``bytes`` cannot, so this helper prevents post-validation mutation
    and also breaks aliases to caller-owned arrays.
    """

    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    result = result.reshape(contiguous.shape)
    result.setflags(write=False)
    return result


__all__ = ["immutable_float64"]
