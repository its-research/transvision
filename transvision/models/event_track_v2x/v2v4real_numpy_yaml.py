"""Decode the release's numerical YAML records without Python object loading.

These exact tags describe data. No named callable, dtype constructor from YAML,
pickle, eval, import lookup, or ndarray reconstruction function is executed.
"""
from dataclasses import dataclass
import struct

import numpy as np

PREFIX = 'tag:yaml.org,2002:'
DTYPE = PREFIX + 'python/object/apply:numpy.dtype'
ARRAY = PREFIX + 'python/object/apply:numpy.core.multiarray._reconstruct'
SCALAR = PREFIX + 'python/object/apply:numpy.core.multiarray.scalar'
ALIAS_TAGS = frozenset((DTYPE, ARRAY))
_ARRAY_SYMBOL = object()


@dataclass(frozen=True)
class FloatFormat:
    endian: str
    width: int


def install_numeric_constructors(loader, error):
    def tuple_data(reader, node):
        return tuple(reader.construct_sequence(node, deep=True))

    def dtype_data(reader, node):
        v = reader.construct_mapping(node, deep=True)
        args, state = v.get('args'), v.get('state')
        if (set(v) != {'args', 'state'} or type(args) is not list or len(args) != 3
                or args[0] not in ('f8', 'f4') or args[1] is not False or args[2] is not True
                or type(state) is not tuple or len(state) != 8
                or type(state[0]) is not int or state[0] != 3 or state[1] not in ('<', '>')
                or state[2:5] != (None, None, None) or state[5:] != (-1, -1, 0)
                or any(type(x) is not int for x in state[5:])):
            raise error('unsupported native numerical dtype record')
        return FloatFormat(state[1], int(args[0][1]))

    def decode(dtype, payload, count):
        if type(dtype) is not FloatFormat or type(payload) is not bytes or len(payload) != count * dtype.width:
            raise error('native numerical payload size/type differs')
        # Format is generated from the closed float-width/endian allowlist.
        return struct.unpack(dtype.endian + str(count) + ('d' if dtype.width == 8 else 'f'), payload)

    def scalar_data(reader, node):
        v = reader.construct_sequence(node, deep=True)
        if len(v) != 2:
            raise error('unsupported native scalar record')
        return decode(v[0], v[1], 1)[0]

    def array_symbol(reader, node):
        if reader.construct_scalar(node) != '':
            raise error('unsupported native array symbol')
        return _ARRAY_SYMBOL

    def array_data(reader, node):
        v = reader.construct_mapping(node, deep=True)
        args, state = v.get('args'), v.get('state')
        if (set(v) != {'args', 'state'} or type(args) is not list or len(args) != 3
                or args[0] is not _ARRAY_SYMBOL or args[1] != (0,) or args[2] != b'b'
                or type(args[1]) is not tuple or any(type(x) is not int for x in args[1])
                or type(state) is not tuple or len(state) != 5
                or type(state[0]) is not int or state[0] != 1
                or type(state[1]) is not tuple or state[1] not in ((4, 4), (6,))
                or any(type(x) is not int for x in state[1]) or state[3] is not False):
            raise error('unsupported native numerical array record')
        count = 16 if state[1] == (4, 4) else 6
        return np.asarray(decode(state[2], state[4], count), dtype=np.float64).reshape(state[1])

    for tag, constructor in ((PREFIX + 'python/tuple', tuple_data), (DTYPE, dtype_data),
                             (SCALAR, scalar_data), (PREFIX + 'python/name:numpy.ndarray', array_symbol),
                             (ARRAY, array_data)):
        loader.add_constructor(tag, constructor)
