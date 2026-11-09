"""Opt-in JSON wrapper reduction for a pinned frozen verifier module.

The same standard-library decoder parses every eligible call. No decoded data,
mathematical result or verifier decision is cached. Bind only the module's local
json name; leave the process-wide json module and frozen source bytes intact.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


class JumpJSONDecoder:
    def __init__(self):
        decoder = json._default_decoder
        if (type(decoder) is not json.JSONDecoder or
                decoder.object_hook is not None or
                decoder.object_pairs_hook is not None or
                decoder.parse_int is not int or decoder.parse_float is not float or
                decoder.strict is not True):
            raise ValueError('unaltered standard JSON decoder required')
        self._decoder = decoder
        self._raw_decode = decoder.raw_decode
        self._loads = json.loads
        self.dumps = json.dumps

    def __getattr__(self, name):
        return getattr(json, name)

    def loads(self, text, *args, **kwargs):
        # No whitespace prefix, BOM, custom hooks or byte decoding to reproduce.
        # Invalid input follows the original path, preserving exception details.
        if (not args and not kwargs and type(text) is str and
                len(text) <= 512 and text.startswith('[') and text.endswith(']')):
            try:
                value, end = self._raw_decode(text)
            except (ValueError, RecursionError):
                return self._loads(text)
            if end == len(text):
                return value
        return self._loads(text, *args, **kwargs)


@contextmanager
def bind_jump_json_decoder(module, expected_sha256):
    path = Path(module.__file__).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('frozen module source differs')
    if module.json is not json:
        raise ValueError('ambient or nested JSON proxy forbidden')
    proxy = JumpJSONDecoder()
    module.json = proxy
    try:
        yield proxy
        if (module.json is not proxy or json._default_decoder is not proxy._decoder or
                json.loads is not proxy._loads or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
            raise ValueError('module, decoder or frozen source changed during binding')
    finally:
        module.json = json
