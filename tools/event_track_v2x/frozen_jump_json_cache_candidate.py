"""Opt-in metadata parse cache; frozen checks and arithmetic still execute.

Only a pinned module's local json reference is proxied. The standard json module
is never patched. Eligible values are short, flat integer lists; each hit returns
a new list. Invalid JSON, hooks and all other data retain the original loader.
Reserve this cache inside the caller's existing total transport cache budget.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys


MAX_BYTES=2*1024**2
MAX_ENTRIES=1024
MAX_TEXT=512
MAX_INTS=64


class JumpJSON:
    def __init__(self, original, max_bytes, max_entries):
        if (type(max_bytes) is not int or not 0<max_bytes<=MAX_BYTES or
                type(max_entries) is not int or not 0<max_entries<=MAX_ENTRIES):
            raise ValueError('bounded metadata cache required')
        self.original=original
        self.max_bytes=max_bytes;self.max_entries=max_entries
        self.cache=OrderedDict()
        self.bytes=0;self.peak=0;self.hits=0;self.misses=0;self.bypasses=0

    def __getattr__(self, name):
        return getattr(self.original,name)

    def loads(self, text, *args, **kwargs):
        if (args or kwargs or type(text) is not str or len(text)>MAX_TEXT or
                not text.startswith('[') or not text.endswith(']')):
            self.bypasses+=1
            return self.original.loads(text,*args,**kwargs)
        if text in self.cache:
            self.hits+=1;self.cache.move_to_end(text)
            return list(self.cache[text][0])
        self.misses+=1
        value=self.original.loads(text)
        if type(value) is not list or len(value)>MAX_INTS or any(type(v) is not int for v in value):
            return value
        frozen=tuple(value)
        # Include immutable payload, key and conservative per-entry table/link
        # allowance. Count repeated immutable integers too rather than sharing.
        size=sys.getsizeof(text)+sys.getsizeof(frozen)+sum(sys.getsizeof(v) for v in frozen)+512
        if size<=self.max_bytes:
            while self.cache and (len(self.cache)>=self.max_entries or self.bytes+size>self.max_bytes):
                _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
            self.cache[text]=(frozen,size);self.bytes+=size;self.peak=max(self.peak,self.bytes)
        return value

    def info(self):
        return {'hits':self.hits,'misses':self.misses,'bypasses':self.bypasses,
            'entries':len(self.cache),'accounted_bytes':self.bytes,'peak_accounted_bytes':self.peak,
            'max_bytes':self.max_bytes,'max_entries':self.max_entries}

    def clear(self):
        self.cache.clear();self.bytes=0


@contextmanager
def bind_jump_json_cache(module, expected_sha256, *, max_bytes=MAX_BYTES,max_entries=MAX_ENTRIES):
    path=Path(module.__file__).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('frozen module source differs')
    if module.json is not json:
        raise ValueError('ambient or nested JSON proxy forbidden')
    proxy=JumpJSON(json,max_bytes,max_entries)
    original=module.json;module.json=proxy
    try:
        yield proxy
        if module.json is not proxy or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError('module or frozen source changed during cache binding')
    finally:
        module.json=original
        proxy.clear()
