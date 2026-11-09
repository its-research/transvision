"""Bounded legacy-payload cache candidate; frozen reader/oracles unchanged.

Cache only byte-identical outputs of the existing immutable-source decoder and
zlib level1 adapter. No numeric recomputation, filtering or row omission. This
is not enabled by default and does not modify any already-running process.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
from pathlib import Path
import zlib

import compact_teacher_reader as baseline

BASELINE_SHA='56d315746ef08186622898e7bf19723951a87b73c89cfcb1c1f6719cbdf84b10'
if hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest()!=BASELINE_SHA:
    raise ValueError('frozen baseline reader differs; qualify before using candidate')


def _validate(max_cache_bytes,max_cache_entries):
    if (type(max_cache_bytes) is not int or not 0<max_cache_bytes<=64*1024**2 or
            type(max_cache_entries) is not int or not 0<max_cache_entries<=1024):
        raise ValueError('explicit bounded payload cache required')


@contextmanager
def _cache(db,max_cache_bytes,max_cache_entries):
    cache=OrderedDict();state={'hits':0,'misses':0,'payload_bytes':0,'peak_payload_bytes':0}
    try:
        def payload(key):
            if key in cache:
                state['hits']+=1;cache.move_to_end(key);return cache[key]
            state['misses']+=1
            value=zlib.compress(baseline.decode(db._witness_source,key),1)
            if len(value)<=max_cache_bytes:
                while cache and (len(cache)>=max_cache_entries or
                        state['payload_bytes']+len(value)>max_cache_bytes):
                    _,old=cache.popitem(last=False);state['payload_bytes']-=len(old)
                cache[key]=value;state['payload_bytes']+=len(value)
                state['peak_payload_bytes']=max(state['peak_payload_bytes'],state['payload_bytes'])
            return value
        def info():
            return dict(state,entries=len(cache),max_cache_bytes=max_cache_bytes,
                        max_cache_entries=max_cache_entries)
        db.create_function('rbf_legacy_witness_payload',1,payload,deterministic=True)
        db.rbf_witness_cache_info=info
        yield db
    finally:
        cache.clear();state['payload_bytes']=0


@contextmanager
def open_compressed(*args,max_cache_bytes=64*1024**2,max_cache_entries=1024,**kwargs):
    _validate(max_cache_bytes,max_cache_entries)
    # Keep the frozen reader's complete SHA, scratch and failure-retention gates.
    with baseline.open_compressed(*args,**kwargs) as db:
        with _cache(db,max_cache_bytes,max_cache_entries):
            yield db


@contextmanager
def open_existing(path,expected_sha256,*,max_cache_bytes=64*1024**2,max_cache_entries=1024):
    """Read a SHA-pinned existing compact DB without copying or removing it."""
    _validate(max_cache_bytes,max_cache_entries)
    db=baseline.connect(path,expected_sha256)
    try:
        with _cache(db,max_cache_bytes,max_cache_entries):
            yield db
    finally:
        db.close()
