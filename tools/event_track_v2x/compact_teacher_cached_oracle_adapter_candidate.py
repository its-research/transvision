"""One bounded cache shared by all connections of pinned frozen oracles.

Original URI restrictions, physical SHA checks, module source pins, connection
cleanup and module restoration are delegated to the frozen adapter unchanged.
Only legacy payload reads are routed through a single immutable owner reader.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import compact_teacher_oracle_adapter as frozen
import compact_teacher_cached_reader_candidate as cached

PINS={frozen:'8fe811b32ba8a1e41813184a90f5eebb58335f06799affc7b485fb65ed24cdcb',
      cached:'cc3938b81c2468bf44d3e600c3b8fc083eb7f56aaa22a43d97247996817f3466'}
for module,expected in PINS.items():
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
        raise ValueError('qualified transport source differs')


@contextmanager
def bind_compact_sqlite(path,compact_sha256,pinned_modules,*,
                        max_cache_bytes=64*1024**2,max_cache_entries=1024):
    with cached.open_existing(path,compact_sha256,max_cache_bytes=max_cache_bytes,
                              max_cache_entries=max_cache_entries) as owner:
        with frozen.bind_compact_sqlite(path,compact_sha256,pinned_modules) as proxy:
            original_connect=proxy.connect
            def payload(key):
                row=owner.execute('SELECT payload FROM temp.priority_probe_witnesses WHERE sha=?',(key,)).fetchone()
                if row is None:raise ValueError('missing SHA-pinned legacy witness')
                return row[0]
            def connect(*args,**kwargs):
                db=original_connect(*args,**kwargs)
                db.create_function('rbf_legacy_witness_payload',1,payload,deterministic=True)
                return db
            proxy.connect=connect
            proxy.cache_info=owner.rbf_witness_cache_info
            try:
                yield proxy
            finally:
                proxy.connect=original_connect
