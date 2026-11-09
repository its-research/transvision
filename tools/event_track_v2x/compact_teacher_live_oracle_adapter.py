"""Scoped SQL transport for explicitly versioned live compact teacher oracles.

The oracle receives the actual compact database SHA, never the original SHA.
Original archive/database provenance and complete storage readback are separate
mandatory caller gates. This module alone cannot grant scientific acceptance.
Only explicitly pinned module-local sqlite3 bindings are replaced, temporarily.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3
from urllib.parse import parse_qsl, unquote, urlsplit

from compact_teacher_live_reader import connect


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


class _PinnedSQLite:
    def __init__(self, path, digest):
        self.path = Path(path).absolute()
        self.digest = digest
        self.connections = []
        self.open_count = 0

    def __getattr__(self, name):
        # Row and exception types keep their standard identities. Connections
        # must always go through the exact read-only entry point below.
        if name in ('Row', 'Error', 'DatabaseError', 'OperationalError',
                    'IntegrityError', 'ProgrammingError'):
            return getattr(sqlite3, name)
        raise AttributeError(name)

    def connect(self, database, *args, **kwargs):
        if args or kwargs != {'uri': True}:
            raise ValueError('only the frozen read-only URI call is supported')
        uri = urlsplit(str(database))
        query = parse_qsl(uri.query, keep_blank_values=True)
        if (uri.scheme != 'file' or uri.netloc or uri.fragment or
                query not in ([('mode', 'ro')], [('mode', 'ro'), ('immutable', '1')]) or
                unquote(uri.path) != str(self.path)):
            raise ValueError('oracle attempted an unbound or writable database')
        db = connect(self.path, self.digest)
        self.connections.append(db)
        self.open_count += 1
        return db

    def close(self):
        for db in self.connections:
            db.close()
        self.connections.clear()


@contextmanager
def bind_compact_sqlite(path, compact_sha256, pinned_modules):
    """Bind (module, source SHA) pairs in a single-threaded oracle process.

    Requires freshly loaded modules with standard sqlite3 bindings. No sha
    function, numerical constant, provenance claim or frozen source is changed.
    All opened connections are closed and bindings restored even after failure.
    The caller must perform the full independent storage and cohort gates.
    """
    path = Path(path).absolute()
    # Validate physical identity and sealing before replacing any binding.
    with_connection = connect(path, compact_sha256)
    with_connection.close()
    modules = list(pinned_modules)
    if not modules or len({id(m) for m, _ in modules}) != len(modules):
        raise ValueError('explicit distinct frozen modules required')
    for module, digest in modules:
        if file_sha(module.__file__) != digest:
            raise ValueError('frozen oracle source SHA differs')
        if getattr(module, 'sqlite3', None) is not sqlite3:
            raise ValueError('ambient or nested sqlite3 override forbidden')
    proxy = _PinnedSQLite(path, compact_sha256)
    try:
        for module, _ in modules:
            module.sqlite3 = proxy
        yield proxy
    finally:
        for module, _ in modules:
            module.sqlite3 = sqlite3
        proxy.close()
