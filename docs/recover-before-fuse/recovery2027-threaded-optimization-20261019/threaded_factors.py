"""Opt-in candidate: bounded one-expansion root reuse and native CPU threads.

Does not replace the frozen production implementation. ctypes.CDLL releases
the GIL during native candidate checks. Search heap/order/budgets stay Python
original. Build the library separately; no automatic compile or deployment.
"""
import ctypes as C


def ints(values):
    values = tuple(values)
    return (C.c_int64 * len(values))(*values)


class ThreadedRestrictedFactors:
    def __init__(self, original, library, threads=4):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError('threads must be explicitly bounded to 1..64')
        self.original, self.nodes = original, original.nodes
        self.threads = threads
        self.lib = C.CDLL(str(library))
        i64 = C.POINTER(C.c_int64)
        u8 = C.POINTER(C.c_uint8)
        self.fn = self.lib.restricted_children
        self.fn.argtypes = [i64, C.c_int64, i64, C.c_int64, u8, i64,
                           C.c_int64, i64, i64, i64, i64, C.c_int, u8]
        self.fn.restype = C.c_int
        offsets, indices, voffsets, variants, counts = [0], [], [], [], []
        for ids, choices in original.restriction.clauses:
            indices.extend(ids); offsets.append(len(indices))
            voffsets.append(len(variants)); counts.append(len(choices))
            for choice in choices:
                variants.extend(choice)
        self.arrays = tuple(map(ints, (offsets, indices, voffsets, variants, counts)))
        self.clauses = len(counts)
        self.cached_prefix = None
        self.cached_roots = ()
        self.cached_parents = frozenset()
        self.max_native_workers = 0

    def roots(self, prefix):
        # Cache only children of the most recent expansion, never whole search.
        if (self.cached_prefix is not None and isinstance(prefix, tuple)
                and len(prefix) == len(self.cached_prefix)+1
                and prefix[:-1] == self.cached_prefix
                and type(prefix[-1]) is int and prefix[-1] in self.cached_parents):
            parent = prefix[-1]
            root = len(self.cached_roots) if parent == -1 else self.cached_roots[parent]
            return self.cached_roots + (root,)
        return self.original.roots(prefix)

    def children(self, prefix):
        roots = self.roots(prefix)
        depth = len(prefix)
        if depth == len(self.nodes):
            return ()
        parents = tuple(p for p, _ in self.original.factors.rows[depth])
        node = self.nodes[depth]
        blocked = (C.c_uint8 * (depth+1))()
        if node.source_id != -1:
            for i, root in enumerate(roots):
                previous = self.nodes[i]
                if (previous.source_id, previous.frame_id) == (node.source_id, node.frame_id):
                    blocked[root] = 1
        offsets, indices, voffsets, variants, counts = self.arrays
        output = (C.c_uint8 * len(parents))()
        workers = self.fn(ints(roots), depth, ints(parents), len(parents), blocked,
                          offsets, self.clauses, indices, voffsets, variants,
                          counts, self.threads, output)
        if workers < 0:
            raise RuntimeError('native candidate check failed: '+str(workers))
        self.max_native_workers = max(self.max_native_workers, workers)
        admitted = tuple(p for p, ok in zip(parents, output) if ok)
        self.cached_prefix, self.cached_roots = prefix, roots
        self.cached_parents = frozenset(admitted)
        return tuple(prefix + (p,) for p in admitted)
