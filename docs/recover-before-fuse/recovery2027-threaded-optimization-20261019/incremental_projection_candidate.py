"""Isolated candidate: retain correlated variants along one immutable prefix path."""
import weakref


def projected_kernel_choice(restriction, kernel, handle, root):
    prefix = kernel._prefix(handle)
    depth = prefix.depth
    clauses = restriction.clauses
    state = getattr(restriction, '_incremental_projection_candidate', None)
    valid = state is not None and state[0]() is kernel and state[1] is clauses
    if not valid:
        positions = {i: (c, j) for c, (indices, _) in enumerate(clauses)
                     for j, i in enumerate(indices)}
    else:
        positions = state[4]
    if valid and state[2] == handle and state[3] == depth:
        survivors = state[5]
    elif (valid and depth == state[3] + 1
          and getattr(prefix, 'parent_handle', None) == state[2]):
        survivors = list(state[5])
        location = positions.get(depth - 1)
        if location is not None:
            c, j = location
            survivors[c] = tuple(v for v in survivors[c] if v[j] == prefix.root)
    else:
        # Branch jumps, owner changes and clause replacement rebuild exact state.
        survivors = []
        for indices, variants in clauses:
            projected = [(j, kernel._prefix(kernel.ancestor(handle, i + 1)).root)
                         for j, i in enumerate(indices) if i < depth]
            survivors.append(tuple(v for v in variants
                                   if all(v[j] == value for j, value in projected)))
    restriction._incremental_projection_candidate = (
        weakref.ref(kernel), clauses, handle, depth, positions, survivors)
    location = positions.get(depth)
    if location is None:
        return True
    c, j = location
    return any(v[j] == root for v in survivors[c])
