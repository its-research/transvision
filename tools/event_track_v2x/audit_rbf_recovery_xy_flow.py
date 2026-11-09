"""Exact maximum-cardinality/minimum-distance diagnostic matching, not metrics."""
import math
from fractions import Fraction


def solve(matrix):
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    if any(len(row) != m for row in matrix):
        raise ValueError('ragged matrix')
    source, sink = n + m, n + m + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(a, b, cost):
        cost = Fraction(cost)
        forward = [b, len(graph[b]), 1, cost]
        reverse = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    for i in range(n):
        edge(source, i, 0.0)
    for j in range(m):
        edge(n + j, sink, 0.0)
    candidates = []
    for i, row in enumerate(matrix):
        for j, distance in enumerate(row):
            if not isinstance(distance, (int, float)) or distance < 0 or math.isnan(distance):
                raise ValueError('invalid distance')
            if distance < 2:
                candidates.append((i, j, distance, edge(i, n + j, distance)))
    while True:
        distances = [math.inf] * len(graph)
        previous = [None] * len(graph)
        distances[source] = Fraction(0)
        # Residual reverse edges may have negative costs. Bellman-Ford keeps
        # each augmentation exact without assuming nonnegative edge weights.
        for _ in range(len(graph) - 1):
            changed = False
            for a, edges in enumerate(graph):
                if not math.isfinite(distances[a]):
                    continue
                for k, e in enumerate(edges):
                    b, _, capacity, cost = e
                    if capacity and distances[a] + cost < distances[b]:
                        distances[b] = distances[a] + cost
                        previous[b] = (a, k)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        b = sink
        while b != source:
            a, k = previous[b]
            e = graph[a][k]
            e[2] = 0
            graph[b][e[1]][2] = 1
            b = a
    pairs = [(i, j, d) for i, j, d, e in candidates if e[2] == 0]
    return sum(d for _, _, d in pairs), pairs
