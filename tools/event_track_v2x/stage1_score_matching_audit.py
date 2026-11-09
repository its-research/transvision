"""Independent assignment optimality check for emitted-score reliability labels.

Uses an alternating-cycle residual graph, not the producer's Hungarian algorithm.
This is a floating-point diagnostic check, not a formal identity-risk certificate.
Real experiment matching/audits execute on admitted L40 CPU only.
"""
import math


def audit_assignment(similarities, labels, *, tolerance=1e-10):
    """Verify feasibility and absence of improving assignment cycles.

    Each unmatched GT is assigned its own zero-cost dummy column. Contracting
    occupied columns gives GT vertices; one extra vertex represents free columns.
    Negative cycles encode every possible cheaper reassignment, including paths
    that exchange an occupied column for a previously unused column.
    """
    ng, np = len(similarities), len(labels)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid predeclared tolerance')
    if any(len(row) != np for row in similarities):
        raise ValueError('nonrectangular similarities')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for row in similarities for x in row):
        raise ValueError('invalid similarity')
    assigned = {}
    for j, i in enumerate(labels):
        if i is None:
            continue
        if type(i) is not int or not 0 <= i < ng or i in assigned or similarities[i][j] < .5:
            raise ValueError('infeasible or duplicate match')
        assigned[i] = j
    if not ng:
        return dict(feasible=True, no_improving_cycle=True, matched_predictions=0,
                    numerical_tolerance=tolerance, formal_certificate=False)
    bonus = min(ng, np) + 1
    def cost(i, j):
        if j >= np or similarities[i][j] < .5:
            return 0.
        return -(bonus + similarities[i][j])
    columns = [assigned.get(i, np + i) for i in range(ng)]
    occupied = set(columns)
    free = [j for j in range(np + ng) if j not in occupied]
    edges = []
    for i, own in enumerate(columns):
        base = cost(i, own)
        edges.extend((i, k, cost(i, other) - base) for k, other in enumerate(columns))
        if free:
            edges.append((i, ng, min(cost(i, j) - base for j in free)))
    if free:
        edges.extend((ng, i, 0.) for i in range(ng))
    distance = [0.] * (ng + 1)
    for step in range(ng + 1):
        changed = False
        for a, b, weight in edges:
            improvement = distance[b] - (distance[a] + weight)
            if improvement > tolerance:
                distance[b] = distance[a] + weight
                changed = True
        if not changed:
            return dict(feasible=True, no_improving_cycle=True,
                        matched_predictions=len(assigned), matched_IoU_sum=math.fsum(
                            similarities[i][j] for i, j in assigned.items()),
                        relaxation_rounds=step + 1, numerical_tolerance=tolerance,
                        formal_certificate=False)
    raise ValueError('improving alternating assignment cycle; labels not optimal')
