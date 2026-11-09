"""Scalar mathematical oracle; does not import the detector motion producer.

Caller separately admits schema and byte provenance. This oracle reconstructs
world centres and unique mutual nearest assignment using Python scalar sums.
No model fitting, initial state substitution or official-test GT is performed.
"""
import math


def scalar_estimate(current, previous, *, max_speed_mps, max_gap_ordinals):
    boxes = current['states_source_legacy7']
    result = dict(velocity_source=[[math.nan, math.nan] for _ in boxes],
                  velocity_valid=[False for _ in boxes],
                  predecessor_candidates=[-1 for _ in boxes],
                  reasons=['no_predecessor' for _ in boxes],
                  dependencies=[current['row_sha256']])
    if previous is None:
        return result
    if (current['sequence_id'], current['side']) != (previous['sequence_id'], previous['side']):
        raise ValueError('cross-source/sequence predecessor')
    gap = current['frame_ordinal'] - previous['frame_ordinal']
    if gap <= 0:
        raise ValueError('noncausal predecessor')
    result['dependencies'].append(previous['row_sha256'])
    reason = 'gap_exceeded' if gap > max_gap_ordinals else 'no_allowed_predecessor'
    result['reasons'] = [reason for _ in boxes]
    if gap > max_gap_ordinals:
        return result

    def centres(frame):
        p = frame['source_to_world']
        return [[math.fsum([float(p[a][b]) * float(box[b]) for b in range(3)] + [float(p[a][3])])
                 for a in range(3)] for box in frame['states_source_legacy7']]

    now, old = centres(current), centres(previous)
    dt = gap / 10.0
    edges = {}
    for i, a in enumerate(now):
        for j, b in enumerate(old):
            if current['class_indices'][i] != previous['class_indices'][j]:
                continue
            distance = math.fsum((a[k] - b[k])**2 for k in range(3))
            if distance <= (max_speed_mps * dt)**2:
                edges[i, j] = distance
    row_best, column_best = {}, {}
    for (i, j), distance in edges.items():
        row_best[i] = min(row_best.get(i, math.inf), distance)
        column_best[j] = min(column_best.get(j, math.inf), distance)
    for i in range(len(now)):
        if i not in row_best:
            continue
        js = [j for (r, j), d in edges.items() if r == i and d == row_best[i]]
        if len(js) != 1:
            result['reasons'][i] = 'ambiguous_nearest'
            continue
        j = js[0]
        rows = [r for (r, c), d in edges.items() if c == j and d == column_best[j]]
        if len(rows) != 1:
            result['reasons'][i] = 'ambiguous_nearest'
        elif rows[0] != i:
            result['reasons'][i] = 'not_mutual_nearest'
        else:
            p = current['source_to_world']
            displacement = [(now[i][k] - old[j][k]) / dt for k in range(3)]
            result['velocity_source'][i] = [math.fsum(displacement[k] * float(p[k][a]) for k in range(3))
                                            for a in range(2)]
            result['velocity_valid'][i] = True
            result['predecessor_candidates'][i] = j
            result['reasons'][i] = 'matched'
    return result
