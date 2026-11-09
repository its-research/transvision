"""Independent scalar train matching, priors and full9 residual mathematics.

No producer fitting or matching functions are imported. Byte admission is a
shared IO-only layer; causal motion is the separate scalar mathematical oracle.
"""
import math
import numpy as np

SIDES = ('vehicle-side', 'infrastructure-side')


def match(boxes, scores, gt):
    used, output = set(), [-1]*len(boxes)
    for i in sorted(range(len(boxes)), key=lambda k: (-float(scores[k]), k)):
        distances = [(math.fsum((float(boxes[i][a])-float(g[a]))**2 for a in range(2)), j)
                     for j, g in enumerate(gt) if j not in used]
        allowed = [(d, j) for d, j in distances if d < 4.]
        if allowed:
            best = min(d for d, _ in allowed)
            winners = [j for d, j in allowed if d == best]
            if len(winners) == 1:
                output[i] = winners[0]
                used.add(winners[0])
    return output


def initial_prior(targets, groups):
    result = {}
    for side in SIDES:
        v = [list(map(float, value)) for r in targets if r['group'] in groups and r['side'] == side
             for value, valid in zip(r['velocity_source'], r['velocity_valid']) if valid]
        if len(v) < 10 or not all(math.isfinite(x) for row in v for x in row):
            raise ValueError('insufficient initial prior supervision')
        result[side] = dict(samples=len(v), velocity_source=[math.fsum(row[j] for row in v)/len(v)
                                                           for j in range(2)])
    return result


def angle(value):
    value = math.atan2(math.sin(value), math.cos(value))
    return value - 2*math.pi if value >= math.pi else value


def errors(row, prediction, gt_matches, prior):
    result = {'matched': [], 'unavailable': []}
    for i, j in enumerate(gt_matches):
        if j < 0 or not row['target_velocity_valid'][j]:
            continue
        g = row['target_states_source'][j]
        target = [float(g[k]) for k in (0, 1, 2, 4, 3, 5)]
        target.append(angle(-float(g[6])-math.pi/2))
        target.extend(map(float, row['target_velocity_source'][j]))
        valid = prediction['velocity_valid'][i]
        velocity = prediction['velocity_source'][i] if valid else prior[row['side']]['velocity_source']
        state = list(map(float, row['states_source_legacy7'][i])) + list(map(float, velocity))
        residual = [x-y for x, y in zip(state, target)]
        residual[6] = angle(residual[6])
        result['matched' if valid else 'unavailable'].append(residual)
    return result


def moments(residuals):
    n = len(residuals)
    if n < 10 or not all(math.isfinite(v) for r in residuals for v in r):
        raise ValueError('insufficient actual full9 residuals')
    mean = [math.fsum(r[a] for r in residuals)/n for a in range(9)]
    second = [[math.fsum(r[a]*r[b] for r in residuals)/n for b in range(9)] for a in range(9)]
    # Off-diagonal entries are independently accumulated, then shrunk by 0.95.
    cov = [[(second[a][b]+1e-6 if a == b else .95*second[a][b]) for b in range(9)]
           for a in range(9)]
    np.linalg.cholesky(np.asarray(cov))
    return dict(samples=n, residual_mean=mean, residual_second_moment=second, covariance=cov)
