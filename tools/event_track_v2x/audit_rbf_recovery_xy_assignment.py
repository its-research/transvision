"""Reproduce an explicitly diagnostic XY assignment; never emits paper metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def solve(matrix):
    if not matrix or len(matrix[0]) > 20:
        raise ValueError('nonempty diagnostic matrix with at most 20 GT columns required')
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError('ragged matrix')
    states = {0: (0.0, [])}
    for i, row in enumerate(matrix):
        next_states = {mask: (cost, pairs[:]) for mask, (cost, pairs) in states.items()}
        for mask, (cost, pairs) in states.items():
            for j, distance in enumerate(row):
                if not isinstance(distance, (float, int)) or distance < 0 or distance != distance:
                    raise ValueError('invalid distance')
                if distance >= 2 or mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = cost + distance
                if new_mask not in next_states or new_cost < next_states[new_mask][0]:
                    next_states[new_mask] = (new_cost, pairs + [(i, j, distance)])
        states = next_states
    _, result = min(states.items(), key=lambda item: (-item[0].bit_count(), item[1][0], item[0]))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    cost, pairs = solve(data['XY_distance_matrix_m'])
    original = data['one_to_one_diagnostic']
    actual = [{'prediction_id': data['prediction_ids'][i], 'GT_id': data['GT_ids'][j],
               'distance_m': distance} for i, j, distance in pairs]
    if actual != original['matched_pairs'] or cost != original['distance_sum_m']:
        raise ValueError('stored assignment differs from recomputation')
    result = {'input_path': str(args.input.resolve()), 'input_sha256': hashlib.sha256(raw).hexdigest(),
              'reader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'assignment_reproduced': True, 'matched': len(pairs),
              'unmatched_predictions': len(data['prediction_ids']) - len(pairs),
              'unmatched_GT': len(data['GT_ids']) - len(pairs),
              'native_metric_engine_executed': False, 'formal_recovery_advantage_proven': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
