"""Check one immutable learned replay database; full-cohort gates are separate."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import rbf_independent_learned_trajectory as oracle


def source_gate(directory):
    freeze = json.loads((directory / 'source-freeze.json').read_bytes())
    assert freeze['kind'] == 'rbf_independent_learned_search_trajectory_source_v1'
    for name, item in freeze['sources'].items():
        path = directory / name
        assert path.is_file() and path.parent.resolve() == directory.resolve()
        assert oracle.numeric.sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    for item in freeze['references']:
        assert oracle.numeric.sha(item['path']) == item['sha256']
    assert Path(oracle.__file__).resolve().parent == directory.resolve()
    return oracle.numeric.sha(directory / 'source-freeze.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('database', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('database-sha256', 'weights-sha256', 'policy-signature'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    freeze_sha = source_gate(directory)
    assert args.output.resolve().is_relative_to(oracle.ROOT / 'artifacts')
    assert not any(p.is_symlink() for p in (args.output, *args.output.parents))
    args.output.mkdir(parents=True, exist_ok=False)
    binding = dict(source_freeze_sha256=freeze_sha, database=str(args.database), weights=str(args.weights),
        expected_database_sha256=args.database_sha256, expected_weights_sha256=args.weights_sha256,
        expected_policy_signature=args.policy_signature, command=sys.argv,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_dataset_and_full_causal_trajectory_accepted=False, paper_performance_complete=False)

    def write(name, value):
        with (args.output / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')

    write('binding.json', binding)
    try:
        result = oracle.verify_database(args.database, args.database_sha256, args.weights,
            args.weights_sha256, args.policy_signature,
            progress=lambda value: print(json.dumps(value, allow_nan=False), flush=True))
        assert source_gate(directory) == freeze_sha
        result.update(binding)
        write('trajectory-check.json', result)
        print(json.dumps(dict(output=str(args.output / 'trajectory-check.json'), events=result['events'],
            candidate_feature_rows=result['candidate_feature_rows'],
            compensated_order_difference_count=len(result['independent_compensated_arithmetic_order_differences']),
            full_experiment_accepted=False)), flush=True)
    except BaseException as error:
        write('failure.json', dict(binding, exception_type=type(error).__name__, message=str(error), accepted=False))
        raise


if __name__ == '__main__':
    main()
