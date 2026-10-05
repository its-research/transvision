"""Capture a committed-prefix counterexample; never grants replay acceptance."""
import argparse
import json
import math
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha


EXECUTION_DIFFERENCES = {
    ('scorer_binding',), ('previous_audit_sha256',), ('factor_rows_sha256',),
    ('cache_ingestion', 'configuration_sha256'),
}


def first_difference(left, right, path=()):
    if path in EXECUTION_DIFFERENCES:
        return None
    def different(reason):
        return dict(path=list(path), reason=reason, serial=left, batched=right)
    if type(left) is not type(right):
        return different('type')
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return different('keys')
        for key in left:
            difference = first_difference(left[key], right[key], path+(key,))
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return dict(path=list(path), reason='length', serial=len(left), batched=len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, path+(index,))
            if difference:
                return difference
    elif type(left) is float:
        tolerance = 1e-4 if path and path[0] == 'appended_rows' else 1e-8
        if not (math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)):
            return dict(**different('numeric'), atol=tolerance, rtol=tolerance,
                absolute_difference=abs(left-right))
    elif left != right:
        return different('discrete')
    return None


def complete_lines(path):
    lines = path.read_bytes().splitlines(keepends=True)
    return [line for line in lines if line.endswith(b'\n')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); pair=args.pair.resolve(); output=args.output.resolve()
    assert pair.is_relative_to(R/'artifacts') and output.is_relative_to(R/'artifacts')
    launch = json.loads((pair/'launch.json').read_bytes())
    assert sha(pair/'input-binding.json') == launch['binding_sha256']
    output.mkdir(exist_ok=False)
    results = {}
    for filename in ('predictions.jsonl', 'audit.jsonl'):
        serial = complete_lines(pair/'serial-replay'/filename)
        batched = complete_lines(pair/'batched-replay'/filename)
        n = min(len(serial), len(batched)); assert n > 0
        snapshots = {}
        for role, lines in [('serial',serial),('batched',batched)]:
            snapshot=output/(role+'-'+filename)
            with snapshot.open('xb') as stream:
                stream.write(b''.join(lines[:n]))
            snapshots[role] = dict(path=str(snapshot), sha256=sha(snapshot))
        difference = None
        for ordinal, (left, right) in enumerate(zip(serial[:n], batched[:n]), 1):
            difference = first_difference(json.loads(left), json.loads(right))
            if difference:
                difference['event_ordinal'] = ordinal
                break
        results[filename] = dict(common_complete_lines=n, first_difference=difference, snapshots=snapshots)
    receipt=output/'committed-prefix-diagnostic.json'
    new(receipt, dict(kind='rbf_serial_batched_committed_prefix_compatibility_diagnostic_v1',
        pair=str(pair), binding_sha256=sha(pair/'input-binding.json'), observer_sha256=sha(__file__),
        results=results, ignored_execution_binding_paths=[list(p) for p in sorted(EXECUTION_DIFFERENCES)],
        factors_atol_rtol=1e-4, other_numeric_atol_rtol=1e-8, discrete_comparison='exact',
        full_sequence_or_termination_claim=False, independent_fresh_state_or_search_acceptance=False,
        production_promotion_allowed=False, GPU_memory_target_admitted=False))
    register(receipt, 'rbf-batched-real-committed-prefix-compatibility-diagnostic')
    print(json.dumps(dict(receipt=str(receipt), results={k:{x:v for x,v in d.items() if x!='snapshots'} for k,d in results.items()})))


if __name__ == '__main__':
    main()
